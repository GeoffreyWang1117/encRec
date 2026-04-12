"""
KDD 2026 Rebuttal: Llama3.1 8B via HuggingFace Transformers.

Uses the SAME evaluation framework as GPT-2 (kdd_triellm_09_large_scale.py)
to ensure perfect controlled-variable comparison. Only the LLM backbone changes.

Supports both:
- 'perplexity' scoring (same as GPT-2 baseline)
- 'generation' scoring (native instruction-following)

Usage:
    CUDA_VISIBLE_DEVICES=1 python experiments/kdd_rebuttal_llama_hf.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --scoring generation \
        --samples 5000 --runs 3
"""

import os
import sys
import json
import time
import logging
import argparse
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import same infrastructure as GPT-2 experiments
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "ablation", str(Path(__file__).parent / "kdd_triellm_01_ablation.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
ExperimentConfig = _mod.ExperimentConfig
TrieStatistics = _mod.TrieStatistics
TrieLLMRecommender = _mod.TrieLLMRecommender
evaluate_recommendations = _mod.evaluate_recommendations


class LlamaRanker:
    """Llama3.1 ranker using HuggingFace Transformers.
    Drop-in replacement for LocalLLMRanker with the same interface."""

    def __init__(self, model_name="meta-llama/Llama-3.1-8B-Instruct",
                 device="cuda:0", scoring="generation"):
        self.device = device
        self.scoring = scoring
        from transformers import AutoModelForCausalLM, AutoTokenizer
        logger.info(f"Loading {model_name} on {device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map=device)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.eval()
        logger.info(f"Model loaded ({sum(p.numel() for p in self.model.parameters())/1e9:.1f}B params)")

    def score_candidates(self, history_text, candidate_texts, use_compression=True):
        """Same interface as LocalLLMRanker.score_candidates"""
        if self.scoring == "generation":
            return self._score_generation(history_text, candidate_texts)
        else:
            return self._score_perplexity(history_text, candidate_texts, use_compression)

    def _score_perplexity(self, history_text, candidate_texts, use_compression):
        """Perplexity scoring — identical to GPT-2 approach."""
        scores = []
        with torch.no_grad():
            for cid, cand_text in candidate_texts:
                if use_compression:
                    prompt = f"User interests: {history_text[:100]}. Recommend: {cand_text[:50]}"
                else:
                    prompt = f"Based on user reading history: {history_text}. Would they like: {cand_text}? Score:"
                try:
                    encoded = self.tokenizer(prompt, return_tensors='pt',
                                             truncation=True, max_length=256)
                    input_ids = encoded['input_ids'].to(self.device)
                    outputs = self.model(input_ids, labels=input_ids)
                    score = -outputs.loss.item()
                    scores.append((cid, score))
                except:
                    scores.append((cid, np.random.random()))
        return scores

    def _score_generation(self, history_text, candidate_texts):
        """Generation scoring — uses instruction-following capability."""
        # Build prompt
        cand_list = "\n".join([f"- [{cid}] {title[:40]}" for cid, title in candidate_texts[:20]])
        prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

Based on reading history: {history_text[:200]}

Rank these candidates by relevance:
{cand_list}

Output ONLY the IDs in order, comma-separated.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        scores = {}
        try:
            encoded = self.tokenizer(prompt, return_tensors='pt',
                                     truncation=True, max_length=512)
            input_ids = encoded['input_ids'].to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids, max_new_tokens=60,
                    temperature=0.1, do_sample=True, top_p=0.95)
            response = self.tokenizer.decode(outputs[0][input_ids.shape[1]:],
                                             skip_special_tokens=True)
            # Parse IDs from response
            for rank, token in enumerate(response.replace('\n', ',').split(',')):
                token = token.strip().strip('[]')
                for cid, _ in candidate_texts:
                    if cid in token or token in cid:
                        if cid not in scores:
                            scores[cid] = 1.0 / (rank + 1)
                        break
        except Exception as e:
            logger.warning(f"Generation failed: {e}")

        result = []
        for cid, title in candidate_texts:
            result.append((cid, scores.get(cid, np.random.random() * 0.01)))
        return result


def run_experiment(samples, news_items, all_items, ranker, seed, pool_size=20):
    """Run experiment with exact same protocol as GPT-2 large-scale."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    config = ExperimentConfig(
        n_samples=len(samples), candidate_pool_size=pool_size,
        device=str(ranker.device), early_exit_threshold=0.5, history_length=15)
    trie_stats = TrieStatistics(news_items, samples)

    # Create a wrapper that makes LlamaRanker look like LocalLLMRanker
    class RankerWrapper:
        def __init__(self, ranker):
            self._ranker = ranker
        def score_candidates(self, *args, **kwargs):
            return self._ranker.score_candidates(*args, **kwargs)

    recommender = TrieLLMRecommender(
        news_items=news_items, trie_stats=trie_stats,
        llm_ranker=RankerWrapper(ranker), config=config)

    all_metrics = defaultdict(list)
    latencies = []
    start_time = time.time()

    for i, sample in enumerate(samples):
        req_start = time.time()
        neg_items = [it for it in all_items if it != sample['ground_truth'] and it not in sample['history']]
        n_neg = min(pool_size - 1, len(neg_items))
        candidates = [sample['ground_truth']] + list(np.random.choice(neg_items, size=n_neg, replace=False))
        np.random.shuffle(candidates)

        recs, meta = recommender.recommend(
            history=sample['history'], candidates=candidates, k=10,
            use_early_exit=False, use_compression=True,
            use_trie_filtering=True, use_ctr_signals=False)

        latencies.append((time.time() - req_start) * 1000)
        metrics = evaluate_recommendations(recs, sample['ground_truth'], [1, 3, 5, 10])
        for key, value in metrics.items():
            all_metrics[key].append(value)

        if (i + 1) % 200 == 0:
            logger.info(f"  [{i+1}/{len(samples)}] Hit@5={np.mean(all_metrics['hit@5']):.4f}, "
                         f"Latency={np.mean(latencies):.1f}ms")

    total_time = time.time() - start_time
    results = {'n_samples': len(samples), 'seed': seed, 'total_time': total_time,
               'avg_latency_ms': float(np.mean(latencies)),
               'throughput_rps': len(samples) / total_time}
    for key, values in all_metrics.items():
        results[key] = float(np.mean(values))
        results[f'{key}_std'] = float(np.std(values))
    logger.info(f"Done: Hit@5={results['hit@5']:.4f}, Latency={results['avg_latency_ms']:.1f}ms")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--scoring", default="generation", choices=["generation", "perplexity"])
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", default="results/kdd_rebuttal_llama")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info(f"Llama Experiment: {args.model}, scoring={args.scoring}")
    logger.info(f"Samples: {args.samples}, Runs: {args.runs}")
    logger.info("=" * 60)

    from src.data.mind_loader import load_mind_for_trie_experiment
    _, news_items, all_items = load_mind_for_trie_experiment(100)
    ranker = LlamaRanker(args.model, args.device, args.scoring)

    seeds = [42, 123, 456][:args.runs]
    all_results = []
    for seed in seeds:
        logger.info(f"\n{'='*40} Run seed={seed} {'='*40}")
        samples, _, _ = load_mind_for_trie_experiment(args.samples, seed=seed)
        result = run_experiment(samples, news_items, all_items, ranker, seed)
        result['model'] = args.model
        result['scoring'] = args.scoring
        all_results.append(result)

    # Aggregate
    agg = {}
    for k in [k for k in all_results[0] if k.startswith(('hit@', 'ndcg@', 'mrr@')) and not k.endswith('_std')]:
        vals = [r[k] for r in all_results]
        agg[k] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals))}

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"llama_{args.scoring}_{ts}.json"
    json.dump({'model': args.model, 'scoring': args.scoring,
               'n_samples': args.samples, 'n_runs': args.runs,
               'aggregate': agg, 'results': all_results,
               'timestamp': datetime.now().isoformat()},
              open(out_path, 'w'), indent=2)

    logger.info("\n" + "=" * 60)
    logger.info("RESULTS:")
    for k in ['hit@5', 'ndcg@5', 'mrr@5']:
        if k in agg:
            logger.info(f"  {k}: {agg[k]['mean']:.4f} ± {agg[k]['std']:.4f}")
    logger.info(f"Saved: {out_path}")

    del ranker
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
