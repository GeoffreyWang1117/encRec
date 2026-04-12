"""
KDD 2026 Rebuttal: Llama3.1 8B Large-Scale Experiment via vLLM.

Uses vLLM for high-throughput inference instead of Ollama/HuggingFace.
Supports both perplexity-based scoring and generation-based scoring.

Usage:
    # Using tensor_parallel across 2 GPUs
    python experiments/kdd_rebuttal_llama_vllm.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --samples 5000 --runs 3 --tp 2

    # Single GPU with smaller model
    python experiments/kdd_rebuttal_llama_vllm.py \
        --model meta-llama/Llama-3.2-3B \
        --samples 5000 --runs 3 --tp 1 --gpu 1
"""

import os
import sys
import json
import time
import hashlib
import logging
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
from datetime import datetime
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VLLMRanker:
    """High-throughput LLM ranker using vLLM.

    Two scoring modes:
    - 'perplexity': Score candidates by prompt perplexity (like GPT-2 baseline)
    - 'generation': Generate rankings via instruction prompting (native LLM mode)
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
        scoring: str = "generation",
        tp_size: int = 2,
        gpu_memory_utilization: float = 0.85,
        max_model_len: int = 1024,
    ):
        from vllm import LLM, SamplingParams
        self.scoring = scoring
        self.SamplingParams = SamplingParams

        logger.info(f"Loading vLLM model: {model_name} (tp={tp_size})")
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tp_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            dtype='half',
            enforce_eager=True,
        )
        self.tokenizer = self.llm.get_tokenizer()
        logger.info("vLLM model loaded successfully")

    def score_candidates_generation(
        self,
        history_titles: List[str],
        candidate_info: List[Tuple[str, str]],  # (id, title)
        k: int = 10,
    ) -> List[Tuple[str, float]]:
        """Score via generation: ask LLM to rank candidates."""
        history_text = ", ".join(history_titles[-5:])

        # Format candidate list
        cand_lines = []
        for i, (cid, title) in enumerate(candidate_info[:20]):
            cand_lines.append(f"{i+1}. [{cid}] {title[:40]}")
        cand_text = "\n".join(cand_lines)

        prompt = f"""Based on the user's reading history, rank the top {k} most relevant candidates.

User history: {history_text}

Candidates:
{cand_text}

Output only the candidate numbers in order of relevance, separated by commas. Example: 3,7,1,5,...
Ranking:"""

        params = self.SamplingParams(
            temperature=0.1,
            max_tokens=50,
            top_p=0.95,
        )
        outputs = self.llm.generate([prompt], params)
        response = outputs[0].outputs[0].text.strip()

        # Parse rankings from response
        scores = {}
        try:
            nums = [int(x.strip()) for x in response.split(",") if x.strip().isdigit()]
            for rank, num in enumerate(nums):
                if 1 <= num <= len(candidate_info):
                    cid = candidate_info[num - 1][0]
                    scores[cid] = 1.0 / (rank + 1)  # MRR-style scoring
        except:
            pass

        # Assign low scores to unranked candidates
        result = []
        for cid, title in candidate_info:
            result.append((cid, scores.get(cid, np.random.random() * 0.01)))

        return result

    def score_candidates_perplexity(
        self,
        history_titles: List[str],
        candidate_info: List[Tuple[str, str]],
    ) -> List[Tuple[str, float]]:
        """Score via perplexity: lower perplexity = better fit."""
        history_text = ", ".join(history_titles[-5:])

        prompts = []
        for cid, title in candidate_info:
            prompt = f"User liked: {history_text}. User will also like: {title}"
            prompts.append(prompt)

        # Use prompt_logprobs to get perplexity
        params = self.SamplingParams(
            temperature=0,
            max_tokens=1,
            prompt_logprobs=0,
        )
        outputs = self.llm.generate(prompts, params)

        scores = []
        for i, output in enumerate(outputs):
            # Sum log probs as perplexity proxy
            if output.prompt_logprobs:
                logprobs = [lp for lp in output.prompt_logprobs if lp is not None]
                if logprobs:
                    avg_logprob = np.mean([
                        max(lp.values()).logprob if lp else 0
                        for lp in logprobs
                    ])
                    scores.append((candidate_info[i][0], avg_logprob))
                    continue
            scores.append((candidate_info[i][0], np.random.random()))

        return scores

    def score_candidates(
        self,
        history_titles: List[str],
        candidate_info: List[Tuple[str, str]],
        k: int = 10,
    ) -> List[Tuple[str, float]]:
        if self.scoring == "generation":
            return self.score_candidates_generation(history_titles, candidate_info, k)
        else:
            return self.score_candidates_perplexity(history_titles, candidate_info)


def evaluate_recommendations(recs, ground_truth, k_values):
    metrics = {}
    for k in k_values:
        top_k = recs[:k]
        hit = 1.0 if ground_truth in top_k else 0.0
        metrics[f'hit@{k}'] = hit
        if ground_truth in top_k:
            rank = top_k.index(ground_truth) + 1
            metrics[f'ndcg@{k}'] = 1.0 / np.log2(rank + 1)
            metrics[f'mrr@{k}'] = 1.0 / rank
        else:
            metrics[f'ndcg@{k}'] = 0.0
            metrics[f'mrr@{k}'] = 0.0
    return metrics


def run_vllm_experiment(
    ranker: VLLMRanker,
    samples: List[Dict],
    news_items: Dict,
    all_items: List[str],
    seed: int = 42,
    candidate_pool_size: int = 20,
) -> Dict:
    """Run recommendation experiment with vLLM ranker."""
    np.random.seed(seed)
    logger.info(f"Running vLLM experiment: {len(samples)} samples, seed={seed}")

    all_metrics = defaultdict(list)
    latencies = []
    start_time = time.time()

    for i, sample in enumerate(samples):
        req_start = time.time()

        # Get history titles
        history_titles = [
            news_items.get(h, {}).get('title', '')
            for h in sample['history'][-5:]
        ]

        # Sample candidates
        neg_items = [it for it in all_items
                     if it != sample['ground_truth'] and it not in sample['history']]
        n_neg = min(candidate_pool_size - 1, len(neg_items))
        candidates = [sample['ground_truth']] + list(
            np.random.choice(neg_items, size=n_neg, replace=False))
        np.random.shuffle(candidates)

        candidate_info = [
            (cid, news_items.get(cid, {}).get('title', '')[:50])
            for cid in candidates
        ]

        # Score and rank
        scores = ranker.score_candidates(history_titles, candidate_info, k=10)
        scores.sort(key=lambda x: x[1], reverse=True)
        recs = [cid for cid, _ in scores[:10]]

        req_latency = (time.time() - req_start) * 1000
        latencies.append(req_latency)

        metrics = evaluate_recommendations(recs, sample['ground_truth'], [1, 3, 5, 10])
        for key, value in metrics.items():
            all_metrics[key].append(value)

        if (i + 1) % 200 == 0:
            hit5 = np.mean(all_metrics['hit@5'])
            avg_lat = np.mean(latencies)
            logger.info(f"  [{i+1}/{len(samples)}] Hit@5={hit5:.4f}, Latency={avg_lat:.1f}ms")

    total_time = time.time() - start_time

    results = {
        'n_samples': len(samples),
        'seed': seed,
        'total_time': total_time,
        'avg_latency_ms': float(np.mean(latencies)),
        'p50_latency_ms': float(np.percentile(latencies, 50)),
        'p95_latency_ms': float(np.percentile(latencies, 95)),
        'throughput_rps': len(samples) / total_time,
    }
    for key, values in all_metrics.items():
        results[key] = float(np.mean(values))
        results[f'{key}_std'] = float(np.std(values))

    logger.info(f"Done: Hit@5={results['hit@5']:.4f}, NDCG@5={results.get('ndcg@5', 0):.4f}, "
                 f"Latency={results['avg_latency_ms']:.1f}ms, Throughput={results['throughput_rps']:.2f}")
    return results


def main():
    parser = argparse.ArgumentParser(description="KDD Rebuttal: Llama3.1 via vLLM")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--scoring", type=str, default="generation",
                        choices=["generation", "perplexity"])
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--tp", type=int, default=2, help="Tensor parallel size")
    parser.add_argument("--gpu", type=int, default=None, help="Single GPU id (if tp=1)")
    parser.add_argument("--output", type=str, default="results/kdd_rebuttal_llama_vllm")
    args = parser.parse_args()

    if args.gpu is not None and args.tp == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info(f"KDD REBUTTAL: {args.model} via vLLM")
    logger.info(f"Scoring: {args.scoring}, Samples: {args.samples}, "
                f"Runs: {args.runs}, TP: {args.tp}")
    logger.info("=" * 60)

    # Load MIND dataset
    from src.data.mind_loader import load_mind_for_trie_experiment
    _, news_items, all_items = load_mind_for_trie_experiment(100)
    logger.info(f"Loaded {len(news_items)} news items")

    # Initialize vLLM ranker
    ranker = VLLMRanker(
        model_name=args.model,
        scoring=args.scoring,
        tp_size=args.tp,
    )

    seeds = [42, 123, 456, 789, 1024][:args.runs]
    all_results = []

    for seed in seeds:
        logger.info(f"\n{'='*40} Run seed={seed} {'='*40}")

        samples, _, _ = load_mind_for_trie_experiment(args.samples, seed=seed)
        samples_dicts = [{'user_id': s['user_id'], 'history': s['history'],
                          'ground_truth': s['ground_truth']} for s in samples]

        result = run_vllm_experiment(
            ranker, samples_dicts, news_items, all_items,
            seed=seed, candidate_pool_size=20)
        result['model'] = args.model
        result['scoring'] = args.scoring
        all_results.append(result)

    # Aggregate
    agg = {}
    metric_keys = [k for k in all_results[0] if k.startswith('hit@') or
                   k.startswith('ndcg@') or k.startswith('mrr@')]
    metric_keys = [k for k in metric_keys if not k.endswith('_std')]
    for k in metric_keys:
        vals = [r[k] for r in all_results]
        agg[k] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals))}

    lat_vals = [r['avg_latency_ms'] for r in all_results]
    tp_vals = [r['throughput_rps'] for r in all_results]

    final = {
        'model': args.model,
        'scoring': args.scoring,
        'n_samples': args.samples,
        'n_runs': args.runs,
        'tensor_parallel': args.tp,
        'timestamp': datetime.now().isoformat(),
        'aggregate': agg,
        'avg_latency_ms': {'mean': float(np.mean(lat_vals)), 'std': float(np.std(lat_vals))},
        'throughput_rps': {'mean': float(np.mean(tp_vals)), 'std': float(np.std(tp_vals))},
        'results': all_results,
    }

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short = args.model.split("/")[-1].replace("-", "_")
    out_path = output_dir / f"llama_vllm_{model_short}_{args.scoring}_{ts}.json"
    with open(out_path, 'w') as f:
        json.dump(final, f, indent=2)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 60)
    for k in ['hit@1', 'hit@5', 'hit@10', 'ndcg@5', 'mrr@5']:
        if k in agg:
            logger.info(f"  {k}: {agg[k]['mean']:.4f} ± {agg[k]['std']:.4f}")
    logger.info(f"  Latency: {np.mean(lat_vals):.1f} ± {np.std(lat_vals):.1f} ms")
    logger.info(f"  Throughput: {np.mean(tp_vals):.2f} ± {np.std(tp_vals):.2f} req/s")
    logger.info(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
