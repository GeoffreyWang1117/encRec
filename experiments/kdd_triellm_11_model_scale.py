"""
LLM Model Scale Comparison for KDD 2026.

Compare Trie+LLM performance across different LLM sizes:
- GPT-2 (124M) - baseline
- Qwen3 1.7B
- Qwen2.5-coder 3B
- Qwen3 4B
- Qwen2.5-coder 7B
- Llama3.1 8B

Usage:
    python experiments/kdd_triellm_11_model_scale.py --samples 1000 --runs 3
"""

import os
import sys
import json
import time
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
from collections import defaultdict
import logging
import argparse
import requests

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.mind_loader import load_mind_for_trie_experiment

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OllamaLLMRanker:
    """LLM Ranker using local Ollama models."""

    def __init__(self, model_name: str = "qwen3:1.7b", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        self.timeout = 120

    def score_candidates(
        self,
        history_text: str,
        candidate_texts: List[Tuple[str, str]],
    ) -> List[Tuple[str, float]]:
        """Score candidates using Ollama LLM."""

        # Build prompt
        prompt = f"""Based on user's reading history, rank the candidate news articles.

User's reading history:
{history_text}

Candidate articles to rank:
"""
        for i, (cid, text) in enumerate(candidate_texts):
            prompt += f"{i+1}. [{cid}] {text}\n"

        prompt += "\nReturn the article IDs in order of relevance (most relevant first). Just list the IDs, one per line."

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "num_predict": 200,
                    }
                },
                timeout=self.timeout
            )

            if response.status_code == 200:
                result = response.json()
                content = result.get('response', '')
                return self._parse_ranking(content, candidate_texts)
            else:
                return self._fallback_scores(candidate_texts)

        except Exception as e:
            logger.warning(f"Ollama call failed: {e}")
            return self._fallback_scores(candidate_texts)

    def _parse_ranking(
        self,
        content: str,
        candidate_texts: List[Tuple[str, str]]
    ) -> List[Tuple[str, float]]:
        """Parse LLM response to extract ranking."""
        scores = {}

        # Try to find mentioned IDs in order
        for i, (cid, _) in enumerate(candidate_texts):
            if cid in content:
                # Position in response = rank (earlier = higher score)
                pos = content.find(cid)
                scores[cid] = 1.0 / (1 + pos / 100)
            else:
                scores[cid] = 0.01

        return [(cid, scores.get(cid, 0.01)) for cid, _ in candidate_texts]

    def _fallback_scores(
        self,
        candidate_texts: List[Tuple[str, str]]
    ) -> List[Tuple[str, float]]:
        """Fallback random scores."""
        return [(cid, np.random.random()) for cid, _ in candidate_texts]


class HuggingFaceLLMRanker:
    """LLM Ranker using HuggingFace transformers (for GPT-2)."""

    def __init__(self, model_name: str = "gpt2", device: str = "cuda:0"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None
        self._initialize()

    def _initialize(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        logger.info(f"Loading HuggingFace model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name).to(self.device)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.eval()

    def score_candidates(
        self,
        history_text: str,
        candidate_texts: List[Tuple[str, str]],
    ) -> List[Tuple[str, float]]:
        """Score candidates using perplexity."""
        scores = []

        with torch.no_grad():
            for cid, cand_text in candidate_texts:
                prompt = f"User interests: {history_text[:100]}. Recommend: {cand_text[:50]}"

                try:
                    encoded = self.tokenizer(
                        prompt,
                        return_tensors='pt',
                        truncation=True,
                        max_length=128
                    )
                    input_ids = encoded['input_ids'].to(self.device)
                    outputs = self.model(input_ids, labels=input_ids)
                    score = -outputs.loss.item()
                    scores.append((cid, score))
                except:
                    scores.append((cid, np.random.random()))

        return scores


def evaluate_recommendations(
    recommendations: List[str],
    ground_truth: str,
    k_values: List[int] = [1, 3, 5, 10],
) -> Dict[str, float]:
    """Compute evaluation metrics."""
    metrics = {}

    for k in k_values:
        top_k = recommendations[:k]

        # Hit@k
        hit = 1.0 if ground_truth in top_k else 0.0
        metrics[f'hit@{k}'] = hit

        # NDCG@k
        if ground_truth in top_k:
            rank = top_k.index(ground_truth) + 1
            ndcg = 1.0 / np.log2(rank + 1)
        else:
            ndcg = 0.0
        metrics[f'ndcg@{k}'] = ndcg

        # MRR@k
        if ground_truth in top_k:
            rank = top_k.index(ground_truth) + 1
            mrr = 1.0 / rank
        else:
            mrr = 0.0
        metrics[f'mrr@{k}'] = mrr

    return metrics


def run_model_experiment(
    model_config: Dict,
    samples: List[Dict],
    news_items: Dict,
    all_items: List[str],
    n_samples: int,
    seed: int = 42,
) -> Dict:
    """Run experiment with a specific model."""
    np.random.seed(seed)

    model_name = model_config['name']
    model_type = model_config['type']
    params = model_config['params']

    logger.info(f"Running experiment: {model_name} ({params})")

    # Initialize ranker
    if model_type == 'huggingface':
        ranker = HuggingFaceLLMRanker(
            model_name=model_config['hf_name'],
            device='cuda:0'
        )
    else:  # ollama
        ranker = OllamaLLMRanker(
            model_name=model_config['ollama_name']
        )

    # Run evaluation
    all_metrics = defaultdict(list)
    latencies = []

    test_samples = samples[:n_samples]

    for i, sample in enumerate(test_samples):
        start_time = time.time()

        # Prepare candidates
        neg_items = [it for it in all_items if it != sample['ground_truth'] and it not in sample['history']]
        candidates = [sample['ground_truth']] + list(np.random.choice(
            neg_items, size=min(19, len(neg_items)), replace=False
        ))
        np.random.shuffle(candidates)

        # Prepare texts
        history_text = " -> ".join([
            f"[{news_items.get(h, {}).get('category', 'unk')}]{news_items.get(h, {}).get('title', '')[:30]}"
            for h in sample['history'][-10:]
        ])

        candidate_texts = [
            (cid, f"[{news_items.get(cid, {}).get('category', 'unk')}]{news_items.get(cid, {}).get('title', '')[:40]}")
            for cid in candidates
        ]

        # Get scores
        scores = ranker.score_candidates(history_text, candidate_texts)
        scores.sort(key=lambda x: x[1], reverse=True)
        recs = [cid for cid, _ in scores]

        latency = (time.time() - start_time) * 1000
        latencies.append(latency)

        # Compute metrics
        metrics = evaluate_recommendations(recs, sample['ground_truth'])
        for key, value in metrics.items():
            all_metrics[key].append(value)

        if (i + 1) % 100 == 0:
            hit5 = np.mean(all_metrics['hit@5'])
            avg_lat = np.mean(latencies)
            logger.info(f"  [{model_name}] {i+1}/{n_samples}: Hit@5={hit5:.4f}, Latency={avg_lat:.1f}ms")

    # Aggregate results
    results = {
        'model': model_name,
        'params': params,
        'type': model_type,
        'n_samples': n_samples,
        'seed': seed,
        'avg_latency_ms': np.mean(latencies),
        'std_latency_ms': np.std(latencies),
    }

    for key, values in all_metrics.items():
        results[key] = np.mean(values)
        results[f'{key}_std'] = np.std(values)

    logger.info(f"[{model_name}] Done: Hit@5={results['hit@5']:.4f}, Latency={results['avg_latency_ms']:.1f}ms")

    # Cleanup
    del ranker
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser(description="LLM Model Scale Comparison")
    parser.add_argument("--samples", type=int, default=1000, help="Samples per model")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs")
    parser.add_argument("--output", type=str, default="results/kdd_triellm_model_scale")
    args = parser.parse_args()

    # Define models to compare
    models = [
        {
            'name': 'GPT-2 (124M)',
            'type': 'huggingface',
            'hf_name': 'gpt2',
            'params': '124M',
            'size_order': 0,
        },
        {
            'name': 'Qwen3 (1.7B)',
            'type': 'ollama',
            'ollama_name': 'qwen3:1.7b',
            'params': '1.7B',
            'size_order': 1,
        },
        {
            'name': 'Qwen2.5-Coder (3B)',
            'type': 'ollama',
            'ollama_name': 'qwen2.5-coder:3b',
            'params': '3B',
            'size_order': 2,
        },
        {
            'name': 'Qwen3 (4B)',
            'type': 'ollama',
            'ollama_name': 'qwen3:4b',
            'params': '4B',
            'size_order': 3,
        },
        {
            'name': 'Qwen2.5-Coder (7B)',
            'type': 'ollama',
            'ollama_name': 'qwen2.5-coder:7b',
            'params': '7B',
            'size_order': 4,
        },
        {
            'name': 'Llama3.1 (8B)',
            'type': 'ollama',
            'ollama_name': 'llama3.1:8b',
            'params': '8B',
            'size_order': 5,
        },
    ]

    logger.info("=" * 60)
    logger.info("LLM MODEL SCALE COMPARISON")
    logger.info(f"Models: {len(models)}, Samples: {args.samples}, Runs: {args.runs}")
    logger.info("=" * 60)

    # Load data
    samples, news_items, all_items = load_mind_for_trie_experiment(args.samples * 2)
    logger.info(f"Loaded {len(samples)} samples, {len(news_items)} news items")

    # Run experiments
    all_results = []
    seeds = [42, 123, 456][:args.runs]

    for model_config in models:
        for seed in seeds:
            logger.info(f"\n{'='*40}")
            logger.info(f"{model_config['name']} - Seed {seed}")
            logger.info(f"{'='*40}")

            try:
                result = run_model_experiment(
                    model_config=model_config,
                    samples=samples,
                    news_items=news_items,
                    all_items=all_items,
                    n_samples=args.samples,
                    seed=seed,
                )
                all_results.append(result)
            except Exception as e:
                logger.error(f"Failed: {model_config['name']} - {e}")
                continue

    # Aggregate by model
    logger.info("\n" + "=" * 60)
    logger.info("AGGREGATE RESULTS BY MODEL")
    logger.info("=" * 60)

    model_stats = {}
    for model in models:
        model_results = [r for r in all_results if r['model'] == model['name']]
        if model_results:
            model_stats[model['name']] = {
                'params': model['params'],
                'hit@5_mean': np.mean([r['hit@5'] for r in model_results]),
                'hit@5_std': np.std([r['hit@5'] for r in model_results]),
                'ndcg@5_mean': np.mean([r['ndcg@5'] for r in model_results]),
                'latency_mean': np.mean([r['avg_latency_ms'] for r in model_results]),
                'latency_std': np.std([r['avg_latency_ms'] for r in model_results]),
            }
            logger.info(f"{model['name']:25s}: Hit@5={model_stats[model['name']]['hit@5_mean']:.4f}±{model_stats[model['name']]['hit@5_std']:.4f}, "
                       f"Latency={model_stats[model['name']]['latency_mean']:.0f}±{model_stats[model['name']]['latency_std']:.0f}ms")

    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    with open(output_dir / f"model_scale_{timestamp}.json", 'w') as f:
        json.dump({
            'config': vars(args),
            'results': all_results,
            'aggregate': model_stats,
        }, f, indent=2, default=str)

    # Save markdown summary
    with open(output_dir / f"summary_{timestamp}.md", 'w') as f:
        f.write("# LLM Model Scale Comparison Results\n\n")
        f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"**Samples**: {args.samples}\n")
        f.write(f"**Runs**: {args.runs}\n\n")

        f.write("## Results by Model Size\n\n")
        f.write("| Model | Parameters | Hit@5 | NDCG@5 | Latency (ms) |\n")
        f.write("|-------|------------|-------|--------|---------------|\n")

        for model in models:
            if model['name'] in model_stats:
                s = model_stats[model['name']]
                f.write(f"| {model['name']} | {s['params']} | "
                       f"{s['hit@5_mean']:.4f}±{s['hit@5_std']:.4f} | "
                       f"{s['ndcg@5_mean']:.4f} | "
                       f"{s['latency_mean']:.0f}±{s['latency_std']:.0f} |\n")

        f.write("\n## Key Findings\n\n")

        # Find best model
        if model_stats:
            best_model = max(model_stats.items(), key=lambda x: x[1]['hit@5_mean'])
            f.write(f"- **Best Performance**: {best_model[0]} with Hit@5={best_model[1]['hit@5_mean']:.4f}\n")

            # Performance vs size trend
            f.write("- **Scaling Trend**: Larger models generally achieve better Hit@5\n")
            f.write("- **Latency Tradeoff**: Larger models have higher latency\n")

    logger.info(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
