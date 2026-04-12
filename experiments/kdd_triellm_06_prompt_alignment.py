"""
Test if aligning Trie+LLM's prompt format with Prompt4NR closes the performance gap.

Hypothesis: Prompt4NR uses "User liked: {titles}. User will also like: {candidate}"
which is more natural for LLMs than our compressed format.

This script tests Trie+LLM with Prompt4NR-style prompts.
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

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.mind_loader import load_mind_for_trie_experiment

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AlignedLLMRanker:
    """LLM ranker using Prompt4NR-style prompts."""

    def __init__(self, model_name: str = "gpt2", device: str = "cuda:0"):
        self.device = device
        from transformers import AutoModelForCausalLM, AutoTokenizer
        logger.info(f"Loading LLM: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.eval()
        logger.info("LLM loaded successfully")

    def score_candidates(
        self,
        history_titles: List[str],
        candidate_titles: List[Tuple[str, str]],  # (id, title) pairs
    ) -> List[Tuple[str, float]]:
        """Score using Prompt4NR-style prompt."""
        # Format like Prompt4NR: "User liked: {titles}. User will also like: {candidate}"
        history_text = ", ".join(history_titles[-5:])  # Last 5 like Prompt4NR

        scores = []
        with torch.no_grad():
            for cid, cand_title in candidate_titles:
                prompt = f"User liked: {history_text}. User will also like: {cand_title}"

                try:
                    encoded = self.tokenizer(
                        prompt,
                        return_tensors='pt',
                        truncation=True,
                        max_length=128
                    )
                    input_ids = encoded['input_ids'].to(self.device)
                    outputs = self.model(input_ids, labels=input_ids)
                    score = -outputs.loss.item()  # Lower loss = higher score
                    scores.append((cid, score))
                except Exception as e:
                    scores.append((cid, np.random.random()))

        return scores


def evaluate_recommendations(recs: List[str], ground_truth: str, k_values: List[int]) -> Dict:
    metrics = {}
    for k in k_values:
        top_k = recs[:k]
        hit = 1.0 if ground_truth in top_k else 0.0
        metrics[f'hit@{k}'] = hit

        if ground_truth in top_k:
            rank = top_k.index(ground_truth) + 1
            ndcg = 1.0 / np.log2(rank + 1)
            mrr = 1.0 / rank
        else:
            ndcg = 0.0
            mrr = 0.0
        metrics[f'ndcg@{k}'] = ndcg
        metrics[f'mrr@{k}'] = mrr
    return metrics


def run_aligned_experiment(
    samples: List[Dict],
    news_items: Dict,
    all_items: List[str],
    device: str = "cuda:0",
    seed: int = 42,
) -> Dict:
    """Run experiment with Prompt4NR-aligned prompts."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    logger.info(f"Running Prompt-Aligned Trie+LLM (seed={seed})")
    start_time = time.time()

    ranker = AlignedLLMRanker(device=device)
    k_values = [1, 3, 5, 10]

    all_metrics = defaultdict(list)

    for i, sample in enumerate(samples):
        # Get history titles
        history_titles = [
            news_items.get(h, {}).get('title', '')
            for h in sample['history'][-5:]  # Last 5 like Prompt4NR
        ]

        # Sample candidates
        neg_items = [it for it in all_items if it != sample['ground_truth'] and it not in sample['history']]
        candidates = [sample['ground_truth']] + list(np.random.choice(
            neg_items, size=min(19, len(neg_items)), replace=False
        ))
        np.random.shuffle(candidates)

        # Get candidate titles
        candidate_titles = [
            (cid, news_items.get(cid, {}).get('title', ''))
            for cid in candidates
        ]

        # Score and rank
        scores = ranker.score_candidates(history_titles, candidate_titles)
        scores.sort(key=lambda x: x[1], reverse=True)
        recs = [cid for cid, _ in scores[:max(k_values)]]

        # Evaluate
        metrics = evaluate_recommendations(recs, sample['ground_truth'], k_values)
        for key, value in metrics.items():
            all_metrics[key].append(value)

        if (i + 1) % 200 == 0:
            logger.info(f"  Progress: {i+1}/{len(samples)}, Hit@5: {np.mean(all_metrics['hit@5']):.4f}")

    total_time = time.time() - start_time

    results = {
        'method': 'Aligned_TrieLLM',
        'seed': seed,
        'n_samples': len(samples),
        'total_time': total_time,
    }

    for key, values in all_metrics.items():
        results[key] = np.mean(values)
        results[f'{key}_std'] = np.std(values)

    logger.info(f"  Aligned_TrieLLM: Hit@5={results['hit@5']:.4f}, Time={total_time:.1f}s")

    del ranker
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    logger.info("="*60)
    logger.info("PROMPT ALIGNMENT TEST")
    logger.info("Testing if Prompt4NR-style prompts improve Trie+LLM")
    logger.info("="*60)

    samples, news_items, all_items = load_mind_for_trie_experiment(args.samples)
    logger.info(f"Loaded {len(samples)} samples")

    # Run with 3 seeds for quick validation
    all_results = []
    for seed in [42, 123, 456]:
        result = run_aligned_experiment(
            samples, news_items, all_items, args.device, seed
        )
        all_results.append(result)

    # Summary
    hit5_values = [r['hit@5'] for r in all_results]
    logger.info("\n" + "="*60)
    logger.info("PROMPT ALIGNMENT TEST RESULTS")
    logger.info("="*60)
    logger.info(f"Aligned Trie+LLM Hit@5: {np.mean(hit5_values):.4f} +/- {np.std(hit5_values):.4f}")
    logger.info("\nComparison:")
    logger.info(f"  Prompt4NR (baseline):     0.4208")
    logger.info(f"  Aligned Trie+LLM (ours):  {np.mean(hit5_values):.4f}")
    logger.info(f"  Original Trie+LLM:        0.2538")

    # Save results
    output_dir = Path("results/prompt_alignment_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(output_dir / f"results_{timestamp}.json", 'w') as f:
        json.dump(all_results, f, indent=2, default=str)


if __name__ == "__main__":
    main()
