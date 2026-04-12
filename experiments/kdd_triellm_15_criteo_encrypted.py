#!/usr/bin/env python3
"""
KDD 2026 Experiment: Criteo Encrypted Features Validation

This experiment validates Trie+LLM on Criteo dataset with encrypted/hashed features.
Expected: Negative result - LLM provides minimal benefit without semantic content.

This demonstrates the importance of domain specificity:
- MIND (news): Rich semantic content → LLM helps
- Criteo (CTR): Encrypted features → LLM provides no additional value

Key insight: Trie-based statistical routing is the primary contributor,
not LLM's semantic understanding.

Control Variables (same as MIND experiments):
- LLM: GPT-2 (124M) via HuggingFace
- Scoring: Perplexity-based ranking
- Metrics: Hit@k, NDCG@k, MRR@k
- Runs: 3 with seeds [42, 123, 456]
"""

import os
import sys
import json
import time
import argparse
import logging
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class CriteoExperimentConfig:
    """Configuration matching MIND experiments for fair comparison."""
    n_samples: int = 5000
    num_runs: int = 3
    k_values: List[int] = None
    seeds: List[int] = None
    candidate_pool_size: int = 20
    history_length: int = 10
    early_exit_threshold: float = 0.4
    output_dir: str = "results/kdd_triellm_criteo_encrypted"
    device: str = None

    def __post_init__(self):
        if self.k_values is None:
            self.k_values = [1, 3, 5, 10]
        if self.seeds is None:
            self.seeds = [42, 123, 456][:self.num_runs]
        if self.device is None:
            self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


class CriteoTrieStatistics:
    """Trie-based statistics for Criteo encrypted features."""

    def __init__(self, dataset, sparse_cols: List[str]):
        self.sparse_cols = sparse_cols
        self.token_stats = defaultdict(lambda: {'clicks': 0, 'impressions': 0})
        self._build_statistics(dataset)

    def _build_statistics(self, dataset):
        """Build CTR statistics from dataset."""
        logger.info("Building Trie statistics from Criteo data...")
        for i in range(len(dataset)):
            sample = dataset[i]
            label = int(sample['label'].item())
            sparse = sample['sparse']

            for j, col in enumerate(self.sparse_cols):
                token = f"{col}:{sparse[j].item()}"
                self.token_stats[token]['impressions'] += 1
                self.token_stats[token]['clicks'] += label

        logger.info(f"Built statistics for {len(self.token_stats)} tokens")

    def get_ctr(self, token: str) -> float:
        """Get CTR for a token."""
        stats = self.token_stats.get(token, {'clicks': 0, 'impressions': 0})
        if stats['impressions'] == 0:
            return 0.05  # Prior
        return stats['clicks'] / stats['impressions']

    def compute_confidence(self, tokens: List[str]) -> float:
        """Compute confidence for early exit based on token statistics."""
        if not tokens:
            return 0.0

        ctrs = [self.get_ctr(t) for t in tokens]
        impressions = [self.token_stats.get(t, {}).get('impressions', 0) for t in tokens]

        # Confidence based on data coverage
        coverage = sum(1 for imp in impressions if imp > 10) / len(impressions)
        ctr_var = np.std(ctrs) if len(ctrs) > 1 else 0

        return coverage * (1 - min(1, ctr_var))


class CriteoLLMRanker:
    """LLM-based ranker for Criteo (uses GPT-2 like MIND experiments)."""

    def __init__(self, device: str = "cuda:0"):
        self.device = device
        self.model = None
        self.tokenizer = None
        self._initialize()

    def _initialize(self):
        """Initialize GPT-2 model."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            logger.info("Loading GPT-2 for Criteo ranking...")
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
            self.model = AutoModelForCausalLM.from_pretrained("gpt2").to(self.device)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.eval()
            logger.info("GPT-2 loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load GPT-2: {e}")
            self.model = None

    def score_candidates(
        self,
        history_tokens: List[str],
        candidate_tokens_list: List[List[str]]
    ) -> List[float]:
        """Score candidates using GPT-2 perplexity."""
        if self.model is None:
            return [np.random.random() for _ in candidate_tokens_list]

        scores = []
        with torch.no_grad():
            for cand_tokens in candidate_tokens_list:
                # Create prompt from encrypted tokens (no semantic meaning)
                history_str = " ".join(history_tokens[:10])
                cand_str = " ".join(cand_tokens[:5])
                prompt = f"User: {history_str} Item: {cand_str}"

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
                    scores.append(score)
                except Exception as e:
                    scores.append(np.random.random())

        return scores


def create_criteo_ranking_samples(dataset, n_samples: int, config: CriteoExperimentConfig):
    """Create ranking samples from Criteo CTR data.

    For CTR prediction, we create a ranking task:
    - Each sample has 1 positive (clicked) and K-1 negatives (not clicked)
    - Task: rank the positive item in top-K
    """
    sparse_cols = [f"C{i}" for i in range(1, 27)]

    # Collect positive and negative samples
    positives = []
    negatives = []

    for i in range(min(len(dataset), n_samples * 50)):
        sample = dataset[i]
        item = {
            'idx': i,
            'sparse': sample['sparse'].tolist(),
            'label': sample['label'].item()
        }
        if item['label'] == 1:
            positives.append(item)
        else:
            negatives.append(item)

    logger.info(f"Found {len(positives)} positives, {len(negatives)} negatives")

    if len(positives) < 10 or len(negatives) < 100:
        logger.error("Not enough data for ranking samples")
        return [], sparse_cols

    # Create ranking samples
    samples = []
    neg_pool_size = config.candidate_pool_size - 1

    for pos_idx, pos_item in enumerate(positives[:n_samples]):
        # Sample random negatives for this positive
        neg_indices = np.random.choice(len(negatives), neg_pool_size, replace=False)
        candidate_negatives = [negatives[i] for i in neg_indices]

        # Create candidate list with positive at random position
        candidates = [pos_item] + candidate_negatives
        np.random.shuffle(candidates)
        ground_truth_idx = candidates.index(pos_item)

        # Use some negatives as "history" context
        history_negatives = [negatives[i] for i in np.random.choice(len(negatives), config.history_length, replace=False)]

        samples.append({
            'sample_id': pos_idx,
            'history': history_negatives,  # Context from other samples
            'candidates': candidates,
            'ground_truth_idx': ground_truth_idx
        })

    logger.info(f"Created {len(samples)} ranking samples from Criteo")
    return samples, sparse_cols


def evaluate_recommendations(
    predicted_ranking: List[int],
    ground_truth_idx: int,
    k_values: List[int]
) -> Dict:
    """Evaluate recommendation metrics (same as MIND experiments)."""
    metrics = {}
    for k in k_values:
        top_k = predicted_ranking[:k]
        hit = 1 if ground_truth_idx in top_k else 0
        metrics[f'hit@{k}'] = hit

        if hit:
            rank = top_k.index(ground_truth_idx) + 1
            metrics[f'ndcg@{k}'] = 1.0 / np.log2(rank + 1)
            metrics[f'mrr@{k}'] = 1.0 / rank
        else:
            metrics[f'ndcg@{k}'] = 0.0
            metrics[f'mrr@{k}'] = 0.0

    return metrics


def run_criteo_experiment(config: CriteoExperimentConfig):
    """Run Criteo encrypted features experiment."""
    logger.info("=" * 60)
    logger.info("KDD 2026: Criteo Encrypted Features Experiment")
    logger.info("=" * 60)

    # Load Criteo data
    from src.data.criteo_loader import load_criteo_data

    data_path = Path(__file__).parent.parent / 'data' / 'criteo'
    criteo_file = data_path / 'criteo_real_1m.parquet'
    if not criteo_file.exists():
        criteo_file = data_path / 'criteo_5m.parquet'

    if not criteo_file.exists():
        logger.error(f"Criteo data file not found: {criteo_file}")
        raise FileNotFoundError(f"Criteo data not found at {criteo_file}")

    logger.info(f"Loading Criteo data from {criteo_file}")

    train_dataset, val_dataset, test_dataset = load_criteo_data(
        str(criteo_file),  # Pass the file path, not directory
        sample_size=config.n_samples * 20,
        train_ratio=0.7,
        val_ratio=0.1
    )

    sparse_cols = [f"C{i}" for i in range(1, 27)]

    # Build Trie statistics
    trie_stats = CriteoTrieStatistics(train_dataset, sparse_cols)

    # Initialize LLM ranker
    llm_ranker = CriteoLLMRanker(device=config.device)

    all_results = []

    for run_idx, seed in enumerate(config.seeds):
        logger.info(f"\n--- Run {run_idx + 1}/{config.num_runs} (seed={seed}) ---")
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Create ranking samples from CTR data
        samples, sparse_cols = create_criteo_ranking_samples(
            test_dataset, config.n_samples, config
        )

        if len(samples) < 100:
            logger.warning(f"Only {len(samples)} samples created, need more data")
            continue

        # Evaluate methods
        methods = ['Trie-only', 'Trie+LLM', 'LLM-only']
        method_metrics = {m: defaultdict(list) for m in methods}

        start_time = time.time()
        cache_hits = 0
        early_exits = 0
        llm_calls = 0

        for sample_idx, sample in enumerate(samples):
            history = sample['history']
            candidates = sample['candidates']
            gt_idx = sample['ground_truth_idx']

            # Extract tokens
            history_tokens = [f"C{i+1}:{h['sparse'][i]}" for h in history for i in range(len(sparse_cols))]
            candidate_tokens_list = [
                [f"C{i+1}:{c['sparse'][i]}" for i in range(len(sparse_cols))]
                for c in candidates
            ]

            # Method 1: Trie-only (CTR-based ranking)
            trie_scores = [
                np.mean([trie_stats.get_ctr(t) for t in cand_tokens])
                for cand_tokens in candidate_tokens_list
            ]
            trie_ranking = np.argsort(trie_scores)[::-1].tolist()

            # Method 2: Trie+LLM (with early exit)
            confidence = trie_stats.compute_confidence(history_tokens)
            if confidence >= config.early_exit_threshold:
                early_exits += 1
                trie_llm_ranking = trie_ranking
            else:
                llm_calls += 1
                llm_scores = llm_ranker.score_candidates(history_tokens, candidate_tokens_list)
                # Combine Trie + LLM scores
                combined_scores = [0.3 * t + 0.7 * l for t, l in zip(trie_scores, llm_scores)]
                trie_llm_ranking = np.argsort(combined_scores)[::-1].tolist()

            # Method 3: LLM-only
            llm_scores = llm_ranker.score_candidates(history_tokens, candidate_tokens_list)
            llm_ranking = np.argsort(llm_scores)[::-1].tolist()

            # Evaluate all methods
            for method, ranking in [
                ('Trie-only', trie_ranking),
                ('Trie+LLM', trie_llm_ranking),
                ('LLM-only', llm_ranking)
            ]:
                metrics = evaluate_recommendations(ranking, gt_idx, config.k_values)
                for k, v in metrics.items():
                    method_metrics[method][k].append(v)

            if (sample_idx + 1) % 100 == 0:
                elapsed = time.time() - start_time
                logger.info(f"  Processed {sample_idx + 1}/{len(samples)} "
                           f"({elapsed:.1f}s, EarlyExit={early_exits}, LLM={llm_calls})")

        # Aggregate run results
        run_result = {
            'seed': seed,
            'n_samples': len(samples),
            'early_exit_rate': early_exits / len(samples),
            'llm_call_rate': llm_calls / len(samples),
            'total_time_s': time.time() - start_time
        }

        for method in methods:
            for metric, values in method_metrics[method].items():
                run_result[f'{method}_{metric}'] = np.mean(values)
                run_result[f'{method}_{metric}_std'] = np.std(values)

        all_results.append(run_result)

        # Print run summary
        logger.info(f"\nRun {run_idx + 1} Results:")
        for method in methods:
            hit5 = run_result.get(f'{method}_hit@5', 0)
            ndcg5 = run_result.get(f'{method}_ndcg@5', 0)
            logger.info(f"  {method}: Hit@5={hit5:.4f}, NDCG@5={ndcg5:.4f}")

    # Aggregate all runs
    aggregate = {}
    for method in methods:
        for metric in ['hit@1', 'hit@5', 'ndcg@5', 'mrr@5']:
            key = f'{method}_{metric}'
            values = [r[key] for r in all_results if key in r]
            if values:
                aggregate[f'{key}_mean'] = np.mean(values)
                aggregate[f'{key}_std'] = np.std(values)

    # Statistical significance: Trie+LLM vs Trie-only
    from scipy import stats
    trie_only_hits = [r['Trie-only_hit@5'] for r in all_results]
    trie_llm_hits = [r['Trie+LLM_hit@5'] for r in all_results]
    llm_only_hits = [r['LLM-only_hit@5'] for r in all_results]

    if len(trie_only_hits) >= 2:
        t_stat, p_value = stats.ttest_rel(trie_llm_hits, trie_only_hits)
        aggregate['trie_llm_vs_trie_only'] = {
            't_stat': float(t_stat),
            'p_value': float(p_value),
            'significant': p_value < 0.05
        }

        t_stat2, p_value2 = stats.ttest_rel(llm_only_hits, trie_only_hits)
        aggregate['llm_only_vs_trie_only'] = {
            't_stat': float(t_stat2),
            'p_value': float(p_value2),
            'significant': p_value2 < 0.05
        }

    results = {
        'config': asdict(config),
        'timestamp': datetime.now().isoformat(),
        'aggregate': aggregate,
        'raw_results': all_results
    }

    # Save results
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f'criteo_encrypted_{timestamp}.json'

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # Print final summary
    logger.info("\n" + "=" * 60)
    logger.info("FINAL SUMMARY: Criteo Encrypted Features")
    logger.info("=" * 60)
    logger.info(f"\n{'Method':<15} {'Hit@5':<20} {'NDCG@5':<20}")
    logger.info("-" * 55)

    for method in methods:
        hit5_mean = aggregate.get(f'{method}_hit@5_mean', 0)
        hit5_std = aggregate.get(f'{method}_hit@5_std', 0)
        ndcg5_mean = aggregate.get(f'{method}_ndcg@5_mean', 0)
        ndcg5_std = aggregate.get(f'{method}_ndcg@5_std', 0)
        logger.info(f"{method:<15} {hit5_mean:.4f}±{hit5_std:.4f}      {ndcg5_mean:.4f}±{ndcg5_std:.4f}")

    logger.info("\n[Expected Negative Result]")
    logger.info("On encrypted features, LLM cannot leverage semantic understanding.")
    logger.info("Trie-based statistical routing should be the primary contributor.")

    logger.info(f"\nResults saved to: {output_file}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Criteo Encrypted Features Experiment")
    parser.add_argument("--samples", type=int, default=2000,
                       help="Number of samples per run")
    parser.add_argument("--runs", type=int, default=3,
                       help="Number of runs")
    parser.add_argument("--device", type=str, default=None,
                       help="Device (cuda:0 or cpu)")
    args = parser.parse_args()

    config = CriteoExperimentConfig(
        n_samples=args.samples,
        num_runs=args.runs,
        device=args.device
    )

    run_criteo_experiment(config)


if __name__ == '__main__':
    main()
