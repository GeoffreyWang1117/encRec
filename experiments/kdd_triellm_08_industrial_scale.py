"""
Industrial-Scale Trie-Augmented LLM Recommendation Experiment.

Target datasets:
1. Criteo 45M (Display Advertising Challenge)
2. Amazon Reviews (large-scale product recommendation)
3. Avazu CTR (40M mobile ad clicks)

Metrics:
- Recommendation quality: AUC, NDCG@K, Hit@K
- Efficiency: Latency, Token usage, LLM call rate
- Cost: API calls, estimated $

This experiment is designed for KDD-level publication.
"""

import os
import sys
import json
import time
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.trie.retrieval_trie import RetrievalTrie, ItemStats
from src.llm.ollama_client import OllamaClient, OllamaConfig
from src.llm.trie_augmented_llm_rec import (
    TrieAugmentedLLMRecommender,
    TrieContextCompressor,
    RecommendationRequest,
    StatisticalEarlyExit,
    TrieResultCache,
)


@dataclass
class ExperimentConfig:
    """Configuration for industrial-scale experiment."""
    # Data
    dataset: str = "criteo"  # criteo, amazon, avazu
    data_path: str = "data/criteo/criteo_5m.parquet"
    sample_size: Optional[int] = None  # None = full data

    # Trie
    min_token_freq: int = 5
    trie_categories: int = 100  # Number of category buckets

    # LLM
    model: str = "glm-4.6"
    api_key: str = ""
    max_concurrent_requests: int = 5

    # Experiment
    num_test_users: int = 1000
    num_recommendations: int = 10
    num_runs: int = 3

    # Optimization
    enable_early_exit: bool = True
    early_exit_threshold: float = 0.8
    enable_cache: bool = True
    cache_ttl: int = 3600
    compression_ratio: float = 0.2

    # Output
    output_dir: str = "results/industrial_experiment"


@dataclass
class ExperimentResult:
    """Result of a single experiment run."""
    run_id: int
    dataset: str
    method: str

    # Quality metrics
    auc: float = 0.0
    ndcg_5: float = 0.0
    ndcg_10: float = 0.0
    hit_5: float = 0.0
    hit_10: float = 0.0

    # Efficiency metrics
    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    avg_tokens: float = 0.0
    llm_call_rate: float = 0.0
    early_exit_rate: float = 0.0
    cache_hit_rate: float = 0.0

    # Cost metrics
    total_llm_calls: int = 0
    total_tokens_used: int = 0
    estimated_cost_usd: float = 0.0

    # Meta
    num_samples: int = 0
    timestamp: str = ""


class CriteoDataProcessor:
    """Process Criteo dataset for Trie-LLM experiment."""

    DENSE_COLS = [f'I{i}' for i in range(1, 14)]
    SPARSE_COLS = [f'C{i}' for i in range(1, 27)]

    def __init__(self, data_path: str, sample_size: Optional[int] = None):
        self.data_path = data_path
        self.sample_size = sample_size

    def load_data(self) -> pd.DataFrame:
        """Load Criteo data."""
        logger.info(f"Loading data from {self.data_path}")
        df = pd.read_parquet(self.data_path)

        if self.sample_size and self.sample_size < len(df):
            df = df.sample(n=self.sample_size, random_state=42)

        logger.info(f"Loaded {len(df)} samples")
        return df

    def build_trie(self, df: pd.DataFrame, min_freq: int = 5) -> RetrievalTrie:
        """Build Trie from Criteo data."""
        logger.info("Building Trie from data...")
        trie = RetrievalTrie(min_count_for_stats=min_freq)

        # Group items by sparse features
        for idx, row in df.iterrows():
            if idx % 100000 == 0:
                logger.info(f"  Processing row {idx}/{len(df)}")

            label = int(row['label'])

            # Create item ID from sparse features
            for col in self.SPARSE_COLS[:5]:  # Use first 5 sparse features
                token = str(row.get(col, 'UNK'))
                if token and token != 'nan':
                    # Infer category from feature column
                    category = f"cat_{col}"
                    trie.insert(token, label, category)

        trie.build_indexes()
        logger.info(f"Trie built with {trie.total_items} unique tokens")
        return trie

    def create_user_sessions(
        self,
        df: pd.DataFrame,
        num_users: int = 1000,
        history_length: int = 10,
    ) -> List[Dict]:
        """Create simulated user sessions from data."""
        logger.info(f"Creating {num_users} user sessions...")

        # Group by first sparse feature as "user"
        user_col = self.SPARSE_COLS[0]
        grouped = df.groupby(user_col)

        sessions = []
        user_ids = list(grouped.groups.keys())[:num_users * 2]

        for user_id in user_ids[:num_users]:
            user_data = grouped.get_group(user_id)
            if len(user_data) < history_length + 5:
                continue

            # History: earlier interactions
            history_items = []
            for _, row in user_data.head(history_length).iterrows():
                for col in self.SPARSE_COLS[1:3]:
                    token = str(row.get(col, ''))
                    if token and token != 'nan':
                        history_items.append(token)

            # Ground truth: later interactions with positive labels
            ground_truth = []
            for _, row in user_data.tail(10).iterrows():
                if row['label'] == 1:
                    for col in self.SPARSE_COLS[1:3]:
                        token = str(row.get(col, ''))
                        if token and token != 'nan':
                            ground_truth.append(token)

            if history_items and ground_truth:
                sessions.append({
                    'user_id': str(user_id),
                    'history': history_items[:history_length],
                    'ground_truth': list(set(ground_truth)),
                })

        logger.info(f"Created {len(sessions)} valid sessions")
        return sessions


class MetricsCalculator:
    """Calculate recommendation quality metrics."""

    @staticmethod
    def ndcg_at_k(recommended: List[str], ground_truth: List[str], k: int) -> float:
        """Calculate NDCG@K."""
        if not ground_truth:
            return 0.0

        dcg = 0.0
        for i, item in enumerate(recommended[:k]):
            if item in ground_truth:
                dcg += 1.0 / np.log2(i + 2)

        # Ideal DCG
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(ground_truth), k)))

        return dcg / idcg if idcg > 0 else 0.0

    @staticmethod
    def hit_at_k(recommended: List[str], ground_truth: List[str], k: int) -> float:
        """Calculate Hit@K (1 if any hit in top-k, else 0)."""
        if not ground_truth:
            return 0.0
        return 1.0 if any(item in ground_truth for item in recommended[:k]) else 0.0

    @staticmethod
    def precision_at_k(recommended: List[str], ground_truth: List[str], k: int) -> float:
        """Calculate Precision@K."""
        if not recommended[:k]:
            return 0.0
        hits = sum(1 for item in recommended[:k] if item in ground_truth)
        return hits / k


class IndustrialExperiment:
    """Run industrial-scale experiment."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.output_path = Path(config.output_dir)
        self.output_path.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.data_processor = CriteoDataProcessor(
            config.data_path,
            config.sample_size
        )
        self.metrics = MetricsCalculator()

    def setup(self):
        """Setup experiment components."""
        # Load data
        self.df = self.data_processor.load_data()

        # Build Trie
        self.trie = self.data_processor.build_trie(
            self.df,
            min_freq=self.config.min_token_freq
        )

        # Create user sessions
        self.sessions = self.data_processor.create_user_sessions(
            self.df,
            num_users=self.config.num_test_users,
        )

        # Initialize LLM client
        llm_config = OllamaConfig(api_key=self.config.api_key)
        self.llm_client = OllamaClient(llm_config)

        # Initialize recommenders
        self.trie_llm_recommender = TrieAugmentedLLMRecommender(
            trie=self.trie,
            llm_client=self.llm_client,
            model=self.config.model,
            enable_cache=self.config.enable_cache,
            enable_early_exit=self.config.enable_early_exit,
            compression_ratio=self.config.compression_ratio,
        )

        if self.config.enable_early_exit:
            self.trie_llm_recommender.early_exit.confidence_threshold = \
                self.config.early_exit_threshold

        logger.info("Experiment setup complete")

    def run_trie_only(self, session: Dict) -> Tuple[List[str], Dict]:
        """Baseline: Trie-only recommendation."""
        from src.trie.retrieval_trie import TrieCandidateRetrieval

        start = time.time()
        retriever = TrieCandidateRetrieval(self.trie)
        recommended = retriever.retrieve(
            session['history'],
            k=self.config.num_recommendations,
        )
        latency = (time.time() - start) * 1000

        return recommended, {
            'latency_ms': latency,
            'tokens_used': 0,
            'llm_called': False,
        }

    def run_trie_llm(self, session: Dict) -> Tuple[List[str], Dict]:
        """Our method: Trie-augmented LLM."""
        request = RecommendationRequest(
            user_id=session['user_id'],
            user_history=session['history'],
            num_recommendations=self.config.num_recommendations,
        )

        result = self.trie_llm_recommender.recommend(request)

        return result.items, {
            'latency_ms': result.latency_ms,
            'tokens_used': result.tokens_used,
            'llm_called': not result.early_exit and not result.cache_hit,
            'early_exit': result.early_exit,
            'cache_hit': result.cache_hit,
        }

    def run_pure_llm(self, session: Dict) -> Tuple[List[str], Dict]:
        """Baseline: Pure LLM without Trie."""
        start = time.time()

        # Get all possible items
        all_items = [s.item_id for s in self.trie.get_top_by_freq(100)]

        # Build verbose prompt
        history_str = ", ".join(session['history'][:10])
        candidates_str = ", ".join(all_items[:30])

        prompt = f"""推荐任务: 根据用户历史选择最佳商品

用户历史: {history_str}

候选商品: {candidates_str}

请选择{self.config.num_recommendations}个最合适的推荐，每行一个商品ID。
"""

        try:
            response = self.llm_client.chat(
                messages=[{"role": "user", "content": prompt}],
                model=self.config.model,
                temperature=0.3,
                max_tokens=256,
            )
            content = response.get('message', {}).get('content', '')

            # Parse response
            recommended = []
            for item in all_items:
                if item in content:
                    recommended.append(item)
                    if len(recommended) >= self.config.num_recommendations:
                        break

            # Fallback
            while len(recommended) < self.config.num_recommendations:
                for item in all_items:
                    if item not in recommended:
                        recommended.append(item)
                        break

        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            recommended = all_items[:self.config.num_recommendations]

        latency = (time.time() - start) * 1000
        tokens = len(prompt.split()) + len(content.split()) if 'content' in dir() else 0

        return recommended, {
            'latency_ms': latency,
            'tokens_used': tokens,
            'llm_called': True,
        }

    def evaluate_single_session(
        self,
        session: Dict,
        method: str,
    ) -> Dict:
        """Evaluate a single session."""
        if method == 'trie_only':
            recommended, metrics = self.run_trie_only(session)
        elif method == 'trie_llm':
            recommended, metrics = self.run_trie_llm(session)
        elif method == 'pure_llm':
            recommended, metrics = self.run_pure_llm(session)
        else:
            raise ValueError(f"Unknown method: {method}")

        ground_truth = session['ground_truth']

        return {
            **metrics,
            'ndcg_5': self.metrics.ndcg_at_k(recommended, ground_truth, 5),
            'ndcg_10': self.metrics.ndcg_at_k(recommended, ground_truth, 10),
            'hit_5': self.metrics.hit_at_k(recommended, ground_truth, 5),
            'hit_10': self.metrics.hit_at_k(recommended, ground_truth, 10),
        }

    def run_method(
        self,
        method: str,
        run_id: int,
        sample_sessions: List[Dict] = None,
    ) -> ExperimentResult:
        """Run experiment for a specific method."""
        logger.info(f"Running {method} (run {run_id})...")

        sessions = sample_sessions or self.sessions
        results = []

        for i, session in enumerate(sessions):
            if i % 100 == 0:
                logger.info(f"  Processing session {i}/{len(sessions)}")

            try:
                result = self.evaluate_single_session(session, method)
                results.append(result)
            except Exception as e:
                logger.error(f"Session {i} failed: {e}")
                continue

        # Aggregate results
        latencies = [r['latency_ms'] for r in results]

        return ExperimentResult(
            run_id=run_id,
            dataset=self.config.dataset,
            method=method,
            ndcg_5=np.mean([r['ndcg_5'] for r in results]),
            ndcg_10=np.mean([r['ndcg_10'] for r in results]),
            hit_5=np.mean([r['hit_5'] for r in results]),
            hit_10=np.mean([r['hit_10'] for r in results]),
            avg_latency_ms=np.mean(latencies),
            p50_latency_ms=np.percentile(latencies, 50),
            p99_latency_ms=np.percentile(latencies, 99),
            avg_tokens=np.mean([r['tokens_used'] for r in results]),
            llm_call_rate=np.mean([r.get('llm_called', False) for r in results]),
            early_exit_rate=np.mean([r.get('early_exit', False) for r in results]),
            cache_hit_rate=np.mean([r.get('cache_hit', False) for r in results]),
            total_llm_calls=sum(r.get('llm_called', False) for r in results),
            total_tokens_used=sum(r['tokens_used'] for r in results),
            estimated_cost_usd=sum(r['tokens_used'] for r in results) * 0.00001,  # Rough estimate
            num_samples=len(results),
            timestamp=datetime.now().isoformat(),
        )

    def run_full_experiment(self):
        """Run the full experiment."""
        logger.info("="*60)
        logger.info("Starting Industrial-Scale Experiment")
        logger.info("="*60)

        self.setup()

        all_results = []
        methods = ['trie_only', 'trie_llm']

        # Add pure_llm for small sample (expensive)
        if len(self.sessions) <= 100:
            methods.append('pure_llm')

        for run_id in range(self.config.num_runs):
            logger.info(f"\n--- Run {run_id + 1}/{self.config.num_runs} ---")

            # Shuffle sessions for each run
            np.random.seed(42 + run_id)
            shuffled = np.random.permutation(len(self.sessions))
            sample_sessions = [self.sessions[i] for i in shuffled]

            for method in methods:
                # For pure_llm, use smaller sample
                if method == 'pure_llm':
                    sample = sample_sessions[:50]
                else:
                    sample = sample_sessions

                result = self.run_method(method, run_id, sample)
                all_results.append(result)

                logger.info(f"{method}: NDCG@10={result.ndcg_10:.4f}, "
                           f"Latency={result.avg_latency_ms:.1f}ms, "
                           f"LLM_rate={result.llm_call_rate:.1%}")

        # Save results
        self.save_results(all_results)
        self.print_summary(all_results)

        return all_results

    def save_results(self, results: List[ExperimentResult]):
        """Save experiment results."""
        # Save as JSON
        results_dict = [asdict(r) for r in results]
        with open(self.output_path / 'results.json', 'w') as f:
            json.dump(results_dict, f, indent=2)

        # Save as CSV
        df = pd.DataFrame(results_dict)
        df.to_csv(self.output_path / 'results.csv', index=False)

        # Save config
        with open(self.output_path / 'config.json', 'w') as f:
            json.dump(asdict(self.config), f, indent=2)

        logger.info(f"Results saved to {self.output_path}")

    def print_summary(self, results: List[ExperimentResult]):
        """Print experiment summary."""
        print("\n" + "="*80)
        print("INDUSTRIAL-SCALE EXPERIMENT SUMMARY")
        print("="*80)

        # Group by method
        by_method = defaultdict(list)
        for r in results:
            by_method[r.method].append(r)

        print(f"\nDataset: {self.config.dataset}")
        print(f"Samples: {results[0].num_samples if results else 0}")
        print(f"Runs: {self.config.num_runs}")

        print("\n" + "-"*80)
        print(f"{'Method':<15} {'NDCG@5':<10} {'NDCG@10':<10} {'Hit@10':<10} "
              f"{'Latency(ms)':<15} {'LLM Rate':<10}")
        print("-"*80)

        for method, method_results in by_method.items():
            ndcg5 = np.mean([r.ndcg_5 for r in method_results])
            ndcg10 = np.mean([r.ndcg_10 for r in method_results])
            hit10 = np.mean([r.hit_10 for r in method_results])
            latency = np.mean([r.avg_latency_ms for r in method_results])
            llm_rate = np.mean([r.llm_call_rate for r in method_results])

            print(f"{method:<15} {ndcg5:<10.4f} {ndcg10:<10.4f} {hit10:<10.4f} "
                  f"{latency:<15.1f} {llm_rate:<10.1%}")

        print("-"*80)

        # Improvement analysis
        if 'trie_llm' in by_method and 'trie_only' in by_method:
            trie_only_ndcg = np.mean([r.ndcg_10 for r in by_method['trie_only']])
            trie_llm_ndcg = np.mean([r.ndcg_10 for r in by_method['trie_llm']])
            improvement = (trie_llm_ndcg - trie_only_ndcg) / trie_only_ndcg * 100

            trie_llm_latency = np.mean([r.avg_latency_ms for r in by_method['trie_llm']])
            trie_only_latency = np.mean([r.avg_latency_ms for r in by_method['trie_only']])

            print(f"\nTrie+LLM vs Trie-only:")
            print(f"  NDCG@10 improvement: {improvement:+.2f}%")
            print(f"  Latency overhead: {trie_llm_latency - trie_only_latency:.1f}ms")

        if 'pure_llm' in by_method and 'trie_llm' in by_method:
            pure_llm_latency = np.mean([r.avg_latency_ms for r in by_method['pure_llm']])
            trie_llm_latency = np.mean([r.avg_latency_ms for r in by_method['trie_llm']])
            speedup = pure_llm_latency / trie_llm_latency

            pure_llm_tokens = np.mean([r.avg_tokens for r in by_method['pure_llm']])
            trie_llm_tokens = np.mean([r.avg_tokens for r in by_method['trie_llm']])
            token_reduction = (pure_llm_tokens - trie_llm_tokens) / pure_llm_tokens * 100

            print(f"\nTrie+LLM vs Pure LLM:")
            print(f"  Speedup: {speedup:.1f}x")
            print(f"  Token reduction: {token_reduction:.1f}%")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Industrial-Scale Experiment")
    parser.add_argument('--dataset', type=str, default='criteo')
    parser.add_argument('--data_path', type=str, default='data/criteo/criteo_5m.parquet')
    parser.add_argument('--sample_size', type=int, default=None)
    parser.add_argument('--api_key', type=str, required=True)
    parser.add_argument('--model', type=str, default='glm-4.6')
    parser.add_argument('--num_test_users', type=int, default=500)
    parser.add_argument('--num_runs', type=int, default=3)
    parser.add_argument('--output_dir', type=str, default='results/industrial_experiment')
    parser.add_argument('--early_exit_threshold', type=float, default=0.8)

    args = parser.parse_args()

    config = ExperimentConfig(
        dataset=args.dataset,
        data_path=args.data_path,
        sample_size=args.sample_size,
        api_key=args.api_key,
        model=args.model,
        num_test_users=args.num_test_users,
        num_runs=args.num_runs,
        output_dir=args.output_dir,
        early_exit_threshold=args.early_exit_threshold,
    )

    experiment = IndustrialExperiment(config)
    experiment.run_full_experiment()


if __name__ == '__main__':
    main()
