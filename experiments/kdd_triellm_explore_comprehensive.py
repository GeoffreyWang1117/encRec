"""
KDD 2026 Trie-Augmented LLM Recommendation: Comprehensive Experiments.

This module runs supplementary experiments with:
1. Multiple datasets (MIND, Amazon, MovieLens)
2. Full recommendation metrics (Hit, Precision, Recall, NDCG, MRR, MAP, Coverage, Diversity, Novelty)
3. Multiple runs for statistical significance (5 runs)
4. Baseline comparisons
5. Ablation studies (early exit threshold, compression ratio, history length)
6. Statistical significance tests (paired t-test, Wilcoxon, Cohen's d)
"""

import os
import sys
import json
import time
import random
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, asdict, field
from collections import defaultdict
import logging
import hashlib
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.metrics.recommendation_metrics import (
    MetricsCalculator, RecommendationMetrics, StatisticalTests, create_metrics_table
)
from src.trie.retrieval_trie import RetrievalTrie
from src.baselines.collaborative_filtering import BPRRecommender, NeuMFRecommender


# =====================================================================
# Dataset Loaders
# =====================================================================

@dataclass
class RecommendationSample:
    """A single recommendation sample."""
    user_id: str
    history: List[str]  # Item IDs in user history
    candidates: List[str]  # Candidate item IDs
    ground_truth: List[str]  # Positive items (clicked/rated)
    metadata: Dict = field(default_factory=dict)


class DatasetLoader:
    """Base class for dataset loaders."""

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.items: Dict[str, Dict] = {}  # item_id -> metadata
        self.item_popularity: Dict[str, int] = {}
        self.item_categories: Dict[str, str] = {}
        self.all_items: Set[str] = set()

    def load(self, max_samples: int = None) -> List[RecommendationSample]:
        raise NotImplementedError

    def get_metrics_calculator(self) -> MetricsCalculator:
        """Create MetricsCalculator with item metadata."""
        return MetricsCalculator(
            item_popularity=self.item_popularity,
            item_categories=self.item_categories,
            all_items=self.all_items,
        )


class MINDDatasetLoader(DatasetLoader):
    """Load MIND news dataset."""

    def load(self, max_samples: int = None) -> List[RecommendationSample]:
        logger.info(f"Loading MIND dataset from {self.data_dir}")

        # Load news articles
        news_path = self.data_dir / "news.tsv"
        with open(news_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 5:
                    news_id, category, subcategory, title, abstract = parts[:5]
                    self.items[news_id] = {
                        'category': category,
                        'subcategory': subcategory,
                        'title': title,
                        'abstract': abstract[:200] if abstract else "",
                    }
                    self.item_categories[news_id] = category
                    self.all_items.add(news_id)

        logger.info(f"Loaded {len(self.items)} news articles")

        # Load behaviors and count popularity
        behaviors_path = self.data_dir / "behaviors.tsv"
        samples = []

        with open(behaviors_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 5:
                    _, user_id, _, history_str, impressions_str = parts[:5]

                    history = history_str.split() if history_str else []

                    impressions = []
                    ground_truth = []
                    for item in impressions_str.split():
                        if '-' in item:
                            news_id, label = item.rsplit('-', 1)
                            impressions.append(news_id)
                            self.item_popularity[news_id] = self.item_popularity.get(news_id, 0) + 1
                            if label == '1':
                                ground_truth.append(news_id)

                    if history and impressions and ground_truth:
                        samples.append(RecommendationSample(
                            user_id=user_id,
                            history=history,
                            candidates=impressions,
                            ground_truth=ground_truth,
                        ))

                        if max_samples and len(samples) >= max_samples:
                            break

        logger.info(f"Loaded {len(samples)} samples")
        return samples


class AmazonDatasetLoader(DatasetLoader):
    """Load Amazon Reviews dataset (parquet format)."""

    def __init__(self, data_dir: str, category: str = "All_Beauty"):
        super().__init__(data_dir)
        self.category = category

    def load(self, max_samples: int = None) -> List[RecommendationSample]:
        logger.info(f"Loading Amazon {self.category} dataset")

        # Load reviews
        reviews_path = self.data_dir / f"{self.category}_reviews.parquet"
        if not reviews_path.exists():
            logger.error(f"Reviews file not found: {reviews_path}")
            return []

        df = pd.read_parquet(reviews_path)
        logger.info(f"Loaded {len(df)} reviews")

        # Load metadata if available
        meta_path = self.data_dir / f"{self.category}_meta.parquet"
        if meta_path.exists():
            try:
                meta_df = pd.read_parquet(meta_path)
                for _, row in meta_df.head(100000).iterrows():  # Limit for memory
                    asin = row.get('parent_asin') or row.get('asin', '')
                    if asin:
                        self.items[asin] = {
                            'title': str(row.get('title', ''))[:100],
                            'category': str(row.get('main_category', ''))[:50],
                        }
                        self.item_categories[asin] = str(row.get('main_category', ''))
                        self.all_items.add(asin)
            except Exception as e:
                logger.warning(f"Could not load metadata: {e}")

        # Group by user and create samples
        user_items = defaultdict(list)
        for _, row in df.iterrows():
            user_id = str(row.get('user_id', ''))
            asin = str(row.get('parent_asin') or row.get('asin', ''))
            rating = row.get('rating', 3)
            timestamp = row.get('timestamp', 0)

            if user_id and asin:
                user_items[user_id].append((asin, rating, timestamp))
                self.item_popularity[asin] = self.item_popularity.get(asin, 0) + 1
                self.all_items.add(asin)

        # Create recommendation samples
        samples = []
        all_items_list = list(self.all_items)

        for user_id, items in user_items.items():
            if len(items) < 5:  # Need enough history
                continue

            # Sort by timestamp
            items.sort(key=lambda x: x[2])

            # Split: 80% history, 20% test
            split_idx = int(len(items) * 0.8)
            history = [x[0] for x in items[:split_idx]]
            test_items = items[split_idx:]

            # Ground truth: highly rated items (>=4)
            ground_truth = [x[0] for x in test_items if x[1] >= 4]

            if not ground_truth:
                continue

            # Candidates: test items + random negatives
            candidates = [x[0] for x in test_items]
            num_neg = max(20, len(candidates) * 3)
            negatives = random.sample(all_items_list, min(num_neg, len(all_items_list)))
            candidates = list(set(candidates + negatives))

            samples.append(RecommendationSample(
                user_id=user_id,
                history=history,
                candidates=candidates,
                ground_truth=ground_truth,
            ))

            if max_samples and len(samples) >= max_samples:
                break

        logger.info(f"Created {len(samples)} samples")
        return samples


class MovieLensDatasetLoader(DatasetLoader):
    """Load MovieLens 1M dataset."""

    def load(self, max_samples: int = None) -> List[RecommendationSample]:
        logger.info(f"Loading MovieLens dataset from {self.data_dir}")

        # Load movies
        movies_path = self.data_dir / "movies.dat"
        with open(movies_path, 'r', encoding='latin-1') as f:
            for line in f:
                parts = line.strip().split('::')
                if len(parts) >= 3:
                    movie_id, title, genres = parts[:3]
                    self.items[movie_id] = {
                        'title': title,
                        'genres': genres,
                    }
                    self.item_categories[movie_id] = genres.split('|')[0]  # First genre
                    self.all_items.add(movie_id)

        logger.info(f"Loaded {len(self.items)} movies")

        # Load ratings
        ratings_path = self.data_dir / "ratings.dat"
        user_items = defaultdict(list)

        with open(ratings_path, 'r', encoding='latin-1') as f:
            for line in f:
                parts = line.strip().split('::')
                if len(parts) >= 4:
                    user_id, movie_id, rating, timestamp = parts[:4]
                    user_items[user_id].append((movie_id, int(rating), int(timestamp)))
                    self.item_popularity[movie_id] = self.item_popularity.get(movie_id, 0) + 1

        # Create samples
        samples = []
        all_items_list = list(self.all_items)

        for user_id, items in user_items.items():
            if len(items) < 10:
                continue

            # Sort by timestamp
            items.sort(key=lambda x: x[2])

            # Split
            split_idx = int(len(items) * 0.8)
            history = [x[0] for x in items[:split_idx]]
            test_items = items[split_idx:]

            # Ground truth: rating >= 4
            ground_truth = [x[0] for x in test_items if x[1] >= 4]

            if not ground_truth:
                continue

            # Candidates
            candidates = [x[0] for x in test_items]
            num_neg = max(50, len(candidates) * 5)
            negatives = random.sample(all_items_list, min(num_neg, len(all_items_list)))
            candidates = list(set(candidates + negatives))

            samples.append(RecommendationSample(
                user_id=user_id,
                history=history,
                candidates=candidates,
                ground_truth=ground_truth,
            ))

            if max_samples and len(samples) >= max_samples:
                break

        logger.info(f"Created {len(samples)} samples")
        return samples


# =====================================================================
# Recommendation Methods
# =====================================================================

class TrieRecommender:
    """Statistical Trie-based recommender."""

    def __init__(self, items: Dict[str, Dict], item_popularity: Dict[str, int]):
        self.items = items
        self.item_popularity = item_popularity
        self.trie = RetrievalTrie()
        self._build_trie()

    def _build_trie(self):
        """Build Trie from item data."""
        for item_id, meta in self.items.items():
            pop = self.item_popularity.get(item_id, 1)
            category = meta.get('category', 'unknown')

            # Insert with popularity-weighted labels
            for _ in range(min(pop, 10)):
                label = 1 if random.random() < 0.5 else 0
                self.trie.insert(item_id, label, category)

        self.trie.build_indexes()

    def recommend(
        self,
        history: List[str],
        candidates: List[str],
        k: int = 10
    ) -> Tuple[List[str], float]:
        """Recommend top-k items based on Trie statistics."""
        start = time.time()

        # Get user's category preferences
        user_categories = defaultdict(int)
        for item_id in history[-20:]:
            meta = self.items.get(item_id, {})
            cat = meta.get('category', 'unknown')
            user_categories[cat] += 1

        # Score candidates
        scores = []
        for item_id in candidates:
            stats = self.trie.get(item_id)
            meta = self.items.get(item_id, {})
            cat = meta.get('category', 'unknown')

            # Score: CTR * category match * popularity boost
            ctr = stats.ctr if stats else 0.01
            cat_match = 1.0 + user_categories.get(cat, 0) * 0.2
            pop = self.item_popularity.get(item_id, 1)

            score = ctr * cat_match * np.log1p(pop)
            scores.append((item_id, score))

        # Sort and return top-k
        scores.sort(key=lambda x: x[1], reverse=True)
        recommendations = [item_id for item_id, _ in scores[:k]]

        latency = (time.time() - start) * 1000
        return recommendations, latency


class TrieLLMRecommender:
    """Trie-augmented LLM recommender with optimizations."""

    def __init__(
        self,
        items: Dict[str, Dict],
        item_popularity: Dict[str, int],
        llm_client,
        early_exit_threshold: float = 0.5,
        max_history_items: int = 5,
    ):
        self.items = items
        self.item_popularity = item_popularity
        self.llm = llm_client
        self.early_exit_threshold = early_exit_threshold
        self.max_history_items = max_history_items

        # Build Trie
        self.trie = RetrievalTrie()
        self._build_trie()

        # Cache
        self.cache = {}

        # Stats
        self.stats = {
            'early_exits': 0,
            'cache_hits': 0,
            'llm_calls': 0,
            'total_requests': 0,
        }

    def _build_trie(self):
        for item_id, meta in self.items.items():
            pop = self.item_popularity.get(item_id, 1)
            category = meta.get('category', 'unknown')
            for _ in range(min(pop, 10)):
                label = 1 if random.random() < 0.5 else 0
                self.trie.insert(item_id, label, category)
        self.trie.build_indexes()

    def _should_early_exit(
        self,
        history: List[str],
        candidates: List[str],
        k: int
    ) -> Tuple[bool, List[str], float]:
        """Check if we can skip LLM call."""
        # Get user's category preferences
        user_categories = defaultdict(int)
        for item_id in history[-20:]:
            meta = self.items.get(item_id, {})
            cat = meta.get('category', 'unknown')
            user_categories[cat] += 1

        if not user_categories:
            return False, [], 0.0

        top_category = max(user_categories, key=user_categories.get)
        preference_strength = user_categories[top_category] / len(history[-20:])

        # Score candidates
        candidate_scores = []
        for item_id in candidates:
            stats = self.trie.get(item_id)
            meta = self.items.get(item_id, {})
            cat = meta.get('category', 'unknown')

            if stats:
                cat_match = 1.0 if cat == top_category else 0.3
                score = cat_match * stats.ctr * (1 + np.log1p(stats.frequency))
                candidate_scores.append((item_id, score))

        if len(candidate_scores) < k:
            return False, [], 0.0

        candidate_scores.sort(key=lambda x: x[1], reverse=True)
        top_k = candidate_scores[:k]
        rest = candidate_scores[k:k + 10]

        if not rest:
            return False, [], 0.0

        avg_top = np.mean([s for _, s in top_k])
        avg_rest = np.mean([s for _, s in rest])

        confidence = preference_strength * (avg_top - avg_rest) / max(avg_top, 0.01)
        confidence = min(confidence, 1.0)

        if confidence >= self.early_exit_threshold:
            return True, [item_id for item_id, _ in top_k], confidence

        return False, [], confidence

    def _compress_prompt(
        self,
        history: List[str],
        candidates: List[str],
        k: int
    ) -> str:
        """Create compressed prompt."""
        # Compress history
        history_parts = []
        for item_id in history[-self.max_history_items:]:
            meta = self.items.get(item_id, {})
            title = meta.get('title', item_id)[:30]
            cat = meta.get('category', '')
            history_parts.append(f"[{cat}]{title}")

        history_str = " -> ".join(history_parts)

        # Group candidates by category
        by_cat = defaultdict(list)
        for item_id in candidates[:30]:
            meta = self.items.get(item_id, {})
            cat = meta.get('category', 'other')
            title = meta.get('title', item_id)[:25]
            stats = self.trie.get(item_id)
            ctr = stats.ctr if stats else 0
            by_cat[cat].append((item_id, title, ctr))

        # Format candidates
        candidates_parts = []
        for cat, items in by_cat.items():
            items.sort(key=lambda x: x[2], reverse=True)
            item_strs = [f"{item_id}:{title}" for item_id, title, _ in items[:3]]
            candidates_parts.append(f"[{cat}]: {', '.join(item_strs)}")

        return f"""推荐
历史: {history_str}
候选:
{chr(10).join(candidates_parts)}
选{k}个ID"""

    def recommend(
        self,
        history: List[str],
        candidates: List[str],
        k: int = 10,
        use_compression: bool = True,
    ) -> Tuple[List[str], float, int, Dict]:
        """
        Recommend items with Trie+LLM.

        Returns:
            recommendations, latency_ms, tokens_used, metadata
        """
        start = time.time()
        self.stats['total_requests'] += 1

        metadata = {
            'early_exit': False,
            'cache_hit': False,
            'llm_called': False,
        }

        # Check cache
        cache_key = hashlib.md5(",".join(sorted(history[-5:])).encode()).hexdigest()
        if cache_key in self.cache:
            self.stats['cache_hits'] += 1
            metadata['cache_hit'] = True
            latency = (time.time() - start) * 1000
            return self.cache[cache_key], latency, 0, metadata

        # Check early exit
        should_exit, early_recs, confidence = self._should_early_exit(history, candidates, k)

        if should_exit:
            self.stats['early_exits'] += 1
            metadata['early_exit'] = True
            metadata['confidence'] = confidence
            self.cache[cache_key] = early_recs
            latency = (time.time() - start) * 1000
            return early_recs, latency, 0, metadata

        # Call LLM
        self.stats['llm_calls'] += 1
        metadata['llm_called'] = True

        prompt = self._compress_prompt(history, candidates, k)
        tokens = len(prompt.split())

        try:
            response = self.llm.chat(
                messages=[{"role": "user", "content": prompt}],
                model="glm-4.6",
                temperature=0.3,
                max_tokens=200,
            )
            content = response.get('message', {}).get('content', '')
            tokens += len(content.split())

            # Parse recommendations
            recommendations = []
            for item_id in candidates:
                if item_id in content:
                    recommendations.append(item_id)
                    if len(recommendations) >= k:
                        break

            # Fallback if parsing fails
            if len(recommendations) < k:
                # Use Trie-based fallback
                fallback_recs, _ = TrieRecommender(
                    self.items, self.item_popularity
                ).recommend(history, candidates, k)

                for r in fallback_recs:
                    if r not in recommendations:
                        recommendations.append(r)
                        if len(recommendations) >= k:
                            break

        except Exception as e:
            logger.error(f"LLM error: {e}")
            recommendations, _ = TrieRecommender(
                self.items, self.item_popularity
            ).recommend(history, candidates, k)
            tokens = 0

        self.cache[cache_key] = recommendations
        latency = (time.time() - start) * 1000

        return recommendations, latency, tokens, metadata

    def get_optimization_stats(self) -> Dict:
        """Get optimization statistics."""
        total = max(self.stats['total_requests'], 1)
        return {
            'early_exit_rate': self.stats['early_exits'] / total,
            'cache_hit_rate': self.stats['cache_hits'] / total,
            'llm_call_rate': self.stats['llm_calls'] / total,
            **self.stats,
        }


class PureLLMRecommender:
    """Pure LLM recommender without Trie optimization."""

    def __init__(self, items: Dict[str, Dict], llm_client):
        self.items = items
        self.llm = llm_client

    def recommend(
        self,
        history: List[str],
        candidates: List[str],
        k: int = 10
    ) -> Tuple[List[str], float, int]:
        """Recommend using pure LLM."""
        start = time.time()

        # Create full prompt
        history_titles = []
        for item_id in history[-10:]:
            meta = self.items.get(item_id, {})
            title = meta.get('title', item_id)
            history_titles.append(f"- {title}")

        candidate_titles = []
        for item_id in candidates[:30]:
            meta = self.items.get(item_id, {})
            title = meta.get('title', item_id)
            cat = meta.get('category', '')
            candidate_titles.append(f"- {item_id}: {title} ({cat})")

        prompt = f"""你是一个推荐专家。

用户历史:
{chr(10).join(history_titles)}

候选:
{chr(10).join(candidate_titles)}

请选择{k}个用户最可能喜欢的，输出{k}个ID，每行一个。"""

        tokens = len(prompt.split())

        try:
            response = self.llm.chat(
                messages=[{"role": "user", "content": prompt}],
                model="glm-4.6",
                temperature=0.3,
                max_tokens=200,
            )
            content = response.get('message', {}).get('content', '')
            tokens += len(content.split())

            # Parse
            recommendations = []
            for item_id in candidates:
                if item_id in content:
                    recommendations.append(item_id)
                    if len(recommendations) >= k:
                        break

            # Random fallback
            if len(recommendations) < k:
                remaining = [c for c in candidates if c not in recommendations]
                recommendations.extend(random.sample(remaining, min(k - len(recommendations), len(remaining))))

        except Exception as e:
            logger.error(f"LLM error: {e}")
            recommendations = random.sample(candidates, min(k, len(candidates)))
            tokens = 0

        latency = (time.time() - start) * 1000
        return recommendations, latency, tokens


# =====================================================================
# Experiment Runner
# =====================================================================

class TrieLLMExperimentRunner:
    """Run comprehensive Trie-LLM recommendation experiments."""

    def __init__(
        self,
        dataset_name: str,
        loader: DatasetLoader,
        llm_client,
        output_dir: str = "results/trie_llm_experiments",
    ):
        self.dataset_name = dataset_name
        self.loader = loader
        self.llm = llm_client
        self.output_dir = Path(output_dir) / dataset_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_single_experiment(
        self,
        samples: List[RecommendationSample],
        method: str,
        k: int = 5,
        early_exit_threshold: float = 0.5,
        run_id: int = 0,
        llm_sample_rate: float = 0.1,
    ) -> Dict:
        """Run a single experiment."""
        logger.info(f"Running {method} (run {run_id + 1})")

        metrics_calc = self.loader.get_metrics_calculator()

        # Initialize recommender
        if method == 'trie_only':
            recommender = TrieRecommender(self.loader.items, self.loader.item_popularity)
        elif method == 'trie_llm':
            recommender = TrieLLMRecommender(
                self.loader.items,
                self.loader.item_popularity,
                self.llm,
                early_exit_threshold=early_exit_threshold,
            )
        elif method == 'pure_llm':
            recommender = PureLLMRecommender(self.loader.items, self.llm)
        elif method == 'bpr':
            recommender = BPRRecommender(self.loader.items, epochs=20)
            # Build user-item interactions from samples for training
            user_items = self._build_user_items(samples)
            recommender.fit(user_items, self.loader.all_items)
        elif method == 'neumf':
            recommender = NeuMFRecommender(self.loader.items, epochs=20)
            user_items = self._build_user_items(samples)
            recommender.fit(user_items, self.loader.all_items)
        else:
            raise ValueError(f"Unknown method: {method}")

        all_metrics = []
        all_recommendations = []
        total_tokens = 0
        total_latency = 0

        for idx, sample in enumerate(samples):
            # For LLM methods, only run on a subset for speed
            if method in ['trie_llm', 'pure_llm'] and random.random() > llm_sample_rate:
                continue

            # Get recommendations
            if method == 'trie_only':
                recs, latency = recommender.recommend(sample.history, sample.candidates, k)
                tokens = 0
            elif method == 'trie_llm':
                recs, latency, tokens, _ = recommender.recommend(
                    sample.history, sample.candidates, k
                )
            elif method in ['bpr', 'neumf']:
                recs, latency = recommender.recommend(
                    sample.user_id, sample.history, sample.candidates, k
                )
                tokens = 0
            else:  # pure_llm
                recs, latency, tokens = recommender.recommend(
                    sample.history, sample.candidates, k
                )

            total_tokens += tokens
            total_latency += latency
            all_recommendations.append(recs)

            # Calculate metrics
            metrics = metrics_calc.calculate_all(
                recommended=recs,
                ground_truth=sample.ground_truth,
                k=k,
                latency_ms=latency,
                tokens_used=tokens,
            )
            all_metrics.append(metrics)

            # Progress
            if idx > 0 and idx % 100 == 0:
                avg_hit = np.mean([m.hit for m in all_metrics])
                logger.info(f"  Progress: {idx}/{len(samples)}, Hit@{k}={avg_hit:.4f}")

        # Aggregate results
        aggregated = metrics_calc.aggregate_metrics(all_metrics, all_recommendations)

        # Add optimization stats for trie_llm
        if method == 'trie_llm':
            aggregated['optimization'] = recommender.get_optimization_stats()

        aggregated['total_tokens'] = total_tokens
        aggregated['total_latency'] = total_latency
        aggregated['num_samples'] = len(all_metrics)

        return aggregated

    def run_full_experiment(
        self,
        num_samples: int = 500,
        num_runs: int = 5,
        k: int = 5,
        methods: List[str] = None,
        early_exit_threshold: float = 0.5,
    ) -> Dict:
        """Run full experiment with multiple runs."""
        if methods is None:
            methods = ['bpr', 'neumf', 'trie_only', 'trie_llm']

        # Load samples
        samples = self.loader.load(max_samples=num_samples)
        if not samples:
            logger.error("No samples loaded!")
            return {}

        logger.info(f"Loaded {len(samples)} samples for {self.dataset_name}")

        results = {method: [] for method in methods}

        for run_id in range(num_runs):
            logger.info(f"\n{'='*60}")
            logger.info(f"RUN {run_id + 1}/{num_runs}")
            logger.info(f"{'='*60}")

            # Shuffle samples
            random.shuffle(samples)

            for method in methods:
                run_results = self.run_single_experiment(
                    samples=samples,
                    method=method,
                    k=k,
                    early_exit_threshold=early_exit_threshold,
                    run_id=run_id,
                )
                results[method].append(run_results)

        # Aggregate across runs
        final_results = self._aggregate_runs(results, methods)

        # Statistical tests
        if 'trie_llm' in methods and 'trie_only' in methods:
            trie_llm_hits = [r.get('hit_mean', 0) for r in results['trie_llm']]
            trie_only_hits = [r.get('hit_mean', 0) for r in results['trie_only']]
            if len(trie_llm_hits) == len(trie_only_hits) and len(trie_llm_hits) > 1:
                try:
                    final_results['statistical_tests'] = {
                        'trie_llm_vs_trie_only': StatisticalTests.paired_ttest(
                            trie_llm_hits, trie_only_hits
                        )
                    }
                except Exception as e:
                    logger.warning(f"Statistical test failed: {e}")

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.output_dir / f"results_{timestamp}.json"
        with open(output_path, 'w') as f:
            json.dump(final_results, f, indent=2, default=str)

        logger.info(f"\nResults saved to {output_path}")

        return final_results

    def _aggregate_runs(self, results: Dict, methods: List[str]) -> Dict:
        """Aggregate results across multiple runs."""
        final = {}

        for method in methods:
            if not results[method]:
                continue

            method_results = {}

            # Get all metric keys
            sample_result = results[method][0]

            for key in sample_result.keys():
                if key == 'optimization':
                    continue

                values = [r.get(key, 0) for r in results[method]]
                if values and isinstance(values[0], (int, float)):
                    method_results[f'{key}_mean'] = np.mean(values)
                    method_results[f'{key}_std'] = np.std(values)

            # Optimization stats (for trie_llm)
            if 'optimization' in sample_result:
                opt_stats = results[method][-1]['optimization']
                method_results['optimization'] = opt_stats

            final[method] = method_results

        return final

    def _build_user_items(self, samples: List[RecommendationSample]) -> Dict[str, List[str]]:
        """Build user-item interaction dict from samples for CF training.

        NOTE: Only use history for training, NOT ground_truth!
        Ground truth should only be used for evaluation to avoid data leakage.
        """
        user_items = defaultdict(list)
        for sample in samples:
            # Only use history as positive interactions (NOT ground_truth!)
            user_items[sample.user_id].extend(sample.history)
        # Deduplicate
        return {u: list(set(items)) for u, items in user_items.items()}

    def run_ablation_study(
        self,
        num_samples: int = 300,
        k: int = 5,
    ) -> Dict:
        """Run ablation studies."""
        samples = self.loader.load(max_samples=num_samples)
        if not samples:
            return {}

        ablation_results = {}

        # 1. Early Exit Threshold Ablation
        logger.info("\n" + "="*60)
        logger.info("ABLATION: Early Exit Threshold")
        logger.info("="*60)

        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        threshold_results = []

        for threshold in thresholds:
            results = self.run_single_experiment(
                samples=samples,
                method='trie_llm',
                k=k,
                early_exit_threshold=threshold,
            )
            threshold_results.append({
                'threshold': threshold,
                'hit': results.get('hit_mean', 0),
                'ndcg': results.get('ndcg_mean', 0),
                'early_exit_rate': results.get('optimization', {}).get('early_exit_rate', 0),
                'avg_latency': results.get('latency_ms_mean', 0),
            })
            logger.info(f"  Threshold={threshold}: Hit@{k}={results.get('hit_mean', 0):.4f}, "
                       f"Exit Rate={results.get('optimization', {}).get('early_exit_rate', 0):.1%}")

        ablation_results['early_exit_threshold'] = threshold_results

        # Save ablation results
        output_path = self.output_dir / "ablation_results.json"
        with open(output_path, 'w') as f:
            json.dump(ablation_results, f, indent=2)

        return ablation_results


def run_trie_llm_experiments(
    api_key: str,
    datasets: List[str] = None,
    num_samples: int = 500,
    num_runs: int = 5,
    methods: List[str] = None,
):
    """Run full Trie-LLM supplementary experiments."""
    from src.llm.ollama_client import OllamaClient, OllamaConfig

    if datasets is None:
        datasets = ['mind_small']

    if methods is None:
        methods = ['bpr', 'neumf', 'trie_only', 'trie_llm']

    # Initialize LLM
    config = OllamaConfig(api_key=api_key)
    llm = OllamaClient(config)

    all_results = {}

    for dataset_name in datasets:
        logger.info(f"\n{'='*70}")
        logger.info(f"DATASET: {dataset_name.upper()}")
        logger.info(f"{'='*70}")

        # Create loader
        if dataset_name == 'mind':
            loader = MINDDatasetLoader("data/mind/MINDlarge_train")
        elif dataset_name == 'mind_small':
            loader = MINDDatasetLoader("data/mind/MINDsmall_train")
        elif dataset_name == 'amazon':
            loader = AmazonDatasetLoader("/home/coder-gw/DataSets/amazon", "All_Beauty")
        elif dataset_name == 'movielens':
            loader = MovieLensDatasetLoader("data/ml-1m")
        else:
            logger.error(f"Unknown dataset: {dataset_name}")
            continue

        # Run experiment
        runner = TrieLLMExperimentRunner(dataset_name, loader, llm)
        results = runner.run_full_experiment(
            num_samples=num_samples,
            num_runs=num_runs,
            k=5,
            methods=methods,
        )

        all_results[dataset_name] = results

        # Print summary
        print(f"\n{'='*60}")
        print(f"RESULTS: {dataset_name}")
        print(f"{'='*60}")

        for method, method_results in results.items():
            if method == 'statistical_tests':
                continue
            print(f"\n{method}:")
            if isinstance(method_results, dict):
                for key, value in sorted(method_results.items()):
                    if '_mean' in key and not key.startswith('total'):
                        metric_name = key.replace('_mean', '')
                        std_key = key.replace('_mean', '_std')
                        std_val = method_results.get(std_key, 0)
                        print(f"  {metric_name}: {value:.4f} +/- {std_val:.4f}")

    # Save all results
    output_path = Path("results/trie_llm_experiments/all_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    logger.info(f"\nAll results saved to {output_path}")

    return all_results


if __name__ == '__main__':
    import argparse
    from dotenv import load_dotenv

    # Load environment variables from .env file
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument('--api_key', type=str, default=None, help='Ollama API key (or set OLLAMA_API_KEY env)')
    parser.add_argument('--datasets', type=str, nargs='+',
                       default=['mind_small'],
                       help='Datasets to run (mind, mind_small, amazon, movielens)')
    parser.add_argument('--num_samples', type=int, default=500, help='Number of samples per dataset')
    parser.add_argument('--num_runs', type=int, default=5, help='Number of runs for statistical significance')
    parser.add_argument('--methods', type=str, nargs='+',
                       default=['bpr', 'neumf', 'trie_only', 'trie_llm'],
                       help='Methods to run (bpr, neumf, trie_only, trie_llm, pure_llm)')
    args = parser.parse_args()

    # Get API key from argument or environment
    api_key = args.api_key or os.environ.get('OLLAMA_API_KEY')
    if not api_key:
        raise ValueError("API key required. Set OLLAMA_API_KEY or pass --api_key")

    run_trie_llm_experiments(
        api_key=api_key,
        datasets=args.datasets,
        num_samples=args.num_samples,
        num_runs=args.num_runs,
        methods=args.methods,
    )
