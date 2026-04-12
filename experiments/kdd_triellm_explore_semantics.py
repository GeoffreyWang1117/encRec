"""
LLM Recommendation Experiment with Semantic Data.

Uses MovieLens dataset where LLM can understand movie titles and genres.
This provides a more meaningful evaluation of Trie-augmented LLM recommendation.
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.trie.retrieval_trie import RetrievalTrie
from src.llm.ollama_client import OllamaClient, OllamaConfig


@dataclass
class Movie:
    """Movie information."""
    movie_id: int
    title: str
    genres: List[str]
    avg_rating: float = 0.0
    num_ratings: int = 0


class MovieLensSemanticData:
    """Load and process MovieLens data with semantic information."""

    def __init__(self, data_dir: str = "data/ml-1m"):
        self.data_dir = Path(data_dir)
        self.movies: Dict[int, Movie] = {}
        self.ratings: pd.DataFrame = None
        self.user_histories: Dict[int, List[int]] = {}

    def load(self):
        """Load all data."""
        # Load movies
        logger.info("Loading movies...")
        movies_df = pd.read_csv(
            self.data_dir / "movies.dat",
            sep="::",
            names=["movie_id", "title", "genres"],
            engine="python",
            encoding="latin-1"
        )

        for _, row in movies_df.iterrows():
            self.movies[row['movie_id']] = Movie(
                movie_id=row['movie_id'],
                title=row['title'],
                genres=row['genres'].split('|')
            )

        # Load ratings
        logger.info("Loading ratings...")
        self.ratings = pd.read_csv(
            self.data_dir / "ratings.dat",
            sep="::",
            names=["user_id", "movie_id", "rating", "timestamp"],
            engine="python"
        )

        # Calculate movie statistics
        movie_stats = self.ratings.groupby('movie_id').agg({
            'rating': ['mean', 'count']
        }).reset_index()
        movie_stats.columns = ['movie_id', 'avg_rating', 'num_ratings']

        for _, row in movie_stats.iterrows():
            if row['movie_id'] in self.movies:
                self.movies[row['movie_id']].avg_rating = row['avg_rating']
                self.movies[row['movie_id']].num_ratings = row['num_ratings']

        # Build user histories (positive ratings >= 4)
        logger.info("Building user histories...")
        positive_ratings = self.ratings[self.ratings['rating'] >= 4]
        self.user_histories = positive_ratings.groupby('user_id')['movie_id'].apply(list).to_dict()

        logger.info(f"Loaded {len(self.movies)} movies, {len(self.user_histories)} users")

    def build_trie(self) -> RetrievalTrie:
        """Build Trie from movie data."""
        trie = RetrievalTrie()

        for movie_id, movie in self.movies.items():
            # Insert each movie multiple times based on ratings
            ctr = movie.avg_rating / 5.0  # Normalize to 0-1
            num_positive = int(movie.num_ratings * ctr)
            num_negative = int(movie.num_ratings - num_positive)

            # Use title as item_id for interpretability
            item_id = f"movie_{movie_id}"
            category = movie.genres[0] if movie.genres else "Unknown"

            for _ in range(num_positive):
                trie.insert(item_id, 1, category)
            for _ in range(num_negative):
                trie.insert(item_id, 0, category)

        trie.build_indexes()
        return trie

    def get_user_sessions(self, num_users: int = 100) -> List[Dict]:
        """Create user sessions for evaluation."""
        sessions = []

        for user_id, movie_ids in list(self.user_histories.items())[:num_users]:
            if len(movie_ids) < 15:
                continue

            # Split into history and ground truth
            history_ids = movie_ids[:10]
            ground_truth_ids = movie_ids[10:15]

            # Get movie titles for LLM
            history_titles = []
            for mid in history_ids:
                if mid in self.movies:
                    history_titles.append(self.movies[mid].title)

            ground_truth_items = [f"movie_{mid}" for mid in ground_truth_ids]

            sessions.append({
                'user_id': user_id,
                'history_ids': history_ids,
                'history_titles': history_titles,
                'history_items': [f"movie_{mid}" for mid in history_ids],
                'ground_truth': ground_truth_items,
            })

        return sessions


class TrieSemanticCompressor:
    """Compress context using Trie stats + semantic info."""

    def __init__(self, trie: RetrievalTrie, movies: Dict[int, Movie]):
        self.trie = trie
        self.movies = movies

    def compress_history(self, history_titles: List[str], history_ids: List[int]) -> str:
        """Create compressed but semantic history."""
        compressed = []
        for title, mid in zip(history_titles[:5], history_ids[:5]):
            movie = self.movies.get(mid)
            if movie:
                stats = self.trie.get(f"movie_{mid}")
                rating = f"★{movie.avg_rating:.1f}" if movie.avg_rating > 0 else ""
                compressed.append(f"{title[:30]}{rating}")
            else:
                compressed.append(title[:30])
        return " → ".join(compressed)

    def compress_candidates(self, candidate_ids: List[str]) -> str:
        """Create compressed candidate list with key stats."""
        lines = []
        by_genre = defaultdict(list)

        for item_id in candidate_ids[:20]:
            mid = int(item_id.replace("movie_", ""))
            movie = self.movies.get(mid)
            if movie:
                genre = movie.genres[0] if movie.genres else "Other"
                stats = self.trie.get(item_id)
                ctr = stats.ctr if stats else 0
                by_genre[genre].append((movie.title[:25], ctr, mid))

        for genre, items in by_genre.items():
            item_strs = [f"{t}(★{c*5:.1f})" for t, c, _ in items[:3]]
            lines.append(f"[{genre}]: {', '.join(item_strs)}")

        return "\n".join(lines)


def run_experiment(
    api_key: str,
    num_users: int = 50,
    num_recs: int = 5,
):
    """Run the semantic LLM recommendation experiment."""
    print("="*70)
    print("Trie-Augmented LLM Recommendation with Semantic Data")
    print("="*70)

    # Load data
    data = MovieLensSemanticData()
    data.load()

    # Build Trie
    trie = data.build_trie()
    compressor = TrieSemanticCompressor(trie, data.movies)

    # Get sessions
    sessions = data.get_user_sessions(num_users)
    print(f"\nEvaluating on {len(sessions)} users")

    # Initialize LLM
    config = OllamaConfig(api_key=api_key)
    llm_client = OllamaClient(config)

    results = {
        'trie_only': [],
        'trie_llm_compressed': [],
        'pure_llm': [],
    }

    for i, session in enumerate(sessions[:20]):  # Limit for cost
        print(f"\n--- User {i+1}/{min(len(sessions), 20)} ---")

        # Get candidates from Trie
        from src.trie.retrieval_trie import TrieCandidateRetrieval
        retriever = TrieCandidateRetrieval(trie)
        candidates = retriever.retrieve(session['history_items'], k=30)

        ground_truth = set(session['ground_truth'])

        # Method 1: Trie-only
        start = time.time()
        trie_recs = candidates[:num_recs]
        trie_latency = (time.time() - start) * 1000
        trie_hits = sum(1 for r in trie_recs if r in ground_truth)
        results['trie_only'].append({
            'latency_ms': trie_latency,
            'hits': trie_hits,
            'tokens': 0,
        })

        # Method 2: Trie+LLM with compression
        start = time.time()
        compressed_history = compressor.compress_history(
            session['history_titles'], session['history_ids']
        )
        compressed_candidates = compressor.compress_candidates(candidates[:15])

        prompt = f"""电影推荐任务

用户喜欢的电影: {compressed_history}

候选电影:
{compressed_candidates}

从候选中选择{num_recs}个最适合该用户的电影，每行输出一个电影名称。
"""
        try:
            response = llm_client.chat(
                messages=[{"role": "user", "content": prompt}],
                model="glm-4.6",
                temperature=0.3,
                max_tokens=200,
            )
            content = response.get('message', {}).get('content', '')

            # Parse recommendations
            trie_llm_recs = []
            for item_id in candidates:
                mid = int(item_id.replace("movie_", ""))
                movie = data.movies.get(mid)
                if movie and movie.title[:20] in content:
                    trie_llm_recs.append(item_id)

            # Fallback
            while len(trie_llm_recs) < num_recs:
                for c in candidates:
                    if c not in trie_llm_recs:
                        trie_llm_recs.append(c)
                        break

            trie_llm_recs = trie_llm_recs[:num_recs]

        except Exception as e:
            logger.error(f"LLM error: {e}")
            trie_llm_recs = candidates[:num_recs]
            content = ""

        trie_llm_latency = (time.time() - start) * 1000
        trie_llm_hits = sum(1 for r in trie_llm_recs if r in ground_truth)
        results['trie_llm_compressed'].append({
            'latency_ms': trie_llm_latency,
            'hits': trie_llm_hits,
            'tokens': len(prompt.split()) + len(content.split()),
        })

        # Method 3: Pure LLM (every 5th user, expensive)
        if i % 5 == 0:
            start = time.time()
            full_history = ", ".join(session['history_titles'])
            all_movies = [data.movies[int(c.replace("movie_", ""))].title
                         for c in candidates[:30] if int(c.replace("movie_", "")) in data.movies]
            full_candidates = ", ".join(all_movies)

            full_prompt = f"""电影推荐任务

用户喜欢的电影: {full_history}

候选电影: {full_candidates}

从候选中选择{num_recs}个最适合该用户的电影，每行输出一个电影名称。
"""
            try:
                response = llm_client.chat(
                    messages=[{"role": "user", "content": full_prompt}],
                    model="glm-4.6",
                    temperature=0.3,
                    max_tokens=200,
                )
                full_content = response.get('message', {}).get('content', '')
            except:
                full_content = ""

            pure_llm_latency = (time.time() - start) * 1000
            results['pure_llm'].append({
                'latency_ms': pure_llm_latency,
                'tokens': len(full_prompt.split()) + len(full_content.split()),
            })

        print(f"  Trie-only: {trie_latency:.1f}ms, hits={trie_hits}/{num_recs}")
        print(f"  Trie+LLM: {trie_llm_latency:.1f}ms, hits={trie_llm_hits}/{num_recs}, "
              f"tokens={results['trie_llm_compressed'][-1]['tokens']}")

    # Summary
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)

    for method, method_results in results.items():
        if not method_results:
            continue
        avg_latency = np.mean([r['latency_ms'] for r in method_results])
        avg_tokens = np.mean([r.get('tokens', 0) for r in method_results])
        avg_hits = np.mean([r.get('hits', 0) for r in method_results]) if 'hits' in method_results[0] else 0

        print(f"\n{method}:")
        print(f"  Avg Latency: {avg_latency:.1f}ms")
        print(f"  Avg Tokens: {avg_tokens:.1f}")
        if avg_hits > 0:
            print(f"  Avg Hits@{num_recs}: {avg_hits:.2f}")

    # Comparison
    if results['trie_llm_compressed'] and results['trie_only']:
        trie_hits = np.mean([r['hits'] for r in results['trie_only']])
        llm_hits = np.mean([r['hits'] for r in results['trie_llm_compressed']])
        improvement = (llm_hits - trie_hits) / max(trie_hits, 0.01) * 100

        trie_latency = np.mean([r['latency_ms'] for r in results['trie_only']])
        llm_latency = np.mean([r['latency_ms'] for r in results['trie_llm_compressed']])

        print(f"\n--- Comparison ---")
        print(f"Trie+LLM vs Trie-only:")
        print(f"  Hit rate improvement: {improvement:+.1f}%")
        print(f"  Latency increase: {llm_latency - trie_latency:.1f}ms")

    if results['pure_llm'] and results['trie_llm_compressed']:
        pure_tokens = np.mean([r['tokens'] for r in results['pure_llm']])
        compressed_tokens = np.mean([r['tokens'] for r in results['trie_llm_compressed']])
        token_reduction = (pure_tokens - compressed_tokens) / pure_tokens * 100

        print(f"\nCompressed vs Pure LLM:")
        print(f"  Token reduction: {token_reduction:.1f}%")

    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--api_key', type=str, required=True)
    parser.add_argument('--num_users', type=int, default=50)
    args = parser.parse_args()

    run_experiment(args.api_key, args.num_users)
