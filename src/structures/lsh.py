"""
Locality Sensitive Hashing (LSH): Approximate nearest neighbor search.

Time: Sub-linear query time O(n^ρ) where ρ < 1
Space: O(nL) where L is number of hash tables

Reference: Indyk & Motwani, "Approximate Nearest Neighbors", 1998
"""

import numpy as np
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict


class LSH:
    """
    Locality Sensitive Hashing for Euclidean distance.

    Uses random projections (hyperplanes) to hash similar
    vectors to the same bucket with high probability.

    Suitable for:
    - Finding similar tokens based on embeddings
    - Cold-start routing (find similar known tokens)
    - Approximate nearest neighbor queries
    """

    def __init__(
        self,
        dim: int,
        num_tables: int = 10,
        hash_size: int = 8,
        bucket_width: float = 4.0,
    ):
        """
        Initialize LSH.

        Args:
            dim: Dimension of input vectors
            num_tables: Number of hash tables (more = better recall, more space)
            hash_size: Number of hash functions per table (more = better precision)
            bucket_width: Width of hash buckets (larger = more collisions)
        """
        self.dim = dim
        self.num_tables = num_tables
        self.hash_size = hash_size
        self.bucket_width = bucket_width

        # Random projection matrices: one per table
        # Each is (hash_size, dim)
        self.projections = [
            np.random.randn(hash_size, dim).astype(np.float32)
            for _ in range(num_tables)
        ]

        # Random offsets for each hash function
        self.offsets = [
            np.random.uniform(0, bucket_width, hash_size).astype(np.float32)
            for _ in range(num_tables)
        ]

        # Hash tables: table_idx -> bucket_key -> set of (id, vector)
        self.tables: List[Dict[tuple, Set[int]]] = [
            defaultdict(set) for _ in range(num_tables)
        ]

        # Store vectors for retrieval
        self.vectors: Dict[int, np.ndarray] = {}

    def _hash(self, vector: np.ndarray, table_idx: int) -> tuple:
        """
        Compute hash key for a vector in given table.

        Uses: h(v) = floor((a·v + b) / w)
        """
        projection = self.projections[table_idx]
        offset = self.offsets[table_idx]

        # Project and discretize
        projected = projection @ vector
        bucket_ids = np.floor((projected + offset) / self.bucket_width).astype(int)

        return tuple(bucket_ids)

    def insert(self, item_id: int, vector: np.ndarray):
        """
        Insert a vector into all hash tables.

        Args:
            item_id: Unique identifier for this vector
            vector: Feature vector of shape (dim,)
        """
        vector = np.asarray(vector, dtype=np.float32)
        self.vectors[item_id] = vector

        for i in range(self.num_tables):
            key = self._hash(vector, i)
            self.tables[i][key].add(item_id)

    def query(
        self,
        vector: np.ndarray,
        k: int = 10,
        return_distances: bool = False,
    ) -> List[int]:
        """
        Find approximate nearest neighbors.

        Args:
            vector: Query vector
            k: Number of neighbors to return
            return_distances: Whether to return distances too

        Returns:
            List of item_ids (and optionally distances)
        """
        vector = np.asarray(vector, dtype=np.float32)
        candidates = set()

        # Collect candidates from all tables
        for i in range(self.num_tables):
            key = self._hash(vector, i)
            candidates.update(self.tables[i].get(key, set()))

        if not candidates:
            return [] if not return_distances else ([], [])

        # Compute exact distances for candidates
        distances = []
        for item_id in candidates:
            dist = np.linalg.norm(self.vectors[item_id] - vector)
            distances.append((item_id, dist))

        # Sort by distance and return top-k
        distances.sort(key=lambda x: x[1])
        top_k = distances[:k]

        if return_distances:
            return [x[0] for x in top_k], [x[1] for x in top_k]
        return [x[0] for x in top_k]

    def delete(self, item_id: int):
        """Remove a vector from all tables."""
        if item_id not in self.vectors:
            return

        vector = self.vectors[item_id]
        for i in range(self.num_tables):
            key = self._hash(vector, i)
            self.tables[i][key].discard(item_id)

        del self.vectors[item_id]

    def __len__(self) -> int:
        return len(self.vectors)


class CosineLSH:
    """
    LSH for Cosine Similarity (SimHash).

    Uses random hyperplanes to hash similar vectors.
    Probability of same hash = 1 - θ/π where θ is angle between vectors.

    Better suited for normalized embeddings.
    """

    def __init__(
        self,
        dim: int,
        num_tables: int = 10,
        hash_size: int = 16,
    ):
        """
        Initialize Cosine LSH.

        Args:
            dim: Dimension of input vectors
            num_tables: Number of hash tables
            hash_size: Bits per hash (more = finer granularity)
        """
        self.dim = dim
        self.num_tables = num_tables
        self.hash_size = hash_size

        # Random hyperplanes for each table
        self.hyperplanes = [
            np.random.randn(hash_size, dim).astype(np.float32)
            for _ in range(num_tables)
        ]

        # Hash tables
        self.tables: List[Dict[int, Set[int]]] = [
            defaultdict(set) for _ in range(num_tables)
        ]

        self.vectors: Dict[int, np.ndarray] = {}

    def _hash(self, vector: np.ndarray, table_idx: int) -> int:
        """
        Compute SimHash for a vector.

        Each bit = sign of dot product with random hyperplane.
        """
        projections = self.hyperplanes[table_idx] @ vector
        bits = (projections > 0).astype(int)

        # Convert bit array to integer
        hash_val = 0
        for bit in bits:
            hash_val = (hash_val << 1) | bit

        return hash_val

    def insert(self, item_id: int, vector: np.ndarray):
        """Insert a vector (will be normalized)."""
        vector = np.asarray(vector, dtype=np.float32)
        # Normalize for cosine similarity
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm

        self.vectors[item_id] = vector

        for i in range(self.num_tables):
            key = self._hash(vector, i)
            self.tables[i][key].add(item_id)

    def query(
        self,
        vector: np.ndarray,
        k: int = 10,
        return_similarities: bool = False,
    ) -> List[int]:
        """
        Find approximate nearest neighbors by cosine similarity.

        Args:
            vector: Query vector
            k: Number of neighbors
            return_similarities: Whether to return similarities

        Returns:
            List of item_ids sorted by similarity
        """
        vector = np.asarray(vector, dtype=np.float32)
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm

        candidates = set()

        # Collect candidates
        for i in range(self.num_tables):
            key = self._hash(vector, i)
            candidates.update(self.tables[i].get(key, set()))

        if not candidates:
            return [] if not return_similarities else ([], [])

        # Compute exact cosine similarities
        similarities = []
        for item_id in candidates:
            sim = np.dot(self.vectors[item_id], vector)
            similarities.append((item_id, sim))

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: -x[1])
        top_k = similarities[:k]

        if return_similarities:
            return [x[0] for x in top_k], [x[1] for x in top_k]
        return [x[0] for x in top_k]

    def __len__(self) -> int:
        return len(self.vectors)


class TokenSimilarityIndex:
    """
    Token similarity index using LSH.

    Specialized for recommendation system use case:
    finding similar tokens for cold-start routing.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_tables: int = 10,
        hash_size: int = 12,
    ):
        self.lsh = CosineLSH(embedding_dim, num_tables, hash_size)
        self.token_stats: Dict[int, Dict] = {}  # token_id -> stats

    def add_token(
        self,
        token_id: int,
        embedding: np.ndarray,
        frequency: int = 0,
        ctr: float = 0.0,
    ):
        """Add a token with its embedding and statistics."""
        self.lsh.insert(token_id, embedding)
        self.token_stats[token_id] = {
            'frequency': frequency,
            'ctr': ctr,
        }

    def find_similar_frequent_tokens(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        min_frequency: int = 10,
    ) -> List[Tuple[int, float, Dict]]:
        """
        Find similar tokens that are also frequent.

        Useful for cold-start: use statistics from similar
        well-observed tokens.

        Returns:
            List of (token_id, similarity, stats)
        """
        # Get more candidates to filter by frequency
        candidates, sims = self.lsh.query(
            query_embedding, k=k*3, return_similarities=True
        )

        # Filter by frequency
        results = []
        for token_id, sim in zip(candidates, sims):
            stats = self.token_stats.get(token_id, {})
            if stats.get('frequency', 0) >= min_frequency:
                results.append((token_id, sim, stats))

        # Return top k
        return results[:k]

    def get_routing_prior(
        self,
        query_embedding: np.ndarray,
        expert_assignments: Dict[int, int],
        k: int = 5,
    ) -> np.ndarray:
        """
        Get routing prior for a cold-start token.

        Based on expert assignments of similar frequent tokens.

        Args:
            query_embedding: Embedding of cold-start token
            expert_assignments: token_id -> expert_id mapping
            k: Number of similar tokens to consider

        Returns:
            Probability distribution over experts
        """
        similar = self.find_similar_frequent_tokens(query_embedding, k)

        if not similar:
            # No similar tokens found, return uniform
            num_experts = max(expert_assignments.values()) + 1 if expert_assignments else 8
            return np.ones(num_experts) / num_experts

        # Weighted voting based on similarity
        num_experts = max(expert_assignments.values()) + 1
        expert_scores = np.zeros(num_experts)

        for token_id, sim, stats in similar:
            if token_id in expert_assignments:
                expert_idx = expert_assignments[token_id]
                # Weight by similarity and token frequency
                weight = sim * np.log(stats.get('frequency', 1) + 1)
                expert_scores[expert_idx] += weight

        # Normalize to probability
        if expert_scores.sum() > 0:
            expert_scores /= expert_scores.sum()
        else:
            expert_scores = np.ones(num_experts) / num_experts

        return expert_scores
