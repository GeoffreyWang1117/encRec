"""
Trie-based Retrieval for Recommendation.

This module implements Trie as a retrieval/indexing structure rather than
a feature generator. Key use cases:
1. Fast candidate retrieval (替代向量ANN)
2. LLM context augmentation (替代RAG)
3. Real-time feature lookup (替代Redis)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
from collections import defaultdict
import heapq
import time


@dataclass
class ItemStats:
    """Statistics for an item in the Trie."""
    item_id: str
    frequency: int = 0
    positive_count: int = 0
    ctr: float = 0.0
    category: str = ""
    embedding_idx: int = -1  # Index into embedding matrix
    last_update: float = 0.0

    def update(self, label: int):
        self.frequency += 1
        self.positive_count += label
        self.ctr = self.positive_count / self.frequency if self.frequency > 0 else 0
        self.last_update = time.time()


@dataclass
class TrieNode:
    """Node in the retrieval Trie."""
    children: Dict[str, 'TrieNode'] = field(default_factory=dict)
    items: Dict[str, ItemStats] = field(default_factory=dict)  # Items ending at this node
    prefix_count: int = 0  # Total items under this prefix
    prefix_ctr_sum: float = 0.0  # Sum of CTRs for aggregation


class RetrievalTrie:
    """
    Trie optimized for fast retrieval operations.

    Key features:
    - O(k) exact lookup
    - O(k + m) prefix search
    - O(1) top-k by CTR (with precomputed index)
    - Hierarchical statistics aggregation
    """

    def __init__(self, min_count_for_stats: int = 10):
        self.root = TrieNode()
        self.total_items = 0
        self.min_count = min_count_for_stats

        # Indexes for fast retrieval
        self._ctr_index: List[Tuple[float, str]] = []  # (ctr, item_id) heap
        self._freq_index: List[Tuple[int, str]] = []   # (freq, item_id) heap
        self._category_index: Dict[str, Set[str]] = defaultdict(set)

    def insert(self, item_id: str, label: int = 0, category: str = ""):
        """Insert or update an item."""
        node = self.root
        for char in item_id:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
            node.prefix_count += 1

        if item_id not in node.items:
            node.items[item_id] = ItemStats(item_id=item_id, category=category)
            self.total_items += 1

        node.items[item_id].update(label)
        if category:
            self._category_index[category].add(item_id)

    def get(self, item_id: str) -> Optional[ItemStats]:
        """O(k) exact lookup."""
        node = self.root
        for char in item_id:
            if char not in node.children:
                return None
            node = node.children[char]
        return node.items.get(item_id)

    def prefix_search(self, prefix: str, max_results: int = 100) -> List[ItemStats]:
        """O(k + m) prefix search."""
        node = self.root
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]

        # Collect all items under this prefix
        results = []
        self._collect_items(node, results, max_results)
        return results

    def _collect_items(self, node: TrieNode, results: List[ItemStats], max_results: int):
        """DFS to collect items."""
        if len(results) >= max_results:
            return

        for item_stats in node.items.values():
            results.append(item_stats)
            if len(results) >= max_results:
                return

        for child in node.children.values():
            self._collect_items(child, results, max_results)
            if len(results) >= max_results:
                return

    def get_prefix_stats(self, prefix: str) -> Optional[Dict[str, float]]:
        """Get aggregated statistics for a prefix (hierarchical aggregation)."""
        node = self.root
        for char in prefix:
            if char not in node.children:
                return None
            node = node.children[char]

        if node.prefix_count < self.min_count:
            return None

        # Aggregate statistics
        total_freq = 0
        total_positive = 0
        items_count = 0

        def aggregate(n: TrieNode):
            nonlocal total_freq, total_positive, items_count
            for stats in n.items.values():
                total_freq += stats.frequency
                total_positive += stats.positive_count
                items_count += 1
            for child in n.children.values():
                aggregate(child)

        aggregate(node)

        return {
            'prefix': prefix,
            'item_count': items_count,
            'total_freq': total_freq,
            'avg_ctr': total_positive / total_freq if total_freq > 0 else 0,
        }

    def build_indexes(self):
        """Build auxiliary indexes for fast retrieval."""
        self._ctr_index = []
        self._freq_index = []

        def collect(node: TrieNode):
            for item_id, stats in node.items.items():
                heapq.heappush(self._ctr_index, (-stats.ctr, item_id))
                heapq.heappush(self._freq_index, (-stats.frequency, item_id))
            for child in node.children.values():
                collect(child)

        collect(self.root)

    def get_top_by_ctr(self, k: int) -> List[ItemStats]:
        """Get top-k items by CTR."""
        if not self._ctr_index:
            self.build_indexes()

        results = []
        temp_heap = list(self._ctr_index)
        heapq.heapify(temp_heap)

        for _ in range(min(k, len(temp_heap))):
            neg_ctr, item_id = heapq.heappop(temp_heap)
            stats = self.get(item_id)
            if stats:
                results.append(stats)

        return results

    def get_top_by_freq(self, k: int) -> List[ItemStats]:
        """Get top-k items by frequency."""
        if not self._freq_index:
            self.build_indexes()

        results = []
        temp_heap = list(self._freq_index)
        heapq.heapify(temp_heap)

        for _ in range(min(k, len(temp_heap))):
            neg_freq, item_id = heapq.heappop(temp_heap)
            stats = self.get(item_id)
            if stats:
                results.append(stats)

        return results

    def get_by_category(self, category: str, max_results: int = 100) -> List[ItemStats]:
        """Get items by category."""
        results = []
        for item_id in list(self._category_index.get(category, []))[:max_results]:
            stats = self.get(item_id)
            if stats:
                results.append(stats)
        return results


class TrieCandidateRetrieval:
    """
    Use Trie for candidate retrieval in recommendation.

    Replaces vector ANN with exact/prefix matching.
    """

    def __init__(self, trie: RetrievalTrie):
        self.trie = trie

    def retrieve(
        self,
        user_history: List[str],
        k: int = 100,
        diversity_factor: float = 0.3,
    ) -> List[str]:
        """
        Retrieve candidate items based on user history.

        Args:
            user_history: List of item IDs user interacted with
            k: Number of candidates to retrieve
            diversity_factor: Fraction of candidates from exploration

        Returns:
            List of candidate item IDs
        """
        candidates = set()

        # 1. Prefix-based retrieval (similar items)
        for item in user_history[-10:]:  # Recent history
            for prefix_len in [6, 4, 2]:
                if len(candidates) >= k * (1 - diversity_factor):
                    break
                prefix = item[:prefix_len]
                similar = self.trie.prefix_search(prefix, max_results=20)
                for stats in similar:
                    if stats.item_id not in user_history:
                        candidates.add(stats.item_id)

        # 2. Category-based retrieval
        for item in user_history[-5:]:
            stats = self.trie.get(item)
            if stats and stats.category:
                category_items = self.trie.get_by_category(stats.category, max_results=20)
                for cat_stats in category_items:
                    if cat_stats.item_id not in user_history:
                        candidates.add(cat_stats.item_id)
                        if len(candidates) >= k * (1 - diversity_factor):
                            break

        # 3. Popular items for exploration
        num_explore = int(k * diversity_factor)
        popular = self.trie.get_top_by_ctr(num_explore * 2)
        for stats in popular:
            if stats.item_id not in user_history and stats.item_id not in candidates:
                candidates.add(stats.item_id)
                if len(candidates) >= k:
                    break

        return list(candidates)[:k]


class TrieFeatureStore:
    """
    Use Trie as real-time feature store.

    Replaces Redis/Feature Service for token-level features.
    """

    def __init__(self, trie: RetrievalTrie):
        self.trie = trie

    def get_features(self, tokens: List[str]) -> np.ndarray:
        """
        Get features for a batch of tokens.

        Returns:
            (N, 4) array: [frequency, ctr, is_hot, is_cold]
        """
        features = np.zeros((len(tokens), 4), dtype=np.float32)

        for i, token in enumerate(tokens):
            stats = self.trie.get(token)

            if stats:
                features[i, 0] = np.log1p(stats.frequency)  # Log frequency
                features[i, 1] = stats.ctr
                features[i, 2] = 1.0 if stats.frequency > 100 else 0.0  # Is hot
                features[i, 3] = 0.0
            else:
                # Unknown token - try prefix aggregation
                for prefix_len in [4, 2]:
                    prefix_stats = self.trie.get_prefix_stats(token[:prefix_len])
                    if prefix_stats:
                        features[i, 0] = np.log1p(prefix_stats['total_freq'] / prefix_stats['item_count'])
                        features[i, 1] = prefix_stats['avg_ctr']
                        features[i, 2] = 0.0
                        features[i, 3] = 1.0  # Is cold (using aggregated stats)
                        break

        return features


class TrieLLMAugmenter:
    """
    Use Trie to augment LLM prompts with structured statistics.

    Replaces vector RAG with statistical context.
    """

    def __init__(self, trie: RetrievalTrie):
        self.trie = trie

    def augment_prompt(
        self,
        query: str,
        user_history: List[str] = None,
        max_context_items: int = 10,
    ) -> str:
        """
        Augment LLM prompt with Trie-based statistics.

        Args:
            query: User query or search term
            user_history: Previous interactions
            max_context_items: Max items to include in context

        Returns:
            Augmented prompt with statistical context
        """
        context_items = []

        # 1. Direct matches
        direct = self.trie.get(query)
        if direct:
            context_items.append(self._format_item(direct, "exact_match"))

        # 2. Prefix matches
        for prefix_len in [6, 4]:
            prefix_items = self.trie.prefix_search(query[:prefix_len], max_results=5)
            for item in prefix_items:
                if len(context_items) < max_context_items:
                    context_items.append(self._format_item(item, "prefix_match"))

        # 3. User history context
        if user_history:
            for item_id in user_history[-3:]:
                stats = self.trie.get(item_id)
                if stats:
                    context_items.append(self._format_item(stats, "user_history"))

        # 4. Popular items for reference
        popular = self.trie.get_top_by_ctr(3)
        for item in popular:
            context_items.append(self._format_item(item, "popular"))

        # Build prompt
        context_str = "\n".join(context_items[:max_context_items])

        return f"""用户查询: {query}

相关商品统计信息:
{context_str}

请根据以上统计信息，推荐最合适的商品并解释原因。
考虑因素: CTR(点击率)、热度(频率)、与用户历史的相关性。
"""

    def _format_item(self, stats: ItemStats, source: str) -> str:
        return f"- [{source}] {stats.item_id}: CTR={stats.ctr:.2%}, 热度={stats.frequency}, 类别={stats.category or 'N/A'}"


# Benchmark utilities
def benchmark_retrieval(trie: RetrievalTrie, n_queries: int = 1000) -> Dict[str, float]:
    """Benchmark retrieval operations."""
    import random
    import string

    # Generate random queries
    queries = [''.join(random.choices(string.ascii_lowercase, k=8)) for _ in range(n_queries)]

    results = {}

    # Exact lookup
    start = time.time()
    for q in queries:
        trie.get(q)
    results['exact_lookup_ms'] = (time.time() - start) * 1000 / n_queries

    # Prefix search
    start = time.time()
    for q in queries:
        trie.prefix_search(q[:4], max_results=10)
    results['prefix_search_ms'] = (time.time() - start) * 1000 / n_queries

    # Top-k by CTR
    trie.build_indexes()
    start = time.time()
    for _ in range(n_queries):
        trie.get_top_by_ctr(100)
    results['top_k_ctr_ms'] = (time.time() - start) * 1000 / n_queries

    return results
