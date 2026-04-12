"""
Count-Min Sketch: Probabilistic frequency estimation.

Space: O(1/ε · log(1/δ)) - independent of number of elements
Time: O(d) = O(1) for update and query
Error: Overestimates by at most εN with probability 1-δ

Reference: Cormode & Muthukrishnan, 2005
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import mmh3  # MurmurHash3


class CountMinSketch:
    """
    Count-Min Sketch for streaming frequency estimation.

    Suitable for:
    - Estimating token frequencies without storing all tokens
    - Finding heavy hitters (top-K frequent items)
    - Memory-constrained environments
    """

    def __init__(
        self,
        width: int = 1000,
        depth: int = 5,
        epsilon: Optional[float] = None,
        delta: Optional[float] = None,
    ):
        """
        Initialize Count-Min Sketch.

        Args:
            width: Number of counters per row (w = ceil(e/ε))
            depth: Number of hash functions (d = ceil(ln(1/δ)))
            epsilon: Error tolerance (alternative to width)
            delta: Failure probability (alternative to depth)
        """
        if epsilon is not None:
            width = int(np.ceil(np.e / epsilon))
        if delta is not None:
            depth = int(np.ceil(np.log(1 / delta)))

        self.width = width
        self.depth = depth
        self.table = np.zeros((depth, width), dtype=np.int64)
        self.total_count = 0

        # Generate random seeds for hash functions (must be 32-bit)
        self._seeds = [(i * 0xDEAD + 0xBEEF) % (2**31) for i in range(depth)]

    def _hash(self, item: int, seed_idx: int) -> int:
        """Hash an item to a bucket index."""
        return mmh3.hash(str(item), self._seeds[seed_idx], signed=False) % self.width

    def update(self, item: int, count: int = 1):
        """
        Update count for an item.

        Args:
            item: Item identifier (token_id)
            count: Count to add (default 1)
        """
        self.total_count += count
        for i in range(self.depth):
            j = self._hash(item, i)
            self.table[i, j] += count

    def estimate(self, item: int) -> int:
        """
        Estimate the count of an item.

        Returns minimum across all hash functions (least overestimate).

        Args:
            item: Item identifier

        Returns:
            Estimated count (may overestimate, never underestimates)
        """
        return min(
            self.table[i, self._hash(item, i)]
            for i in range(self.depth)
        )

    def estimate_frequency(self, item: int) -> float:
        """Estimate relative frequency (0 to 1)."""
        if self.total_count == 0:
            return 0.0
        return self.estimate(item) / self.total_count

    def merge(self, other: 'CountMinSketch') -> 'CountMinSketch':
        """
        Merge another sketch into this one.

        Useful for distributed/parallel processing.
        """
        if self.width != other.width or self.depth != other.depth:
            raise ValueError("Sketches must have same dimensions")

        result = CountMinSketch(self.width, self.depth)
        result.table = self.table + other.table
        result.total_count = self.total_count + other.total_count
        return result

    def memory_usage_bytes(self) -> int:
        """Return memory usage in bytes."""
        return self.table.nbytes

    def error_bound(self) -> Tuple[float, float]:
        """
        Return theoretical error bounds.

        Returns:
            (epsilon, delta): Error and failure probability
        """
        epsilon = np.e / self.width
        delta = np.exp(-self.depth)
        return epsilon, delta


class ConservativeCountMinSketch(CountMinSketch):
    """
    Conservative Update variant of Count-Min Sketch.

    Reduces overestimation by only incrementing counters
    that are at the current minimum estimate.
    """

    def update(self, item: int, count: int = 1):
        """Conservative update: only increment minimum counters."""
        self.total_count += count

        # Find current minimum
        indices = [self._hash(item, i) for i in range(self.depth)]
        values = [self.table[i, indices[i]] for i in range(self.depth)]
        min_val = min(values)

        # Only increment counters at minimum
        new_val = min_val + count
        for i in range(self.depth):
            if self.table[i, indices[i]] < new_val:
                self.table[i, indices[i]] = new_val


class CountMinSketchWithHeap(CountMinSketch):
    """
    Count-Min Sketch with min-heap for top-K tracking.

    Maintains approximate top-K heavy hitters.
    """

    def __init__(
        self,
        width: int = 1000,
        depth: int = 5,
        top_k: int = 100,
    ):
        super().__init__(width, depth)
        self.top_k = top_k
        self.heap: Dict[int, int] = {}  # item -> count
        self.min_in_heap = 0

    def update(self, item: int, count: int = 1):
        """Update with top-K tracking."""
        super().update(item, count)

        estimated = self.estimate(item)

        if item in self.heap:
            self.heap[item] = estimated
        elif len(self.heap) < self.top_k:
            self.heap[item] = estimated
            self._update_min()
        elif estimated > self.min_in_heap:
            # Remove minimum, add new item
            min_item = min(self.heap, key=self.heap.get)
            del self.heap[min_item]
            self.heap[item] = estimated
            self._update_min()

    def _update_min(self):
        """Update minimum value in heap."""
        if self.heap:
            self.min_in_heap = min(self.heap.values())
        else:
            self.min_in_heap = 0

    def get_top_k(self) -> List[Tuple[int, int]]:
        """Return top-K items with their estimated counts."""
        return sorted(self.heap.items(), key=lambda x: -x[1])
