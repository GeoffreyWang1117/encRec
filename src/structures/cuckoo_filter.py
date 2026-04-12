"""
Cuckoo Filter: Space-efficient probabilistic set membership with deletion.

Space: ~7 bits per entry at 3% FPP
Time: O(1) for lookup, O(1) amortized for insert
Advantage over Bloom Filter: Supports deletion

Reference: Fan et al., "Cuckoo Filter: Practically Better Than Bloom", 2014
"""

import numpy as np
from typing import Optional, Tuple
import mmh3


class CuckooFilter:
    """
    Cuckoo Filter for probabilistic set membership testing.

    Suitable for:
    - Fast cold-start detection (is token seen in training?)
    - Deduplication with deletion support
    - Memory-efficient set membership
    """

    def __init__(
        self,
        capacity: int = 10000,
        bucket_size: int = 4,
        fingerprint_bits: int = 8,
        max_kicks: int = 500,
    ):
        """
        Initialize Cuckoo Filter.

        Args:
            capacity: Number of buckets
            bucket_size: Entries per bucket (typically 4)
            fingerprint_bits: Bits per fingerprint (8 = 3% FPP)
            max_kicks: Maximum relocations before failure
        """
        self.capacity = capacity
        self.bucket_size = bucket_size
        self.fingerprint_bits = fingerprint_bits
        self.max_kicks = max_kicks

        # Storage: 2D array of fingerprints
        # 0 = empty slot
        self.buckets = np.zeros(
            (capacity, bucket_size),
            dtype=np.uint8 if fingerprint_bits <= 8 else np.uint16
        )

        self.size = 0
        self.fingerprint_mask = (1 << fingerprint_bits) - 1

    def _fingerprint(self, item: int) -> int:
        """Generate fingerprint for an item."""
        fp = mmh3.hash(str(item), seed=12345678, signed=False) & self.fingerprint_mask
        # Ensure fingerprint is non-zero (0 = empty)
        return fp if fp != 0 else 1

    def _hash1(self, item: int) -> int:
        """First hash function for bucket index."""
        return mmh3.hash(str(item), seed=87654321, signed=False) % self.capacity

    def _hash2(self, index1: int, fingerprint: int) -> int:
        """
        Second hash function using partial-key cuckoo hashing.

        index2 = index1 XOR hash(fingerprint)
        This allows computing alternate bucket from just fingerprint.
        """
        return (index1 ^ mmh3.hash(str(fingerprint), seed=11223344, signed=False)) % self.capacity

    def _get_bucket_indices(self, item: int) -> Tuple[int, int, int]:
        """Get both bucket indices and fingerprint for an item."""
        fp = self._fingerprint(item)
        i1 = self._hash1(item)
        i2 = self._hash2(i1, fp)
        return i1, i2, fp

    def insert(self, item: int) -> bool:
        """
        Insert an item into the filter.

        Args:
            item: Item to insert

        Returns:
            True if successful, False if filter is full
        """
        i1, i2, fp = self._get_bucket_indices(item)

        # Try to insert in first bucket
        for j in range(self.bucket_size):
            if self.buckets[i1, j] == 0:
                self.buckets[i1, j] = fp
                self.size += 1
                return True

        # Try to insert in second bucket
        for j in range(self.bucket_size):
            if self.buckets[i2, j] == 0:
                self.buckets[i2, j] = fp
                self.size += 1
                return True

        # Both buckets full, need to relocate
        # Randomly choose one of the two buckets
        i = i1 if np.random.random() < 0.5 else i2

        for _ in range(self.max_kicks):
            # Randomly select an entry to kick
            j = np.random.randint(self.bucket_size)

            # Swap fingerprints
            fp, self.buckets[i, j] = self.buckets[i, j], fp

            # Find alternate bucket for kicked fingerprint
            i = self._hash2(i, fp)

            # Try to insert in alternate bucket
            for j in range(self.bucket_size):
                if self.buckets[i, j] == 0:
                    self.buckets[i, j] = fp
                    self.size += 1
                    return True

        # Too many kicks, filter is too full
        return False

    def lookup(self, item: int) -> bool:
        """
        Check if an item might be in the filter.

        Args:
            item: Item to check

        Returns:
            True if possibly in set, False if definitely not
        """
        i1, i2, fp = self._get_bucket_indices(item)

        # Check first bucket
        if fp in self.buckets[i1]:
            return True

        # Check second bucket
        if fp in self.buckets[i2]:
            return True

        return False

    def delete(self, item: int) -> bool:
        """
        Delete an item from the filter.

        Args:
            item: Item to delete

        Returns:
            True if deleted, False if not found
        """
        i1, i2, fp = self._get_bucket_indices(item)

        # Try to delete from first bucket
        for j in range(self.bucket_size):
            if self.buckets[i1, j] == fp:
                self.buckets[i1, j] = 0
                self.size -= 1
                return True

        # Try to delete from second bucket
        for j in range(self.bucket_size):
            if self.buckets[i2, j] == fp:
                self.buckets[i2, j] = 0
                self.size -= 1
                return True

        return False

    def __contains__(self, item: int) -> bool:
        """Support 'in' operator."""
        return self.lookup(item)

    def load_factor(self) -> float:
        """Return current load factor."""
        return self.size / (self.capacity * self.bucket_size)

    def memory_usage_bytes(self) -> int:
        """Return memory usage in bytes."""
        return self.buckets.nbytes

    def false_positive_rate(self) -> float:
        """
        Theoretical false positive rate.

        FPP ≈ 8b/2^f where b = bucket_size, f = fingerprint_bits
        """
        return (8 * self.bucket_size) / (2 ** self.fingerprint_bits)


class ScalableCuckooFilter:
    """
    Scalable Cuckoo Filter that grows automatically.

    Creates new filters when capacity is reached.
    """

    def __init__(
        self,
        initial_capacity: int = 10000,
        bucket_size: int = 4,
        fingerprint_bits: int = 8,
        growth_factor: float = 2.0,
    ):
        self.bucket_size = bucket_size
        self.fingerprint_bits = fingerprint_bits
        self.growth_factor = growth_factor
        self.initial_capacity = initial_capacity

        self.filters = [
            CuckooFilter(initial_capacity, bucket_size, fingerprint_bits)
        ]

    def insert(self, item: int) -> bool:
        """Insert with automatic scaling."""
        # Try current filter
        if self.filters[-1].insert(item):
            return True

        # Current filter full, create new one
        new_capacity = int(self.filters[-1].capacity * self.growth_factor)
        new_filter = CuckooFilter(
            new_capacity, self.bucket_size, self.fingerprint_bits
        )

        if new_filter.insert(item):
            self.filters.append(new_filter)
            return True

        return False

    def lookup(self, item: int) -> bool:
        """Check all filters."""
        return any(f.lookup(item) for f in self.filters)

    def __contains__(self, item: int) -> bool:
        return self.lookup(item)

    @property
    def size(self) -> int:
        return sum(f.size for f in self.filters)
