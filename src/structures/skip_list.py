"""
Skip List: Probabilistic sorted data structure.

Time: O(log n) expected for search, insert, delete
Space: O(n) expected
Advantage: Simpler than balanced trees, good cache locality

Reference: Pugh, "Skip Lists: A Probabilistic Alternative to Balanced Trees", 1990
"""

import random
from typing import Any, Generator, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class SkipListNode:
    """Node in a skip list."""
    key: float
    value: Any
    forward: List['SkipListNode']

    def __init__(self, key: float, value: Any, level: int):
        self.key = key
        self.value = value
        self.forward = [None] * (level + 1)


class SkipList:
    """
    Skip List for maintaining sorted elements with O(log n) operations.

    Suitable for:
    - Dynamic frequency ranking (token_id -> frequency)
    - Range queries (all tokens with frequency in [a, b])
    - Percentile queries (top 10% by frequency)
    """

    def __init__(self, max_level: int = 16, p: float = 0.5):
        """
        Initialize Skip List.

        Args:
            max_level: Maximum number of levels
            p: Probability of adding a level (typically 0.5)
        """
        self.max_level = max_level
        self.p = p
        self.level = 0
        self.size = 0

        # Header node with minimum key
        self.header = SkipListNode(float('-inf'), None, max_level)

    def _random_level(self) -> int:
        """Generate random level for new node."""
        level = 0
        while random.random() < self.p and level < self.max_level:
            level += 1
        return level

    def search(self, key: float) -> Optional[Any]:
        """
        Search for a key.

        Args:
            key: Key to search for

        Returns:
            Value if found, None otherwise
        """
        current = self.header

        # Start from highest level
        for i in range(self.level, -1, -1):
            while current.forward[i] and current.forward[i].key < key:
                current = current.forward[i]

        current = current.forward[0]

        if current and current.key == key:
            return current.value
        return None

    def insert(self, key: float, value: Any) -> bool:
        """
        Insert a key-value pair.

        Args:
            key: Key to insert
            value: Associated value

        Returns:
            True if inserted, False if key already exists (value updated)
        """
        update = [None] * (self.max_level + 1)
        current = self.header

        # Find insert position at each level
        for i in range(self.level, -1, -1):
            while current.forward[i] and current.forward[i].key < key:
                current = current.forward[i]
            update[i] = current

        current = current.forward[0]

        # Key already exists, update value
        if current and current.key == key:
            current.value = value
            return False

        # Generate level for new node
        new_level = self._random_level()

        # Expand list level if needed
        if new_level > self.level:
            for i in range(self.level + 1, new_level + 1):
                update[i] = self.header
            self.level = new_level

        # Create and insert new node
        new_node = SkipListNode(key, value, new_level)
        for i in range(new_level + 1):
            new_node.forward[i] = update[i].forward[i]
            update[i].forward[i] = new_node

        self.size += 1
        return True

    def delete(self, key: float) -> bool:
        """
        Delete a key.

        Args:
            key: Key to delete

        Returns:
            True if deleted, False if not found
        """
        update = [None] * (self.max_level + 1)
        current = self.header

        for i in range(self.level, -1, -1):
            while current.forward[i] and current.forward[i].key < key:
                current = current.forward[i]
            update[i] = current

        current = current.forward[0]

        if current and current.key == key:
            for i in range(self.level + 1):
                if update[i].forward[i] != current:
                    break
                update[i].forward[i] = current.forward[i]

            # Reduce level if needed
            while self.level > 0 and self.header.forward[self.level] is None:
                self.level -= 1

            self.size -= 1
            return True

        return False

    def range_query(
        self,
        low: float,
        high: float,
    ) -> Generator[Tuple[float, Any], None, None]:
        """
        Find all elements with keys in [low, high].

        Time: O(log n + k) where k is number of results

        Args:
            low: Lower bound (inclusive)
            high: Upper bound (inclusive)

        Yields:
            (key, value) pairs in sorted order
        """
        current = self.header

        # Find first element >= low
        for i in range(self.level, -1, -1):
            while current.forward[i] and current.forward[i].key < low:
                current = current.forward[i]

        current = current.forward[0]

        # Iterate while key <= high
        while current and current.key <= high:
            yield current.key, current.value
            current = current.forward[0]

    def get_percentile(self, percentile: float) -> Optional[Tuple[float, Any]]:
        """
        Get element at given percentile.

        Args:
            percentile: Value between 0 and 100

        Returns:
            (key, value) at that percentile, or None if empty
        """
        if self.size == 0:
            return None

        target_rank = int(self.size * percentile / 100)
        target_rank = max(0, min(target_rank, self.size - 1))

        current = self.header.forward[0]
        for _ in range(target_rank):
            if current.forward[0]:
                current = current.forward[0]
            else:
                break

        if current:
            return current.key, current.value
        return None

    def get_top_k(self, k: int) -> List[Tuple[float, Any]]:
        """
        Get top-k elements by key (highest keys).

        Args:
            k: Number of elements

        Returns:
            List of (key, value) pairs in descending order
        """
        result = []
        current = self.header.forward[0]

        # Traverse to end
        elements = []
        while current:
            elements.append((current.key, current.value))
            current = current.forward[0]

        # Return last k elements in reverse
        return elements[-k:][::-1]

    def __len__(self) -> int:
        return self.size

    def __contains__(self, key: float) -> bool:
        return self.search(key) is not None


class FrequencySkipList:
    """
    Skip List specialized for token frequency management.

    Maintains tokens sorted by frequency for efficient:
    - Frequency-based bucketing
    - Range queries (tokens with freq in [a, b])
    - Top-K queries (most/least frequent tokens)
    """

    def __init__(self):
        # Primary index: frequency -> list of token_ids
        self.freq_to_tokens = SkipList()
        # Reverse index: token_id -> frequency
        self.token_to_freq = {}

    def update_frequency(self, token_id: int, new_freq: int):
        """
        Update frequency for a token.

        Handles moving token to new frequency bucket.
        """
        old_freq = self.token_to_freq.get(token_id)

        if old_freq == new_freq:
            return

        # Remove from old frequency bucket
        if old_freq is not None:
            tokens = self.freq_to_tokens.search(old_freq)
            if tokens:
                tokens.discard(token_id)
                if not tokens:
                    self.freq_to_tokens.delete(old_freq)

        # Add to new frequency bucket
        tokens = self.freq_to_tokens.search(new_freq)
        if tokens is None:
            tokens = set()
            self.freq_to_tokens.insert(new_freq, tokens)
        tokens.add(token_id)

        self.token_to_freq[token_id] = new_freq

    def get_frequency(self, token_id: int) -> int:
        """Get frequency of a token."""
        return self.token_to_freq.get(token_id, 0)

    def get_tokens_in_range(
        self,
        low_freq: int,
        high_freq: int,
    ) -> List[int]:
        """Get all tokens with frequency in [low, high]."""
        result = []
        for freq, tokens in self.freq_to_tokens.range_query(low_freq, high_freq):
            result.extend(tokens)
        return result

    def get_frequency_bucket(
        self,
        token_id: int,
        num_buckets: int = 10,
    ) -> int:
        """
        Get frequency bucket for a token.

        Uses percentile-based bucketing.
        """
        freq = self.get_frequency(token_id)
        if freq == 0:
            return 0  # Cold-start bucket

        # Find percentile of this frequency
        lower_count = 0
        for f, tokens in self.freq_to_tokens.range_query(0, freq - 1):
            lower_count += len(tokens)

        percentile = lower_count / max(len(self.token_to_freq), 1) * 100
        return min(int(percentile / (100 / num_buckets)), num_buckets - 1)

    def get_top_k_tokens(self, k: int) -> List[Tuple[int, int]]:
        """Get k most frequent tokens."""
        result = []
        for freq, tokens in self.freq_to_tokens.get_top_k(k):
            for token_id in tokens:
                result.append((token_id, freq))
                if len(result) >= k:
                    return result
        return result

    def __len__(self) -> int:
        return len(self.token_to_freq)
