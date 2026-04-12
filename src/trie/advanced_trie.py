"""
Advanced Trie Implementations for Statistical Routing.

This module implements memory-efficient Trie variants:
1. Adaptive Radix Trie (ART) - Uses adaptive node sizes
2. Patricia Trie - Path compression for sparse paths
3. HOT (Height-Optimized Trie) - Compound nodes for cache efficiency

Reference:
- Leis et al., "The Adaptive Radix Tree: ARTful Indexing for Main-Memory Databases", ICDE 2013
- Morrison, "PATRICIA—Practical Algorithm To Retrieve Information Coded in Alphanumeric", JACM 1968
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from collections import defaultdict
from abc import ABC, abstractmethod
import sys

from .statistics import NodeStatistics


# =============================================================================
# Base Classes
# =============================================================================

@dataclass
class BaseTrieNode(ABC):
    """Abstract base class for Trie nodes."""
    node_id: str
    depth: int
    path: Tuple[str, ...]

    # Token statistics
    tokens: Set[str] = field(default_factory=set)
    avg_ctr: float = 0.0
    avg_lift: float = 0.0
    avg_frequency: float = 0.0
    token_count: int = 0

    # Expert assignment
    expert_affinity: Optional[int] = None

    def add_token(self, token: str, stats: NodeStatistics):
        """Add a token and update running statistics."""
        self.tokens.add(token)
        n = self.token_count
        self.avg_ctr = (self.avg_ctr * n + stats.ctr) / (n + 1)
        self.avg_lift = (self.avg_lift * n + stats.ctr_lift) / (n + 1)
        self.avg_frequency = (self.avg_frequency * n + stats.frequency) / (n + 1)
        self.token_count += 1

    @abstractmethod
    def get_memory_bytes(self) -> int:
        """Estimate memory usage in bytes."""
        pass


# =============================================================================
# Adaptive Radix Tree (ART)
# =============================================================================

@dataclass
class ARTNode4(BaseTrieNode):
    """
    ART Node with capacity for 4 children.
    Most memory-efficient for sparse nodes.
    """
    keys: List[str] = field(default_factory=list)  # Max 4 keys
    children: List['ARTNode'] = field(default_factory=list)  # Max 4 children

    def get_child(self, key: str) -> Optional['ARTNode']:
        """Get child by key."""
        for i, k in enumerate(self.keys):
            if k == key:
                return self.children[i]
        return None

    def set_child(self, key: str, child: 'ARTNode'):
        """Set child, growing node type if needed."""
        if key in self.keys:
            idx = self.keys.index(key)
            self.children[idx] = child
        else:
            self.keys.append(key)
            self.children.append(child)

    def should_grow(self) -> bool:
        """Check if node should grow to Node16."""
        return len(self.keys) > 4

    def get_memory_bytes(self) -> int:
        # Estimate: base + 4 * (key_ptr + child_ptr)
        return 64 + 4 * 16


@dataclass
class ARTNode16(BaseTrieNode):
    """
    ART Node with capacity for 16 children.
    Uses sorted keys for binary search.
    """
    keys: List[str] = field(default_factory=list)  # Max 16 keys
    children: List['ARTNode'] = field(default_factory=list)

    def get_child(self, key: str) -> Optional['ARTNode']:
        """Binary search for key."""
        import bisect
        idx = bisect.bisect_left(self.keys, key)
        if idx < len(self.keys) and self.keys[idx] == key:
            return self.children[idx]
        return None

    def set_child(self, key: str, child: 'ARTNode'):
        """Insert child maintaining sorted order."""
        import bisect
        idx = bisect.bisect_left(self.keys, key)
        if idx < len(self.keys) and self.keys[idx] == key:
            self.children[idx] = child
        else:
            self.keys.insert(idx, key)
            self.children.insert(idx, child)

    def should_grow(self) -> bool:
        """Check if node should grow to Node48."""
        return len(self.keys) > 16

    def get_memory_bytes(self) -> int:
        return 64 + 16 * 16


@dataclass
class ARTNode48(BaseTrieNode):
    """
    ART Node with capacity for 48 children.
    Uses 256-entry key index for O(1) lookup.
    """
    key_index: Dict[str, int] = field(default_factory=dict)  # key -> child index
    children: List[Optional['ARTNode']] = field(default_factory=lambda: [None] * 48)
    next_slot: int = 0

    def get_child(self, key: str) -> Optional['ARTNode']:
        """O(1) lookup via index."""
        idx = self.key_index.get(key)
        return self.children[idx] if idx is not None else None

    def set_child(self, key: str, child: 'ARTNode'):
        """Add child to next available slot."""
        if key in self.key_index:
            self.children[self.key_index[key]] = child
        else:
            self.key_index[key] = self.next_slot
            self.children[self.next_slot] = child
            self.next_slot += 1

    def should_grow(self) -> bool:
        """Check if node should grow to Node256."""
        return self.next_slot > 48

    def get_memory_bytes(self) -> int:
        return 64 + 256 + 48 * 8


@dataclass
class ARTNode256(BaseTrieNode):
    """
    ART Node with capacity for 256 children.
    Direct indexing for maximum speed.
    """
    children: Dict[str, 'ARTNode'] = field(default_factory=dict)

    def get_child(self, key: str) -> Optional['ARTNode']:
        """Direct O(1) lookup."""
        return self.children.get(key)

    def set_child(self, key: str, child: 'ARTNode'):
        """Direct assignment."""
        self.children[key] = child

    def should_grow(self) -> bool:
        """Node256 doesn't grow."""
        return False

    def get_memory_bytes(self) -> int:
        return 64 + len(self.children) * 16


# Union type for ART nodes
ARTNode = Union[ARTNode4, ARTNode16, ARTNode48, ARTNode256]


class AdaptiveRadixTrie:
    """
    Adaptive Radix Tree for statistical token organization.

    Uses adaptive node sizes based on fanout:
    - 1-4 children: Node4 (64 bytes)
    - 5-16 children: Node16 (128 bytes)
    - 17-48 children: Node48 (656 bytes)
    - 49-256 children: Node256 (2KB+)

    Memory savings: 2-10x compared to standard Trie with fixed node sizes.
    """

    def __init__(
        self,
        field_name: str,
        hierarchy_config: Optional[List[str]] = None,
    ):
        self.field_name = field_name
        self.hierarchy_config = hierarchy_config or ["frequency", "info"]

        self.root = self._create_node("root", 0, ())
        self.token_to_node: Dict[str, ARTNode] = {}
        self.leaf_nodes: List[ARTNode] = []

        # Statistics
        self.num_nodes = {4: 0, 16: 0, 48: 0, 256: 0}

    def _create_node(self, node_id: str, depth: int, path: Tuple[str, ...]) -> ARTNode4:
        """Create initial Node4 (will grow as needed)."""
        return ARTNode4(node_id=node_id, depth=depth, path=path)

    def _grow_node(self, node: ARTNode) -> ARTNode:
        """Grow node to larger capacity."""
        if isinstance(node, ARTNode4):
            new_node = ARTNode16(
                node_id=node.node_id,
                depth=node.depth,
                path=node.path,
                tokens=node.tokens,
                avg_ctr=node.avg_ctr,
                avg_lift=node.avg_lift,
                avg_frequency=node.avg_frequency,
                token_count=node.token_count,
                expert_affinity=node.expert_affinity,
            )
            new_node.keys = list(node.keys)
            new_node.children = list(node.children)
            self.num_nodes[4] -= 1
            self.num_nodes[16] += 1
            return new_node

        elif isinstance(node, ARTNode16):
            new_node = ARTNode48(
                node_id=node.node_id,
                depth=node.depth,
                path=node.path,
                tokens=node.tokens,
                avg_ctr=node.avg_ctr,
                avg_lift=node.avg_lift,
                avg_frequency=node.avg_frequency,
                token_count=node.token_count,
                expert_affinity=node.expert_affinity,
            )
            for i, (k, c) in enumerate(zip(node.keys, node.children)):
                new_node.key_index[k] = i
                new_node.children[i] = c
            new_node.next_slot = len(node.keys)
            self.num_nodes[16] -= 1
            self.num_nodes[48] += 1
            return new_node

        elif isinstance(node, ARTNode48):
            new_node = ARTNode256(
                node_id=node.node_id,
                depth=node.depth,
                path=node.path,
                tokens=node.tokens,
                avg_ctr=node.avg_ctr,
                avg_lift=node.avg_lift,
                avg_frequency=node.avg_frequency,
                token_count=node.token_count,
                expert_affinity=node.expert_affinity,
            )
            for k, idx in node.key_index.items():
                if node.children[idx] is not None:
                    new_node.children[k] = node.children[idx]
            self.num_nodes[48] -= 1
            self.num_nodes[256] += 1
            return new_node

        return node

    def _get_bucket(self, level: str, stats: NodeStatistics) -> str:
        """Get bucket value for hierarchy level."""
        if level == "frequency":
            return stats.frequency_bucket
        elif level == "info":
            return stats.info_bucket
        elif level == "stability":
            if stats.stability_score > 0.8:
                return "stable"
            elif stats.stability_score > 0.5:
                return "moderate"
            return "drifting"
        raise ValueError(f"Unknown level: {level}")

    def build(self, statistics: Dict[str, NodeStatistics]):
        """Build ART from token statistics."""
        self.num_nodes[4] = 1  # Root

        for token, stats in statistics.items():
            path = [self._get_bucket(l, stats) for l in self.hierarchy_config]

            current = self.root
            parent = None
            parent_key = None

            for depth, bucket in enumerate(path, start=1):
                # Check if need to grow current node
                if current.should_grow():
                    new_current = self._grow_node(current)
                    if parent:
                        parent.set_child(parent_key, new_current)
                    else:
                        self.root = new_current
                    current = new_current

                child = current.get_child(bucket)
                if child is None:
                    node_id = f"{self.field_name}_{'_'.join(path[:depth])}"
                    child = self._create_node(node_id, depth, tuple(path[:depth]))
                    current.set_child(bucket, child)
                    self.num_nodes[4] += 1

                parent = current
                parent_key = bucket
                current = child

            current.add_token(token, stats)
            self.token_to_node[token] = current

        self._collect_leaves()

    def _collect_leaves(self):
        """Collect all leaf nodes."""
        def traverse(node):
            children = []
            if isinstance(node, ARTNode4):
                children = node.children
            elif isinstance(node, ARTNode16):
                children = node.children
            elif isinstance(node, ARTNode48):
                children = [c for c in node.children if c]
            elif isinstance(node, ARTNode256):
                children = list(node.children.values())

            if not children:
                self.leaf_nodes.append(node)
            else:
                for child in children:
                    if child:
                        traverse(child)

        traverse(self.root)

    def get_memory_usage(self) -> Dict[str, int]:
        """Get detailed memory usage."""
        total = 0
        for size, count in self.num_nodes.items():
            if size == 4:
                total += count * 96
            elif size == 16:
                total += count * 192
            elif size == 48:
                total += count * 720
            elif size == 256:
                total += count * 2048

        return {
            'total_bytes': total,
            'total_kb': total / 1024,
            'node_counts': dict(self.num_nodes),
            'num_tokens': len(self.token_to_node),
        }

    def assign_experts(self, num_experts: int, strategy: str = "balanced"):
        """Assign experts to leaf nodes."""
        if strategy == "balanced":
            for i, node in enumerate(self.leaf_nodes):
                node.expert_affinity = i % num_experts
        elif strategy == "frequency_aware":
            tail_nodes = [n for n in self.leaf_nodes if "tail" in n.path]
            other_nodes = [n for n in self.leaf_nodes if "tail" not in n.path]

            tail_experts = max(1, int(num_experts * 0.5))
            for i, node in enumerate(tail_nodes):
                node.expert_affinity = i % tail_experts
            for i, node in enumerate(other_nodes):
                node.expert_affinity = tail_experts + (i % (num_experts - tail_experts))


# =============================================================================
# Patricia Trie (Path Compressed)
# =============================================================================

@dataclass
class PatriciaNode(BaseTrieNode):
    """
    Patricia Trie node with path compression.

    Instead of one node per character, compresses paths when
    there's only one child, storing the full path segment.
    """
    # Compressed path segment (multiple buckets combined)
    path_segment: Tuple[str, ...] = field(default_factory=tuple)

    # Children (branching points only)
    children: Dict[str, 'PatriciaNode'] = field(default_factory=dict)

    def get_memory_bytes(self) -> int:
        return 64 + len(self.path_segment) * 8 + len(self.children) * 16


class PatriciaTrie:
    """
    Patricia Trie with path compression.

    Collapses single-child chains into compressed path segments,
    reducing memory usage for sparse distributions.

    Example:
        Standard: root -> head -> high_info -> stable
        Patricia: root -> [head, high_info, stable] (single node)
    """

    def __init__(
        self,
        field_name: str,
        hierarchy_config: Optional[List[str]] = None,
    ):
        self.field_name = field_name
        self.hierarchy_config = hierarchy_config or ["frequency", "info"]

        self.root = PatriciaNode(node_id="root", depth=0, path=())
        self.token_to_node: Dict[str, PatriciaNode] = {}
        self.leaf_nodes: List[PatriciaNode] = []
        self.compression_ratio: float = 1.0

    def _get_bucket(self, level: str, stats: NodeStatistics) -> str:
        """Get bucket value for hierarchy level."""
        if level == "frequency":
            return stats.frequency_bucket
        elif level == "info":
            return stats.info_bucket
        elif level == "stability":
            if stats.stability_score > 0.8:
                return "stable"
            elif stats.stability_score > 0.5:
                return "moderate"
            return "drifting"
        raise ValueError(f"Unknown level: {level}")

    def build(self, statistics: Dict[str, NodeStatistics]):
        """Build Patricia Trie with path compression."""
        # First, build standard trie structure
        paths = []
        for token, stats in statistics.items():
            path = tuple(self._get_bucket(l, stats) for l in self.hierarchy_config)
            paths.append((token, path, stats))

        # Group by path
        from collections import defaultdict
        path_groups = defaultdict(list)
        for token, path, stats in paths:
            path_groups[path].append((token, stats))

        # Build compressed structure
        total_uncompressed = 0
        total_compressed = 0

        for path, tokens_stats in path_groups.items():
            total_uncompressed += len(path)

            # Find compression point
            current = self.root
            i = 0

            while i < len(path):
                bucket = path[i]

                if bucket in current.children:
                    child = current.children[bucket]
                    # Check if path matches compressed segment
                    seg_len = len(child.path_segment)
                    if path[i:i+seg_len] == child.path_segment:
                        current = child
                        i += seg_len
                    else:
                        # Need to split
                        # Find divergence point
                        j = 0
                        while j < seg_len and i + j < len(path):
                            if child.path_segment[j] != path[i + j]:
                                break
                            j += 1

                        if j > 0:
                            # Split node
                            split_node = PatriciaNode(
                                node_id=f"{self.field_name}_split_{i}",
                                depth=i,
                                path=path[:i+j],
                                path_segment=child.path_segment[:j],
                            )
                            child.path_segment = child.path_segment[j:]
                            child.depth = i + j
                            split_node.children[child.path_segment[0] if child.path_segment else ''] = child
                            current.children[bucket] = split_node
                            current = split_node
                            i += j
                        else:
                            i += 1
                else:
                    # Create new compressed node
                    remaining = path[i:]
                    node_id = f"{self.field_name}_{'_'.join(path)}"
                    new_node = PatriciaNode(
                        node_id=node_id,
                        depth=len(path),
                        path=path,
                        path_segment=remaining,
                    )
                    current.children[bucket] = new_node
                    current = new_node
                    total_compressed += 1
                    break

            # Add tokens to leaf
            for token, stats in tokens_stats:
                current.add_token(token, stats)
                self.token_to_node[token] = current

        if total_uncompressed > 0:
            self.compression_ratio = total_compressed / total_uncompressed

        self._collect_leaves()

    def _collect_leaves(self):
        """Collect all leaf nodes."""
        def traverse(node):
            if not node.children:
                self.leaf_nodes.append(node)
            else:
                for child in node.children.values():
                    traverse(child)
        traverse(self.root)

    def get_memory_usage(self) -> Dict[str, int]:
        """Get memory usage statistics."""
        def count_nodes(node):
            return 1 + sum(count_nodes(c) for c in node.children.values())

        total_nodes = count_nodes(self.root)
        avg_segment = sum(len(n.path_segment) for n in self.leaf_nodes) / max(len(self.leaf_nodes), 1)

        return {
            'total_nodes': total_nodes,
            'leaf_nodes': len(self.leaf_nodes),
            'compression_ratio': self.compression_ratio,
            'avg_segment_length': avg_segment,
            'num_tokens': len(self.token_to_node),
        }

    def assign_experts(self, num_experts: int, strategy: str = "balanced"):
        """Assign experts to leaf nodes."""
        for i, node in enumerate(self.leaf_nodes):
            node.expert_affinity = i % num_experts


# =============================================================================
# Comparison Utilities
# =============================================================================

def compare_trie_implementations(
    statistics: Dict[str, NodeStatistics],
    field_name: str = "test_field",
    hierarchy_config: List[str] = None,
) -> Dict[str, Any]:
    """
    Compare memory usage and structure of different Trie implementations.
    """
    config = hierarchy_config or ["frequency", "info"]

    # Standard Trie (from builder.py)
    from .builder import StatisticalTrie
    standard = StatisticalTrie(field_name, config)
    standard.build(statistics)

    # ART
    art = AdaptiveRadixTrie(field_name, config)
    art.build(statistics)

    # Patricia
    patricia = PatriciaTrie(field_name, config)
    patricia.build(statistics)

    # Estimate standard trie memory (64 bytes per node, fixed size)
    def count_standard_nodes(node):
        return 1 + sum(count_standard_nodes(c) for c in node.children.values())

    standard_nodes = count_standard_nodes(standard.root)
    standard_memory = standard_nodes * 64

    art_memory = art.get_memory_usage()
    patricia_memory = patricia.get_memory_usage()

    return {
        'num_tokens': len(statistics),
        'standard': {
            'nodes': standard_nodes,
            'memory_bytes': standard_memory,
            'leaf_nodes': len(standard.leaf_nodes),
        },
        'art': {
            'memory_bytes': art_memory['total_bytes'],
            'node_distribution': art_memory['node_counts'],
            'leaf_nodes': len(art.leaf_nodes),
            'memory_reduction': 1 - art_memory['total_bytes'] / max(standard_memory, 1),
        },
        'patricia': {
            'nodes': patricia_memory['total_nodes'],
            'leaf_nodes': len(patricia.leaf_nodes),
            'compression_ratio': patricia_memory['compression_ratio'],
            'avg_segment_length': patricia_memory['avg_segment_length'],
        },
    }
