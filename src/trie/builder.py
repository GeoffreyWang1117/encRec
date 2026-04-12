"""
Trie structure builder for encrypted token organization.

In this project, Trie is NOT a character prefix tree, but a
HIERARCHICAL STATISTICAL INDEX where paths represent statistical
properties rather than semantic meaning.

The hierarchy is constructed based on:
- Frequency tier (Head/Mid/Tail)
- Information tier (High/Neutral/Low CTR lift)
- Co-occurrence cluster
- Temporal stability
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
from collections import defaultdict
from .statistics import TrieStatistics, NodeStatistics


@dataclass
class TrieNode:
    """
    A node in the statistical Trie.

    Unlike traditional tries where nodes represent characters,
    here nodes represent statistical buckets/clusters.
    """
    # Node identification
    node_id: str
    depth: int
    path: Tuple[str, ...]  # Path from root (e.g., ("head", "high_info", "stable"))

    # Tokens assigned to this node
    tokens: Set[str] = field(default_factory=set)

    # Aggregated statistics for this node
    avg_ctr: float = 0.0
    avg_lift: float = 0.0
    avg_frequency: float = 0.0
    token_count: int = 0

    # Children nodes
    children: Dict[str, 'TrieNode'] = field(default_factory=dict)

    # For MoE routing
    expert_affinity: Optional[int] = None  # Which expert this node maps to

    def add_token(self, token: str, stats: NodeStatistics):
        """Add a token to this node and update aggregated statistics."""
        self.tokens.add(token)
        n = self.token_count
        # Incremental mean update
        self.avg_ctr = (self.avg_ctr * n + stats.ctr) / (n + 1)
        self.avg_lift = (self.avg_lift * n + stats.ctr_lift) / (n + 1)
        self.avg_frequency = (self.avg_frequency * n + stats.frequency) / (n + 1)
        self.token_count += 1

    def to_dict(self) -> Dict[str, Any]:
        """Serialize node for storage/debugging."""
        return {
            'node_id': self.node_id,
            'depth': self.depth,
            'path': self.path,
            'token_count': self.token_count,
            'avg_ctr': self.avg_ctr,
            'avg_lift': self.avg_lift,
            'avg_frequency': self.avg_frequency,
            'expert_affinity': self.expert_affinity,
            'children': list(self.children.keys()),
        }


class StatisticalTrie:
    """
    A Trie organized by statistical properties of encrypted tokens.

    Structure example:
    root
    ├── head (high frequency)
    │   ├── high_info (high CTR lift)
    │   │   ├── stable
    │   │   └── drifting
    │   └── low_info
    │       ├── stable
    │       └── drifting
    ├── mid
    │   └── ...
    └── tail (low frequency)
        └── ...
    """

    def __init__(
        self,
        field_name: str,
        hierarchy_config: Optional[List[str]] = None,
    ):
        """
        Args:
            field_name: Name of the feature field this Trie is for
            hierarchy_config: List of hierarchy levels to use.
                Default: ["frequency", "info", "stability"]
        """
        self.field_name = field_name
        self.hierarchy_config = hierarchy_config or ["frequency", "info"]

        # Root node
        self.root = TrieNode(
            node_id="root",
            depth=0,
            path=()
        )

        # Token to node mapping for fast lookup
        self.token_to_node: Dict[str, TrieNode] = {}

        # All leaf nodes (for MoE expert assignment)
        self.leaf_nodes: List[TrieNode] = []

    def _get_bucket_value(
        self,
        level: str,
        stats: NodeStatistics
    ) -> str:
        """Get the bucket value for a hierarchy level."""
        if level == "frequency":
            return stats.frequency_bucket  # head/mid/tail
        elif level == "info":
            return stats.info_bucket  # high_info/neutral/low_info
        elif level == "stability":
            if stats.stability_score > 0.8:
                return "stable"
            elif stats.stability_score > 0.5:
                return "moderate"
            else:
                return "drifting"
        else:
            raise ValueError(f"Unknown hierarchy level: {level}")

    def build(self, statistics: Dict[str, NodeStatistics]):
        """
        Build the Trie from token statistics.

        Args:
            statistics: Dict mapping token -> NodeStatistics
        """
        for token, stats in statistics.items():
            # Determine path through hierarchy
            path = []
            for level in self.hierarchy_config:
                bucket = self._get_bucket_value(level, stats)
                path.append(bucket)

            # Traverse/create path in Trie
            current = self.root
            for depth, bucket in enumerate(path, start=1):
                if bucket not in current.children:
                    node_id = f"{self.field_name}_{'_'.join(path[:depth])}"
                    current.children[bucket] = TrieNode(
                        node_id=node_id,
                        depth=depth,
                        path=tuple(path[:depth])
                    )
                current = current.children[bucket]

            # Add token to leaf node
            current.add_token(token, stats)
            self.token_to_node[token] = current

        # Collect leaf nodes
        self._collect_leaves(self.root)

    def _collect_leaves(self, node: TrieNode):
        """Recursively collect all leaf nodes."""
        if not node.children:
            self.leaf_nodes.append(node)
        else:
            for child in node.children.values():
                self._collect_leaves(child)

    def get_node_for_token(self, token: str) -> Optional[TrieNode]:
        """Get the Trie node containing a token."""
        return self.token_to_node.get(token)

    def get_path_for_token(self, token: str) -> Optional[Tuple[str, ...]]:
        """Get the hierarchical path for a token."""
        node = self.token_to_node.get(token)
        return node.path if node else None

    def assign_experts(self, num_experts: int, strategy: str = "balanced"):
        """
        Assign MoE experts to leaf nodes.

        Args:
            num_experts: Number of experts in MoE
            strategy: Assignment strategy
                - "balanced": Distribute evenly
                - "frequency_aware": More experts for tail
                - "info_aware": More experts for high-info
        """
        if strategy == "balanced":
            for i, node in enumerate(self.leaf_nodes):
                node.expert_affinity = i % num_experts

        elif strategy == "frequency_aware":
            # Assign more experts to tail (they need more specialization)
            tail_nodes = [n for n in self.leaf_nodes if "tail" in n.path]
            head_nodes = [n for n in self.leaf_nodes if "head" in n.path]
            mid_nodes = [n for n in self.leaf_nodes if "mid" in n.path]

            # 50% experts for tail, 30% for mid, 20% for head
            tail_experts = max(1, int(num_experts * 0.5))
            mid_experts = max(1, int(num_experts * 0.3))
            head_experts = num_experts - tail_experts - mid_experts

            expert_idx = 0
            for nodes, n_experts in [
                (head_nodes, head_experts),
                (mid_nodes, mid_experts),
                (tail_nodes, tail_experts)
            ]:
                for i, node in enumerate(nodes):
                    node.expert_affinity = expert_idx + (i % max(1, n_experts))
                expert_idx += n_experts

        elif strategy == "info_aware":
            # Assign dedicated experts to high-info nodes
            high_info = [n for n in self.leaf_nodes if "high_info" in n.path]
            others = [n for n in self.leaf_nodes if "high_info" not in n.path]

            high_info_experts = max(1, int(num_experts * 0.4))
            other_experts = num_experts - high_info_experts

            for i, node in enumerate(high_info):
                node.expert_affinity = i % high_info_experts

            for i, node in enumerate(others):
                node.expert_affinity = high_info_experts + (i % max(1, other_experts))

    def get_expert_for_token(self, token: str) -> int:
        """Get the assigned expert index for a token."""
        node = self.token_to_node.get(token)
        if node and node.expert_affinity is not None:
            return node.expert_affinity
        return 0  # Default expert

    def summarize(self) -> Dict[str, Any]:
        """Get summary statistics for the Trie."""
        def count_nodes(node):
            return 1 + sum(count_nodes(c) for c in node.children.values())

        return {
            'field': self.field_name,
            'hierarchy': self.hierarchy_config,
            'total_nodes': count_nodes(self.root),
            'leaf_nodes': len(self.leaf_nodes),
            'total_tokens': len(self.token_to_node),
            'avg_tokens_per_leaf': len(self.token_to_node) / max(len(self.leaf_nodes), 1),
            'leaf_summaries': [
                {
                    'path': node.path,
                    'token_count': node.token_count,
                    'avg_ctr': node.avg_ctr,
                    'expert': node.expert_affinity,
                }
                for node in self.leaf_nodes
            ]
        }


class TrieBuilder:
    """
    High-level builder for constructing Statistical Tries from data.
    """

    def __init__(
        self,
        hierarchy_config: Optional[List[str]] = None,
        num_experts: int = 8,
        expert_strategy: str = "frequency_aware",
    ):
        self.hierarchy_config = hierarchy_config or ["frequency", "info"]
        self.num_experts = num_experts
        self.expert_strategy = expert_strategy

        self.tries: Dict[str, StatisticalTrie] = {}
        self.statistics = TrieStatistics()

    def fit(
        self,
        data: List[Dict[str, Any]],
        sparse_fields: List[str],
        label_key: str = "label",
    ):
        """
        Build Tries from data.

        Args:
            data: List of samples, each with sparse features and label
            sparse_fields: Names of sparse/categorical fields
            label_key: Key for the label in each sample
        """
        # Collect statistics
        for sample in data:
            label = sample[label_key]
            for field in sparse_fields:
                if field in sample:
                    token = str(sample[field])
                    self.statistics.update(field, token, label)

        # Compute statistics
        field_stats = self.statistics.compute_statistics()

        # Build Trie for each field
        for field in sparse_fields:
            if field in field_stats:
                trie = StatisticalTrie(
                    field_name=field,
                    hierarchy_config=self.hierarchy_config
                )
                trie.build(field_stats[field])
                trie.assign_experts(self.num_experts, self.expert_strategy)
                self.tries[field] = trie

        return self

    def get_trie(self, field: str) -> Optional[StatisticalTrie]:
        """Get the Trie for a specific field."""
        return self.tries.get(field)

    def get_expert_indices(
        self,
        sample: Dict[str, Any],
        sparse_fields: List[str]
    ) -> List[int]:
        """
        Get expert indices for a sample based on its tokens.

        Args:
            sample: Sample with sparse features
            sparse_fields: Fields to consider

        Returns:
            List of expert indices (one per field, or aggregated)
        """
        experts = []
        for field in sparse_fields:
            if field in sample and field in self.tries:
                token = str(sample[field])
                expert = self.tries[field].get_expert_for_token(token)
                experts.append(expert)
        return experts

    def export_for_llm(self, field: str) -> str:
        """Export Trie structure for LLM prompt."""
        trie = self.tries.get(field)
        if not trie:
            return f"No Trie built for field {field}"

        summary = trie.summarize()
        lines = [
            f"=== Trie Structure for {field} ===",
            f"Hierarchy: {' -> '.join(summary['hierarchy'])}",
            f"Total tokens: {summary['total_tokens']}",
            f"Leaf nodes: {summary['leaf_nodes']}",
            "",
            "Leaf Node Details:",
        ]

        for leaf in summary['leaf_summaries']:
            path_str = " > ".join(leaf['path'])
            lines.append(
                f"  [{path_str}]: {leaf['token_count']} tokens, "
                f"avg_ctr={leaf['avg_ctr']:.4f}, expert={leaf['expert']}"
            )

        return "\n".join(lines)
