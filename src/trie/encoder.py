"""
Trie-based encoding for MoE routing.

Converts Trie node statistics into routing signals for the MoE layer.
This is the bridge between statistical structure and conditional computation.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from .builder import StatisticalTrie, TrieNode
from .statistics import NodeStatistics


@dataclass
class TrieRoutingVector:
    """
    A structured routing vector derived from Trie statistics.

    This replaces raw sparse embeddings as the routing signal,
    providing stable, interpretable, low-dimensional routing information.
    """
    # Per-field statistics
    field_vectors: Dict[str, np.ndarray]

    # Aggregated statistics
    aggregated: np.ndarray

    # Interpretable features
    head_ratio: float  # Fraction of tokens from head
    tail_ratio: float  # Fraction of tokens from tail
    high_info_ratio: float  # Fraction of high-info tokens
    avg_lift: float  # Average CTR lift
    structure_entropy: float  # Diversity of Trie paths

    # Expert hints (from Trie assignment)
    suggested_experts: List[int]

    def to_tensor(self) -> torch.Tensor:
        """Convert to PyTorch tensor for model input."""
        return torch.from_numpy(self.aggregated).float()


class TrieEncoder(nn.Module):
    """
    Encodes sample features into Trie-based routing vectors.

    This module:
    1. Looks up Trie nodes for each token in the sample
    2. Aggregates node statistics into a fixed-size routing vector
    3. Optionally learns a projection for the routing signal
    """

    def __init__(
        self,
        tries: Dict[str, StatisticalTrie],
        routing_dim: int = 32,
        aggregation: str = "mean",  # mean, max, attention
        learnable_projection: bool = True,
    ):
        """
        Args:
            tries: Dict of field_name -> StatisticalTrie
            routing_dim: Output dimension for routing vector
            aggregation: How to aggregate across fields
            learnable_projection: Whether to learn a projection layer
        """
        super().__init__()

        self.tries = tries
        self.routing_dim = routing_dim
        self.aggregation = aggregation

        # Dimension of raw statistics per token
        self.stats_dim = 7  # From NodeStatistics.to_vector()

        # Number of fields
        self.num_fields = len(tries)

        # Input dimension: stats_dim per field + global features
        self.input_dim = self.stats_dim * self.num_fields + 5  # +5 for global features

        # Optional learnable projection
        if learnable_projection:
            self.projection = nn.Sequential(
                nn.Linear(self.input_dim, routing_dim * 2),
                nn.ReLU(),
                nn.Linear(routing_dim * 2, routing_dim),
                nn.LayerNorm(routing_dim),
            )
        else:
            self.projection = nn.Linear(self.input_dim, routing_dim)

        # Pre-compute default statistics for unknown tokens
        self._default_stats = np.zeros(self.stats_dim, dtype=np.float32)

    def _get_token_stats(
        self,
        field: str,
        token: str
    ) -> Tuple[np.ndarray, Optional[TrieNode]]:
        """Get statistics vector for a token."""
        trie = self.tries.get(field)
        if trie is None:
            return self._default_stats, None

        node = trie.get_node_for_token(str(token))
        if node is None:
            return self._default_stats, None

        # Get statistics from node (aggregated stats)
        # We use node-level stats rather than individual token stats
        # for better generalization
        stats = np.array([
            node.avg_frequency,
            node.avg_ctr,
            node.avg_lift,
            0.5,  # placeholder for entropy
            0.1,  # placeholder for mutual_info
            0.8,  # placeholder for stability
            len(node.tokens) / 1000,  # normalized token count
        ], dtype=np.float32)

        return stats, node

    def encode_sample(
        self,
        sparse_features: Dict[str, str],
    ) -> TrieRoutingVector:
        """
        Encode a single sample into a routing vector.

        Args:
            sparse_features: Dict mapping field_name -> token_value

        Returns:
            TrieRoutingVector with all routing information
        """
        field_vectors = {}
        all_stats = []
        nodes = []

        head_count = 0
        tail_count = 0
        high_info_count = 0
        total_lift = 0.0

        for field, trie in self.tries.items():
            token = sparse_features.get(field, "__UNK__")
            stats, node = self._get_token_stats(field, token)

            field_vectors[field] = stats
            all_stats.append(stats)

            if node:
                nodes.append(node)
                if "head" in node.path:
                    head_count += 1
                if "tail" in node.path:
                    tail_count += 1
                if "high_info" in node.path:
                    high_info_count += 1
                total_lift += node.avg_lift

        # Aggregate
        if self.aggregation == "mean":
            aggregated_stats = np.mean(all_stats, axis=0) if all_stats else self._default_stats
        elif self.aggregation == "max":
            aggregated_stats = np.max(all_stats, axis=0) if all_stats else self._default_stats
        else:
            aggregated_stats = np.mean(all_stats, axis=0) if all_stats else self._default_stats

        # Global features
        n_fields = max(len(sparse_features), 1)
        head_ratio = head_count / n_fields
        tail_ratio = tail_count / n_fields
        high_info_ratio = high_info_count / n_fields
        avg_lift = total_lift / n_fields if n_fields > 0 else 1.0

        # Structure entropy (diversity of paths)
        if nodes:
            paths = [n.path for n in nodes]
            unique_paths = len(set(paths))
            structure_entropy = unique_paths / len(paths)
        else:
            structure_entropy = 0.0

        # Combine all into aggregated vector
        global_features = np.array([
            head_ratio,
            tail_ratio,
            high_info_ratio,
            avg_lift,
            structure_entropy,
        ], dtype=np.float32)

        full_vector = np.concatenate([
            np.concatenate(all_stats) if all_stats else np.zeros(self.stats_dim * self.num_fields),
            global_features
        ])

        # Suggested experts
        suggested_experts = [
            n.expert_affinity for n in nodes
            if n.expert_affinity is not None
        ]

        return TrieRoutingVector(
            field_vectors=field_vectors,
            aggregated=full_vector,
            head_ratio=head_ratio,
            tail_ratio=tail_ratio,
            high_info_ratio=high_info_ratio,
            avg_lift=avg_lift,
            structure_entropy=structure_entropy,
            suggested_experts=suggested_experts,
        )

    def forward(
        self,
        sparse_indices: torch.Tensor,
        sparse_fields: List[str],
        vocab_maps: Dict[str, Dict[int, str]],  # idx -> token mapping
    ) -> torch.Tensor:
        """
        Forward pass for batch encoding.

        Args:
            sparse_indices: (batch_size, num_fields) tensor of token indices
            sparse_fields: List of field names
            vocab_maps: Reverse vocabulary mapping (index -> token)

        Returns:
            (batch_size, routing_dim) routing vectors
        """
        batch_size = sparse_indices.shape[0]
        device = sparse_indices.device

        # Encode each sample
        routing_vectors = []

        for i in range(batch_size):
            # Convert indices to tokens
            sparse_features = {}
            for j, field in enumerate(sparse_fields):
                idx = sparse_indices[i, j].item()
                token = vocab_maps.get(field, {}).get(idx, "__UNK__")
                sparse_features[field] = token

            # Encode
            routing_vec = self.encode_sample(sparse_features)
            routing_vectors.append(routing_vec.aggregated)

        # Stack and project
        routing_tensor = torch.from_numpy(
            np.stack(routing_vectors)
        ).float().to(device)

        # Apply projection
        output = self.projection(routing_tensor)

        return output

    def get_expert_hints(
        self,
        sparse_indices: torch.Tensor,
        sparse_fields: List[str],
        vocab_maps: Dict[str, Dict[int, str]],
    ) -> torch.Tensor:
        """
        Get expert hints from Trie structure.

        Returns a (batch_size,) tensor of suggested expert indices.
        """
        batch_size = sparse_indices.shape[0]
        hints = []

        for i in range(batch_size):
            sparse_features = {}
            for j, field in enumerate(sparse_fields):
                idx = sparse_indices[i, j].item()
                token = vocab_maps.get(field, {}).get(idx, "__UNK__")
                sparse_features[field] = token

            routing_vec = self.encode_sample(sparse_features)

            # Use mode of suggested experts
            if routing_vec.suggested_experts:
                from collections import Counter
                most_common = Counter(routing_vec.suggested_experts).most_common(1)
                hints.append(most_common[0][0])
            else:
                hints.append(0)

        return torch.tensor(hints, dtype=torch.long, device=sparse_indices.device)
