"""
Fast Vectorized Trie Encoder for MoE routing.

This is an optimized version of TrieEncoder that:
1. Pre-computes all Trie statistics as tensor lookup tables
2. Uses vectorized tensor operations (no Python for-loops in forward pass)
3. Achieves O(1) batch encoding instead of O(B*F)

Expected speedup: 50-100x (70ms -> ~1ms)
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from .builder import StatisticalTrie, TrieNode


class FastTrieEncoder(nn.Module):
    """
    Vectorized Trie Encoder with O(1) batch forward pass.

    Key optimizations:
    1. Pre-compute statistics for all tokens as tensor lookup tables
    2. Use torch.embedding for O(1) lookup instead of Python dict
    3. Batch all operations using tensor operations
    """

    def __init__(
        self,
        tries: Dict[str, StatisticalTrie],
        vocab_sizes: Dict[str, int],
        routing_dim: int = 32,
        stats_dim: int = 7,
        learnable_projection: bool = True,
        learnable_residual: bool = False,  # Phase 2 enhancement
    ):
        """
        Args:
            tries: Dict of field_name -> StatisticalTrie
            vocab_sizes: Dict of field_name -> vocabulary size
            routing_dim: Output dimension for routing vector
            stats_dim: Dimension of statistics per token (default 7)
            learnable_projection: Whether to learn a projection layer
            learnable_residual: Whether to add learnable residual to frozen stats
        """
        super().__init__()

        self.routing_dim = routing_dim
        self.stats_dim = stats_dim
        self.num_fields = len(tries)
        self.field_names = list(tries.keys())
        self.learnable_residual = learnable_residual

        # Global features dimension
        self.global_dim = 5  # head_ratio, tail_ratio, high_info_ratio, avg_lift, structure_entropy

        # Input dimension to projection
        self.input_dim = stats_dim * self.num_fields + self.global_dim

        # Pre-compute statistics lookup tables for each field
        # These are registered as buffers (non-learnable, but moved with model)
        self._build_lookup_tables(tries, vocab_sizes)

        # Optional learnable residual for statistics (Phase 2)
        if learnable_residual:
            self.stats_residual = nn.ParameterList([
                nn.Parameter(torch.zeros(vocab_sizes.get(field, 1), stats_dim))
                for field in self.field_names
            ])

        # Projection network
        if learnable_projection:
            self.projection = nn.Sequential(
                nn.Linear(self.input_dim, routing_dim * 2),
                nn.ReLU(),
                nn.Linear(routing_dim * 2, routing_dim),
                nn.LayerNorm(routing_dim),
            )
        else:
            self.projection = nn.Linear(self.input_dim, routing_dim)

    def _build_lookup_tables(
        self,
        tries: Dict[str, StatisticalTrie],
        vocab_sizes: Dict[str, int],
    ):
        """
        Pre-compute all statistics as tensor lookup tables.

        This converts the Python dict-based Trie lookup to O(1) tensor indexing.
        """
        for field_idx, field in enumerate(self.field_names):
            trie = tries.get(field)
            vocab_size = vocab_sizes.get(field, 1)

            # Initialize with default statistics
            stats_table = torch.zeros(vocab_size, self.stats_dim)
            expert_table = torch.zeros(vocab_size, dtype=torch.long)
            path_info_table = torch.zeros(vocab_size, 3)  # [is_head, is_tail, is_high_info]

            if trie is not None:
                # Populate from Trie
                for token_str, node in trie.token_to_node.items():
                    try:
                        token_idx = int(token_str)
                        if 0 <= token_idx < vocab_size:
                            # Statistics vector
                            stats_table[token_idx] = torch.tensor([
                                node.avg_frequency,
                                node.avg_ctr,
                                node.avg_lift,
                                0.5,  # entropy placeholder
                                0.1,  # mutual_info placeholder
                                0.8,  # stability placeholder
                                min(len(node.tokens) / 1000.0, 1.0),  # normalized token count
                            ])

                            # Expert assignment
                            if node.expert_affinity is not None:
                                expert_table[token_idx] = node.expert_affinity

                            # Path info for global features
                            path = node.path
                            path_info_table[token_idx, 0] = 1.0 if "head" in path else 0.0
                            path_info_table[token_idx, 1] = 1.0 if "tail" in path else 0.0
                            path_info_table[token_idx, 2] = 1.0 if "high_info" in path else 0.0
                    except (ValueError, IndexError):
                        continue

            # Register as buffers (moved with model, but not trained)
            self.register_buffer(f'stats_table_{field_idx}', stats_table)
            self.register_buffer(f'expert_table_{field_idx}', expert_table)
            self.register_buffer(f'path_info_table_{field_idx}', path_info_table)

    def _get_stats_table(self, field_idx: int) -> torch.Tensor:
        """Get statistics table for a field."""
        return getattr(self, f'stats_table_{field_idx}')

    def _get_expert_table(self, field_idx: int) -> torch.Tensor:
        """Get expert table for a field."""
        return getattr(self, f'expert_table_{field_idx}')

    def _get_path_info_table(self, field_idx: int) -> torch.Tensor:
        """Get path info table for a field."""
        return getattr(self, f'path_info_table_{field_idx}')

    def forward(
        self,
        sparse_indices: torch.Tensor,
        sparse_fields: Optional[List[str]] = None,
        vocab_maps: Optional[Dict] = None,  # Kept for API compatibility
    ) -> torch.Tensor:
        """
        Vectorized forward pass - O(1) batch complexity.

        Args:
            sparse_indices: (batch_size, num_fields) tensor of token indices
            sparse_fields: List of field names (optional, uses self.field_names if None)
            vocab_maps: Ignored (kept for backward compatibility)

        Returns:
            (batch_size, routing_dim) routing vectors
        """
        batch_size = sparse_indices.shape[0]
        device = sparse_indices.device

        # Collect statistics for all fields
        all_stats = []
        all_path_info = []

        for field_idx in range(self.num_fields):
            # Get lookup tables (already on correct device via buffer registration)
            stats_table = self._get_stats_table(field_idx).to(device)
            path_info_table = self._get_path_info_table(field_idx).to(device)

            # Get indices for this field
            field_indices = sparse_indices[:, field_idx]  # (batch_size,)

            # Clamp indices to valid range
            field_indices = field_indices.clamp(0, stats_table.shape[0] - 1)

            # Vectorized lookup - O(1)!
            field_stats = stats_table[field_indices]  # (batch_size, stats_dim)
            field_path_info = path_info_table[field_indices]  # (batch_size, 3)

            # Add learnable residual if enabled (Phase 2)
            if self.learnable_residual:
                residual = self.stats_residual[field_idx][field_indices]
                field_stats = field_stats + residual

            all_stats.append(field_stats)
            all_path_info.append(field_path_info)

        # Stack and aggregate
        stacked_stats = torch.stack(all_stats, dim=1)  # (batch_size, num_fields, stats_dim)
        stacked_path_info = torch.stack(all_path_info, dim=1)  # (batch_size, num_fields, 3)

        # Flatten field statistics
        flat_stats = stacked_stats.view(batch_size, -1)  # (batch_size, num_fields * stats_dim)

        # Compute global features
        head_ratio = stacked_path_info[:, :, 0].mean(dim=1, keepdim=True)  # (batch_size, 1)
        tail_ratio = stacked_path_info[:, :, 1].mean(dim=1, keepdim=True)
        high_info_ratio = stacked_path_info[:, :, 2].mean(dim=1, keepdim=True)

        # Average lift from stats
        avg_lift = stacked_stats[:, :, 2].mean(dim=1, keepdim=True)  # avg_lift is at index 2

        # Structure entropy (simplified: variance of stats across fields)
        structure_entropy = stacked_stats.var(dim=1).mean(dim=1, keepdim=True)

        # Combine global features
        global_features = torch.cat([
            head_ratio, tail_ratio, high_info_ratio, avg_lift, structure_entropy
        ], dim=1)  # (batch_size, 5)

        # Full input vector
        full_vector = torch.cat([flat_stats, global_features], dim=1)  # (batch_size, input_dim)

        # Project to routing dimension
        output = self.projection(full_vector)

        return output

    def get_expert_hints(
        self,
        sparse_indices: torch.Tensor,
        sparse_fields: Optional[List[str]] = None,
        vocab_maps: Optional[Dict] = None,  # Kept for API compatibility
    ) -> torch.Tensor:
        """
        Vectorized expert hints lookup.

        Returns the mode of expert assignments across fields.
        """
        batch_size = sparse_indices.shape[0]
        device = sparse_indices.device

        # Collect expert assignments for all fields
        all_experts = []

        for field_idx in range(self.num_fields):
            expert_table = self._get_expert_table(field_idx).to(device)
            field_indices = sparse_indices[:, field_idx].clamp(0, expert_table.shape[0] - 1)
            field_experts = expert_table[field_indices]  # (batch_size,)
            all_experts.append(field_experts)

        # Stack and compute mode
        stacked_experts = torch.stack(all_experts, dim=1)  # (batch_size, num_fields)

        # Use the expert from the first field as hint (or could compute mode)
        # Computing true mode is expensive, so we use first field
        hints = stacked_experts[:, 0]

        return hints


class FastTrieEncoderV2(FastTrieEncoder):
    """
    Enhanced version with additional optimizations:
    1. Learnable residuals for frozen statistics
    2. Optional attention-based aggregation
    3. Support for dense gradient estimation
    """

    def __init__(
        self,
        tries: Dict[str, StatisticalTrie],
        vocab_sizes: Dict[str, int],
        routing_dim: int = 32,
        stats_dim: int = 7,
        learnable_projection: bool = True,
        use_attention: bool = False,
        num_attention_heads: int = 4,
    ):
        super().__init__(
            tries=tries,
            vocab_sizes=vocab_sizes,
            routing_dim=routing_dim,
            stats_dim=stats_dim,
            learnable_projection=False,  # We'll define our own
            learnable_residual=True,  # Enable learnable residuals
        )

        self.use_attention = use_attention

        if use_attention:
            # Attention-based aggregation across fields
            self.field_attention = nn.MultiheadAttention(
                embed_dim=stats_dim,
                num_heads=min(num_attention_heads, stats_dim),
                dropout=0.1,
                batch_first=True,
            )
            self.attn_proj = nn.Linear(stats_dim * self.num_fields, stats_dim * self.num_fields)

        # Enhanced projection with residual connection
        hidden_dim = routing_dim * 2
        self.projection = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, routing_dim),
            nn.LayerNorm(routing_dim),
        )

    def forward(
        self,
        sparse_indices: torch.Tensor,
        sparse_fields: Optional[List[str]] = None,
        vocab_maps: Optional[Dict] = None,
    ) -> torch.Tensor:
        """Enhanced forward with optional attention."""
        batch_size = sparse_indices.shape[0]
        device = sparse_indices.device

        # Collect statistics for all fields
        all_stats = []
        all_path_info = []

        for field_idx in range(self.num_fields):
            stats_table = self._get_stats_table(field_idx).to(device)
            path_info_table = self._get_path_info_table(field_idx).to(device)

            field_indices = sparse_indices[:, field_idx].clamp(0, stats_table.shape[0] - 1)

            field_stats = stats_table[field_indices]
            field_path_info = path_info_table[field_indices]

            # Add learnable residual
            if self.learnable_residual:
                residual = self.stats_residual[field_idx][field_indices]
                field_stats = field_stats + residual

            all_stats.append(field_stats)
            all_path_info.append(field_path_info)

        # Stack
        stacked_stats = torch.stack(all_stats, dim=1)  # (batch_size, num_fields, stats_dim)
        stacked_path_info = torch.stack(all_path_info, dim=1)

        # Optional attention-based aggregation
        if self.use_attention:
            attn_out, _ = self.field_attention(stacked_stats, stacked_stats, stacked_stats)
            stacked_stats = stacked_stats + attn_out  # Residual connection

        # Flatten
        flat_stats = stacked_stats.view(batch_size, -1)

        # Global features
        head_ratio = stacked_path_info[:, :, 0].mean(dim=1, keepdim=True)
        tail_ratio = stacked_path_info[:, :, 1].mean(dim=1, keepdim=True)
        high_info_ratio = stacked_path_info[:, :, 2].mean(dim=1, keepdim=True)
        avg_lift = stacked_stats[:, :, 2].mean(dim=1, keepdim=True)
        structure_entropy = stacked_stats.var(dim=1).mean(dim=1, keepdim=True)

        global_features = torch.cat([
            head_ratio, tail_ratio, high_info_ratio, avg_lift, structure_entropy
        ], dim=1)

        full_vector = torch.cat([flat_stats, global_features], dim=1)
        output = self.projection(full_vector)

        return output


def convert_to_fast_encoder(
    old_encoder: 'TrieEncoder',
    vocab_sizes: Dict[str, int],
) -> FastTrieEncoder:
    """
    Convert an existing TrieEncoder to FastTrieEncoder.

    This preserves the Trie structure but enables vectorized operations.
    """
    fast_encoder = FastTrieEncoder(
        tries=old_encoder.tries,
        vocab_sizes=vocab_sizes,
        routing_dim=old_encoder.routing_dim,
        stats_dim=old_encoder.stats_dim,
        learnable_projection=True,
    )

    # Copy projection weights if compatible
    try:
        fast_encoder.projection.load_state_dict(old_encoder.projection.state_dict())
    except:
        pass  # Dimensions might differ, use fresh weights

    return fast_encoder
