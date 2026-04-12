"""
Expert networks for MoE layers.

Experts are specialized for different statistical patterns:
- Head experts: Handle high-frequency, stable patterns
- Tail experts: Handle long-tail, sparse patterns
- High-info experts: Handle strong-signal patterns
- Drift experts: Handle temporally unstable patterns
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class Expert(nn.Module):
    """
    A single expert network.

    Experts are intentionally small (compared to full model)
    to ensure computational efficiency.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        expert_type: str = 'general',  # For potential specialization
    ):
        super().__init__()

        self.expert_type = expert_type

        layers = []
        prev_dim = input_dim

        for i in range(num_layers):
            is_last = (i == num_layers - 1)
            out_dim = output_dim if is_last else hidden_dim

            layers.append(nn.Linear(prev_dim, out_dim))
            if not is_last:
                layers.append(nn.LayerNorm(out_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
            prev_dim = out_dim

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ExpertLayer(nn.Module):
    """
    A collection of expert networks with optional specialization.
    """

    def __init__(
        self,
        num_experts: int,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        expert_types: Optional[List[str]] = None,
    ):
        super().__init__()

        self.num_experts = num_experts
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Assign expert types
        if expert_types is None:
            # Default: balanced distribution
            expert_types = ['general'] * num_experts
        elif len(expert_types) != num_experts:
            # Extend/truncate to match
            expert_types = (expert_types * (num_experts // len(expert_types) + 1))[:num_experts]

        # Create experts
        self.experts = nn.ModuleList([
            Expert(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                num_layers=num_layers,
                dropout=dropout,
                expert_type=expert_types[i],
            )
            for i in range(num_experts)
        ])

    def forward(
        self,
        x: torch.Tensor,
        expert_weights: torch.Tensor,  # (batch_size, num_experts)
    ) -> torch.Tensor:
        """
        Compute weighted combination of expert outputs.

        Args:
            x: Input tensor (batch_size, input_dim)
            expert_weights: Soft weights for each expert

        Returns:
            Combined output (batch_size, output_dim)
        """
        batch_size = x.shape[0]

        # Compute all expert outputs
        expert_outputs = torch.stack([
            expert(x) for expert in self.experts
        ], dim=1)  # (batch, num_experts, output_dim)

        # Weighted combination
        weights = expert_weights.unsqueeze(-1)  # (batch, num_experts, 1)
        output = (expert_outputs * weights).sum(dim=1)  # (batch, output_dim)

        return output

    def forward_sparse(
        self,
        x: torch.Tensor,
        expert_indices: torch.Tensor,  # (batch_size, top_k)
        expert_weights: torch.Tensor,  # (batch_size, top_k)
    ) -> torch.Tensor:
        """
        Sparse forward pass (only compute selected experts).

        More efficient when top_k << num_experts.
        """
        batch_size = x.shape[0]
        top_k = expert_indices.shape[1]
        device = x.device

        # Initialize output
        output = torch.zeros(batch_size, self.output_dim, device=device)

        # Process each expert that's selected
        for expert_idx in range(self.num_experts):
            # Find samples that selected this expert
            mask = (expert_indices == expert_idx).any(dim=1)  # (batch,)

            if mask.sum() == 0:
                continue

            # Get indices and weights for this expert
            selected_x = x[mask]
            expert_out = self.experts[expert_idx](selected_x)

            # Get weights for this expert
            weight_mask = (expert_indices == expert_idx)  # (batch, top_k)
            weights = (expert_weights * weight_mask.float()).sum(dim=1)  # (batch,)

            # Add weighted contribution
            output[mask] += expert_out * weights[mask].unsqueeze(-1)

        return output


class SpecializedExpertLayer(ExpertLayer):
    """
    Expert layer with explicit specialization based on Trie structure.

    Expert assignments:
    - Experts 0-1: Head patterns (high frequency, stable)
    - Experts 2-4: Mid patterns (balanced)
    - Experts 5-7: Tail patterns (low frequency, needs generalization)
    """

    def __init__(
        self,
        num_experts: int = 8,
        input_dim: int = 128,
        hidden_dim: int = 64,
        output_dim: int = 64,
        dropout: float = 0.1,
    ):
        # Define specialization
        expert_types = []
        for i in range(num_experts):
            if i < num_experts * 0.25:
                expert_types.append('head')
            elif i < num_experts * 0.6:
                expert_types.append('mid')
            else:
                expert_types.append('tail')

        super().__init__(
            num_experts=num_experts,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=2,
            dropout=dropout,
            expert_types=expert_types,
        )

        # Specialized architectures for different types
        for i, expert in enumerate(self.experts):
            if expert.expert_type == 'tail':
                # Tail experts have more capacity for generalization
                expert.network = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim * 2),
                    nn.LayerNorm(hidden_dim * 2),
                    nn.ReLU(),
                    nn.Dropout(dropout * 1.5),  # More dropout for regularization
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, output_dim),
                )
            elif expert.expert_type == 'head':
                # Head experts are simpler (patterns are clearer)
                expert.network = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, output_dim),
                )
