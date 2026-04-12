"""
Adaptive Alpha MoE: Frequency-aware routing with learnable alpha.

Core innovation: α(t) = σ(w·log(n+1) + b)
- Low-frequency tokens: higher α, trust prior more
- High-frequency tokens: lower α, trust learned routing more

This solves the "cold-start paradox" where tokens needing most help
have the least reliable statistics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class AdaptiveAlphaRouter(nn.Module):
    """
    Router with frequency-adaptive mixing of prior and learned routing.

    Mathematical formulation:
        α(n) = σ(w·log(n+1) + b)
        g(x) = α·g_prior(x) + (1-α)·g_learned(x)

    Where:
        - n: token frequency
        - w, b: learnable parameters
        - g_prior: prior-based routing (from statistics)
        - g_learned: neural network routing
    """

    def __init__(
        self,
        input_dim: int,
        num_experts: int = 8,
        hidden_dim: int = 32,
        alpha_init_weight: float = -0.5,
        alpha_init_bias: float = 0.5,
    ):
        super().__init__()

        self.num_experts = num_experts

        # Learnable alpha parameters
        # α(n) = σ(w·log(n+1) + b)
        self.alpha_weight = nn.Parameter(torch.tensor(alpha_init_weight))
        self.alpha_bias = nn.Parameter(torch.tensor(alpha_init_bias))

        # Prior-based router (simple linear, uses statistics)
        self.prior_router = nn.Linear(input_dim, num_experts)

        # Learned router (more complex, learns from data)
        self.learned_router = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_experts),
        )

    def compute_alpha(self, token_freqs: torch.Tensor) -> torch.Tensor:
        """
        Compute adaptive alpha based on token frequencies.

        Args:
            token_freqs: (batch_size,) token frequency counts

        Returns:
            alpha: (batch_size, 1) mixing weights
        """
        log_freq = torch.log(token_freqs.float() + 1)
        alpha = torch.sigmoid(self.alpha_weight * log_freq + self.alpha_bias)
        return alpha.unsqueeze(1)

    def forward(
        self,
        x: torch.Tensor,
        token_freqs: torch.Tensor,
        return_alpha: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute adaptive routing.

        Args:
            x: Input features (batch, input_dim)
            token_freqs: Token frequencies (batch,)
            return_alpha: Whether to return alpha values

        Returns:
            expert_indices: (batch, top_k)
            expert_weights: (batch, top_k)
            alpha: Optional (batch,) alpha values
        """
        # Compute adaptive alpha
        alpha = self.compute_alpha(token_freqs)  # (batch, 1)

        # Compute both routing logits
        prior_logits = self.prior_router(x)
        learned_logits = self.learned_router(x)

        # Mix with adaptive alpha
        router_logits = alpha * prior_logits + (1 - alpha) * learned_logits

        # Top-2 selection
        top_k_logits, expert_indices = torch.topk(router_logits, 2, dim=-1)
        expert_weights = F.softmax(top_k_logits, dim=-1)

        if return_alpha:
            return expert_indices, expert_weights, alpha.squeeze(1)
        return expert_indices, expert_weights, None


class AdaptiveAlphaMoE(nn.Module):
    """
    MoE layer with adaptive alpha routing.

    Combines:
    1. Adaptive alpha routing based on token frequency
    2. Top-k expert selection
    3. Optional load balancing
    """

    def __init__(
        self,
        input_dim: int,
        num_experts: int = 8,
        expert_hidden_dim: int = 64,
        expert_output_dim: int = 64,
        router_hidden_dim: int = 32,
        alpha_init_weight: float = -0.5,
        alpha_init_bias: float = 0.5,
        load_balance_weight: float = 0.01,
    ):
        super().__init__()

        self.num_experts = num_experts
        self.load_balance_weight = load_balance_weight

        # Adaptive router
        self.router = AdaptiveAlphaRouter(
            input_dim=input_dim,
            num_experts=num_experts,
            hidden_dim=router_hidden_dim,
            alpha_init_weight=alpha_init_weight,
            alpha_init_bias=alpha_init_bias,
        )

        # Expert networks
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, expert_hidden_dim),
                nn.ReLU(),
                nn.Linear(expert_hidden_dim, expert_output_dim),
            )
            for _ in range(num_experts)
        ])

        self.output_dim = expert_output_dim

    def forward(
        self,
        x: torch.Tensor,
        token_freqs: torch.Tensor,
        return_routing: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with adaptive routing.

        Args:
            x: Input features (batch, input_dim)
            token_freqs: Token frequencies (batch,)
            return_routing: Whether to return routing details

        Returns:
            Dict with 'output', 'load_balance_loss', optionally 'routing_info'
        """
        batch_size = x.shape[0]
        device = x.device

        # Get routing decisions
        expert_indices, expert_weights, alpha = self.router(
            x, token_freqs, return_alpha=return_routing
        )

        # Compute all expert outputs
        all_expert_outputs = torch.stack(
            [expert(x) for expert in self.experts], dim=1
        )  # (batch, num_experts, output_dim)

        # Gather selected expert outputs
        output = torch.zeros(batch_size, self.output_dim, device=device)
        for k in range(2):  # top-2
            expert_idx = expert_indices[:, k]
            weight = expert_weights[:, k:k+1]
            expert_out = all_expert_outputs[
                torch.arange(batch_size, device=device), expert_idx
            ]
            output = output + weight * expert_out

        # Load balance loss
        router_logits = self.router.prior_router(x)  # Use prior for balance
        expert_mask = F.one_hot(expert_indices[:, 0], self.num_experts).float()
        fraction = expert_mask.mean(dim=0)
        prob = F.softmax(router_logits, dim=-1).mean(dim=0)
        lb_loss = self.num_experts * (fraction * prob).sum() * self.load_balance_weight

        result = {
            'output': output,
            'load_balance_loss': lb_loss,
            'alpha_mean': alpha.mean().item() if alpha is not None else None,
        }

        if return_routing:
            result['routing_info'] = {
                'expert_indices': expert_indices,
                'expert_weights': expert_weights,
                'alpha': alpha,
            }

        return result


class AdaptiveAlphaRecommender(nn.Module):
    """
    Full recommendation model with Adaptive Alpha MoE.

    Architecture:
    1. Embedding layer for sparse features
    2. BatchNorm for dense features
    3. Feature concatenation
    4. Adaptive Alpha MoE
    5. Prediction head
    """

    def __init__(
        self,
        dense_dim: int,
        sparse_dims: Dict[str, int],
        embedding_dim: int = 16,
        num_experts: int = 8,
        expert_hidden_dim: int = 64,
        expert_output_dim: int = 64,
        hidden_dims: List[int] = [64, 32],
        dropout: float = 0.1,
        alpha_init_weight: float = -0.5,
        alpha_init_bias: float = 0.5,
    ):
        super().__init__()

        self.num_sparse_fields = len(sparse_dims)
        self.sparse_field_names = list(sparse_dims.keys())

        # Embeddings
        self.embeddings = nn.ModuleDict({
            name: nn.Embedding(dim, embedding_dim)
            for name, dim in sparse_dims.items()
        })

        # Dense processing
        self.dense_bn = nn.BatchNorm1d(dense_dim)

        # MoE input dimension
        moe_input_dim = dense_dim + self.num_sparse_fields * embedding_dim

        # Adaptive Alpha MoE
        self.moe = AdaptiveAlphaMoE(
            input_dim=moe_input_dim,
            num_experts=num_experts,
            expert_hidden_dim=expert_hidden_dim,
            expert_output_dim=expert_output_dim,
            alpha_init_weight=alpha_init_weight,
            alpha_init_bias=alpha_init_bias,
        )

        # Prediction head
        layers = []
        prev_dim = expert_output_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.prediction_head = nn.Sequential(*layers)

    def forward(
        self,
        dense: torch.Tensor,
        sparse: torch.Tensor,
        token_freqs: torch.Tensor,
        return_routing: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            dense: Dense features (batch, dense_dim)
            sparse: Sparse feature indices (batch, num_sparse_fields)
            token_freqs: Token frequencies for routing (batch,)
            return_routing: Whether to return routing info

        Returns:
            Dict with 'logits', 'load_balance_loss', optionally 'routing_info'
        """
        batch_size = dense.shape[0]

        # Process dense features
        dense = self.dense_bn(dense)

        # Process sparse features
        sparse_embs = [
            self.embeddings[name](sparse[:, i])
            for i, name in enumerate(self.sparse_field_names)
        ]
        sparse_flat = torch.cat(sparse_embs, dim=1)

        # Combine features
        combined = torch.cat([dense, sparse_flat], dim=1)

        # MoE layer
        moe_output = self.moe(combined, token_freqs, return_routing)

        # Prediction
        logits = self.prediction_head(moe_output['output']).squeeze(-1)

        result = {
            'logits': logits,
            'load_balance_loss': moe_output['load_balance_loss'],
            'alpha_mean': moe_output.get('alpha_mean'),
        }

        if return_routing:
            result['routing_info'] = moe_output.get('routing_info')

        return result

    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute total loss."""
        bce_loss = F.binary_cross_entropy_with_logits(outputs['logits'], labels)
        lb_loss = outputs.get('load_balance_loss', 0)
        return bce_loss + lb_loss


def compute_token_frequencies(
    dataset,
    field_idx: int = 1,
) -> Dict[int, int]:
    """
    Compute token frequency dictionary from dataset.

    Args:
        dataset: Dataset with 'sparse' field
        field_idx: Index of the field to compute frequencies for

    Returns:
        Dict mapping token_id -> frequency count
    """
    from collections import defaultdict

    freq_dict = defaultdict(int)
    for i in range(len(dataset)):
        sample = dataset[i]
        token_id = sample['sparse'][field_idx].item()
        freq_dict[token_id] += 1

    return dict(freq_dict)


def get_batch_token_freqs(
    batch_sparse: torch.Tensor,
    field_idx: int,
    freq_dict: Dict[int, int],
) -> torch.Tensor:
    """
    Get token frequencies for a batch.

    Args:
        batch_sparse: Sparse features (batch, num_fields)
        field_idx: Index of the field to get frequencies for
        freq_dict: Token frequency dictionary

    Returns:
        Tensor of frequencies (batch,)
    """
    freqs = [
        freq_dict.get(batch_sparse[i, field_idx].item(), 1)
        for i in range(batch_sparse.shape[0])
    ]
    return torch.tensor(freqs, device=batch_sparse.device)
