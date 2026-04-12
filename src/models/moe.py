"""
Mixture of Experts (MoE) layers for conditional computation.

Key innovation: Trie-guided routing that uses statistical structure
instead of learned embeddings for routing decisions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from .experts import ExpertLayer, SpecializedExpertLayer


class MoELayer(nn.Module):
    """
    Standard MoE layer with learned routing.

    Baseline for comparison with Trie-guided routing.
    """

    def __init__(
        self,
        input_dim: int,
        num_experts: int = 8,
        expert_hidden_dim: int = 64,
        expert_output_dim: int = 64,
        top_k: int = 2,
        router_hidden_dim: int = 32,
        load_balance_weight: float = 0.01,
        noise_std: float = 0.1,
    ):
        super().__init__()

        self.num_experts = num_experts
        self.top_k = top_k
        self.load_balance_weight = load_balance_weight
        self.noise_std = noise_std

        # Router network (learned)
        self.router = nn.Sequential(
            nn.Linear(input_dim, router_hidden_dim),
            nn.ReLU(),
            nn.Linear(router_hidden_dim, num_experts),
        )

        # Expert layer
        self.experts = ExpertLayer(
            num_experts=num_experts,
            input_dim=input_dim,
            hidden_dim=expert_hidden_dim,
            output_dim=expert_output_dim,
        )

    def _compute_routing(
        self,
        x: torch.Tensor,
        training: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute routing decisions.

        Returns:
            expert_weights: (batch, top_k) - weights for selected experts
            expert_indices: (batch, top_k) - indices of selected experts
            router_logits: (batch, num_experts) - raw logits for load balancing
        """
        # Router logits
        router_logits = self.router(x)  # (batch, num_experts)

        # Add noise during training for exploration
        if training and self.noise_std > 0:
            noise = torch.randn_like(router_logits) * self.noise_std
            router_logits = router_logits + noise

        # Top-k selection
        top_k_logits, expert_indices = torch.topk(router_logits, self.top_k, dim=-1)

        # Softmax over selected experts
        expert_weights = F.softmax(top_k_logits, dim=-1)

        return expert_weights, expert_indices, router_logits

    def _compute_load_balance_loss(
        self,
        router_logits: torch.Tensor,
        expert_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute load balancing loss to encourage even expert usage.

        Based on Switch Transformer's load balancing loss.
        """
        batch_size = router_logits.shape[0]

        # Fraction of samples assigned to each expert
        expert_mask = F.one_hot(expert_indices[:, 0], self.num_experts).float()
        fraction_per_expert = expert_mask.mean(dim=0)

        # Probability assigned to each expert
        router_probs = F.softmax(router_logits, dim=-1)
        prob_per_expert = router_probs.mean(dim=0)

        # Load balance loss: minimize CV (coefficient of variation)
        load_balance_loss = self.num_experts * (fraction_per_expert * prob_per_expert).sum()

        return load_balance_loss

    def forward(
        self,
        x: torch.Tensor,
        return_routing: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through MoE layer.

        Args:
            x: Input tensor (batch_size, input_dim)
            return_routing: Whether to return routing information

        Returns:
            Dict with 'output' and optionally 'routing_info'
        """
        # Compute routing
        expert_weights, expert_indices, router_logits = self._compute_routing(
            x, training=self.training
        )

        # Compute expert outputs (sparse)
        output = self.experts.forward_sparse(x, expert_indices, expert_weights)

        # Compute load balance loss
        lb_loss = self._compute_load_balance_loss(router_logits, expert_indices)

        result = {
            'output': output,
            'load_balance_loss': lb_loss,
        }

        if return_routing:
            result['routing_info'] = {
                'expert_weights': expert_weights,
                'expert_indices': expert_indices,
                'router_logits': router_logits,
            }

        return result


class TrieGuidedMoE(nn.Module):
    """
    MoE layer with Trie-guided routing.

    Key innovation: Uses Trie structure statistics as routing signal
    instead of (or in addition to) learned embeddings.

    Benefits:
    1. More stable routing (based on pre-computed statistics)
    2. Interpretable routing decisions
    3. Better handling of long-tail tokens
    4. Reduced training instability
    """

    def __init__(
        self,
        input_dim: int,
        trie_routing_dim: int,  # Dimension of Trie-based routing vector
        num_experts: int = 8,
        expert_hidden_dim: int = 64,
        expert_output_dim: int = 64,
        top_k: int = 2,
        routing_mode: str = 'hybrid',  # 'trie_only', 'learned_only', 'hybrid'
        trie_weight: float = 0.7,  # Weight for Trie routing in hybrid mode
        use_expert_hints: bool = True,  # Use Trie's expert assignments as hints
        load_balance_weight: float = 0.01,
    ):
        super().__init__()

        self.num_experts = num_experts
        self.top_k = top_k
        self.routing_mode = routing_mode
        self.trie_weight = trie_weight
        self.use_expert_hints = use_expert_hints
        self.load_balance_weight = load_balance_weight

        # Trie-based router
        self.trie_router = nn.Sequential(
            nn.Linear(trie_routing_dim, num_experts * 2),
            nn.ReLU(),
            nn.Linear(num_experts * 2, num_experts),
        )

        # Learned router (for hybrid mode)
        if routing_mode in ['learned_only', 'hybrid']:
            self.learned_router = nn.Sequential(
                nn.Linear(input_dim, 32),
                nn.ReLU(),
                nn.Linear(32, num_experts),
            )

        # Expert layer (specialized based on Trie structure)
        self.experts = SpecializedExpertLayer(
            num_experts=num_experts,
            input_dim=input_dim,
            hidden_dim=expert_hidden_dim,
            output_dim=expert_output_dim,
        )

        # Temperature for routing (learnable)
        self.temperature = nn.Parameter(torch.ones(1))

    def _compute_routing(
        self,
        x: torch.Tensor,
        trie_routing_vec: torch.Tensor,
        expert_hints: Optional[torch.Tensor] = None,
        training: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute routing with Trie guidance.

        Args:
            x: Input features (batch, input_dim)
            trie_routing_vec: Trie-based routing vector (batch, trie_routing_dim)
            expert_hints: Optional hints from Trie (batch,) - suggested expert index
            training: Whether in training mode

        Returns:
            expert_weights, expert_indices, router_logits
        """
        batch_size = x.shape[0]
        device = x.device

        # Trie-based routing logits
        trie_logits = self.trie_router(trie_routing_vec)  # (batch, num_experts)

        if self.routing_mode == 'trie_only':
            router_logits = trie_logits

        elif self.routing_mode == 'learned_only':
            router_logits = self.learned_router(x)

        else:  # hybrid
            learned_logits = self.learned_router(x)
            # Combine with learnable weight
            router_logits = self.trie_weight * trie_logits + (1 - self.trie_weight) * learned_logits

        # Apply expert hints as soft bias
        if self.use_expert_hints and expert_hints is not None:
            hint_bias = F.one_hot(expert_hints, self.num_experts).float() * 2.0
            router_logits = router_logits + hint_bias

        # Temperature scaling
        router_logits = router_logits / (self.temperature + 1e-6)

        # Top-k selection
        top_k_logits, expert_indices = torch.topk(router_logits, self.top_k, dim=-1)
        expert_weights = F.softmax(top_k_logits, dim=-1)

        return expert_weights, expert_indices, router_logits

    def forward(
        self,
        x: torch.Tensor,
        trie_routing_vec: torch.Tensor,
        expert_hints: Optional[torch.Tensor] = None,
        return_routing: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with Trie-guided routing.

        Args:
            x: Input features
            trie_routing_vec: Trie-based routing vector from TrieEncoder
            expert_hints: Optional expert hints from Trie assignment
            return_routing: Whether to return routing details

        Returns:
            Dict with output and auxiliary information
        """
        # Compute routing
        expert_weights, expert_indices, router_logits = self._compute_routing(
            x, trie_routing_vec, expert_hints, self.training
        )

        # Expert computation
        output = self.experts.forward_sparse(x, expert_indices, expert_weights)

        # Load balance loss
        expert_mask = F.one_hot(expert_indices[:, 0], self.num_experts).float()
        fraction_per_expert = expert_mask.mean(dim=0)
        router_probs = F.softmax(router_logits, dim=-1).mean(dim=0)
        lb_loss = self.num_experts * (fraction_per_expert * router_probs).sum()

        result = {
            'output': output,
            'load_balance_loss': lb_loss * self.load_balance_weight,
        }

        if return_routing:
            result['routing_info'] = {
                'expert_weights': expert_weights,
                'expert_indices': expert_indices,
                'router_logits': router_logits,
                'trie_contribution': self.trie_weight if self.routing_mode == 'hybrid' else 1.0,
            }

        return result


class TrieMoERecommender(nn.Module):
    """
    Full recommendation model with Trie-guided MoE.

    Architecture:
    1. Embedding layer for sparse features
    2. Trie encoder for routing signals
    3. Feature interaction (FM-style)
    4. Trie-guided MoE for conditional computation
    5. Prediction head
    """

    def __init__(
        self,
        dense_dim: int,
        sparse_dims: Dict[str, int],
        embedding_dim: int = 16,
        trie_routing_dim: int = 32,
        num_experts: int = 8,
        expert_hidden_dim: int = 64,
        expert_output_dim: int = 64,
        top_k: int = 2,
        routing_mode: str = 'hybrid',
        hidden_dims: List[int] = [128, 64],
        dropout: float = 0.1,
    ):
        super().__init__()

        self.dense_dim = dense_dim
        self.sparse_dims = sparse_dims
        self.num_sparse_fields = len(sparse_dims)

        # Embedding
        from .base import EmbeddingLayer, FeatureInteraction, MLPBlock
        self.embedding = EmbeddingLayer(sparse_dims, embedding_dim)

        # Dense processing
        self.dense_bn = nn.BatchNorm1d(dense_dim)

        # Feature interaction
        self.interaction = FeatureInteraction(reduce=True)

        # Input dimension for MoE
        moe_input_dim = dense_dim + self.num_sparse_fields * embedding_dim + embedding_dim

        # Trie-guided MoE
        self.moe = TrieGuidedMoE(
            input_dim=moe_input_dim,
            trie_routing_dim=trie_routing_dim,
            num_experts=num_experts,
            expert_hidden_dim=expert_hidden_dim,
            expert_output_dim=expert_output_dim,
            top_k=top_k,
            routing_mode=routing_mode,
        )

        # Prediction head
        self.prediction_head = MLPBlock(
            input_dim=expert_output_dim,
            hidden_dims=hidden_dims,
            output_dim=1,
            dropout=dropout,
        )

    def forward(
        self,
        dense: torch.Tensor,
        sparse: torch.Tensor,
        trie_routing_vec: torch.Tensor,
        expert_hints: Optional[torch.Tensor] = None,
        return_routing: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        batch_size = dense.shape[0]

        # Process features
        dense = self.dense_bn(dense)
        sparse_emb = self.embedding(sparse)  # (batch, num_fields, emb_dim)
        sparse_flat = sparse_emb.view(batch_size, -1)

        # Feature interaction
        interaction = self.interaction(sparse_emb)  # (batch, emb_dim)

        # Combine all features
        combined = torch.cat([dense, sparse_flat, interaction], dim=1)

        # MoE layer
        moe_output = self.moe(
            combined,
            trie_routing_vec,
            expert_hints,
            return_routing=return_routing,
        )

        # Prediction
        logits = self.prediction_head(moe_output['output']).squeeze(-1)

        result = {
            'logits': logits,
            'load_balance_loss': moe_output['load_balance_loss'],
        }

        if return_routing:
            result['routing_info'] = moe_output['routing_info']

        return result

    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute total loss including load balance regularization."""
        bce_loss = F.binary_cross_entropy_with_logits(
            outputs['logits'], labels
        )
        lb_loss = outputs.get('load_balance_loss', 0)
        return bce_loss + lb_loss
