"""
Trie-MoE: Unified model for privacy-preserving & data-scarce recommendation.

This model integrates:
1. Statistical Trie for token organization
2. Trie-based encoder for routing signals
3. Adaptive alpha mechanism for frequency-aware routing
4. MoE layer for conditional computation

Key innovation: Uses statistical priors to guide routing when learned routing fails.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np


class TrieFeatureEncoder(nn.Module):
    """
    Encodes sparse tokens into Trie-based statistical features.

    Unlike embeddings which are learned per-token, this encoder uses
    pre-computed statistical properties that work even for unseen tokens.
    """

    def __init__(
        self,
        num_fields: int,
        stats_dim: int = 7,  # freq, ctr, lift, entropy, MI, stability, count
        output_dim: int = 32,
    ):
        super().__init__()
        self.num_fields = num_fields
        self.stats_dim = stats_dim
        self.output_dim = output_dim

        # Per-field statistical feature vectors will be registered as buffers
        # These are computed from data, not learned
        self.register_buffer('default_stats', torch.zeros(stats_dim))

        # Learnable projection
        self.projection = nn.Sequential(
            nn.Linear(num_fields * stats_dim + 5, output_dim * 2),
            nn.LayerNorm(output_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(
        self,
        trie_features: torch.Tensor,  # (batch, num_fields * stats_dim + 5)
    ) -> torch.Tensor:
        """
        Project Trie statistics to routing dimension.

        Args:
            trie_features: Pre-computed statistical features

        Returns:
            (batch, output_dim) routing signals
        """
        return self.projection(trie_features)


class AdaptiveAlphaGate(nn.Module):
    """
    Computes adaptive blending weight α(n) = σ(w·log(n+1) + b)

    For low-frequency (cold) tokens: α high, trust Trie prior more
    For high-frequency (warm) tokens: α low, trust learned routing more
    """

    def __init__(
        self,
        alpha_init_weight: float = -0.5,  # Negative: higher freq -> lower alpha
        alpha_init_bias: float = 0.5,     # Default alpha around 0.5
    ):
        super().__init__()
        self.alpha_weight = nn.Parameter(torch.tensor(alpha_init_weight))
        self.alpha_bias = nn.Parameter(torch.tensor(alpha_init_bias))

    def forward(self, token_freqs: torch.Tensor) -> torch.Tensor:
        """
        Compute adaptive alpha.

        Args:
            token_freqs: (batch,) token frequencies

        Returns:
            (batch, 1) alpha values in [0, 1]
        """
        log_freq = torch.log(token_freqs.float() + 1)
        alpha = torch.sigmoid(self.alpha_weight * log_freq + self.alpha_bias)
        return alpha.unsqueeze(1)


class TrieGuidedRouter(nn.Module):
    """
    Router combining Trie prior with learned routing.

    routing = α·trie_routing + (1-α)·learned_routing
    """

    def __init__(
        self,
        trie_dim: int,
        embedding_dim: int,
        num_experts: int,
        hidden_dim: int = 64,
        routing_mode: str = "hybrid",  # trie_only, learned_only, hybrid
    ):
        super().__init__()
        self.num_experts = num_experts
        self.routing_mode = routing_mode

        # Trie-based router (uses statistical features)
        self.trie_router = nn.Sequential(
            nn.Linear(trie_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_experts),
        )

        # Learned router (uses embeddings)
        self.learned_router = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_experts),
        )

        # Adaptive alpha
        self.alpha_gate = AdaptiveAlphaGate()

    def forward(
        self,
        trie_features: torch.Tensor,   # (batch, trie_dim)
        embeddings: torch.Tensor,      # (batch, embedding_dim)
        token_freqs: torch.Tensor,     # (batch,)
        return_info: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Compute hybrid routing logits.

        Returns:
            router_logits: (batch, num_experts)
            routing_info: Optional dict with diagnostic info
        """
        # Compute both routing signals
        trie_logits = self.trie_router(trie_features)
        learned_logits = self.learned_router(embeddings)

        # Compute adaptive alpha
        alpha = self.alpha_gate(token_freqs)

        # Blend based on routing mode
        if self.routing_mode == "trie_only":
            router_logits = trie_logits
            effective_alpha = torch.ones_like(alpha)
        elif self.routing_mode == "learned_only":
            router_logits = learned_logits
            effective_alpha = torch.zeros_like(alpha)
        else:  # hybrid
            router_logits = alpha * trie_logits + (1 - alpha) * learned_logits
            effective_alpha = alpha

        if return_info:
            info = {
                'trie_logits': trie_logits,
                'learned_logits': learned_logits,
                'alpha': effective_alpha.squeeze(-1),
                'routing_entropy': self._compute_entropy(router_logits),
            }
            return router_logits, info

        return router_logits, None

    def _compute_entropy(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute entropy of routing distribution."""
        probs = F.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
        return entropy


class TrieMoE(nn.Module):
    """
    Trie-Guided Mixture of Experts for CTR Prediction.

    Architecture:
    1. Embedding layer for sparse features
    2. Trie feature encoder for routing signals
    3. Adaptive alpha router (trie prior + learned routing)
    4. Expert networks
    5. Prediction head

    Key properties:
    - Works with encrypted/hashed features (no semantics needed)
    - Stable routing in data-scarce scenarios (<10K samples)
    - Interpretable routing decisions via Trie path
    """

    def __init__(
        self,
        dense_dim: int,
        sparse_dims: Dict[str, int],
        embedding_dim: int = 16,
        num_experts: int = 8,
        expert_hidden_dim: int = 64,
        expert_output_dim: int = 64,
        trie_stats_dim: int = 7,
        trie_output_dim: int = 32,
        routing_mode: str = "hybrid",
        hidden_dims: List[int] = [64, 32],
        dropout: float = 0.1,
        load_balance_weight: float = 0.01,
    ):
        super().__init__()

        self.num_experts = num_experts
        self.load_balance_weight = load_balance_weight
        self.sparse_field_names = list(sparse_dims.keys())
        self.num_sparse_fields = len(sparse_dims)

        # Sparse embeddings
        self.embeddings = nn.ModuleDict({
            name: nn.Embedding(dim, embedding_dim)
            for name, dim in sparse_dims.items()
        })

        # Dense processing
        self.dense_bn = nn.BatchNorm1d(dense_dim)

        # Trie feature encoder
        self.trie_encoder = TrieFeatureEncoder(
            num_fields=self.num_sparse_fields,
            stats_dim=trie_stats_dim,
            output_dim=trie_output_dim,
        )

        # Total embedding dimension
        total_emb_dim = self.num_sparse_fields * embedding_dim

        # Trie-guided router
        self.router = TrieGuidedRouter(
            trie_dim=trie_output_dim,
            embedding_dim=total_emb_dim,
            num_experts=num_experts,
            routing_mode=routing_mode,
        )

        # Expert networks
        expert_input_dim = dense_dim + total_emb_dim
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(expert_input_dim, expert_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(expert_hidden_dim, expert_output_dim),
            )
            for _ in range(num_experts)
        ])

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

        self.output_dim = expert_output_dim

    def forward(
        self,
        dense: torch.Tensor,
        sparse: torch.Tensor,
        trie_features: torch.Tensor,
        token_freqs: torch.Tensor,
        return_routing: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            dense: (batch, dense_dim) dense features
            sparse: (batch, num_sparse_fields) sparse feature indices
            trie_features: (batch, num_fields * stats_dim + 5) pre-computed Trie stats
            token_freqs: (batch,) token frequencies for adaptive alpha
            return_routing: Whether to return routing diagnostics

        Returns:
            Dict with 'logits', 'load_balance_loss', optionally 'routing_info'
        """
        batch_size = dense.shape[0]
        device = dense.device

        # Process dense features
        dense_out = self.dense_bn(dense)

        # Process sparse features
        sparse_embs = [
            self.embeddings[name](sparse[:, i])
            for i, name in enumerate(self.sparse_field_names)
        ]
        sparse_flat = torch.cat(sparse_embs, dim=1)  # (batch, total_emb_dim)

        # Encode Trie features
        trie_encoded = self.trie_encoder(trie_features)  # (batch, trie_output_dim)

        # Get routing decisions
        router_logits, routing_info = self.router(
            trie_encoded, sparse_flat, token_freqs, return_info=return_routing
        )

        # Top-2 expert selection
        top_k_logits, expert_indices = torch.topk(router_logits, 2, dim=-1)
        expert_weights = F.softmax(top_k_logits, dim=-1)

        # Combine features for experts
        combined = torch.cat([dense_out, sparse_flat], dim=1)

        # Compute expert outputs
        all_expert_outputs = torch.stack(
            [expert(combined) for expert in self.experts], dim=1
        )  # (batch, num_experts, expert_output_dim)

        # Weighted combination of top-2 experts
        output = torch.zeros(batch_size, self.output_dim, device=device)
        for k in range(2):
            expert_idx = expert_indices[:, k]
            weight = expert_weights[:, k:k+1]
            expert_out = all_expert_outputs[
                torch.arange(batch_size, device=device), expert_idx
            ]
            output = output + weight * expert_out

        # Prediction
        logits = self.prediction_head(output).squeeze(-1)

        # Load balance loss
        expert_mask = F.one_hot(expert_indices[:, 0], self.num_experts).float()
        fraction = expert_mask.mean(dim=0)
        prob = F.softmax(router_logits, dim=-1).mean(dim=0)
        lb_loss = self.num_experts * (fraction * prob).sum() * self.load_balance_weight

        result = {
            'logits': logits,
            'load_balance_loss': lb_loss,
        }

        if return_routing:
            result['routing_info'] = {
                'expert_indices': expert_indices,
                'expert_weights': expert_weights,
                'alpha': routing_info['alpha'] if routing_info else None,
                'routing_entropy': routing_info['routing_entropy'] if routing_info else None,
            }

        return result

    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute total loss (BCE + load balance)."""
        bce_loss = F.binary_cross_entropy_with_logits(outputs['logits'], labels)
        lb_loss = outputs.get('load_balance_loss', 0)
        return bce_loss + lb_loss


class StandardMoE(nn.Module):
    """
    Standard MoE for comparison (no Trie guidance).

    Uses purely learned routing - expected to collapse with small data.
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
        load_balance_weight: float = 0.01,
    ):
        super().__init__()

        self.num_experts = num_experts
        self.load_balance_weight = load_balance_weight
        self.sparse_field_names = list(sparse_dims.keys())
        self.num_sparse_fields = len(sparse_dims)

        # Sparse embeddings
        self.embeddings = nn.ModuleDict({
            name: nn.Embedding(dim, embedding_dim)
            for name, dim in sparse_dims.items()
        })

        # Dense processing
        self.dense_bn = nn.BatchNorm1d(dense_dim)

        # Total embedding dimension
        total_emb_dim = self.num_sparse_fields * embedding_dim
        expert_input_dim = dense_dim + total_emb_dim

        # Learned router (no Trie guidance)
        self.router = nn.Sequential(
            nn.Linear(total_emb_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_experts),
        )

        # Expert networks
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(expert_input_dim, expert_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(expert_hidden_dim, expert_output_dim),
            )
            for _ in range(num_experts)
        ])

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

        self.output_dim = expert_output_dim

    def forward(
        self,
        dense: torch.Tensor,
        sparse: torch.Tensor,
        return_routing: bool = False,
        **kwargs,  # Ignore trie_features, token_freqs for compatibility
    ) -> Dict[str, torch.Tensor]:
        batch_size = dense.shape[0]
        device = dense.device

        # Process features
        dense_out = self.dense_bn(dense)
        sparse_embs = [
            self.embeddings[name](sparse[:, i])
            for i, name in enumerate(self.sparse_field_names)
        ]
        sparse_flat = torch.cat(sparse_embs, dim=1)

        # Routing
        router_logits = self.router(sparse_flat)

        # Top-2 selection
        top_k_logits, expert_indices = torch.topk(router_logits, 2, dim=-1)
        expert_weights = F.softmax(top_k_logits, dim=-1)

        # Expert computation
        combined = torch.cat([dense_out, sparse_flat], dim=1)
        all_expert_outputs = torch.stack(
            [expert(combined) for expert in self.experts], dim=1
        )

        output = torch.zeros(batch_size, self.output_dim, device=device)
        for k in range(2):
            expert_idx = expert_indices[:, k]
            weight = expert_weights[:, k:k+1]
            expert_out = all_expert_outputs[
                torch.arange(batch_size, device=device), expert_idx
            ]
            output = output + weight * expert_out

        logits = self.prediction_head(output).squeeze(-1)

        # Load balance loss
        expert_mask = F.one_hot(expert_indices[:, 0], self.num_experts).float()
        fraction = expert_mask.mean(dim=0)
        prob = F.softmax(router_logits, dim=-1).mean(dim=0)
        lb_loss = self.num_experts * (fraction * prob).sum() * self.load_balance_weight

        result = {
            'logits': logits,
            'load_balance_loss': lb_loss,
        }

        if return_routing:
            # Compute routing entropy
            probs = F.softmax(router_logits, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)

            result['routing_info'] = {
                'expert_indices': expert_indices,
                'expert_weights': expert_weights,
                'routing_entropy': entropy,
            }

        return result

    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor,
    ) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(outputs['logits'], labels)
        lb_loss = outputs.get('load_balance_loss', 0)
        return bce_loss + lb_loss
