"""
Trie-based routing mechanisms.

These routers use statistical structure from Trie to make
routing decisions without relying on semantic information.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np


class TrieRouter(nn.Module):
    """
    Router that uses Trie statistics for expert selection.

    Unlike learned routers that struggle with encrypted/hashed tokens,
    this router uses stable statistical features:
    - Frequency distribution
    - CTR patterns
    - Information content
    - Temporal stability
    """

    def __init__(
        self,
        trie_feature_dim: int,
        num_experts: int,
        hidden_dim: int = 64,
        use_attention: bool = True,
        temperature: float = 1.0,
    ):
        super().__init__()

        self.num_experts = num_experts
        self.use_attention = use_attention
        self.temperature = temperature

        # Main routing network
        self.router_mlp = nn.Sequential(
            nn.Linear(trie_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_experts),
        )

        # Optional attention over Trie features
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=trie_feature_dim,
                num_heads=4,
                dropout=0.1,
                batch_first=True,
            )
            self.attn_proj = nn.Linear(trie_feature_dim, trie_feature_dim)

        # Expert embeddings for interpretability
        self.expert_embeddings = nn.Parameter(
            torch.randn(num_experts, hidden_dim) * 0.02
        )

    def forward(
        self,
        trie_features: torch.Tensor,  # (batch, trie_feature_dim)
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute routing logits.

        Args:
            trie_features: Statistical features from Trie
            return_attention: Whether to return attention weights

        Returns:
            router_logits: (batch, num_experts)
            attention_weights: Optional attention weights for interpretability
        """
        attention_weights = None

        if self.use_attention:
            # Self-attention over feature dimensions
            # Reshape for attention: (batch, seq_len=1, dim)
            x = trie_features.unsqueeze(1)
            attn_out, attention_weights = self.attention(x, x, x)
            trie_features = trie_features + self.attn_proj(attn_out.squeeze(1))

        # Compute routing logits
        router_logits = self.router_mlp(trie_features)

        # Temperature scaling
        router_logits = router_logits / self.temperature

        if return_attention:
            return router_logits, attention_weights
        return router_logits, None

    def get_expert_selection(
        self,
        trie_features: torch.Tensor,
        top_k: int = 2,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get top-k expert selections.

        Returns:
            expert_indices: (batch, top_k)
            expert_weights: (batch, top_k)
        """
        router_logits, _ = self.forward(trie_features)

        top_k_logits, expert_indices = torch.topk(router_logits, top_k, dim=-1)
        expert_weights = F.softmax(top_k_logits, dim=-1)

        return expert_indices, expert_weights


class HybridRouter(nn.Module):
    """
    Hybrid router combining Trie statistics with learned embeddings.

    This allows the model to:
    1. Start with stable Trie-based routing
    2. Gradually learn refinements from data
    3. Fall back to Trie routing for OOV/rare tokens
    """

    def __init__(
        self,
        trie_feature_dim: int,
        embedding_dim: int,
        num_experts: int,
        hidden_dim: int = 64,
        trie_weight_init: float = 0.8,  # Initial weight for Trie routing
        learnable_weight: bool = True,
    ):
        super().__init__()

        self.num_experts = num_experts

        # Trie-based router
        self.trie_router = TrieRouter(
            trie_feature_dim=trie_feature_dim,
            num_experts=num_experts,
            hidden_dim=hidden_dim,
            use_attention=True,
        )

        # Learned router (from embeddings)
        self.learned_router = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_experts),
        )

        # Mixing weight
        if learnable_weight:
            # Sigmoid-constrained weight
            self.trie_weight_logit = nn.Parameter(
                torch.tensor(np.log(trie_weight_init / (1 - trie_weight_init)))
            )
        else:
            self.register_buffer(
                'trie_weight_logit',
                torch.tensor(np.log(trie_weight_init / (1 - trie_weight_init)))
            )

        # Gate network to adaptively weight Trie vs learned
        self.gate = nn.Sequential(
            nn.Linear(trie_feature_dim + embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    @property
    def trie_weight(self) -> torch.Tensor:
        return torch.sigmoid(self.trie_weight_logit)

    def forward(
        self,
        trie_features: torch.Tensor,
        embeddings: torch.Tensor,
        use_adaptive_gate: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute hybrid routing.

        Args:
            trie_features: (batch, trie_feature_dim)
            embeddings: (batch, embedding_dim)
            use_adaptive_gate: Whether to use sample-wise gating

        Returns:
            router_logits: (batch, num_experts)
            routing_info: Dict with debugging information
        """
        # Trie routing
        trie_logits, attn_weights = self.trie_router(trie_features, return_attention=True)

        # Learned routing
        learned_logits = self.learned_router(embeddings)

        if use_adaptive_gate:
            # Sample-wise adaptive gating
            combined = torch.cat([trie_features, embeddings], dim=1)
            gate_weight = self.gate(combined)  # (batch, 1)
            router_logits = gate_weight * trie_logits + (1 - gate_weight) * learned_logits
            effective_trie_weight = gate_weight.squeeze(-1)
        else:
            # Global fixed weight
            router_logits = self.trie_weight * trie_logits + (1 - self.trie_weight) * learned_logits
            effective_trie_weight = self.trie_weight.expand(trie_features.shape[0])

        routing_info = {
            'trie_logits': trie_logits,
            'learned_logits': learned_logits,
            'effective_trie_weight': effective_trie_weight,
            'attention_weights': attn_weights,
        }

        return router_logits, routing_info


class FrequencyAwareRouter(nn.Module):
    """
    Router with explicit frequency-awareness.

    Uses different routing strategies for head vs tail tokens:
    - Head tokens: More confident, stable routing
    - Tail tokens: More exploration, multiple experts
    """

    def __init__(
        self,
        trie_feature_dim: int,
        num_experts: int,
        hidden_dim: int = 64,
        head_top_k: int = 1,  # Top-k for head tokens
        tail_top_k: int = 3,  # Top-k for tail tokens (more exploration)
    ):
        super().__init__()

        self.num_experts = num_experts
        self.head_top_k = head_top_k
        self.tail_top_k = tail_top_k

        # Base router
        self.router = TrieRouter(
            trie_feature_dim=trie_feature_dim,
            num_experts=num_experts,
            hidden_dim=hidden_dim,
        )

        # Frequency classifier (determines head vs tail)
        self.freq_classifier = nn.Sequential(
            nn.Linear(trie_feature_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        trie_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Frequency-aware routing.

        Returns:
            expert_indices: (batch, max_top_k)
            expert_weights: (batch, max_top_k)
            info: Additional information
        """
        batch_size = trie_features.shape[0]
        device = trie_features.device

        # Compute routing logits
        router_logits, _ = self.router(trie_features)

        # Determine frequency bucket (0 = tail, 1 = head)
        freq_score = self.freq_classifier(trie_features).squeeze(-1)  # (batch,)
        is_head = freq_score > 0.5

        # Different top-k for head vs tail
        max_top_k = max(self.head_top_k, self.tail_top_k)

        # Get top-k experts
        top_k_logits, top_k_indices = torch.topk(
            router_logits, max_top_k, dim=-1
        )

        # Compute weights
        weights = F.softmax(top_k_logits, dim=-1)

        # Mask out extra experts for head tokens
        if self.head_top_k < max_top_k:
            head_mask = torch.zeros(batch_size, max_top_k, device=device)
            head_mask[:, :self.head_top_k] = 1.0
            head_mask = head_mask * is_head.unsqueeze(-1).float()

            tail_mask = torch.ones(batch_size, max_top_k, device=device)
            tail_mask = tail_mask * (~is_head).unsqueeze(-1).float()

            mask = head_mask + tail_mask
            weights = weights * mask
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)

        info = {
            'freq_score': freq_score,
            'is_head': is_head,
            'router_logits': router_logits,
        }

        return top_k_indices, weights, info
