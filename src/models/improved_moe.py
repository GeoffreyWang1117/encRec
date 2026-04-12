"""
Improved MoE implementations based on theoretical analysis.

Key improvements:
1. Loss-Free Balancing (DeepSeek style)
2. Expert Choice Routing (NeurIPS 2022)
3. Task-Aware Trie Integration
4. Soft MoE baseline
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import math


class LossFreeBalancedMoE(nn.Module):
    """
    MoE with dynamic bias balancing instead of auxiliary loss.

    Based on DeepSeek's Loss-Free Balancing strategy.

    Key insight: Auxiliary loss can conflict with task loss gradient.
    Solution: Use dynamic expert bias to maintain balance without loss.
    """

    def __init__(
        self,
        input_dim: int,
        num_experts: int = 8,
        expert_hidden_dim: int = 64,
        expert_output_dim: int = 64,
        top_k: int = 1,
        bias_update_rate: float = 0.01,
        ema_decay: float = 0.9,
    ):
        super().__init__()

        self.num_experts = num_experts
        self.top_k = top_k
        self.bias_update_rate = bias_update_rate
        self.ema_decay = ema_decay

        # Router (simple learned router)
        self.router = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, num_experts),
        )

        # Dynamic expert bias (learned, but also updated dynamically)
        self.expert_bias = nn.Parameter(torch.zeros(num_experts))

        # EMA load tracking (not a parameter, just state)
        self.register_buffer('ema_load', torch.ones(num_experts) / num_experts)

        # Experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, expert_hidden_dim),
                nn.ReLU(),
                nn.Linear(expert_hidden_dim, expert_output_dim),
            )
            for _ in range(num_experts)
        ])

    def _update_bias(self, expert_indices: torch.Tensor):
        """Update expert bias based on current load."""
        with torch.no_grad():
            # Compute current load
            one_hot = F.one_hot(expert_indices[:, 0], self.num_experts).float()
            current_load = one_hot.mean(dim=0)

            # Update EMA
            self.ema_load = self.ema_decay * self.ema_load + (1 - self.ema_decay) * current_load

            # Adjust bias: underloaded experts get positive bias, overloaded get negative
            target_load = 1.0 / self.num_experts
            self.expert_bias.data += self.bias_update_rate * (target_load - self.ema_load)

    def forward(
        self,
        x: torch.Tensor,
        return_routing: bool = False,
    ) -> Dict[str, torch.Tensor]:
        batch_size = x.shape[0]

        # Compute router logits
        router_logits = self.router(x)

        # Apply dynamic bias
        adjusted_logits = router_logits + self.expert_bias

        # Top-K selection
        top_k_logits, expert_indices = torch.topk(adjusted_logits, self.top_k, dim=-1)
        expert_weights = F.softmax(top_k_logits, dim=-1)

        # Update bias during training
        if self.training:
            self._update_bias(expert_indices)

        # Compute expert outputs
        output = torch.zeros(batch_size, self.experts[0][-1].out_features, device=x.device)
        for k in range(self.top_k):
            for expert_idx in range(self.num_experts):
                mask = expert_indices[:, k] == expert_idx
                if mask.any():
                    expert_out = self.experts[expert_idx](x[mask])
                    output[mask] += expert_weights[mask, k:k+1] * expert_out

        result = {'output': output}

        if return_routing:
            result['routing_info'] = {
                'expert_indices': expert_indices,
                'expert_weights': expert_weights,
                'expert_bias': self.expert_bias.clone(),
                'ema_load': self.ema_load.clone(),
            }

        return result


class ExpertChoiceMoE(nn.Module):
    """
    Expert Choice Routing: Experts select tokens instead of tokens selecting experts.

    Reference: "Mixture-of-Experts with Expert Choice Routing" (NeurIPS 2022)

    Key advantages:
    1. Perfect load balancing (no auxiliary loss needed)
    2. Each token can be processed by variable number of experts
    3. Faster convergence (2x+ speedup reported)
    """

    def __init__(
        self,
        input_dim: int,
        num_experts: int = 8,
        expert_hidden_dim: int = 64,
        expert_output_dim: int = 64,
        capacity_factor: float = 1.25,  # Each expert processes batch/num_experts * capacity_factor tokens
    ):
        super().__init__()

        self.num_experts = num_experts
        self.capacity_factor = capacity_factor

        # Router computes expert preferences for each token
        self.router = nn.Linear(input_dim, num_experts)

        # Experts
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
        return_routing: bool = False,
    ) -> Dict[str, torch.Tensor]:
        batch_size = x.shape[0]
        device = x.device

        # Compute router scores: (batch, num_experts)
        router_scores = self.router(x)
        router_probs = F.softmax(router_scores, dim=0)  # Softmax over batch dimension!

        # Each expert selects top-k tokens
        tokens_per_expert = int(batch_size / self.num_experts * self.capacity_factor)
        tokens_per_expert = max(1, min(tokens_per_expert, batch_size))

        # Initialize output
        output = torch.zeros(batch_size, self.output_dim, device=device)
        token_counts = torch.zeros(batch_size, device=device)

        expert_assignments = []

        for expert_idx in range(self.num_experts):
            # Expert selects its top tokens
            expert_prefs = router_probs[:, expert_idx]
            top_k_probs, top_k_indices = torch.topk(expert_prefs, tokens_per_expert)

            # Compute expert output for selected tokens
            selected_tokens = x[top_k_indices]
            expert_out = self.experts[expert_idx](selected_tokens)

            # Weighted accumulation
            output[top_k_indices] += top_k_probs.unsqueeze(-1) * expert_out
            token_counts[top_k_indices] += top_k_probs

            expert_assignments.append(top_k_indices)

        # Normalize by total weight received
        output = output / (token_counts.unsqueeze(-1) + 1e-8)

        result = {'output': output}

        if return_routing:
            result['routing_info'] = {
                'router_probs': router_probs,
                'expert_assignments': expert_assignments,
                'token_counts': token_counts,
            }

        return result


class SoftMoE(nn.Module):
    """
    Soft MoE: Fully differentiable routing using softmax over all experts.

    Advantages:
    - All experts receive gradients
    - More stable training
    - No load balancing issues

    Disadvantages:
    - Higher computation (runs all experts)
    - Less sparse
    """

    def __init__(
        self,
        input_dim: int,
        num_experts: int = 8,
        expert_hidden_dim: int = 64,
        expert_output_dim: int = 64,
        temperature: float = 1.0,
    ):
        super().__init__()

        self.num_experts = num_experts
        self.temperature = temperature

        # Router
        self.router = nn.Linear(input_dim, num_experts)

        # Experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, expert_hidden_dim),
                nn.ReLU(),
                nn.Linear(expert_hidden_dim, expert_output_dim),
            )
            for _ in range(num_experts)
        ])

    def forward(
        self,
        x: torch.Tensor,
        return_routing: bool = False,
    ) -> Dict[str, torch.Tensor]:
        # Compute routing weights
        router_logits = self.router(x)
        router_weights = F.softmax(router_logits / self.temperature, dim=-1)

        # Compute all expert outputs
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)

        # Weighted combination
        output = (router_weights.unsqueeze(-1) * expert_outputs).sum(dim=1)

        result = {'output': output}

        if return_routing:
            result['routing_info'] = {
                'router_weights': router_weights,
                'router_logits': router_logits,
            }

        return result


class TaskAwareTrieEncoder(nn.Module):
    """
    Task-Aware Trie Encoder: Learns to integrate statistical features with task objectives.

    Key insight: Pure statistical features (freq, CTR) may not align with optimal routing.
    Solution: Use cross-attention to let the model learn task-relevant combinations.
    """

    def __init__(
        self,
        stat_dim: int,  # Dimension of statistical features from Trie
        embedding_dim: int,  # Dimension of learned embeddings
        output_dim: int = 64,
        num_heads: int = 4,
    ):
        super().__init__()

        self.hidden_dim = max(stat_dim, embedding_dim)

        # Project statistics to hidden dim
        self.stat_proj = nn.Linear(stat_dim, self.hidden_dim)

        # Project embeddings to hidden dim
        self.emb_proj = nn.Linear(embedding_dim, self.hidden_dim)

        # Cross-attention: embeddings attend to statistics
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True,
        )

        # Self-attention for refinement
        self.self_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True,
        )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, output_dim),
        )

        # Layer norms
        self.ln1 = nn.LayerNorm(self.hidden_dim)
        self.ln2 = nn.LayerNorm(self.hidden_dim)

    def forward(
        self,
        stat_features: torch.Tensor,  # (batch, stat_dim)
        embeddings: torch.Tensor,  # (batch, num_fields, embedding_dim)
    ) -> torch.Tensor:
        batch_size = stat_features.shape[0]

        # Project inputs
        stat_emb = self.stat_proj(stat_features).unsqueeze(1)  # (batch, 1, hidden)
        emb_proj = self.emb_proj(embeddings)  # (batch, num_fields, hidden)

        # Cross-attention: embeddings attend to statistics
        attended, _ = self.cross_attention(
            query=emb_proj,
            key=stat_emb,
            value=stat_emb,
        )
        emb_refined = self.ln1(emb_proj + attended)

        # Self-attention among fields
        self_attended, _ = self.self_attention(
            query=emb_refined,
            key=emb_refined,
            value=emb_refined,
        )
        emb_final = self.ln2(emb_refined + self_attended)

        # Pool and combine
        emb_pooled = emb_final.mean(dim=1)  # (batch, hidden)
        stat_pooled = stat_emb.squeeze(1)  # (batch, hidden)

        # Concatenate and project
        combined = torch.cat([emb_pooled, stat_pooled], dim=-1)
        output = self.output_proj(combined)

        return output


class ImprovedTrieMoERecommender(nn.Module):
    """
    Improved Trie-MoE Recommender with all enhancements.

    Key improvements:
    1. Loss-Free Balancing (no auxiliary loss)
    2. Task-Aware Trie encoding
    3. Optional Expert Choice routing
    4. Progressive temperature annealing
    """

    def __init__(
        self,
        dense_dim: int,
        sparse_dims: Dict[str, int],
        embedding_dim: int = 16,
        trie_stat_dim: int = 32,  # Dimension of Trie statistics
        num_experts: int = 8,
        expert_hidden_dim: int = 64,
        expert_output_dim: int = 64,
        top_k: int = 1,
        moe_type: str = 'loss_free',  # 'loss_free', 'expert_choice', 'soft'
        hidden_dims: List[int] = [128, 64],
        dropout: float = 0.1,
    ):
        super().__init__()

        self.dense_dim = dense_dim
        self.sparse_dims = sparse_dims
        self.num_sparse_fields = len(sparse_dims)
        self.moe_type = moe_type

        # Embeddings
        self.embeddings = nn.ModuleDict({
            field: nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
            for field, vocab_size in sparse_dims.items()
        })
        self.field_names = list(sparse_dims.keys())

        # Dense processing
        self.dense_bn = nn.BatchNorm1d(dense_dim)
        self.dense_proj = nn.Linear(dense_dim, embedding_dim)

        # Task-Aware Trie Encoder
        self.trie_encoder = TaskAwareTrieEncoder(
            stat_dim=trie_stat_dim,
            embedding_dim=embedding_dim,
            output_dim=64,
            num_heads=4,
        )

        # MoE input dimension
        moe_input_dim = (self.num_sparse_fields + 1) * embedding_dim + 64  # +64 for trie encoding

        # Select MoE type
        if moe_type == 'loss_free':
            self.moe = LossFreeBalancedMoE(
                input_dim=moe_input_dim,
                num_experts=num_experts,
                expert_hidden_dim=expert_hidden_dim,
                expert_output_dim=expert_output_dim,
                top_k=top_k,
            )
        elif moe_type == 'expert_choice':
            self.moe = ExpertChoiceMoE(
                input_dim=moe_input_dim,
                num_experts=num_experts,
                expert_hidden_dim=expert_hidden_dim,
                expert_output_dim=expert_output_dim,
            )
        else:  # soft
            self.moe = SoftMoE(
                input_dim=moe_input_dim,
                num_experts=num_experts,
                expert_hidden_dim=expert_hidden_dim,
                expert_output_dim=expert_output_dim,
            )

        # Prediction head
        pred_input_dim = expert_output_dim
        layers = []
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(pred_input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            pred_input_dim = hidden_dim
        layers.append(nn.Linear(pred_input_dim, 1))
        self.prediction_head = nn.Sequential(*layers)

    def forward(
        self,
        dense: torch.Tensor,
        sparse: torch.Tensor,
        trie_routing_vec: Optional[torch.Tensor] = None,  # Trie statistics
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        batch_size = dense.shape[0]

        # Process dense
        dense = self.dense_bn(dense)
        dense_emb = self.dense_proj(dense)  # (batch, embedding_dim)

        # Process sparse
        emb_list = []
        for i, field in enumerate(self.field_names):
            emb_list.append(self.embeddings[field](sparse[:, i]))
        sparse_emb = torch.stack(emb_list, dim=1)  # (batch, num_fields, embedding_dim)

        # Task-Aware Trie encoding
        if trie_routing_vec is not None:
            trie_encoded = self.trie_encoder(trie_routing_vec, sparse_emb)
        else:
            # If no Trie features, use zeros
            trie_encoded = torch.zeros(batch_size, 64, device=dense.device)

        # Flatten sparse embeddings
        sparse_flat = sparse_emb.view(batch_size, -1)

        # Combine all features
        combined = torch.cat([dense_emb, sparse_flat, trie_encoded], dim=-1)

        # MoE
        moe_output = self.moe(combined)

        # Prediction
        logits = self.prediction_head(moe_output['output']).squeeze(-1)

        return {'logits': logits}


# Alias for compatibility
ImprovedMoERecommender = ImprovedTrieMoERecommender
