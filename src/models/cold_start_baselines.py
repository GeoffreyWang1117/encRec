"""
Cold-Start Specific Baselines for Comparison.

This module implements:
1. DropoutNet (NeurIPS 2017) - Dropout-based cold-start handling
2. MeLU (KDD 2019) - Meta-learning for cold-start
3. MetaEmb (SIGIR 2019) - Meta-embedding for cold-start ads
4. MMoE (KDD 2018) - Multi-gate Mixture of Experts
5. PLE (RecSys 2020) - Progressive Layered Extraction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import copy

from .base import BaseRecommender, EmbeddingLayer, MLPBlock


class DropoutNet(BaseRecommender):
    """
    DropoutNet: Addressing Cold Start in Recommender Systems.

    Reference: Volkovs et al., "DropoutNet: Addressing Cold Start in
    Recommender Systems", NeurIPS 2017.

    Key idea: Apply input dropout during training to simulate cold-start
    conditions, forcing the model to leverage content features.
    """

    def __init__(
        self,
        dense_dim: int,
        sparse_dims: Dict[str, int],
        embedding_dim: int = 16,
        hidden_dims: List[int] = [256, 128, 64],
        dropout: float = 0.1,
        input_dropout: float = 0.5,  # Key: high dropout for cold-start simulation
        content_dim: int = 0,  # Optional content features
    ):
        super().__init__(dense_dim, sparse_dims, embedding_dim)

        self.input_dropout = nn.Dropout(input_dropout)
        self.content_dim = content_dim

        # Content encoder (if content features available)
        if content_dim > 0:
            self.content_encoder = nn.Sequential(
                nn.Linear(content_dim, embedding_dim),
                nn.ReLU(),
                nn.Linear(embedding_dim, embedding_dim),
            )

        # Main network
        input_dim = dense_dim + self.num_sparse_fields * embedding_dim
        if content_dim > 0:
            input_dim += embedding_dim  # Add content embedding

        self.mlp = MLPBlock(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=1,
            dropout=dropout,
        )

    def forward(
        self,
        dense: torch.Tensor,
        sparse: torch.Tensor,
        content: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        batch_size = dense.shape[0]

        # Normalize dense
        dense = self.dense_bn(dense)

        # Sparse embeddings with INPUT dropout (key for cold-start)
        sparse_emb = self.embedding(sparse)  # (batch, num_fields, emb_dim)

        # Apply input dropout to simulate cold-start during training
        if self.training:
            sparse_emb = self.input_dropout(sparse_emb)

        sparse_flat = sparse_emb.view(batch_size, -1)

        # Combine features
        features = [dense, sparse_flat]

        # Add content if available
        if self.content_dim > 0 and content is not None:
            content_emb = self.content_encoder(content)
            features.append(content_emb)

        x = torch.cat(features, dim=1)
        logits = self.mlp(x).squeeze(-1)

        return {'logits': logits}


class MeLU(BaseRecommender):
    """
    MeLU: Meta-Learned User Preference Estimator for Cold-Start Recommendation.

    Reference: Lee et al., "MeLU: Meta-Learned User Preference Estimator for
    Cold-Start Recommendation", KDD 2019.

    Key idea: Use MAML-style meta-learning to quickly adapt to new users/items
    with few interactions.

    Note: This is a simplified version. Full MeLU requires episode-based training.
    """

    def __init__(
        self,
        dense_dim: int,
        sparse_dims: Dict[str, int],
        embedding_dim: int = 16,
        hidden_dims: List[int] = [128, 64],
        dropout: float = 0.1,
        inner_lr: float = 0.01,
        num_inner_steps: int = 1,
    ):
        super().__init__(dense_dim, sparse_dims, embedding_dim)

        self.inner_lr = inner_lr
        self.num_inner_steps = num_inner_steps

        # Preference estimator network (to be adapted per user)
        input_dim = dense_dim + self.num_sparse_fields * embedding_dim
        self.preference_net = MLPBlock(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=1,
            dropout=dropout,
        )

    def forward(
        self,
        dense: torch.Tensor,
        sparse: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        batch_size = dense.shape[0]

        # Normalize dense
        dense = self.dense_bn(dense)

        # Sparse embeddings
        sparse_emb = self.embedding(sparse)
        sparse_flat = sparse_emb.view(batch_size, -1)

        # Concatenate
        x = torch.cat([dense, sparse_flat], dim=1)

        # Standard forward (meta-adaptation happens during training)
        logits = self.preference_net(x).squeeze(-1)

        return {'logits': logits}

    def adapt(
        self,
        support_dense: torch.Tensor,
        support_sparse: torch.Tensor,
        support_labels: torch.Tensor,
    ) -> 'MeLU':
        """
        Adapt model to a new user/item using support set.
        Returns adapted model copy.
        """
        # Clone model for adaptation
        adapted_model = copy.deepcopy(self)

        for _ in range(self.num_inner_steps):
            # Forward on support set
            output = adapted_model(support_dense, support_sparse)
            logits = output['logits']

            # Compute loss
            loss = F.binary_cross_entropy_with_logits(logits, support_labels)

            # Compute gradients
            grads = torch.autograd.grad(
                loss,
                adapted_model.preference_net.parameters(),
                create_graph=True,
            )

            # Update preference network
            for param, grad in zip(adapted_model.preference_net.parameters(), grads):
                param.data = param.data - self.inner_lr * grad

        return adapted_model


class MetaEmb(BaseRecommender):
    """
    MetaEmb: Learning Graph Meta Embeddings for Cold-start Ads.

    Reference: Pan et al., "Warm Up Cold-start Advertisements: Improving CTR
    Predictions via Learning to Learn ID Embeddings", SIGIR 2019.

    Key idea: Use meta-learning to generate initial embeddings for cold-start items
    based on their attributes/content.
    """

    def __init__(
        self,
        dense_dim: int,
        sparse_dims: Dict[str, int],
        embedding_dim: int = 16,
        hidden_dims: List[int] = [256, 128, 64],
        meta_hidden_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__(dense_dim, sparse_dims, embedding_dim)

        # Meta embedding generator: generates embeddings from attributes
        # Input: dense features, Output: initial embedding for cold items
        self.meta_generator = nn.Sequential(
            nn.Linear(dense_dim, meta_hidden_dim),
            nn.ReLU(),
            nn.Linear(meta_hidden_dim, embedding_dim),
        )

        # Gating mechanism to blend meta and learned embeddings
        self.gate = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.Sigmoid(),
        )

        # Main prediction network
        input_dim = dense_dim + self.num_sparse_fields * embedding_dim
        self.mlp = MLPBlock(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=1,
            dropout=dropout,
        )

    def forward(
        self,
        dense: torch.Tensor,
        sparse: torch.Tensor,
        is_cold: Optional[torch.Tensor] = None,  # (batch,) bool mask
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        batch_size = dense.shape[0]

        # Normalize dense
        dense_normed = self.dense_bn(dense)

        # Sparse embeddings
        sparse_emb = self.embedding(sparse)  # (batch, num_fields, emb_dim)

        # Generate meta embeddings for first sparse field (assumed to be item)
        meta_emb = self.meta_generator(dense_normed)  # (batch, emb_dim)

        # Blend meta and learned embeddings using gate
        learned_emb = sparse_emb[:, 0, :]  # First field embedding
        combined = torch.cat([meta_emb, learned_emb], dim=-1)
        gate_weight = self.gate(combined)

        # Blended embedding: cold items use more meta, warm items use more learned
        blended_emb = gate_weight * meta_emb + (1 - gate_weight) * learned_emb

        # Replace first field with blended embedding
        sparse_emb = sparse_emb.clone()
        sparse_emb[:, 0, :] = blended_emb

        sparse_flat = sparse_emb.view(batch_size, -1)
        x = torch.cat([dense_normed, sparse_flat], dim=1)
        logits = self.mlp(x).squeeze(-1)

        return {
            'logits': logits,
            'meta_emb': meta_emb,
            'gate_weight': gate_weight,
        }


class Expert(nn.Module):
    """Single expert network."""

    def __init__(self, input_dim: int, hidden_dims: List[int], dropout: float = 0.1):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        self.net = nn.Sequential(*layers)
        self.output_dim = hidden_dims[-1] if hidden_dims else input_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MMoE(BaseRecommender):
    """
    MMoE: Modeling Task Relationships in Multi-task Learning with
    Multi-gate Mixture-of-Experts.

    Reference: Ma et al., "Modeling Task Relationships in Multi-task Learning
    with Multi-gate Mixture-of-Experts", KDD 2018.

    For single-task CTR, we use one gate but multiple experts to capture
    different feature interaction patterns.
    """

    def __init__(
        self,
        dense_dim: int,
        sparse_dims: Dict[str, int],
        embedding_dim: int = 16,
        num_experts: int = 8,
        expert_hidden_dims: List[int] = [128, 64],
        tower_hidden_dims: List[int] = [64, 32],
        dropout: float = 0.1,
    ):
        super().__init__(dense_dim, sparse_dims, embedding_dim)

        self.num_experts = num_experts
        input_dim = dense_dim + self.num_sparse_fields * embedding_dim

        # Expert networks
        self.experts = nn.ModuleList([
            Expert(input_dim, expert_hidden_dims, dropout)
            for _ in range(num_experts)
        ])

        expert_output_dim = expert_hidden_dims[-1]

        # Gate network
        self.gate = nn.Sequential(
            nn.Linear(input_dim, num_experts),
            nn.Softmax(dim=-1),
        )

        # Tower network
        self.tower = MLPBlock(
            input_dim=expert_output_dim,
            hidden_dims=tower_hidden_dims,
            output_dim=1,
            dropout=dropout,
        )

    def forward(
        self,
        dense: torch.Tensor,
        sparse: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        batch_size = dense.shape[0]

        # Normalize dense
        dense = self.dense_bn(dense)

        # Sparse embeddings
        sparse_emb = self.embedding(sparse)
        sparse_flat = sparse_emb.view(batch_size, -1)

        # Concatenate
        x = torch.cat([dense, sparse_flat], dim=1)

        # Expert outputs
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        # (batch, num_experts, expert_dim)

        # Gate weights
        gate_weights = self.gate(x)  # (batch, num_experts)

        # Weighted sum of expert outputs
        gate_weights = gate_weights.unsqueeze(-1)  # (batch, num_experts, 1)
        mixed_output = (expert_outputs * gate_weights).sum(dim=1)  # (batch, expert_dim)

        # Tower
        logits = self.tower(mixed_output).squeeze(-1)

        return {
            'logits': logits,
            'gate_weights': gate_weights.squeeze(-1),
            'expert_outputs': expert_outputs,
        }


class PLE(BaseRecommender):
    """
    PLE: Progressive Layered Extraction for Multi-task Learning.

    Reference: Tang et al., "Progressive Layered Extraction (PLE): A Novel
    Multi-Task Learning (MTL) Model for Personalized Recommendations", RecSys 2020.

    For single-task CTR, we adapt PLE to have shared and task-specific experts
    that progressively extract features.
    """

    def __init__(
        self,
        dense_dim: int,
        sparse_dims: Dict[str, int],
        embedding_dim: int = 16,
        num_shared_experts: int = 4,
        num_specific_experts: int = 4,
        num_extraction_layers: int = 2,
        expert_hidden_dims: List[int] = [128],
        tower_hidden_dims: List[int] = [64, 32],
        dropout: float = 0.1,
    ):
        super().__init__(dense_dim, sparse_dims, embedding_dim)

        self.num_layers = num_extraction_layers
        input_dim = dense_dim + self.num_sparse_fields * embedding_dim

        # Build extraction layers
        self.shared_experts = nn.ModuleList()
        self.specific_experts = nn.ModuleList()
        self.gates = nn.ModuleList()

        current_dim = input_dim
        expert_output_dim = expert_hidden_dims[-1]

        for layer_idx in range(num_extraction_layers):
            # Shared experts
            layer_shared = nn.ModuleList([
                Expert(current_dim, expert_hidden_dims, dropout)
                for _ in range(num_shared_experts)
            ])
            self.shared_experts.append(layer_shared)

            # Task-specific experts
            layer_specific = nn.ModuleList([
                Expert(current_dim, expert_hidden_dims, dropout)
                for _ in range(num_specific_experts)
            ])
            self.specific_experts.append(layer_specific)

            # Gate for this layer
            num_total_experts = num_shared_experts + num_specific_experts
            gate = nn.Sequential(
                nn.Linear(current_dim, num_total_experts),
                nn.Softmax(dim=-1),
            )
            self.gates.append(gate)

            current_dim = expert_output_dim

        # Tower
        self.tower = MLPBlock(
            input_dim=expert_output_dim,
            hidden_dims=tower_hidden_dims,
            output_dim=1,
            dropout=dropout,
        )

    def forward(
        self,
        dense: torch.Tensor,
        sparse: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        batch_size = dense.shape[0]

        # Normalize dense
        dense = self.dense_bn(dense)

        # Sparse embeddings
        sparse_emb = self.embedding(sparse)
        sparse_flat = sparse_emb.view(batch_size, -1)

        # Concatenate
        x = torch.cat([dense, sparse_flat], dim=1)

        # Progressive extraction
        all_gate_weights = []
        for layer_idx in range(self.num_layers):
            # Get expert outputs
            shared_outputs = [expert(x) for expert in self.shared_experts[layer_idx]]
            specific_outputs = [expert(x) for expert in self.specific_experts[layer_idx]]

            all_expert_outputs = shared_outputs + specific_outputs
            expert_outputs = torch.stack(all_expert_outputs, dim=1)
            # (batch, num_experts, expert_dim)

            # Gate
            gate_weights = self.gates[layer_idx](x)  # (batch, num_experts)
            all_gate_weights.append(gate_weights)

            # Weighted sum
            gate_weights = gate_weights.unsqueeze(-1)
            x = (expert_outputs * gate_weights).sum(dim=1)

        # Tower
        logits = self.tower(x).squeeze(-1)

        return {
            'logits': logits,
            'gate_weights': all_gate_weights,
        }


class WarmUp(BaseRecommender):
    """
    Warming Up Cold-Start CTR Prediction by Learning Item-Specific
    Feature Interactions.

    Reference: KDD 2024 paper on learning item-specific feature interaction
    patterns for cold-start.

    Key idea: Use hypernetworks to generate item-specific feature interaction
    graphs processed by GNN.

    This is a simplified version focusing on the core idea.
    """

    def __init__(
        self,
        dense_dim: int,
        sparse_dims: Dict[str, int],
        embedding_dim: int = 16,
        hidden_dims: List[int] = [256, 128, 64],
        hyper_hidden_dim: int = 64,
        num_interaction_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__(dense_dim, sparse_dims, embedding_dim)

        self.num_interaction_layers = num_interaction_layers

        # Hypernetwork: generates interaction weights based on item features
        # Input: item embedding, Output: interaction weight matrix
        self.hyper_net = nn.Sequential(
            nn.Linear(embedding_dim, hyper_hidden_dim),
            nn.ReLU(),
            nn.Linear(hyper_hidden_dim, self.num_sparse_fields * self.num_sparse_fields),
        )

        # Feature interaction layers
        self.interaction_layers = nn.ModuleList([
            nn.Linear(embedding_dim, embedding_dim)
            for _ in range(num_interaction_layers)
        ])

        # Main network
        input_dim = dense_dim + self.num_sparse_fields * embedding_dim
        self.mlp = MLPBlock(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=1,
            dropout=dropout,
        )

    def forward(
        self,
        dense: torch.Tensor,
        sparse: torch.Tensor,
        item_field_idx: int = 0,  # Which field is the item
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        batch_size = dense.shape[0]

        # Normalize dense
        dense = self.dense_bn(dense)

        # Sparse embeddings
        sparse_emb = self.embedding(sparse)  # (batch, num_fields, emb_dim)

        # Get item embedding
        item_emb = sparse_emb[:, item_field_idx, :]  # (batch, emb_dim)

        # Generate item-specific interaction weights
        interaction_weights = self.hyper_net(item_emb)  # (batch, num_fields^2)
        interaction_weights = interaction_weights.view(
            batch_size, self.num_sparse_fields, self.num_sparse_fields
        )
        interaction_weights = F.softmax(interaction_weights, dim=-1)

        # Apply item-specific interactions
        enhanced_emb = sparse_emb
        for layer in self.interaction_layers:
            # Aggregate neighbor embeddings with learned weights
            aggregated = torch.bmm(interaction_weights, enhanced_emb)
            enhanced_emb = F.relu(layer(aggregated)) + enhanced_emb

        # Flatten and predict
        sparse_flat = enhanced_emb.view(batch_size, -1)
        x = torch.cat([dense, sparse_flat], dim=1)
        logits = self.mlp(x).squeeze(-1)

        return {
            'logits': logits,
            'interaction_weights': interaction_weights,
        }
