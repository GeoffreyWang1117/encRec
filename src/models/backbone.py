"""
Backbone recommendation models (DeepFM, DLRM).

These serve as:
1. Baselines for comparison
2. Foundation for Trie-MoE extensions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from .base import BaseRecommender, EmbeddingLayer, MLPBlock, FeatureInteraction


class DeepFM(BaseRecommender):
    """
    DeepFM: Factorization Machine + Deep Neural Network.

    Reference: Guo et al., "DeepFM: A Factorization-Machine based Neural Network
    for CTR Prediction", IJCAI 2017.
    """

    def __init__(
        self,
        dense_dim: int,
        sparse_dims: Dict[str, int],
        embedding_dim: int = 16,
        hidden_dims: List[int] = [256, 128, 64],
        dropout: float = 0.1,
    ):
        super().__init__(dense_dim, sparse_dims, embedding_dim)

        self.hidden_dims = hidden_dims

        # FM component
        self.fm_interaction = FeatureInteraction(reduce=False)

        # First-order weights
        self.first_order = nn.ModuleDict({
            field: nn.Embedding(vocab_size, 1)
            for field, vocab_size in sparse_dims.items()
        })

        # Deep component
        deep_input_dim = dense_dim + self.num_sparse_fields * embedding_dim
        self.deep = MLPBlock(
            input_dim=deep_input_dim,
            hidden_dims=hidden_dims,
            output_dim=1,
            dropout=dropout,
        )

        # Output
        self.output_layer = nn.Linear(2, 1)  # FM + Deep

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
        sparse_emb = self.embedding(sparse)  # (batch, num_fields, emb_dim)

        # --- FM Component ---
        # First-order
        first_order_sum = 0
        for i, field in enumerate(self.embedding.field_names):
            first_order_sum = first_order_sum + self.first_order[field](sparse[:, i])

        # Second-order (FM interaction)
        fm_interaction = self.fm_interaction(sparse_emb)  # (batch, 1)

        fm_output = first_order_sum + fm_interaction  # (batch, 1)

        # --- Deep Component ---
        sparse_flat = sparse_emb.view(batch_size, -1)  # (batch, num_fields * emb_dim)
        deep_input = torch.cat([dense, sparse_flat], dim=1)
        deep_output = self.deep(deep_input)  # (batch, 1)

        # --- Combine ---
        combined = torch.cat([fm_output, deep_output], dim=1)  # (batch, 2)
        logits = self.output_layer(combined).squeeze(-1)  # (batch,)

        return {
            'logits': logits,
            'fm_output': fm_output.squeeze(-1),
            'deep_output': deep_output.squeeze(-1),
        }


class DLRM(BaseRecommender):
    """
    DLRM: Deep Learning Recommendation Model.

    Reference: Naumov et al., "Deep Learning Recommendation Model for
    Personalization and Recommendation Systems", arXiv 2019.
    """

    def __init__(
        self,
        dense_dim: int,
        sparse_dims: Dict[str, int],
        embedding_dim: int = 16,
        bottom_mlp_dims: List[int] = [128, 64],
        top_mlp_dims: List[int] = [256, 128, 64],
        dropout: float = 0.1,
        interaction_type: str = 'dot',  # 'dot' or 'cat'
    ):
        super().__init__(dense_dim, sparse_dims, embedding_dim)

        self.interaction_type = interaction_type

        # Bottom MLP for dense features
        self.bottom_mlp = MLPBlock(
            input_dim=dense_dim,
            hidden_dims=bottom_mlp_dims,
            output_dim=embedding_dim,  # Project to same dim as embeddings
            dropout=dropout,
        )

        # Interaction layer
        # Number of interactions: (num_fields + 1) choose 2
        num_vectors = self.num_sparse_fields + 1  # +1 for dense
        if interaction_type == 'dot':
            interaction_dim = num_vectors * (num_vectors - 1) // 2
        else:  # concat
            interaction_dim = num_vectors * embedding_dim

        # Top MLP
        self.top_mlp = MLPBlock(
            input_dim=interaction_dim + embedding_dim,  # interactions + dense
            hidden_dims=top_mlp_dims,
            output_dim=1,
            dropout=dropout,
        )

    def _compute_interactions(
        self,
        vectors: torch.Tensor,  # (batch, num_vectors, emb_dim)
    ) -> torch.Tensor:
        """Compute pairwise interactions."""
        if self.interaction_type == 'dot':
            # Dot product interactions
            # (batch, num_vectors, emb_dim) x (batch, emb_dim, num_vectors)
            # -> (batch, num_vectors, num_vectors)
            interactions = torch.bmm(vectors, vectors.transpose(1, 2))

            # Extract upper triangular (excluding diagonal)
            batch_size = vectors.shape[0]
            num_vectors = vectors.shape[1]

            # Create mask for upper triangular
            triu_indices = torch.triu_indices(num_vectors, num_vectors, offset=1)
            flat_interactions = interactions[:, triu_indices[0], triu_indices[1]]

            return flat_interactions  # (batch, num_interactions)
        else:
            # Concatenation
            return vectors.view(vectors.shape[0], -1)

    def forward(
        self,
        dense: torch.Tensor,
        sparse: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        batch_size = dense.shape[0]

        # Normalize and project dense
        dense = self.dense_bn(dense)
        dense_emb = self.bottom_mlp(dense)  # (batch, emb_dim)

        # Sparse embeddings
        sparse_emb = self.embedding(sparse)  # (batch, num_fields, emb_dim)

        # Combine dense and sparse
        all_vectors = torch.cat([
            dense_emb.unsqueeze(1),  # (batch, 1, emb_dim)
            sparse_emb,  # (batch, num_fields, emb_dim)
        ], dim=1)  # (batch, num_fields + 1, emb_dim)

        # Compute interactions
        interactions = self._compute_interactions(all_vectors)

        # Top MLP input
        top_input = torch.cat([interactions, dense_emb], dim=1)

        # Final prediction
        logits = self.top_mlp(top_input).squeeze(-1)

        return {
            'logits': logits,
            'interactions': interactions,
        }


class DenseBaseline(BaseRecommender):
    """
    Simple dense MLP baseline (no feature interactions).

    Useful for ablation studies.
    """

    def __init__(
        self,
        dense_dim: int,
        sparse_dims: Dict[str, int],
        embedding_dim: int = 16,
        hidden_dims: List[int] = [256, 128, 64],
        dropout: float = 0.1,
    ):
        super().__init__(dense_dim, sparse_dims, embedding_dim)

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
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        # Normalize dense
        dense = self.dense_bn(dense)

        # Sparse embeddings
        sparse_emb = self.embedding(sparse)
        sparse_flat = sparse_emb.view(sparse.shape[0], -1)

        # Concatenate and predict
        x = torch.cat([dense, sparse_flat], dim=1)
        logits = self.mlp(x).squeeze(-1)

        return {'logits': logits}


class CrossNetV2(nn.Module):
    """
    Cross Network V2 with low-rank matrix decomposition.

    Reference: Wang et al., "DCN V2: Improved Deep & Cross Network and
    Practical Lessons for Web-scale Learning to Rank Systems", WWW 2021.
    """

    def __init__(
        self,
        input_dim: int,
        num_layers: int = 3,
        low_rank: int = 32,
    ):
        super().__init__()
        self.num_layers = num_layers

        # Low-rank decomposition: W = U @ V^T
        self.U = nn.ParameterList([
            nn.Parameter(torch.randn(input_dim, low_rank) * 0.01)
            for _ in range(num_layers)
        ])
        self.V = nn.ParameterList([
            nn.Parameter(torch.randn(input_dim, low_rank) * 0.01)
            for _ in range(num_layers)
        ])
        self.bias = nn.ParameterList([
            nn.Parameter(torch.zeros(input_dim))
            for _ in range(num_layers)
        ])

    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x0: Input tensor (batch_size, input_dim)
        Returns:
            Cross network output (batch_size, input_dim)
        """
        x = x0
        for i in range(self.num_layers):
            # x_{l+1} = x_0 * (W_l @ x_l + b_l) + x_l
            # With low-rank: W_l @ x_l = U_l @ (V_l^T @ x_l)
            v_out = torch.matmul(x, self.V[i])  # (batch, low_rank)
            uv_out = torch.matmul(v_out, self.U[i].T)  # (batch, input_dim)
            x = x0 * (uv_out + self.bias[i]) + x
        return x


class DCNv2(BaseRecommender):
    """
    DCN V2: Deep & Cross Network V2.

    Reference: Wang et al., "DCN V2: Improved Deep & Cross Network and
    Practical Lessons for Web-scale Learning to Rank Systems", WWW 2021.

    Key improvements over DCN:
    1. Low-rank matrix decomposition for cross network
    2. Mixture of experts for cross layers (optional)
    3. Stacked or parallel structure
    """

    def __init__(
        self,
        dense_dim: int,
        sparse_dims: Dict[str, int],
        embedding_dim: int = 16,
        cross_num_layers: int = 3,
        cross_low_rank: int = 32,
        deep_hidden_dims: List[int] = [256, 128, 64],
        dropout: float = 0.1,
        structure: str = 'parallel',  # 'stacked' or 'parallel'
    ):
        super().__init__(dense_dim, sparse_dims, embedding_dim)

        self.structure = structure
        input_dim = dense_dim + self.num_sparse_fields * embedding_dim

        # Cross Network V2
        self.cross_net = CrossNetV2(
            input_dim=input_dim,
            num_layers=cross_num_layers,
            low_rank=cross_low_rank,
        )

        # Deep Network
        if structure == 'stacked':
            # Stacked: Cross -> Deep
            self.deep_net = MLPBlock(
                input_dim=input_dim,
                hidden_dims=deep_hidden_dims,
                output_dim=1,
                dropout=dropout,
            )
        else:
            # Parallel: Cross || Deep -> Combine
            self.deep_net = MLPBlock(
                input_dim=input_dim,
                hidden_dims=deep_hidden_dims,
                output_dim=deep_hidden_dims[-1],
                dropout=dropout,
            )
            # Output layer combines cross and deep
            self.output_layer = nn.Linear(input_dim + deep_hidden_dims[-1], 1)

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

        # Concatenate features
        x0 = torch.cat([dense, sparse_flat], dim=1)

        # Cross Network
        cross_out = self.cross_net(x0)

        if self.structure == 'stacked':
            # Stacked: Cross -> Deep
            logits = self.deep_net(cross_out).squeeze(-1)
        else:
            # Parallel: Cross || Deep
            deep_out = self.deep_net(x0)
            combined = torch.cat([cross_out, deep_out], dim=1)
            logits = self.output_layer(combined).squeeze(-1)

        return {
            'logits': logits,
            'cross_output': cross_out,
        }


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention for feature interaction.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_residual: bool = True,
    ):
        super().__init__()

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.use_residual = use_residual

        # Q, K, V projections
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)

        # Output projection
        self.W_o = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)

        self.scale = self.head_dim ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, num_fields, embed_dim)
        Returns:
            (batch_size, num_fields, embed_dim)
        """
        batch_size, num_fields, _ = x.shape

        # Linear projections
        Q = self.W_q(x)  # (batch, num_fields, embed_dim)
        K = self.W_k(x)
        V = self.W_v(x)

        # Reshape for multi-head attention
        # (batch, num_fields, num_heads, head_dim) -> (batch, num_heads, num_fields, head_dim)
        Q = Q.view(batch_size, num_fields, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, num_fields, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, num_fields, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (batch, heads, fields, fields)
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        # Attention output
        attn_out = torch.matmul(attn_probs, V)  # (batch, heads, fields, head_dim)

        # Reshape back
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, num_fields, self.embed_dim)

        # Output projection
        out = self.W_o(attn_out)

        # Residual connection and layer norm
        if self.use_residual:
            out = self.layer_norm(out + x)

        return out


class AutoInt(BaseRecommender):
    """
    AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks.

    Reference: Song et al., "AutoInt: Automatic Feature Interaction Learning via
    Self-Attentive Neural Networks", CIKM 2019.

    Key features:
    1. Multi-head self-attention for automatic feature interaction
    2. Residual connections for training stability
    3. Multiple attention layers for high-order interactions
    """

    def __init__(
        self,
        dense_dim: int,
        sparse_dims: Dict[str, int],
        embedding_dim: int = 16,
        num_attention_layers: int = 3,
        num_heads: int = 4,
        attention_dim: int = 32,
        deep_hidden_dims: List[int] = [256, 128, 64],
        dropout: float = 0.1,
        use_deep: bool = True,
    ):
        super().__init__(dense_dim, sparse_dims, embedding_dim)

        self.use_deep = use_deep
        self.attention_dim = attention_dim

        # Project embeddings to attention dimension if different
        if embedding_dim != attention_dim:
            self.embed_proj = nn.Linear(embedding_dim, attention_dim)
        else:
            self.embed_proj = None

        # Dense feature projection to embedding space
        self.dense_proj = nn.Linear(dense_dim, attention_dim)

        # Multi-head self-attention layers
        self.attention_layers = nn.ModuleList([
            MultiHeadSelfAttention(
                embed_dim=attention_dim,
                num_heads=num_heads,
                dropout=dropout,
                use_residual=True,
            )
            for _ in range(num_attention_layers)
        ])

        # Attention output dimension
        # num_fields = num_sparse_fields + 1 (for dense)
        attn_output_dim = (self.num_sparse_fields + 1) * attention_dim

        # Deep network (optional)
        if use_deep:
            deep_input_dim = dense_dim + self.num_sparse_fields * embedding_dim
            self.deep_net = MLPBlock(
                input_dim=deep_input_dim,
                hidden_dims=deep_hidden_dims,
                output_dim=deep_hidden_dims[-1],
                dropout=dropout,
            )
            output_input_dim = attn_output_dim + deep_hidden_dims[-1]
        else:
            self.deep_net = None
            output_input_dim = attn_output_dim

        # Output layer
        self.output_layer = nn.Linear(output_input_dim, 1)

    def forward(
        self,
        dense: torch.Tensor,
        sparse: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        batch_size = dense.shape[0]

        # Normalize dense
        dense_normed = self.dense_bn(dense)

        # Sparse embeddings: (batch, num_sparse_fields, embed_dim)
        sparse_emb = self.embedding(sparse)

        # Project to attention dimension if needed
        if self.embed_proj is not None:
            sparse_emb = self.embed_proj(sparse_emb)

        # Dense to embedding: (batch, 1, attention_dim)
        dense_emb = self.dense_proj(dense_normed).unsqueeze(1)

        # Combine: (batch, num_fields, attention_dim)
        combined_emb = torch.cat([dense_emb, sparse_emb], dim=1)

        # Multi-head self-attention layers
        attn_out = combined_emb
        for attn_layer in self.attention_layers:
            attn_out = attn_layer(attn_out)

        # Flatten attention output
        attn_flat = attn_out.view(batch_size, -1)

        # Deep network (optional)
        if self.use_deep:
            sparse_flat = self.embedding(sparse).view(batch_size, -1)
            deep_input = torch.cat([dense_normed, sparse_flat], dim=1)
            deep_out = self.deep_net(deep_input)
            final_input = torch.cat([attn_flat, deep_out], dim=1)
        else:
            final_input = attn_flat

        # Output
        logits = self.output_layer(final_input).squeeze(-1)

        return {
            'logits': logits,
            'attention_output': attn_flat,
        }


class FeatureGate(nn.Module):
    """Feature selection gate for FinalMLP."""

    def __init__(self, input_dim: int, gate_dim: int = 64):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(input_dim, gate_dim),
            nn.ReLU(),
            nn.Linear(gate_dim, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gate(x)


class FinalMLP(BaseRecommender):
    """
    FinalMLP: An Enhanced Two-Stream MLP Model for CTR Prediction.

    Reference: Mao et al., "FinalMLP: An Enhanced Two-Stream MLP Model for
    CTR Prediction", AAAI 2023.

    Key features:
    1. Two-stream MLP architecture
    2. Feature selection gate
    3. Bilinear fusion layer
    """

    def __init__(
        self,
        dense_dim: int,
        sparse_dims: Dict[str, int],
        embedding_dim: int = 16,
        stream1_dims: List[int] = [256, 128],
        stream2_dims: List[int] = [256, 128],
        gate_dim: int = 64,
        dropout: float = 0.1,
        use_gate: bool = True,
    ):
        super().__init__(dense_dim, sparse_dims, embedding_dim)

        self.use_gate = use_gate
        input_dim = dense_dim + self.num_sparse_fields * embedding_dim

        # Feature gate
        if use_gate:
            self.feature_gate = FeatureGate(input_dim, gate_dim)

        # Stream 1: Standard MLP
        self.stream1 = nn.Sequential()
        prev_dim = input_dim
        for i, hidden_dim in enumerate(stream1_dims):
            self.stream1.add_module(f'linear_{i}', nn.Linear(prev_dim, hidden_dim))
            self.stream1.add_module(f'bn_{i}', nn.BatchNorm1d(hidden_dim))
            self.stream1.add_module(f'relu_{i}', nn.ReLU())
            self.stream1.add_module(f'dropout_{i}', nn.Dropout(dropout))
            prev_dim = hidden_dim

        # Stream 2: MLP with different architecture
        self.stream2 = nn.Sequential()
        prev_dim = input_dim
        for i, hidden_dim in enumerate(stream2_dims):
            self.stream2.add_module(f'linear_{i}', nn.Linear(prev_dim, hidden_dim))
            self.stream2.add_module(f'bn_{i}', nn.BatchNorm1d(hidden_dim))
            self.stream2.add_module(f'relu_{i}', nn.ReLU())
            self.stream2.add_module(f'dropout_{i}', nn.Dropout(dropout))
            prev_dim = hidden_dim

        # Bilinear fusion
        stream1_out = stream1_dims[-1]
        stream2_out = stream2_dims[-1]

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(stream1_out + stream2_out, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
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

        # Concatenate features
        x = torch.cat([dense, sparse_flat], dim=1)

        # Feature gate
        if self.use_gate:
            x = self.feature_gate(x)

        # Two streams
        stream1_out = self.stream1(x)
        stream2_out = self.stream2(x)

        # Fusion
        combined = torch.cat([stream1_out, stream2_out], dim=1)
        logits = self.fusion(combined).squeeze(-1)

        return {
            'logits': logits,
            'stream1_output': stream1_out,
            'stream2_output': stream2_out,
        }


class SelfMask(nn.Module):
    """Self-Mask operation for DCNv3 to filter noise and reduce parameters."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.mask = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.mask(x)


class CrossNetV3(nn.Module):
    """
    Cross Network V3 with Self-Mask and exponential feature interactions.

    Reference: DCNv3: Towards Next Generation Deep Cross Network for
    Click-Through Rate Prediction, arXiv 2024.
    """

    def __init__(
        self,
        input_dim: int,
        num_layers: int = 3,
        low_rank: int = 32,
        use_self_mask: bool = True,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.use_self_mask = use_self_mask

        # Cross layers with low-rank decomposition
        self.U = nn.ParameterList([
            nn.Parameter(torch.randn(input_dim, low_rank) * 0.01)
            for _ in range(num_layers)
        ])
        self.V = nn.ParameterList([
            nn.Parameter(torch.randn(input_dim, low_rank) * 0.01)
            for _ in range(num_layers)
        ])
        self.bias = nn.ParameterList([
            nn.Parameter(torch.zeros(input_dim))
            for _ in range(num_layers)
        ])

        # Self-Mask for each layer
        if use_self_mask:
            self.self_masks = nn.ModuleList([
                SelfMask(input_dim) for _ in range(num_layers)
            ])

        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(input_dim) for _ in range(num_layers)
        ])

    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        x = x0
        for i in range(self.num_layers):
            # Cross operation with low-rank
            v_out = torch.matmul(x, self.V[i])
            uv_out = torch.matmul(v_out, self.U[i].T)

            # Apply self-mask
            if self.use_self_mask:
                uv_out = self.self_masks[i](uv_out)

            # Cross connection
            x = x0 * (uv_out + self.bias[i]) + x

            # Layer normalization
            x = self.layer_norms[i](x)

        return x


class DCNv3(BaseRecommender):
    """
    DCNv3: Next Generation Deep Cross Network for CTR Prediction.

    Reference: DCNv3: Towards Next Generation Deep Cross Network for
    Click-Through Rate Prediction, arXiv 2024.

    Key improvements:
    1. Self-Mask operation to filter noise
    2. Exponentially increasing feature interaction order
    3. Tri-BCE loss (optional, handled in training)
    4. More parameter efficient
    """

    def __init__(
        self,
        dense_dim: int,
        sparse_dims: Dict[str, int],
        embedding_dim: int = 16,
        cross_num_layers: int = 3,
        cross_low_rank: int = 32,
        deep_hidden_dims: List[int] = [256, 128, 64],
        dropout: float = 0.1,
        use_self_mask: bool = True,
        structure: str = 'parallel',
    ):
        super().__init__(dense_dim, sparse_dims, embedding_dim)

        self.structure = structure
        input_dim = dense_dim + self.num_sparse_fields * embedding_dim

        # Cross Network V3
        self.cross_net = CrossNetV3(
            input_dim=input_dim,
            num_layers=cross_num_layers,
            low_rank=cross_low_rank,
            use_self_mask=use_self_mask,
        )

        # Deep Network
        if structure == 'stacked':
            self.deep_net = MLPBlock(
                input_dim=input_dim,
                hidden_dims=deep_hidden_dims,
                output_dim=1,
                dropout=dropout,
            )
        else:
            self.deep_net = MLPBlock(
                input_dim=input_dim,
                hidden_dims=deep_hidden_dims,
                output_dim=deep_hidden_dims[-1],
                dropout=dropout,
            )
            self.output_layer = nn.Linear(input_dim + deep_hidden_dims[-1], 1)

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

        # Concatenate features
        x0 = torch.cat([dense, sparse_flat], dim=1)

        # Cross Network V3
        cross_out = self.cross_net(x0)

        if self.structure == 'stacked':
            logits = self.deep_net(cross_out).squeeze(-1)
        else:
            deep_out = self.deep_net(x0)
            combined = torch.cat([cross_out, deep_out], dim=1)
            logits = self.output_layer(combined).squeeze(-1)

        return {
            'logits': logits,
            'cross_output': cross_out,
        }


class CIN(nn.Module):
    """
    Compressed Interaction Network for xDeepFM.

    Reference: Lian et al., "xDeepFM: Combining Explicit and Implicit Feature
    Interactions for Recommender Systems", KDD 2018.
    """

    def __init__(
        self,
        num_fields: int,
        embedding_dim: int,
        cin_layer_sizes: List[int] = [128, 128],
        activation: str = 'relu',
        split_half: bool = True,
    ):
        super().__init__()
        self.num_fields = num_fields
        self.embedding_dim = embedding_dim
        self.cin_layer_sizes = cin_layer_sizes
        self.split_half = split_half

        self.conv_layers = nn.ModuleList()
        self.field_nums = [num_fields]

        for i, layer_size in enumerate(cin_layer_sizes):
            self.conv_layers.append(
                nn.Conv1d(
                    self.field_nums[-1] * self.field_nums[0],
                    layer_size,
                    kernel_size=1,
                )
            )
            if split_half and i < len(cin_layer_sizes) - 1:
                self.field_nums.append(layer_size // 2)
            else:
                self.field_nums.append(layer_size)

        if split_half:
            self.output_dim = sum(cin_layer_sizes[:-1]) // 2 + cin_layer_sizes[-1]
        else:
            self.output_dim = sum(cin_layer_sizes)

        self.activation = nn.ReLU() if activation == 'relu' else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        hidden_layers = [x]
        final_outputs = []

        for i, conv in enumerate(self.conv_layers):
            x0 = hidden_layers[0]
            xk = hidden_layers[-1]
            outer = torch.einsum('bmd,bhd->bmhd', x0, xk)
            outer = outer.reshape(batch_size, -1, self.embedding_dim)
            out = conv(outer)
            out = self.activation(out)

            if self.split_half and i < len(self.conv_layers) - 1:
                next_hidden, direct = torch.split(out, out.shape[1] // 2, dim=1)
                hidden_layers.append(next_hidden)
                final_outputs.append(direct)
            else:
                hidden_layers.append(out)
                final_outputs.append(out)

        result = torch.cat([layer.sum(dim=-1) for layer in final_outputs], dim=1)
        return result


class xDeepFM(BaseRecommender):
    """
    xDeepFM: eXtreme Deep Factorization Machine.

    Reference: Lian et al., "xDeepFM: Combining Explicit and Implicit Feature
    Interactions for Recommender Systems", KDD 2018.
    """

    def __init__(
        self,
        dense_dim: int,
        sparse_dims: Dict[str, int],
        embedding_dim: int = 16,
        cin_layer_sizes: List[int] = [128, 128],
        deep_hidden_dims: List[int] = [256, 128, 64],
        dropout: float = 0.1,
        split_half: bool = True,
    ):
        super().__init__(dense_dim, sparse_dims, embedding_dim)

        self.linear = nn.Linear(dense_dim + self.num_sparse_fields * embedding_dim, 1)

        self.cin = CIN(
            num_fields=self.num_sparse_fields,
            embedding_dim=embedding_dim,
            cin_layer_sizes=cin_layer_sizes,
            split_half=split_half,
        )

        dnn_input_dim = dense_dim + self.num_sparse_fields * embedding_dim
        self.dnn = MLPBlock(
            input_dim=dnn_input_dim,
            hidden_dims=deep_hidden_dims,
            output_dim=deep_hidden_dims[-1],
            dropout=dropout,
        )

        self.output_layer = nn.Linear(
            1 + self.cin.output_dim + deep_hidden_dims[-1], 1
        )

    def forward(
        self,
        dense: torch.Tensor,
        sparse: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        batch_size = dense.shape[0]
        dense = self.dense_bn(dense)
        sparse_emb = self.embedding(sparse)
        sparse_flat = sparse_emb.view(batch_size, -1)
        x = torch.cat([dense, sparse_flat], dim=1)

        linear_out = self.linear(x)
        cin_out = self.cin(sparse_emb)
        dnn_out = self.dnn(x)

        combined = torch.cat([linear_out, cin_out, dnn_out], dim=1)
        logits = self.output_layer(combined).squeeze(-1)

        return {
            'logits': logits,
            'cin_output': cin_out,
        }


class TransformerCTR(BaseRecommender):
    """
    Transformer-based CTR model.

    Reference:
    - Song et al., "AutoInt: Automatic Feature Interaction Learning", CIKM 2019
    - Chen et al., "Behavior Sequence Transformer", KDD 2019
    """

    def __init__(
        self,
        dense_dim: int,
        sparse_dims: Dict[str, int],
        embedding_dim: int = 16,
        num_heads: int = 4,
        num_layers: int = 3,
        ffn_dim: int = 256,
        dropout: float = 0.1,
        use_residual: bool = True,
    ):
        super().__init__(dense_dim, sparse_dims, embedding_dim)
        self.use_residual = use_residual

        self.dense_proj = nn.Linear(dense_dim, embedding_dim)
        num_tokens = 1 + self.num_sparse_fields

        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_tokens, embedding_dim) * 0.02
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_norm = nn.LayerNorm(embedding_dim)
        self.output_mlp = nn.Sequential(
            nn.Linear(num_tokens * embedding_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(
        self,
        dense: torch.Tensor,
        sparse: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        batch_size = dense.shape[0]
        dense = self.dense_bn(dense)
        dense_emb = self.dense_proj(dense).unsqueeze(1)
        sparse_emb = self.embedding(sparse)

        x = torch.cat([dense_emb, sparse_emb], dim=1)
        x = x + self.pos_embedding

        if self.use_residual:
            x_orig = x
            x = self.transformer(x)
            x = x + x_orig
        else:
            x = self.transformer(x)

        x = self.output_norm(x)
        x = x.view(batch_size, -1)
        logits = self.output_mlp(x).squeeze(-1)

        return {'logits': logits}
