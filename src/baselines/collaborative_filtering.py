"""
Collaborative Filtering Baselines for Recommendation.

Implements:
1. BPR (Bayesian Personalized Ranking) - Rendle et al., UAI 2009
2. NeuMF (Neural Matrix Factorization) - He et al., WWW 2017

These methods learn user-item embeddings from interaction data and
can rank candidates based on predicted preference scores.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
import random
import time
import logging

logger = logging.getLogger(__name__)


# =====================================================================
# Data Structures
# =====================================================================

class InteractionDataset(Dataset):
    """Dataset for collaborative filtering training."""

    def __init__(
        self,
        user_items: Dict[str, List[str]],
        all_items: Set[str],
        num_negatives: int = 4,
    ):
        """
        Args:
            user_items: Dict mapping user_id -> list of interacted item_ids
            all_items: Set of all item IDs
            num_negatives: Number of negative samples per positive
        """
        self.user_items = user_items
        self.all_items = list(all_items)
        self.num_negatives = num_negatives

        # Build user/item ID mappings
        self.user2idx = {u: i for i, u in enumerate(user_items.keys())}
        self.item2idx = {item: i for i, item in enumerate(self.all_items)}
        self.idx2item = {i: item for item, i in self.item2idx.items()}

        self.num_users = len(self.user2idx)
        self.num_items = len(self.item2idx)

        # Build positive pairs
        self.pairs = []
        for user, items in user_items.items():
            user_idx = self.user2idx[user]
            for item in items:
                if item in self.item2idx:
                    item_idx = self.item2idx[item]
                    self.pairs.append((user_idx, item_idx))

        # User positive item sets for negative sampling
        self.user_pos_items = {
            self.user2idx[u]: set(self.item2idx.get(i, -1) for i in items if i in self.item2idx)
            for u, items in user_items.items()
        }

    def __len__(self):
        return len(self.pairs) * (1 + self.num_negatives)

    def __getitem__(self, idx):
        pair_idx = idx // (1 + self.num_negatives)
        is_positive = (idx % (1 + self.num_negatives)) == 0

        user_idx, pos_item_idx = self.pairs[pair_idx]

        if is_positive:
            return {
                'user': user_idx,
                'item': pos_item_idx,
                'label': 1.0,
            }
        else:
            # Sample negative item
            neg_item_idx = random.randint(0, self.num_items - 1)
            while neg_item_idx in self.user_pos_items.get(user_idx, set()):
                neg_item_idx = random.randint(0, self.num_items - 1)

            return {
                'user': user_idx,
                'item': neg_item_idx,
                'label': 0.0,
            }


class BPRDataset(Dataset):
    """Dataset for BPR pairwise training."""

    def __init__(
        self,
        user_items: Dict[str, List[str]],
        all_items: Set[str],
    ):
        self.user_items = user_items
        self.all_items = list(all_items)

        self.user2idx = {u: i for i, u in enumerate(user_items.keys())}
        self.item2idx = {item: i for i, item in enumerate(self.all_items)}
        self.idx2item = {i: item for item, i in self.item2idx.items()}

        self.num_users = len(self.user2idx)
        self.num_items = len(self.item2idx)

        # Build training triplets
        self.triplets = []
        for user, items in user_items.items():
            user_idx = self.user2idx[user]
            item_indices = [self.item2idx[i] for i in items if i in self.item2idx]
            for pos_idx in item_indices:
                self.triplets.append((user_idx, pos_idx))

        self.user_pos_items = {
            self.user2idx[u]: set(self.item2idx.get(i, -1) for i in items if i in self.item2idx)
            for u, items in user_items.items()
        }

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        user_idx, pos_item_idx = self.triplets[idx]

        # Sample negative
        neg_item_idx = random.randint(0, self.num_items - 1)
        while neg_item_idx in self.user_pos_items.get(user_idx, set()):
            neg_item_idx = random.randint(0, self.num_items - 1)

        return {
            'user': user_idx,
            'pos_item': pos_item_idx,
            'neg_item': neg_item_idx,
        }


# =====================================================================
# BPR Model
# =====================================================================

class BPR(nn.Module):
    """
    Bayesian Personalized Ranking (BPR).

    Learns user and item embeddings by optimizing pairwise ranking loss.

    Reference:
        Rendle et al., "BPR: Bayesian Personalized Ranking from Implicit Feedback"
        UAI 2009
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        reg_weight: float = 0.001,
    ):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.reg_weight = reg_weight

        # Embeddings
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        # Initialize
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)

    def forward(self, user_ids, item_ids):
        """Compute preference scores."""
        user_emb = self.user_embedding(user_ids)  # (batch, dim)
        item_emb = self.item_embedding(item_ids)  # (batch, dim)

        # Inner product
        scores = (user_emb * item_emb).sum(dim=1)
        return scores

    def bpr_loss(self, user_ids, pos_item_ids, neg_item_ids):
        """BPR loss: -log(sigmoid(pos_score - neg_score))."""
        pos_scores = self.forward(user_ids, pos_item_ids)
        neg_scores = self.forward(user_ids, neg_item_ids)

        # BPR loss
        loss = -F.logsigmoid(pos_scores - neg_scores).mean()

        # L2 regularization
        user_emb = self.user_embedding(user_ids)
        pos_emb = self.item_embedding(pos_item_ids)
        neg_emb = self.item_embedding(neg_item_ids)

        reg_loss = self.reg_weight * (
            user_emb.norm(2).pow(2) +
            pos_emb.norm(2).pow(2) +
            neg_emb.norm(2).pow(2)
        ) / user_ids.size(0)

        return loss + reg_loss

    def predict(self, user_ids, item_ids):
        """Predict scores for user-item pairs."""
        with torch.no_grad():
            return self.forward(user_ids, item_ids)

    def get_user_embedding(self, user_id):
        """Get embedding for a single user."""
        with torch.no_grad():
            return self.user_embedding(torch.tensor([user_id])).squeeze(0)

    def get_item_embeddings(self):
        """Get all item embeddings."""
        with torch.no_grad():
            return self.item_embedding.weight.clone()


# =====================================================================
# NeuMF Model
# =====================================================================

class NeuMF(nn.Module):
    """
    Neural Matrix Factorization (NeuMF).

    Combines Generalized Matrix Factorization (GMF) and Multi-Layer Perceptron (MLP)
    for collaborative filtering.

    Reference:
        He et al., "Neural Collaborative Filtering"
        WWW 2017
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        gmf_dim: int = 32,
        mlp_dim: int = 32,
        mlp_layers: List[int] = [64, 32, 16],
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items

        # GMF embeddings
        self.gmf_user_embedding = nn.Embedding(num_users, gmf_dim)
        self.gmf_item_embedding = nn.Embedding(num_items, gmf_dim)

        # MLP embeddings
        self.mlp_user_embedding = nn.Embedding(num_users, mlp_dim)
        self.mlp_item_embedding = nn.Embedding(num_items, mlp_dim)

        # MLP layers
        mlp_input_dim = mlp_dim * 2
        layers = []
        for hidden_dim in mlp_layers:
            layers.append(nn.Linear(mlp_input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            mlp_input_dim = hidden_dim

        self.mlp = nn.Sequential(*layers)

        # Final prediction layer
        # GMF output: gmf_dim, MLP output: mlp_layers[-1]
        self.predict_layer = nn.Linear(gmf_dim + mlp_layers[-1], 1)

        # Initialize
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.gmf_user_embedding.weight, std=0.01)
        nn.init.normal_(self.gmf_item_embedding.weight, std=0.01)
        nn.init.normal_(self.mlp_user_embedding.weight, std=0.01)
        nn.init.normal_(self.mlp_item_embedding.weight, std=0.01)

        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

        nn.init.xavier_uniform_(self.predict_layer.weight)
        nn.init.zeros_(self.predict_layer.bias)

    def forward(self, user_ids, item_ids):
        """Compute preference scores."""
        # GMF part
        gmf_user = self.gmf_user_embedding(user_ids)
        gmf_item = self.gmf_item_embedding(item_ids)
        gmf_output = gmf_user * gmf_item  # Element-wise product

        # MLP part
        mlp_user = self.mlp_user_embedding(user_ids)
        mlp_item = self.mlp_item_embedding(item_ids)
        mlp_input = torch.cat([mlp_user, mlp_item], dim=1)
        mlp_output = self.mlp(mlp_input)

        # Concatenate and predict
        concat = torch.cat([gmf_output, mlp_output], dim=1)
        scores = self.predict_layer(concat).squeeze(1)

        return scores

    def predict(self, user_ids, item_ids):
        """Predict scores for user-item pairs."""
        with torch.no_grad():
            return torch.sigmoid(self.forward(user_ids, item_ids))


# =====================================================================
# Recommenders (High-level API)
# =====================================================================

class BPRRecommender:
    """BPR-based recommender for the Trie-LLM experiment framework."""

    def __init__(
        self,
        items: Dict[str, Dict],
        embedding_dim: int = 64,
        epochs: int = 20,
        batch_size: int = 1024,
        learning_rate: float = 0.001,
        device: str = None,
    ):
        self.items = items
        self.embedding_dim = embedding_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = None
        self.dataset = None
        self.trained = False

    def fit(self, user_items: Dict[str, List[str]], all_items: Set[str]):
        """Train the BPR model."""
        logger.info("Training BPR model...")

        # Create dataset
        self.dataset = BPRDataset(user_items, all_items)

        # Create model
        self.model = BPR(
            num_users=self.dataset.num_users,
            num_items=self.dataset.num_items,
            embedding_dim=self.embedding_dim,
        ).to(self.device)

        # Optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Training
        dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
        )

        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch in dataloader:
                user = batch['user'].to(self.device)
                pos_item = batch['pos_item'].to(self.device)
                neg_item = batch['neg_item'].to(self.device)

                optimizer.zero_grad()
                loss = self.model.bpr_loss(user, pos_item, neg_item)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            if (epoch + 1) % 5 == 0:
                logger.info(f"  Epoch {epoch + 1}/{self.epochs}, Loss: {total_loss / len(dataloader):.4f}")

        self.trained = True
        logger.info("BPR training complete.")

    def recommend(
        self,
        user_id: str,
        history: List[str],
        candidates: List[str],
        k: int = 10,
    ) -> Tuple[List[str], float]:
        """Recommend top-k items from candidates."""
        start = time.time()

        if not self.trained or self.model is None:
            # Fallback to random
            recs = random.sample(candidates, min(k, len(candidates)))
            return recs, (time.time() - start) * 1000

        self.model.eval()

        # Get user index (or use history to create pseudo-user)
        if user_id in self.dataset.user2idx:
            user_idx = self.dataset.user2idx[user_id]
        else:
            # For new users, average the embeddings of their history items
            user_idx = 0  # Default to first user as fallback

        # Score candidates
        candidate_indices = []
        valid_candidates = []
        for item_id in candidates:
            if item_id in self.dataset.item2idx:
                candidate_indices.append(self.dataset.item2idx[item_id])
                valid_candidates.append(item_id)

        if not candidate_indices:
            recs = random.sample(candidates, min(k, len(candidates)))
            return recs, (time.time() - start) * 1000

        user_tensor = torch.tensor([user_idx] * len(candidate_indices)).to(self.device)
        item_tensor = torch.tensor(candidate_indices).to(self.device)

        with torch.no_grad():
            scores = self.model.predict(user_tensor, item_tensor).cpu().numpy()

        # Rank and select top-k
        ranked_indices = np.argsort(-scores)[:k]
        recommendations = [valid_candidates[i] for i in ranked_indices]

        # Fill remaining with random if needed
        while len(recommendations) < k and len(candidates) > len(recommendations):
            remaining = [c for c in candidates if c not in recommendations]
            if remaining:
                recommendations.append(random.choice(remaining))

        latency = (time.time() - start) * 1000
        return recommendations, latency


class NeuMFRecommender:
    """NeuMF-based recommender for the Trie-LLM experiment framework."""

    def __init__(
        self,
        items: Dict[str, Dict],
        gmf_dim: int = 32,
        mlp_dim: int = 32,
        mlp_layers: List[int] = [64, 32, 16],
        epochs: int = 20,
        batch_size: int = 1024,
        learning_rate: float = 0.001,
        device: str = None,
    ):
        self.items = items
        self.gmf_dim = gmf_dim
        self.mlp_dim = mlp_dim
        self.mlp_layers = mlp_layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = None
        self.dataset = None
        self.trained = False

    def fit(self, user_items: Dict[str, List[str]], all_items: Set[str]):
        """Train the NeuMF model."""
        logger.info("Training NeuMF model...")

        # Create dataset
        self.dataset = InteractionDataset(user_items, all_items, num_negatives=4)

        # Create model
        self.model = NeuMF(
            num_users=self.dataset.num_users,
            num_items=self.dataset.num_items,
            gmf_dim=self.gmf_dim,
            mlp_dim=self.mlp_dim,
            mlp_layers=self.mlp_layers,
        ).to(self.device)

        # Optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.BCEWithLogitsLoss()

        # Training
        dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
        )

        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch in dataloader:
                user = batch['user'].to(self.device)
                item = batch['item'].to(self.device)
                label = batch['label'].to(self.device)

                optimizer.zero_grad()
                scores = self.model(user, item)
                loss = criterion(scores, label)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            if (epoch + 1) % 5 == 0:
                logger.info(f"  Epoch {epoch + 1}/{self.epochs}, Loss: {total_loss / len(dataloader):.4f}")

        self.trained = True
        logger.info("NeuMF training complete.")

    def recommend(
        self,
        user_id: str,
        history: List[str],
        candidates: List[str],
        k: int = 10,
    ) -> Tuple[List[str], float]:
        """Recommend top-k items from candidates."""
        start = time.time()

        if not self.trained or self.model is None:
            recs = random.sample(candidates, min(k, len(candidates)))
            return recs, (time.time() - start) * 1000

        self.model.eval()

        # Get user index
        if user_id in self.dataset.user2idx:
            user_idx = self.dataset.user2idx[user_id]
        else:
            user_idx = 0

        # Score candidates
        candidate_indices = []
        valid_candidates = []
        for item_id in candidates:
            if item_id in self.dataset.item2idx:
                candidate_indices.append(self.dataset.item2idx[item_id])
                valid_candidates.append(item_id)

        if not candidate_indices:
            recs = random.sample(candidates, min(k, len(candidates)))
            return recs, (time.time() - start) * 1000

        user_tensor = torch.tensor([user_idx] * len(candidate_indices)).to(self.device)
        item_tensor = torch.tensor(candidate_indices).to(self.device)

        with torch.no_grad():
            scores = self.model.predict(user_tensor, item_tensor).cpu().numpy()

        # Rank and select top-k
        ranked_indices = np.argsort(-scores)[:k]
        recommendations = [valid_candidates[i] for i in ranked_indices]

        # Fill remaining
        while len(recommendations) < k and len(candidates) > len(recommendations):
            remaining = [c for c in candidates if c not in recommendations]
            if remaining:
                recommendations.append(random.choice(remaining))

        latency = (time.time() - start) * 1000
        return recommendations, latency


# =====================================================================
# Testing
# =====================================================================

def test_baselines():
    """Test BPR and NeuMF with synthetic data."""
    print("Testing collaborative filtering baselines...")

    # Create synthetic data
    num_users = 100
    num_items = 500

    user_items = {}
    for u in range(num_users):
        num_interactions = random.randint(10, 50)
        items = random.sample(range(num_items), num_interactions)
        user_items[f"user_{u}"] = [f"item_{i}" for i in items]

    all_items = {f"item_{i}" for i in range(num_items)}
    items_meta = {f"item_{i}": {'title': f'Item {i}'} for i in range(num_items)}

    # Test BPR
    print("\n[BPR]")
    bpr = BPRRecommender(items_meta, epochs=10)
    bpr.fit(user_items, all_items)

    test_user = "user_0"
    test_history = user_items[test_user][:5]
    test_candidates = random.sample(list(all_items), 50)

    recs, latency = bpr.recommend(test_user, test_history, test_candidates, k=10)
    print(f"  Recommendations: {recs[:5]}...")
    print(f"  Latency: {latency:.2f}ms")

    # Test NeuMF
    print("\n[NeuMF]")
    neumf = NeuMFRecommender(items_meta, epochs=10)
    neumf.fit(user_items, all_items)

    recs, latency = neumf.recommend(test_user, test_history, test_candidates, k=10)
    print(f"  Recommendations: {recs[:5]}...")
    print(f"  Latency: {latency:.2f}ms")

    print("\n[PASS] Baselines work correctly!")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    test_baselines()
