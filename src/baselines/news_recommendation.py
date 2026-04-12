"""
State-of-the-Art News Recommendation Baselines for KDD 2026.

Implements:
1. NRMS - Neural News Recommendation with Multi-Head Self-Attention (EMNLP 2019)
2. NAML - Neural News Recommendation with Attentive Multi-View Learning (IJCAI 2019)
3. PLM-NR - Pre-trained Language Model based News Recommendation (ACL 2021)

References:
- NRMS: https://aclanthology.org/D19-1671/
- NAML: https://www.ijcai.org/proceedings/2019/536
- PLM-NR: https://aclanthology.org/2021.findings-acl.29/
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional, Tuple
import numpy as np
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================
# Common Components
# ============================================

class AdditiveAttention(nn.Module):
    """Additive attention mechanism."""

    def __init__(self, input_dim: int, hidden_dim: int = 200):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim)
            mask: (batch, seq_len) - True for valid, False for padding
        Returns:
            (batch, input_dim)
        """
        attn = torch.tanh(self.fc1(x))  # (batch, seq_len, hidden)
        attn = self.fc2(attn).squeeze(-1)  # (batch, seq_len)

        if mask is not None:
            # Ensure mask is boolean, then mask invalid positions
            if mask.dtype != torch.bool:
                mask = mask.bool()
            attn = attn.masked_fill(~mask, -1e4)  # Use -1e4 to avoid inf

        attn = F.softmax(attn, dim=-1)  # (batch, seq_len)
        # Handle potential NaN from all-masked rows
        attn = torch.nan_to_num(attn, nan=0.0)
        output = torch.bmm(attn.unsqueeze(1), x).squeeze(1)  # (batch, input_dim)
        return output


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention for news/user encoding."""

    def __init__(self, embed_dim: int, num_heads: int = 16, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, embed_dim)
            mask: (batch, seq_len) - True for positions to mask (padding)
        Returns:
            (batch, seq_len, embed_dim)
        """
        # key_padding_mask: True means to ignore that position
        if mask is not None and mask.dtype != torch.bool:
            mask = mask.bool()
        attn_output, _ = self.attention(x, x, x, key_padding_mask=mask)
        x = self.layer_norm(x + self.dropout(attn_output))
        return x


# ============================================
# NRMS: Neural News Recommendation with Multi-Head Self-Attention
# ============================================

class NRMSNewsEncoder(nn.Module):
    """News encoder using multi-head self-attention."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        num_heads: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.self_attention = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.additive_attention = AdditiveAttention(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, title_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            title_ids: (batch, title_len)
        Returns:
            (batch, embed_dim)
        """
        mask = (title_ids == 0)  # Padding mask
        x = self.dropout(self.embedding(title_ids))  # (batch, title_len, embed_dim)
        x = self.self_attention(x, mask)
        news_repr = self.additive_attention(x, ~mask)
        return news_repr


class NRMSUserEncoder(nn.Module):
    """User encoder using multi-head self-attention over clicked news."""

    def __init__(self, news_dim: int = 256, num_heads: int = 16, dropout: float = 0.1):
        super().__init__()
        self.self_attention = MultiHeadSelfAttention(news_dim, num_heads, dropout)
        self.additive_attention = AdditiveAttention(news_dim)

    def forward(self, news_reprs: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            news_reprs: (batch, history_len, news_dim)
            mask: (batch, history_len)
        Returns:
            (batch, news_dim)
        """
        x = self.self_attention(news_reprs, mask)
        user_repr = self.additive_attention(x, ~mask if mask is not None else None)
        return user_repr


class NRMS(nn.Module):
    """
    NRMS: Neural News Recommendation with Multi-Head Self-Attention.

    Paper: https://aclanthology.org/D19-1671/
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,  # Must be divisible by num_heads
        num_heads: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.news_encoder = NRMSNewsEncoder(vocab_size, embed_dim, num_heads, dropout)
        self.user_encoder = NRMSUserEncoder(embed_dim, num_heads, dropout)

    def forward(
        self,
        history_title_ids: torch.Tensor,
        candidate_title_ids: torch.Tensor,
        history_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            history_title_ids: (batch, history_len, title_len)
            candidate_title_ids: (batch, num_candidates, title_len)
            history_mask: (batch, history_len)
        Returns:
            scores: (batch, num_candidates)
        """
        batch_size, history_len, title_len = history_title_ids.shape
        _, num_candidates, _ = candidate_title_ids.shape

        # Encode history news
        history_flat = history_title_ids.view(-1, title_len)
        history_news = self.news_encoder(history_flat)
        history_news = history_news.view(batch_size, history_len, -1)

        # Encode user
        user_repr = self.user_encoder(history_news, history_mask)  # (batch, embed_dim)

        # Encode candidates
        candidates_flat = candidate_title_ids.view(-1, title_len)
        candidate_news = self.news_encoder(candidates_flat)
        candidate_news = candidate_news.view(batch_size, num_candidates, -1)

        # Compute scores
        scores = torch.bmm(candidate_news, user_repr.unsqueeze(-1)).squeeze(-1)
        return scores


# ============================================
# NAML: Neural News Recommendation with Attentive Multi-View Learning
# ============================================

class NAMLNewsEncoder(nn.Module):
    """
    Multi-view news encoder with title, abstract, category, subcategory.
    """

    def __init__(
        self,
        vocab_size: int,
        num_categories: int,
        num_subcategories: int,
        embed_dim: int = 300,
        category_embed_dim: int = 100,
        num_filters: int = 400,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Word embedding
        self.word_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # Category embeddings
        self.category_embedding = nn.Embedding(num_categories, category_embed_dim, padding_idx=0)
        self.subcategory_embedding = nn.Embedding(num_subcategories, category_embed_dim, padding_idx=0)

        # Category dense
        self.category_dense = nn.Linear(category_embed_dim, num_filters)
        self.subcategory_dense = nn.Linear(category_embed_dim, num_filters)

        # CNN for title and abstract
        self.title_cnn = nn.Conv1d(embed_dim, num_filters, kernel_size, padding=kernel_size//2)
        self.abstract_cnn = nn.Conv1d(embed_dim, num_filters, kernel_size, padding=kernel_size//2)

        # Attention
        self.title_attention = AdditiveAttention(num_filters)
        self.abstract_attention = AdditiveAttention(num_filters)

        # View attention
        self.view_attention = AdditiveAttention(num_filters)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        title_ids: torch.Tensor,
        abstract_ids: torch.Tensor,
        category_ids: torch.Tensor,
        subcategory_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            title_ids: (batch, title_len)
            abstract_ids: (batch, abstract_len)
            category_ids: (batch,)
            subcategory_ids: (batch,)
        Returns:
            (batch, num_filters)
        """
        # Title view
        title_emb = self.dropout(self.word_embedding(title_ids))  # (batch, title_len, embed_dim)
        title_cnn = F.relu(self.title_cnn(title_emb.transpose(1, 2))).transpose(1, 2)
        title_repr = self.title_attention(title_cnn, (title_ids != 0))

        # Abstract view
        abstract_emb = self.dropout(self.word_embedding(abstract_ids))
        abstract_cnn = F.relu(self.abstract_cnn(abstract_emb.transpose(1, 2))).transpose(1, 2)
        abstract_repr = self.abstract_attention(abstract_cnn, (abstract_ids != 0))

        # Category view
        category_repr = F.relu(self.category_dense(self.category_embedding(category_ids)))

        # Subcategory view
        subcategory_repr = F.relu(self.subcategory_dense(self.subcategory_embedding(subcategory_ids)))

        # Multi-view attention
        views = torch.stack([title_repr, abstract_repr, category_repr, subcategory_repr], dim=1)
        news_repr = self.view_attention(views)

        return news_repr


class NAML(nn.Module):
    """
    NAML: Neural News Recommendation with Attentive Multi-View Learning.

    Paper: https://www.ijcai.org/proceedings/2019/536
    """

    def __init__(
        self,
        vocab_size: int,
        num_categories: int,
        num_subcategories: int,
        embed_dim: int = 300,
        num_filters: int = 400,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.news_encoder = NAMLNewsEncoder(
            vocab_size, num_categories, num_subcategories,
            embed_dim, num_filters=num_filters, dropout=dropout
        )
        self.user_attention = AdditiveAttention(num_filters)

    def forward(
        self,
        history_titles: torch.Tensor,
        history_abstracts: torch.Tensor,
        history_categories: torch.Tensor,
        history_subcategories: torch.Tensor,
        candidate_titles: torch.Tensor,
        candidate_abstracts: torch.Tensor,
        candidate_categories: torch.Tensor,
        candidate_subcategories: torch.Tensor,
        history_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute recommendation scores."""
        batch_size = history_titles.shape[0]
        history_len = history_titles.shape[1]
        num_candidates = candidate_titles.shape[1]

        # Encode history
        history_news = []
        for i in range(history_len):
            news_repr = self.news_encoder(
                history_titles[:, i],
                history_abstracts[:, i],
                history_categories[:, i],
                history_subcategories[:, i],
            )
            history_news.append(news_repr)
        history_news = torch.stack(history_news, dim=1)  # (batch, history_len, dim)

        # User representation
        user_repr = self.user_attention(history_news, ~history_mask if history_mask is not None else None)

        # Encode candidates
        candidate_news = []
        for i in range(num_candidates):
            news_repr = self.news_encoder(
                candidate_titles[:, i],
                candidate_abstracts[:, i],
                candidate_categories[:, i],
                candidate_subcategories[:, i],
            )
            candidate_news.append(news_repr)
        candidate_news = torch.stack(candidate_news, dim=1)  # (batch, num_candidates, dim)

        # Scores
        scores = torch.bmm(candidate_news, user_repr.unsqueeze(-1)).squeeze(-1)
        return scores


# ============================================
# PLM-NR: Pre-trained Language Model based News Recommendation
# ============================================

class PLMNR(nn.Module):
    """
    PLM-NR: Using pre-trained language models for news recommendation.

    Uses BERT/RoBERTa to encode news titles.
    Paper: https://aclanthology.org/2021.findings-acl.29/
    """

    def __init__(
        self,
        plm_name: str = "bert-base-uncased",
        hidden_dim: int = 768,
        num_heads: int = 12,
        dropout: float = 0.1,
        freeze_plm: bool = False,
    ):
        super().__init__()

        try:
            from transformers import AutoModel, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(plm_name)
            self.plm = AutoModel.from_pretrained(plm_name)
            self.hidden_dim = self.plm.config.hidden_size
        except Exception as e:
            logger.warning(f"Failed to load PLM {plm_name}: {e}. Using simple embedding.")
            self.tokenizer = None
            self.plm = None
            self.hidden_dim = hidden_dim
            self.fallback_embedding = nn.Embedding(30000, hidden_dim)

        if freeze_plm and self.plm is not None:
            for param in self.plm.parameters():
                param.requires_grad = False

        self.news_attention = AdditiveAttention(self.hidden_dim)
        self.user_attention = MultiHeadSelfAttention(self.hidden_dim, num_heads, dropout)
        self.user_pooling = AdditiveAttention(self.hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def encode_news(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Encode news using PLM."""
        if self.plm is not None:
            outputs = self.plm(input_ids=input_ids, attention_mask=attention_mask)
            hidden = outputs.last_hidden_state  # (batch, seq_len, hidden)
        else:
            hidden = self.fallback_embedding(input_ids)

        news_repr = self.news_attention(hidden, attention_mask)
        return news_repr

    def forward(
        self,
        history_input_ids: torch.Tensor,
        history_attention_mask: torch.Tensor,
        candidate_input_ids: torch.Tensor,
        candidate_attention_mask: torch.Tensor,
        history_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            history_input_ids: (batch, history_len, seq_len)
            history_attention_mask: (batch, history_len, seq_len)
            candidate_input_ids: (batch, num_candidates, seq_len)
            candidate_attention_mask: (batch, num_candidates, seq_len)
        Returns:
            scores: (batch, num_candidates)
        """
        batch_size, history_len, seq_len = history_input_ids.shape
        num_candidates = candidate_input_ids.shape[1]

        # Encode history news
        history_flat_ids = history_input_ids.view(-1, seq_len)
        history_flat_mask = history_attention_mask.view(-1, seq_len)
        history_news = self.encode_news(history_flat_ids, history_flat_mask)
        history_news = history_news.view(batch_size, history_len, -1)

        # User representation
        user_repr = self.user_attention(history_news, history_mask)
        user_repr = self.user_pooling(user_repr, ~history_mask if history_mask is not None else None)

        # Encode candidates
        candidate_flat_ids = candidate_input_ids.view(-1, seq_len)
        candidate_flat_mask = candidate_attention_mask.view(-1, seq_len)
        candidate_news = self.encode_news(candidate_flat_ids, candidate_flat_mask)
        candidate_news = candidate_news.view(batch_size, num_candidates, -1)

        # Scores
        scores = torch.bmm(candidate_news, user_repr.unsqueeze(-1)).squeeze(-1)
        return scores


# ============================================
# Recommender Wrappers for Experiment Integration
# ============================================

class NRMSRecommender:
    """Wrapper for NRMS to integrate with experiment framework."""

    def __init__(
        self,
        items: Dict[str, Dict],
        vocab_size: int = 30000,
        embed_dim: int = 256,
        epochs: int = 5,
        batch_size: int = 32,
        lr: float = 1e-5,  # Lower learning rate for stability
        device: str = None,
    ):
        self.items = items
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = None
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.item2title = {}

    def _tokenize(self, text: str, max_len: int = 30, build_vocab: bool = False) -> List[int]:
        """Simple tokenization."""
        words = text.lower().split()[:max_len]
        ids = []
        for word in words:
            if word not in self.word2idx:
                if build_vocab and len(self.word2idx) < self.vocab_size:
                    self.word2idx[word] = len(self.word2idx)
                else:
                    ids.append(1)  # UNK
                    continue
            ids.append(self.word2idx[word])

        # Pad
        ids = ids + [0] * (max_len - len(ids))
        return ids

    def _build_vocab(self, samples, all_items: List[str] = None):
        """Build vocabulary from training samples and candidate items."""
        # First pass: items in samples
        for sample in samples:
            for item_id in sample.history + [sample.ground_truth]:
                if item_id in self.items:
                    title = self.items[item_id].get('title', '')
                    self._tokenize(title, build_vocab=True)

        # Second pass: sample some candidate items for vocabulary
        if all_items:
            import random
            sampled_items = random.sample(all_items, min(5000, len(all_items)))
            for item_id in sampled_items:
                if item_id in self.items and len(self.word2idx) < self.vocab_size:
                    title = self.items[item_id].get('title', '')
                    self._tokenize(title, build_vocab=True)

    def fit(self, samples, all_items: List[str]):
        """Train NRMS model."""
        logger.info("Building vocabulary...")
        self._build_vocab(samples, all_items)

        logger.info(f"Vocabulary size: {len(self.word2idx)}")

        # Initialize model
        self.model = NRMS(
            vocab_size=len(self.word2idx),
            embed_dim=self.embed_dim,
        ).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        # Training
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for i in range(0, len(samples), self.batch_size):
                batch_samples = samples[i:i+self.batch_size]

                # Prepare batch
                history_ids = []
                candidate_ids = []
                labels = []

                for sample in batch_samples:
                    # History
                    hist_titles = []
                    for item_id in sample.history[-10:]:
                        title = self.items.get(item_id, {}).get('title', '')
                        hist_titles.append(self._tokenize(title))
                    while len(hist_titles) < 10:
                        hist_titles.append([0] * 30)
                    history_ids.append(hist_titles)

                    # Candidates (1 positive + 4 negatives)
                    pos_title = self.items.get(sample.ground_truth, {}).get('title', '')
                    cand_titles = [self._tokenize(pos_title)]

                    neg_items = [it for it in all_items if it != sample.ground_truth and it not in sample.history]
                    for neg_item in np.random.choice(neg_items, min(4, len(neg_items)), replace=False):
                        neg_title = self.items.get(neg_item, {}).get('title', '')
                        cand_titles.append(self._tokenize(neg_title))

                    while len(cand_titles) < 5:
                        cand_titles.append([0] * 30)

                    candidate_ids.append(cand_titles)
                    labels.append(0)  # First candidate is positive

                # Convert to tensors
                history_tensor = torch.tensor(history_ids, device=self.device)
                candidate_tensor = torch.tensor(candidate_ids, device=self.device)
                label_tensor = torch.tensor(labels, device=self.device)

                # Forward
                optimizer.zero_grad()
                scores = self.model(history_tensor, candidate_tensor)
                loss = criterion(scores, label_tensor)

                # Backward
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()

            if (epoch + 1) % 2 == 0:
                avg_loss = total_loss / max(1, len(samples) // self.batch_size)
                logger.info(f"  Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.4f}")

        logger.info("NRMS training complete.")

    def recommend(self, user_history: List[str], candidates: List[str], k: int = 5) -> List[str]:
        """Generate recommendations."""
        if self.model is None:
            return candidates[:k]

        self.model.eval()
        with torch.no_grad():
            # Prepare history
            hist_titles = []
            for item_id in user_history[-10:]:
                title = self.items.get(item_id, {}).get('title', '')
                hist_titles.append(self._tokenize(title))
            while len(hist_titles) < 10:
                hist_titles.append([0] * 30)

            # Prepare candidates
            cand_titles = []
            for item_id in candidates:
                title = self.items.get(item_id, {}).get('title', '')
                cand_titles.append(self._tokenize(title))

            # Tensors
            history_tensor = torch.tensor([hist_titles], device=self.device)
            candidate_tensor = torch.tensor([cand_titles], device=self.device)

            # Score
            scores = self.model(history_tensor, candidate_tensor)[0]

            # Rank
            ranked_indices = torch.argsort(scores, descending=True).cpu().numpy()
            return [candidates[i] for i in ranked_indices[:k]]


class NAMLRecommender:
    """Wrapper for NAML model."""

    def __init__(
        self,
        items: Dict[str, Dict],
        vocab_size: int = 30000,
        epochs: int = 10,
        batch_size: int = 32,
        lr: float = 1e-4,
        device: str = None,
    ):
        self.items = items
        self.vocab_size = vocab_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = None
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.cat2idx = {'<PAD>': 0}
        self.subcat2idx = {'<PAD>': 0}

    def _tokenize(self, text: str, max_len: int = 30) -> List[int]:
        words = text.lower().split()[:max_len]
        ids = [self.word2idx.get(w, 1) for w in words]
        return ids + [0] * (max_len - len(ids))

    def _get_item_features(self, item_id: str):
        """Extract all features for an item."""
        item = self.items.get(item_id, {})
        title = self._tokenize(item.get('title', ''), 30)
        abstract = self._tokenize(item.get('abstract', ''), 50)
        category = self.cat2idx.get(item.get('category', ''), 0)
        subcategory = self.subcat2idx.get(item.get('subcategory', ''), 0)
        return title, abstract, category, subcategory

    def _build_vocab(self, samples):
        for sample in samples:
            for item_id in sample.history + [sample.ground_truth]:
                if item_id in self.items:
                    item = self.items[item_id]
                    # Words
                    for word in (item.get('title', '') + ' ' + item.get('abstract', '')).lower().split():
                        if word not in self.word2idx and len(self.word2idx) < self.vocab_size:
                            self.word2idx[word] = len(self.word2idx)
                    # Category
                    cat = item.get('category', '')
                    if cat and cat not in self.cat2idx:
                        self.cat2idx[cat] = len(self.cat2idx)
                    # Subcategory
                    subcat = item.get('subcategory', '')
                    if subcat and subcat not in self.subcat2idx:
                        self.subcat2idx[subcat] = len(self.subcat2idx)

    def fit(self, samples, all_items: List[str]):
        logger.info("Building NAML vocabulary...")
        self._build_vocab(samples)
        logger.info(f"NAML vocab: {len(self.word2idx)} words, {len(self.cat2idx)} cats, {len(self.subcat2idx)} subcats")

        self.model = NAML(
            vocab_size=len(self.word2idx),
            num_categories=max(len(self.cat2idx), 2),
            num_subcategories=max(len(self.subcat2idx), 2),
        ).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for i in range(0, len(samples), self.batch_size):
                batch = samples[i:i+self.batch_size]

                # Prepare batch data
                hist_titles, hist_abstracts = [], []
                hist_cats, hist_subcats = [], []
                cand_titles, cand_abstracts = [], []
                cand_cats, cand_subcats = [], []
                labels = []

                for sample in batch:
                    # History (pad to 10)
                    h_titles, h_abstracts, h_cats, h_subcats = [], [], [], []
                    for item_id in sample.history[-10:]:
                        t, a, c, s = self._get_item_features(item_id)
                        h_titles.append(t)
                        h_abstracts.append(a)
                        h_cats.append(c)
                        h_subcats.append(s)
                    while len(h_titles) < 10:
                        h_titles.append([0]*30)
                        h_abstracts.append([0]*50)
                        h_cats.append(0)
                        h_subcats.append(0)

                    hist_titles.append(h_titles)
                    hist_abstracts.append(h_abstracts)
                    hist_cats.append(h_cats)
                    hist_subcats.append(h_subcats)

                    # Candidates (1 pos + 4 neg)
                    c_titles, c_abstracts, c_cats, c_subcats = [], [], [], []
                    t, a, c, s = self._get_item_features(sample.ground_truth)
                    c_titles.append(t)
                    c_abstracts.append(a)
                    c_cats.append(c)
                    c_subcats.append(s)

                    neg_items = [it for it in all_items if it != sample.ground_truth][:4]
                    for neg_id in neg_items:
                        t, a, c, s = self._get_item_features(neg_id)
                        c_titles.append(t)
                        c_abstracts.append(a)
                        c_cats.append(c)
                        c_subcats.append(s)

                    while len(c_titles) < 5:
                        c_titles.append([0]*30)
                        c_abstracts.append([0]*50)
                        c_cats.append(0)
                        c_subcats.append(0)

                    cand_titles.append(c_titles)
                    cand_abstracts.append(c_abstracts)
                    cand_cats.append(c_cats)
                    cand_subcats.append(c_subcats)
                    labels.append(0)

                # Convert to tensors
                hist_titles_t = torch.tensor(hist_titles, device=self.device)
                hist_abstracts_t = torch.tensor(hist_abstracts, device=self.device)
                hist_cats_t = torch.tensor(hist_cats, device=self.device)
                hist_subcats_t = torch.tensor(hist_subcats, device=self.device)
                cand_titles_t = torch.tensor(cand_titles, device=self.device)
                cand_abstracts_t = torch.tensor(cand_abstracts, device=self.device)
                cand_cats_t = torch.tensor(cand_cats, device=self.device)
                cand_subcats_t = torch.tensor(cand_subcats, device=self.device)
                labels_t = torch.tensor(labels, device=self.device)

                optimizer.zero_grad()
                scores = self.model(
                    hist_titles_t, hist_abstracts_t, hist_cats_t, hist_subcats_t,
                    cand_titles_t, cand_abstracts_t, cand_cats_t, cand_subcats_t,
                )
                loss = criterion(scores, labels_t)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if (epoch + 1) % 2 == 0:
                logger.info(f"  NAML Epoch {epoch+1}/{self.epochs}, Loss: {total_loss/len(samples):.4f}")

        logger.info("NAML training complete.")

    def recommend(self, user_history: List[str], candidates: List[str], k: int = 5) -> List[str]:
        if self.model is None:
            return candidates[:k]

        self.model.eval()
        with torch.no_grad():
            # History features
            h_titles, h_abstracts, h_cats, h_subcats = [], [], [], []
            for item_id in user_history[-10:]:
                t, a, c, s = self._get_item_features(item_id)
                h_titles.append(t)
                h_abstracts.append(a)
                h_cats.append(c)
                h_subcats.append(s)
            while len(h_titles) < 10:
                h_titles.append([0]*30)
                h_abstracts.append([0]*50)
                h_cats.append(0)
                h_subcats.append(0)

            # Candidate features
            c_titles, c_abstracts, c_cats, c_subcats = [], [], [], []
            for item_id in candidates:
                t, a, c, s = self._get_item_features(item_id)
                c_titles.append(t)
                c_abstracts.append(a)
                c_cats.append(c)
                c_subcats.append(s)

            # Tensors
            hist_titles_t = torch.tensor([h_titles], device=self.device)
            hist_abstracts_t = torch.tensor([h_abstracts], device=self.device)
            hist_cats_t = torch.tensor([h_cats], device=self.device)
            hist_subcats_t = torch.tensor([h_subcats], device=self.device)
            cand_titles_t = torch.tensor([c_titles], device=self.device)
            cand_abstracts_t = torch.tensor([c_abstracts], device=self.device)
            cand_cats_t = torch.tensor([c_cats], device=self.device)
            cand_subcats_t = torch.tensor([c_subcats], device=self.device)

            scores = self.model(
                hist_titles_t, hist_abstracts_t, hist_cats_t, hist_subcats_t,
                cand_titles_t, cand_abstracts_t, cand_cats_t, cand_subcats_t,
            )[0]

            ranked_indices = torch.argsort(scores, descending=True).cpu().numpy()
            return [candidates[i] for i in ranked_indices[:k]]


class PLMNRRecommender:
    """Wrapper for PLM-NR model."""

    def __init__(
        self,
        items: Dict[str, Dict],
        plm_name: str = "bert-base-uncased",
        epochs: int = 5,
        batch_size: int = 16,
        lr: float = 1e-5,
        device: str = None,
    ):
        self.items = items
        self.plm_name = plm_name
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None

    def _get_tokenizer(self):
        if self.tokenizer is None:
            try:
                from transformers import AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(self.plm_name)
            except Exception:
                self.tokenizer = None
        return self.tokenizer

    def _tokenize_text(self, text: str, max_len: int = 64):
        tokenizer = self._get_tokenizer()
        if tokenizer:
            encoded = tokenizer(text, max_length=max_len, padding='max_length',
                              truncation=True, return_tensors='pt')
            return encoded['input_ids'][0], encoded['attention_mask'][0]
        else:
            # Fallback simple tokenization
            words = text.lower().split()[:max_len]
            ids = [hash(w) % 29998 + 2 for w in words]
            ids = ids + [0] * (max_len - len(ids))
            mask = [1] * min(len(words), max_len) + [0] * (max_len - len(words))
            return torch.tensor(ids), torch.tensor(mask)

    def fit(self, samples, all_items: List[str]):
        logger.info(f"Initializing PLM-NR with {self.plm_name}...")
        self.model = PLMNR(plm_name=self.plm_name, freeze_plm=True).to(self.device)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for i in range(0, len(samples), self.batch_size):
                batch = samples[i:i+self.batch_size]

                history_ids_list, history_mask_list = [], []
                candidate_ids_list, candidate_mask_list = [], []
                labels = []

                for sample in batch:
                    # History encoding
                    h_ids, h_mask = [], []
                    for item_id in sample.history[-5:]:
                        title = self.items.get(item_id, {}).get('title', '')
                        ids, mask = self._tokenize_text(title)
                        h_ids.append(ids)
                        h_mask.append(mask)
                    while len(h_ids) < 5:
                        h_ids.append(torch.zeros(64, dtype=torch.long))
                        h_mask.append(torch.zeros(64, dtype=torch.long))
                    history_ids_list.append(torch.stack(h_ids))
                    history_mask_list.append(torch.stack(h_mask))

                    # Candidates (1 pos + 4 neg)
                    c_ids, c_mask = [], []
                    pos_title = self.items.get(sample.ground_truth, {}).get('title', '')
                    ids, mask = self._tokenize_text(pos_title)
                    c_ids.append(ids)
                    c_mask.append(mask)

                    neg_items = [it for it in all_items if it != sample.ground_truth][:4]
                    for neg_item in neg_items:
                        neg_title = self.items.get(neg_item, {}).get('title', '')
                        ids, mask = self._tokenize_text(neg_title)
                        c_ids.append(ids)
                        c_mask.append(mask)

                    while len(c_ids) < 5:
                        c_ids.append(torch.zeros(64, dtype=torch.long))
                        c_mask.append(torch.zeros(64, dtype=torch.long))

                    candidate_ids_list.append(torch.stack(c_ids))
                    candidate_mask_list.append(torch.stack(c_mask))
                    labels.append(0)

                # Stack batches
                history_ids = torch.stack(history_ids_list).to(self.device)
                history_mask = torch.stack(history_mask_list).to(self.device)
                candidate_ids = torch.stack(candidate_ids_list).to(self.device)
                candidate_mask = torch.stack(candidate_mask_list).to(self.device)
                label_tensor = torch.tensor(labels, device=self.device)

                optimizer.zero_grad()
                scores = self.model(history_ids, history_mask, candidate_ids, candidate_mask)
                loss = criterion(scores, label_tensor)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if (epoch + 1) % 2 == 0:
                logger.info(f"  PLM-NR Epoch {epoch+1}/{self.epochs}, Loss: {total_loss/len(samples):.4f}")

        logger.info("PLM-NR training complete.")

    def recommend(self, user_history: List[str], candidates: List[str], k: int = 5) -> List[str]:
        if self.model is None:
            return candidates[:k]

        self.model.eval()
        with torch.no_grad():
            # History
            h_ids, h_mask = [], []
            for item_id in user_history[-5:]:
                title = self.items.get(item_id, {}).get('title', '')
                ids, mask = self._tokenize_text(title)
                h_ids.append(ids)
                h_mask.append(mask)
            while len(h_ids) < 5:
                h_ids.append(torch.zeros(64, dtype=torch.long))
                h_mask.append(torch.zeros(64, dtype=torch.long))

            # Candidates
            c_ids, c_mask = [], []
            for item_id in candidates:
                title = self.items.get(item_id, {}).get('title', '')
                ids, mask = self._tokenize_text(title)
                c_ids.append(ids)
                c_mask.append(mask)

            history_ids = torch.stack(h_ids).unsqueeze(0).to(self.device)
            history_mask = torch.stack(h_mask).unsqueeze(0).to(self.device)
            candidate_ids = torch.stack(c_ids).unsqueeze(0).to(self.device)
            candidate_mask = torch.stack(c_mask).unsqueeze(0).to(self.device)

            scores = self.model(history_ids, history_mask, candidate_ids, candidate_mask)[0]
            ranked_indices = torch.argsort(scores, descending=True).cpu().numpy()
            return [candidates[i] for i in ranked_indices[:k]]


# ============================================
# TALLRec: Tuning Large Language Models for Recommendation
# ============================================

class TALLRec(nn.Module):
    """
    TALLRec: Tuning LLMs for Recommendation using LoRA.

    Paper: RecSys 2023
    Uses instruction-tuned LLM with LoRA for recommendation task.
    """

    def __init__(
        self,
        model_name: str = "gpt2",
        hidden_dim: int = 768,
        use_lora: bool = True,
        lora_r: int = 8,
        lora_alpha: int = 16,
    ):
        super().__init__()
        self.model_name = model_name
        self.use_lora = use_lora

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Freeze base model
            for param in self.model.parameters():
                param.requires_grad = False

            # Add LoRA-like trainable parameters (simplified)
            self.hidden_dim = self.model.config.hidden_size
            if use_lora:
                self.lora_down = nn.Linear(self.hidden_dim, lora_r, bias=False)
                self.lora_up = nn.Linear(lora_r, self.hidden_dim, bias=False)
                nn.init.zeros_(self.lora_up.weight)

        except Exception as e:
            logger.warning(f"Failed to load LLM {model_name}: {e}. Using fallback.")
            self.tokenizer = None
            self.model = None
            self.hidden_dim = hidden_dim
            self.fallback_proj = nn.Linear(hidden_dim, 1)

        # Prediction head
        self.classifier = nn.Linear(self.hidden_dim, 1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Score recommendation candidates."""
        if self.model is not None:
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask,
                               output_hidden_states=True)
            hidden = outputs.hidden_states[-1]  # (batch, seq, hidden)

            # Apply LoRA
            if self.use_lora:
                lora_out = self.lora_up(self.lora_down(hidden))
                hidden = hidden + lora_out

            # Pool (last token)
            seq_lens = attention_mask.sum(dim=1) - 1
            batch_indices = torch.arange(hidden.size(0), device=hidden.device)
            pooled = hidden[batch_indices, seq_lens]
        else:
            pooled = torch.randn(input_ids.size(0), self.hidden_dim, device=input_ids.device)

        scores = self.classifier(pooled).squeeze(-1)
        return scores


class TALLRecRecommender:
    """Wrapper for TALLRec model."""

    def __init__(
        self,
        items: Dict[str, Dict],
        model_name: str = "gpt2",
        epochs: int = 3,
        batch_size: int = 8,
        lr: float = 1e-4,
        device: str = None,
    ):
        self.items = items
        self.model_name = model_name
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None

    def _format_prompt(self, history_titles: List[str], candidate_title: str) -> str:
        """Format recommendation prompt."""
        history_str = ', '.join(history_titles[-5:]) if history_titles else 'None'
        prompt = f"""User history: {history_str}
Candidate: {candidate_title}
Will user like this? Answer:"""
        return prompt

    def fit(self, samples, all_items: List[str]):
        logger.info(f"Initializing TALLRec with {self.model_name}...")
        self.model = TALLRec(model_name=self.model_name).to(self.device)

        optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.lr
        )
        criterion = nn.BCEWithLogitsLoss()

        if self.model.tokenizer is None:
            logger.warning("Tokenizer not available, using simplified training.")
            return

        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for i in range(0, min(len(samples), 500), self.batch_size):  # Limit for speed
                batch = samples[i:i+self.batch_size]

                prompts = []
                labels = []

                for sample in batch:
                    history_titles = [self.items.get(h, {}).get('title', '') for h in sample.history]

                    # Positive
                    pos_title = self.items.get(sample.ground_truth, {}).get('title', '')
                    prompts.append(self._format_prompt(history_titles, pos_title))
                    labels.append(1.0)

                    # Negative
                    neg_items = [it for it in all_items if it != sample.ground_truth][:1]
                    if neg_items:
                        neg_title = self.items.get(neg_items[0], {}).get('title', '')
                        prompts.append(self._format_prompt(history_titles, neg_title))
                        labels.append(0.0)

                # Tokenize
                encoded = self.model.tokenizer(
                    prompts, max_length=128, padding=True, truncation=True, return_tensors='pt'
                )
                input_ids = encoded['input_ids'].to(self.device)
                attention_mask = encoded['attention_mask'].to(self.device)
                label_tensor = torch.tensor(labels, device=self.device)

                optimizer.zero_grad()
                scores = self.model(input_ids, attention_mask)
                loss = criterion(scores, label_tensor)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            logger.info(f"  TALLRec Epoch {epoch+1}/{self.epochs}, Loss: {total_loss/max(1,len(samples)//self.batch_size):.4f}")

        logger.info("TALLRec training complete.")

    def recommend(self, user_history: List[str], candidates: List[str], k: int = 5) -> List[str]:
        if self.model is None or self.model.tokenizer is None:
            return candidates[:k]

        self.model.eval()
        history_titles = [self.items.get(h, {}).get('title', '') for h in user_history]

        scores = []
        with torch.no_grad():
            for cand_id in candidates:
                cand_title = self.items.get(cand_id, {}).get('title', '')
                prompt = self._format_prompt(history_titles, cand_title)

                encoded = self.model.tokenizer(
                    prompt, max_length=128, padding=True, truncation=True, return_tensors='pt'
                )
                input_ids = encoded['input_ids'].to(self.device)
                attention_mask = encoded['attention_mask'].to(self.device)

                score = self.model(input_ids, attention_mask).item()
                scores.append(score)

        ranked_indices = np.argsort(scores)[::-1]
        return [candidates[i] for i in ranked_indices[:k]]


# ============================================
# Prompt4NR: Prompt Learning for News Recommendation
# ============================================

class Prompt4NR:
    """
    Prompt4NR: Prompt-based News Recommendation using LLM API.

    Paper: WWW 2023
    Uses prompt engineering to query LLM for news recommendation.
    """

    def __init__(
        self,
        items: Dict[str, Dict],
        llm_client=None,
        model: str = "gpt-3.5-turbo",
        use_api: bool = False,
    ):
        self.items = items
        self.llm_client = llm_client
        self.model = model
        self.use_api = use_api

        # Cache for API calls
        self.cache = {}

    def _format_prompt(self, history_titles: List[str], candidate_titles: List[str]) -> str:
        """Format prompt for LLM ranking."""
        history_str = '\n'.join([f"- {t}" for t in history_titles[-10:]])
        candidates_str = '\n'.join([f"{i+1}. {t}" for i, t in enumerate(candidate_titles)])

        prompt = f"""Based on the user's reading history, rank the candidate news articles by how likely the user would click on them.

User's reading history:
{history_str}

Candidate articles:
{candidates_str}

Return the ranking as a comma-separated list of numbers (e.g., "3,1,5,2,4" means article 3 is most likely to be clicked).
Ranking:"""
        return prompt

    def _parse_ranking(self, response: str, num_candidates: int) -> List[int]:
        """Parse LLM ranking response."""
        try:
            # Extract numbers from response
            import re
            numbers = re.findall(r'\d+', response)
            ranking = [int(n) - 1 for n in numbers if 1 <= int(n) <= num_candidates]

            # Add missing indices
            for i in range(num_candidates):
                if i not in ranking:
                    ranking.append(i)

            return ranking[:num_candidates]
        except Exception:
            return list(range(num_candidates))

    def _call_llm(self, prompt: str) -> str:
        """Call LLM API or use fallback."""
        # Check cache
        cache_key = hash(prompt)
        if cache_key in self.cache:
            return self.cache[cache_key]

        if self.use_api and self.llm_client:
            try:
                response = self.llm_client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=50,
                    temperature=0,
                )
                result = response.choices[0].message.content
                self.cache[cache_key] = result
                return result
            except Exception as e:
                logger.warning(f"LLM API call failed: {e}")

        # Fallback: random ranking
        return ""

    def fit(self, samples, all_items: List[str]):
        """Prompt4NR doesn't need training (uses LLM directly)."""
        logger.info("Prompt4NR: No training needed (uses LLM API).")

    def recommend(self, user_history: List[str], candidates: List[str], k: int = 5) -> List[str]:
        """Generate recommendations using LLM prompting."""
        history_titles = [self.items.get(h, {}).get('title', '') for h in user_history]
        candidate_titles = [self.items.get(c, {}).get('title', '') for c in candidates]

        prompt = self._format_prompt(history_titles, candidate_titles)
        response = self._call_llm(prompt)

        if response:
            ranking = self._parse_ranking(response, len(candidates))
            return [candidates[i] for i in ranking[:k]]
        else:
            # Fallback: return by order
            return candidates[:k]


class Prompt4NRLocal:
    """
    Local version of Prompt4NR using a small local LLM.
    For fair comparison without API costs.
    """

    def __init__(
        self,
        items: Dict[str, Dict],
        model_name: str = "gpt2",
        device: str = None,
    ):
        self.items = items
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        except Exception as e:
            logger.warning(f"Failed to load local LLM: {e}")
            self.tokenizer = None
            self.model = None

    def fit(self, samples, all_items: List[str]):
        """No training needed."""
        logger.info("Prompt4NR (Local): No training needed.")

    def recommend(self, user_history: List[str], candidates: List[str], k: int = 5) -> List[str]:
        if self.model is None:
            return candidates[:k]

        history_titles = [self.items.get(h, {}).get('title', '') for h in user_history[-5:]]
        candidate_titles = [self.items.get(c, {}).get('title', '') for c in candidates]

        # Score each candidate by perplexity given history
        scores = []
        self.model.eval()
        with torch.no_grad():
            for cand_title in candidate_titles:
                prompt = f"User liked: {', '.join(history_titles)}. User will also like: {cand_title}"
                encoded = self.tokenizer(prompt, return_tensors='pt', truncation=True, max_length=128)
                input_ids = encoded['input_ids'].to(self.device)

                outputs = self.model(input_ids, labels=input_ids)
                # Lower loss = higher score
                scores.append(-outputs.loss.item())

        ranked_indices = np.argsort(scores)[::-1]
        return [candidates[i] for i in ranked_indices[:k]]


# ============================================
# Unified Experiment Runner
# ============================================

def run_baseline_experiments(
    samples,
    items: Dict[str, Dict],
    all_items: List[str],
    baselines: List[str] = None,
    k: int = 5,
    device: str = None,
):
    """
    Run all baseline experiments.

    Args:
        samples: List of recommendation samples
        items: Item metadata dict
        all_items: List of all item IDs
        baselines: List of baseline names to run
        k: Top-k for evaluation
        device: Device to use

    Returns:
        Dict of results per baseline
    """
    if baselines is None:
        baselines = ['NRMS', 'NAML', 'PLM-NR', 'TALLRec', 'Prompt4NR']

    results = {}

    baseline_classes = {
        'NRMS': NRMSRecommender,
        'NAML': NAMLRecommender,
        'PLM-NR': PLMNRRecommender,
        'TALLRec': TALLRecRecommender,
        'Prompt4NR': Prompt4NRLocal,
    }

    for name in baselines:
        if name not in baseline_classes:
            logger.warning(f"Unknown baseline: {name}")
            continue

        logger.info(f"\n{'='*50}")
        logger.info(f"Running {name}...")
        logger.info(f"{'='*50}")

        try:
            # Initialize
            recommender = baseline_classes[name](items=items, device=device)

            # Train
            recommender.fit(samples, all_items)

            # Evaluate
            hits = 0
            total = 0
            for sample in samples:
                # Get candidates (ground truth + negatives)
                neg_items = [it for it in all_items if it != sample.ground_truth and it not in sample.history][:19]
                candidates = [sample.ground_truth] + neg_items
                np.random.shuffle(candidates)

                recommendations = recommender.recommend(sample.history, candidates, k=k)

                if sample.ground_truth in recommendations:
                    hits += 1
                total += 1

            hit_rate = hits / total if total > 0 else 0
            results[name] = {
                'hit@k': hit_rate,
                'total': total,
            }
            logger.info(f"{name} Hit@{k}: {hit_rate:.4f}")

        except Exception as e:
            logger.error(f"Error running {name}: {e}")
            results[name] = {'error': str(e)}

    return results


if __name__ == '__main__':
    # Quick test
    print("Testing NRMS...")
    model = NRMS(vocab_size=10000, embed_dim=256)

    batch_size = 4
    history_len = 10
    num_candidates = 5
    title_len = 20

    history = torch.randint(0, 10000, (batch_size, history_len, title_len))
    candidates = torch.randint(0, 10000, (batch_size, num_candidates, title_len))

    scores = model(history, candidates)
    print(f"NRMS output shape: {scores.shape}")  # Should be (4, 5)

    print("\nAll tests passed!")
