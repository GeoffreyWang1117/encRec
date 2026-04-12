#!/usr/bin/env python3
"""
小规模验证实验：测试各种数学改进方案
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
from collections import defaultdict
import yaml
import time
from typing import Dict, List, Optional, Tuple

from src.data.amazon_loader import load_amazon_data
from src.models.backbone import DeepFM, DLRM
from src.utils.metrics import compute_metrics


#####################################################################
# 方案一：自适应 α 路由
#####################################################################

class AdaptiveAlphaMoE(nn.Module):
    """
    核心改进：α(x) = σ(w·log(n_min + 1) + b)
    """
    def __init__(self, input_dim: int, num_experts: int = 8):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = 2
        
        self.alpha_weight = nn.Parameter(torch.tensor(-0.5))
        self.alpha_bias = nn.Parameter(torch.tensor(0.5))
        
        self.prior_router = nn.Linear(input_dim, num_experts)
        self.learned_router = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_experts),
        )
        
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
            ) for _ in range(num_experts)
        ])
    
    def forward(self, x: torch.Tensor, token_freqs: torch.Tensor) -> Dict:
        batch_size = x.shape[0]
        
        log_freq = torch.log(token_freqs.float() + 1)
        alpha = torch.sigmoid(self.alpha_weight * log_freq + self.alpha_bias).unsqueeze(1)
        
        prior_logits = self.prior_router(x)
        learned_logits = self.learned_router(x)
        router_logits = alpha * prior_logits + (1 - alpha) * learned_logits
        
        top_k_logits, expert_indices = torch.topk(router_logits, self.top_k, dim=-1)
        expert_weights = F.softmax(top_k_logits, dim=-1)
        
        all_expert_outputs = torch.stack([e(x) for e in self.experts], dim=1)
        
        output = torch.zeros(batch_size, 64, device=x.device)
        for k in range(self.top_k):
            expert_idx = expert_indices[:, k]
            weight = expert_weights[:, k:k+1]
            expert_out = all_expert_outputs[torch.arange(batch_size), expert_idx]
            output = output + weight * expert_out
        
        return {'output': output, 'alpha_mean': alpha.mean().item()}


#####################################################################
# 方案二：上下文聚类路由
#####################################################################

class ContextClusterRouter(nn.Module):
    def __init__(self, input_dim: int, num_clusters: int = 8, num_experts: int = 8):
        super().__init__()
        self.num_experts = num_experts
        
        self.cluster_centers = nn.Parameter(torch.randn(num_clusters, input_dim) * 0.1)
        self.cluster_to_expert = nn.Linear(num_clusters, num_experts)
        self.learned_router = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_experts),
        )
        
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
            ) for _ in range(num_experts)
        ])
        
        self.alpha = nn.Parameter(torch.tensor(0.5))
    
    def forward(self, x: torch.Tensor) -> Dict:
        batch_size = x.shape[0]
        
        distances = torch.cdist(x.unsqueeze(0), self.cluster_centers.unsqueeze(0)).squeeze(0)
        cluster_weights = F.softmax(-distances / 0.5, dim=-1)
        cluster_logits = self.cluster_to_expert(cluster_weights)
        
        learned_logits = self.learned_router(x)
        alpha = torch.sigmoid(self.alpha)
        router_logits = alpha * cluster_logits + (1 - alpha) * learned_logits
        
        top_k_logits, expert_indices = torch.topk(router_logits, 2, dim=-1)
        expert_weights = F.softmax(top_k_logits, dim=-1)
        
        all_expert_outputs = torch.stack([e(x) for e in self.experts], dim=1)
        
        output = torch.zeros(batch_size, 64, device=x.device)
        for k in range(2):
            expert_idx = expert_indices[:, k]
            weight = expert_weights[:, k:k+1]
            expert_out = all_expert_outputs[torch.arange(batch_size), expert_idx]
            output = output + weight * expert_out
        
        return {'output': output, 'alpha': alpha.item()}


#####################################################################
# 方案三：组合方案
#####################################################################

class CombinedImprovedMoE(nn.Module):
    def __init__(self, input_dim: int, num_clusters: int = 8, num_experts: int = 8):
        super().__init__()
        self.num_experts = num_experts
        
        self.alpha_weight = nn.Parameter(torch.tensor(-0.3))
        self.alpha_bias = nn.Parameter(torch.tensor(0.3))
        
        self.cluster_centers = nn.Parameter(torch.randn(num_clusters, input_dim) * 0.1)
        self.cluster_to_expert = nn.Linear(num_clusters, num_experts)
        
        self.learned_router = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_experts),
        )
        
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
            ) for _ in range(num_experts)
        ])
    
    def forward(self, x: torch.Tensor, token_freqs: torch.Tensor) -> Dict:
        batch_size = x.shape[0]
        
        log_freq = torch.log(token_freqs.float() + 1)
        alpha = torch.sigmoid(self.alpha_weight * log_freq + self.alpha_bias).unsqueeze(1)
        
        distances = torch.cdist(x.unsqueeze(0), self.cluster_centers.unsqueeze(0)).squeeze(0)
        cluster_weights = F.softmax(-distances / 0.5, dim=-1)
        cluster_logits = self.cluster_to_expert(cluster_weights)
        
        learned_logits = self.learned_router(x)
        router_logits = alpha * cluster_logits + (1 - alpha) * learned_logits
        
        top_k_logits, expert_indices = torch.topk(router_logits, 2, dim=-1)
        expert_weights = F.softmax(top_k_logits, dim=-1)
        
        all_expert_outputs = torch.stack([e(x) for e in self.experts], dim=1)
        
        output = torch.zeros(batch_size, 64, device=x.device)
        for k in range(2):
            expert_idx = expert_indices[:, k]
            weight = expert_weights[:, k:k+1]
            expert_out = all_expert_outputs[torch.arange(batch_size), expert_idx]
            output = output + weight * expert_out
        
        return {'output': output, 'alpha_mean': alpha.mean().item()}


#####################################################################
# 完整模型
#####################################################################

class ImprovedMoEModel(nn.Module):
    def __init__(self, dense_dim, sparse_dims, embedding_dim=16, moe_type='adaptive_alpha', num_experts=8):
        super().__init__()
        
        self.moe_type = moe_type
        self.num_sparse_fields = len(sparse_dims)
        
        self.embeddings = nn.ModuleDict({
            name: nn.Embedding(dim, embedding_dim)
            for name, dim in sparse_dims.items()
        })
        self.sparse_field_names = list(sparse_dims.keys())
        self.dense_bn = nn.BatchNorm1d(dense_dim)
        
        moe_input_dim = dense_dim + self.num_sparse_fields * embedding_dim
        
        if moe_type == 'adaptive_alpha':
            self.moe = AdaptiveAlphaMoE(moe_input_dim, num_experts)
        elif moe_type == 'context_cluster':
            self.moe = ContextClusterRouter(moe_input_dim, num_experts)
        elif moe_type == 'combined':
            self.moe = CombinedImprovedMoE(moe_input_dim, num_experts)
        
        self.prediction_head = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )
    
    def forward(self, dense, sparse, token_freqs=None):
        batch_size = dense.shape[0]
        dense = self.dense_bn(dense)
        
        sparse_embs = []
        for i, name in enumerate(self.sparse_field_names):
            emb = self.embeddings[name](sparse[:, i])
            sparse_embs.append(emb)
        sparse_flat = torch.cat(sparse_embs, dim=1)
        
        combined = torch.cat([dense, sparse_flat], dim=1)
        
        if self.moe_type in ['adaptive_alpha', 'combined']:
            if token_freqs is None:
                token_freqs = torch.ones(batch_size, device=dense.device) * 10
            moe_out = self.moe(combined, token_freqs)
        else:
            moe_out = self.moe(combined)
        
        logits = self.prediction_head(moe_out['output']).squeeze(-1)
        
        return {'logits': logits, 'alpha_mean': moe_out.get('alpha_mean', None)}
    
    def compute_loss(self, outputs, labels):
        return F.binary_cross_entropy_with_logits(outputs['logits'], labels)


#####################################################################
# 实验函数
#####################################################################

def compute_item_frequencies(dataset, field_idx=1):
    freq_counter = defaultdict(int)
    for i in range(len(dataset)):
        sample = dataset[i]
        item_id = sample['sparse'][field_idx].item()
        freq_counter[item_id] += 1
    return freq_counter


def get_token_freqs_for_batch(batch_sparse, item_field_idx, freq_dict):
    freqs = []
    for i in range(batch_sparse.shape[0]):
        item_id = batch_sparse[i, item_field_idx].item()
        freq = freq_dict.get(item_id, 1)
        freqs.append(freq)
    return torch.tensor(freqs)


def train_epoch_improved(model, dataloader, optimizer, device, freq_dict, item_field_idx, use_freq=True):
    model.train()
    total_loss = 0
    for batch in dataloader:
        dense = batch['dense'].to(device)
        sparse = batch['sparse'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        
        if use_freq:
            token_freqs = get_token_freqs_for_batch(sparse, item_field_idx, freq_dict).to(device)
            outputs = model(dense, sparse, token_freqs)
        else:
            outputs = model(dense, sparse)
        
        loss = model.compute_loss(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def train_epoch_baseline(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in dataloader:
        dense = batch['dense'].to(device)
        sparse = batch['sparse'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        outputs = model(dense, sparse)
        loss = model.compute_loss(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def evaluate_improved(model, dataloader, device, freq_dict, item_field_idx, use_freq=True):
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for batch in dataloader:
            dense = batch['dense'].to(device)
            sparse = batch['sparse'].to(device)
            labels = batch['label'].to(device)
            
            if use_freq:
                token_freqs = get_token_freqs_for_batch(sparse, item_field_idx, freq_dict).to(device)
                outputs = model(dense, sparse, token_freqs)
            else:
                outputs = model(dense, sparse)
            
            preds = torch.sigmoid(outputs['logits']).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
    
    return compute_metrics(np.array(all_preds), np.array(all_labels))


def evaluate_baseline(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for batch in dataloader:
            dense = batch['dense'].to(device)
            sparse = batch['sparse'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(dense, sparse)
            preds = torch.sigmoid(outputs['logits']).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
    
    return compute_metrics(np.array(all_preds), np.array(all_labels))


def evaluate_by_bucket(model, dataset, freq_dict, device, item_field_idx, is_baseline=False, is_improved_with_freq=False):
    buckets = {
        'very_rare': (1, 5),
        'rare': (6, 20),
        'moderate': (21, 100),
        'frequent': (101, float('inf')),
    }
    
    bucket_indices = defaultdict(list)
    for i in range(len(dataset)):
        sample = dataset[i]
        item_id = sample['sparse'][item_field_idx].item()
        freq = freq_dict.get(item_id, 0)
        
        for name, (low, high) in buckets.items():
            if low <= freq <= high:
                bucket_indices[name].append(i)
                break
    
    results = {}
    for name, indices in bucket_indices.items():
        if len(indices) < 50:
            continue
        subset = Subset(dataset, indices[:1000])
        loader = DataLoader(subset, batch_size=512, shuffle=False)
        
        if is_baseline:
            metrics = evaluate_baseline(model, loader, device)
        else:
            metrics = evaluate_improved(model, loader, device, freq_dict, item_field_idx, use_freq=is_improved_with_freq)
        
        metrics['count'] = len(indices)
        results[name] = metrics
    
    return results


def run_validation_experiment():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    with open('configs/amazon.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    print("Loading data (small scale)...")
    train_dataset, val_dataset, test_dataset = load_amazon_data(
        reviews_path=config['data']['reviews_path'],
        mode=config['data']['mode'],
        sample_size=50000,
    )
    
    feature_dims = train_dataset.get_feature_dims()
    sparse_fields = train_dataset.actual_sparse_cols
    item_field_idx = sparse_fields.index('asin') if 'asin' in sparse_fields else 1
    
    print(f"Train: {len(train_dataset)}, Test: {len(test_dataset)}")
    
    print("Computing frequencies...")
    freq_dict = compute_item_frequencies(train_dataset, item_field_idx)
    
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False, num_workers=2)
    
    results = {}
    
    # 基线: DeepFM
    print(f"\n{'='*60}")
    print("Testing: DeepFM (baseline)")
    print('='*60)
    
    torch.manual_seed(42)
    model = DeepFM(
        dense_dim=feature_dims['dense_dim'],
        sparse_dims=feature_dims['sparse_dims'],
        embedding_dim=16, hidden_dims=[128, 64], dropout=0.1,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    start_time = time.time()
    for epoch in range(3):
        loss = train_epoch_baseline(model, train_loader, optimizer, device)
        print(f"  Epoch {epoch+1}/3, Loss: {loss:.4f}")
    
    overall = evaluate_baseline(model, test_loader, device)
    buckets = evaluate_by_bucket(model, test_dataset, freq_dict, device, item_field_idx, is_baseline=True)
    results['DeepFM'] = {'overall': overall, 'buckets': buckets, 'time': time.time() - start_time}
    print(f"  Overall: AUC={overall['auc']:.4f}, LogLoss={overall['logloss']:.4f}")
    
    # 基线: DLRM
    print(f"\n{'='*60}")
    print("Testing: DLRM (baseline)")
    print('='*60)
    
    torch.manual_seed(42)
    model = DLRM(
        dense_dim=feature_dims['dense_dim'],
        sparse_dims=feature_dims['sparse_dims'],
        embedding_dim=16, dropout=0.1,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    start_time = time.time()
    for epoch in range(3):
        loss = train_epoch_baseline(model, train_loader, optimizer, device)
        print(f"  Epoch {epoch+1}/3, Loss: {loss:.4f}")
    
    overall = evaluate_baseline(model, test_loader, device)
    buckets = evaluate_by_bucket(model, test_dataset, freq_dict, device, item_field_idx, is_baseline=True)
    results['DLRM'] = {'overall': overall, 'buckets': buckets, 'time': time.time() - start_time}
    print(f"  Overall: AUC={overall['auc']:.4f}, LogLoss={overall['logloss']:.4f}")
    
    # 改进方案
    for method in ['adaptive_alpha', 'context_cluster', 'combined']:
        print(f"\n{'='*60}")
        print(f"Testing: {method}")
        print('='*60)
        
        torch.manual_seed(42)
        model = ImprovedMoEModel(
            dense_dim=feature_dims['dense_dim'],
            sparse_dims=feature_dims['sparse_dims'],
            embedding_dim=16,
            moe_type=method,
            num_experts=8,
        ).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        use_freq = method in ['adaptive_alpha', 'combined']
        
        start_time = time.time()
        for epoch in range(3):
            loss = train_epoch_improved(model, train_loader, optimizer, device, freq_dict, item_field_idx, use_freq)
            print(f"  Epoch {epoch+1}/3, Loss: {loss:.4f}")
        
        overall = evaluate_improved(model, test_loader, device, freq_dict, item_field_idx, use_freq)
        buckets = evaluate_by_bucket(model, test_dataset, freq_dict, device, item_field_idx, 
                                     is_baseline=False, is_improved_with_freq=use_freq)
        
        results[method] = {'overall': overall, 'buckets': buckets, 'time': time.time() - start_time}
        print(f"  Overall: AUC={overall['auc']:.4f}, LogLoss={overall['logloss']:.4f}")
        if 'very_rare' in buckets:
            print(f"  Cold-start: AUC={buckets['very_rare']['auc']:.4f}, LogLoss={buckets['very_rare']['logloss']:.4f}")
    
    # 汇总
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    print("\n整体性能 (按LogLoss排序):")
    sorted_results = sorted(results.items(), key=lambda x: x[1]['overall']['logloss'])
    for i, (name, res) in enumerate(sorted_results, 1):
        print(f"  {i}. {name}: AUC={res['overall']['auc']:.4f}, LogLoss={res['overall']['logloss']:.4f}")
    
    print("\n冷启动性能 (very_rare, 按LogLoss排序):")
    cold_results = [(m, r['buckets'].get('very_rare', {}).get('logloss', float('inf'))) 
                    for m, r in results.items()]
    for i, (name, ll) in enumerate(sorted(cold_results, key=lambda x: x[1]), 1):
        if ll < float('inf'):
            print(f"  {i}. {name}: LogLoss={ll:.4f}")
    
    print("\n" + "="*70)
    best_overall = sorted_results[0][0]
    best_cold = min(cold_results, key=lambda x: x[1])[0]
    print(f"最佳整体方案: {best_overall}")
    print(f"最佳冷启动方案: {best_cold}")
    
    return results


if __name__ == '__main__':
    results = run_validation_experiment()
