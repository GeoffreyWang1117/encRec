#!/usr/bin/env python3
"""
大规模验证：确认改进方案的效果
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
import json

from src.data.amazon_loader import load_amazon_data
from src.models.backbone import DeepFM, DLRM
from src.utils.metrics import compute_metrics


class AdaptiveAlphaMoE(nn.Module):
    """自适应 α 路由 MoE"""
    def __init__(self, input_dim, num_experts=8):
        super().__init__()
        self.num_experts = num_experts
        
        # 可学习的自适应权重参数
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
    
    def forward(self, x, token_freqs):
        batch_size = x.shape[0]
        
        # α(t) = σ(w·log(n+1) + b)
        log_freq = torch.log(token_freqs.float() + 1)
        alpha = torch.sigmoid(self.alpha_weight * log_freq + self.alpha_bias).unsqueeze(1)
        
        prior_logits = self.prior_router(x)
        learned_logits = self.learned_router(x)
        router_logits = alpha * prior_logits + (1 - alpha) * learned_logits
        
        top_k_logits, expert_indices = torch.topk(router_logits, 2, dim=-1)
        expert_weights = F.softmax(top_k_logits, dim=-1)
        
        all_expert_outputs = torch.stack([e(x) for e in self.experts], dim=1)
        
        output = torch.zeros(batch_size, 64, device=x.device)
        for k in range(2):
            expert_idx = expert_indices[:, k]
            weight = expert_weights[:, k:k+1]
            expert_out = all_expert_outputs[torch.arange(batch_size, device=x.device), expert_idx]
            output = output + weight * expert_out
        
        return {'output': output, 'alpha_mean': alpha.mean().item()}


class AdaptiveAlphaModel(nn.Module):
    """完整的自适应 α 模型"""
    def __init__(self, dense_dim, sparse_dims, embedding_dim=16, num_experts=8):
        super().__init__()
        
        self.num_sparse_fields = len(sparse_dims)
        self.embeddings = nn.ModuleDict({
            name: nn.Embedding(dim, embedding_dim)
            for name, dim in sparse_dims.items()
        })
        self.sparse_field_names = list(sparse_dims.keys())
        self.dense_bn = nn.BatchNorm1d(dense_dim)
        
        moe_input_dim = dense_dim + self.num_sparse_fields * embedding_dim
        self.moe = AdaptiveAlphaMoE(moe_input_dim, num_experts)
        
        self.prediction_head = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )
    
    def forward(self, dense, sparse, token_freqs):
        batch_size = dense.shape[0]
        dense = self.dense_bn(dense)
        
        sparse_embs = [self.embeddings[name](sparse[:, i]) 
                       for i, name in enumerate(self.sparse_field_names)]
        sparse_flat = torch.cat(sparse_embs, dim=1)
        combined = torch.cat([dense, sparse_flat], dim=1)
        
        moe_out = self.moe(combined, token_freqs)
        logits = self.prediction_head(moe_out['output']).squeeze(-1)
        
        return {'logits': logits, 'alpha_mean': moe_out.get('alpha_mean')}
    
    def compute_loss(self, outputs, labels):
        return F.binary_cross_entropy_with_logits(outputs['logits'], labels)


def compute_item_frequencies(dataset, field_idx=1):
    freq_counter = defaultdict(int)
    for i in range(len(dataset)):
        sample = dataset[i]
        item_id = sample['sparse'][field_idx].item()
        freq_counter[item_id] += 1
    return freq_counter


def get_token_freqs(batch_sparse, item_field_idx, freq_dict):
    freqs = [freq_dict.get(batch_sparse[i, item_field_idx].item(), 1) 
             for i in range(batch_sparse.shape[0])]
    return torch.tensor(freqs)


def train_epoch(model, dataloader, optimizer, device, freq_dict=None, item_field_idx=1, is_baseline=False):
    model.train()
    total_loss = 0
    
    for batch in dataloader:
        dense = batch['dense'].to(device)
        sparse = batch['sparse'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        
        if is_baseline:
            outputs = model(dense, sparse)
        else:
            token_freqs = get_token_freqs(sparse, item_field_idx, freq_dict).to(device)
            outputs = model(dense, sparse, token_freqs)
        
        loss = model.compute_loss(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, device, freq_dict=None, item_field_idx=1, is_baseline=False):
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for batch in dataloader:
            dense = batch['dense'].to(device)
            sparse = batch['sparse'].to(device)
            labels = batch['label'].to(device)
            
            if is_baseline:
                outputs = model(dense, sparse)
            else:
                token_freqs = get_token_freqs(sparse, item_field_idx, freq_dict).to(device)
                outputs = model(dense, sparse, token_freqs)
            
            preds = torch.sigmoid(outputs['logits']).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
    
    return compute_metrics(np.array(all_preds), np.array(all_labels))


def evaluate_by_bucket(model, dataset, freq_dict, device, item_field_idx, is_baseline=False):
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
        if len(indices) < 100:
            continue
        subset = Subset(dataset, indices)
        loader = DataLoader(subset, batch_size=512, shuffle=False, num_workers=2)
        metrics = evaluate(model, loader, device, freq_dict, item_field_idx, is_baseline)
        metrics['count'] = len(indices)
        results[name] = metrics
    
    return results


def run_single_experiment(model_type, train_dataset, test_dataset, feature_dims, 
                          freq_dict, item_field_idx, device, seed=42):
    """运行单次实验"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False, num_workers=4)
    
    is_baseline = model_type in ['DeepFM', 'DLRM']
    
    if model_type == 'DeepFM':
        model = DeepFM(
            dense_dim=feature_dims['dense_dim'],
            sparse_dims=feature_dims['sparse_dims'],
            embedding_dim=16, hidden_dims=[128, 64], dropout=0.1,
        ).to(device)
    elif model_type == 'DLRM':
        model = DLRM(
            dense_dim=feature_dims['dense_dim'],
            sparse_dims=feature_dims['sparse_dims'],
            embedding_dim=16, dropout=0.1,
        ).to(device)
    else:  # adaptive_alpha
        model = AdaptiveAlphaModel(
            dense_dim=feature_dims['dense_dim'],
            sparse_dims=feature_dims['sparse_dims'],
            embedding_dim=16, num_experts=8,
        ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练 5 epochs
    for epoch in range(5):
        train_epoch(model, train_loader, optimizer, device, freq_dict, item_field_idx, is_baseline)
    
    # 评估
    overall = evaluate(model, test_loader, device, freq_dict, item_field_idx, is_baseline)
    buckets = evaluate_by_bucket(model, test_dataset, freq_dict, device, item_field_idx, is_baseline)
    
    return {'overall': overall, 'buckets': buckets}


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    with open('configs/amazon.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    print("Loading full data...")
    train_dataset, val_dataset, test_dataset = load_amazon_data(
        reviews_path=config['data']['reviews_path'],
        mode=config['data']['mode'],
        sample_size=config['data'].get('sample_size'),
    )
    
    feature_dims = train_dataset.get_feature_dims()
    sparse_fields = train_dataset.actual_sparse_cols
    item_field_idx = sparse_fields.index('asin') if 'asin' in sparse_fields else 1
    
    print(f"Train: {len(train_dataset)}, Test: {len(test_dataset)}")
    
    print("Computing frequencies...")
    freq_dict = compute_item_frequencies(train_dataset, item_field_idx)
    
    NUM_RUNS = 5
    models = ['DeepFM', 'DLRM', 'adaptive_alpha']
    
    all_results = defaultdict(list)
    
    for model_type in models:
        print(f"\n{'='*60}")
        print(f"Testing: {model_type} ({NUM_RUNS} runs)")
        print('='*60)
        
        for run in range(NUM_RUNS):
            print(f"  Run {run+1}/{NUM_RUNS}...", end=" ", flush=True)
            start = time.time()
            
            result = run_single_experiment(
                model_type, train_dataset, test_dataset, feature_dims,
                freq_dict, item_field_idx, device, seed=42+run
            )
            
            elapsed = time.time() - start
            all_results[model_type].append(result)
            
            print(f"AUC={result['overall']['auc']:.4f}, LogLoss={result['overall']['logloss']:.4f}, Time={elapsed:.1f}s")
    
    # 统计汇总
    print("\n" + "="*70)
    print("FINAL SUMMARY (5 runs)")
    print("="*70)
    
    summary = {}
    for model_type in models:
        runs = all_results[model_type]
        aucs = [r['overall']['auc'] for r in runs]
        loglosses = [r['overall']['logloss'] for r in runs]
        
        # 冷启动
        cold_loglosses = [r['buckets'].get('very_rare', {}).get('logloss', float('inf')) for r in runs]
        cold_loglosses = [ll for ll in cold_loglosses if ll < float('inf')]
        
        summary[model_type] = {
            'auc_mean': np.mean(aucs),
            'auc_std': np.std(aucs),
            'logloss_mean': np.mean(loglosses),
            'logloss_std': np.std(loglosses),
            'cold_logloss_mean': np.mean(cold_loglosses) if cold_loglosses else float('inf'),
            'cold_logloss_std': np.std(cold_loglosses) if cold_loglosses else 0,
        }
        
        print(f"\n{model_type}:")
        print(f"  Overall: AUC={summary[model_type]['auc_mean']:.4f}±{summary[model_type]['auc_std']:.4f}, "
              f"LogLoss={summary[model_type]['logloss_mean']:.4f}±{summary[model_type]['logloss_std']:.4f}")
        print(f"  Cold-start: LogLoss={summary[model_type]['cold_logloss_mean']:.4f}±{summary[model_type]['cold_logloss_std']:.4f}")
    
    # 比较改进
    print("\n" + "="*70)
    print("IMPROVEMENT ANALYSIS")
    print("="*70)
    
    baseline = summary['DeepFM']['logloss_mean']
    for model_type in ['DLRM', 'adaptive_alpha']:
        improvement = (baseline - summary[model_type]['logloss_mean']) / baseline * 100
        print(f"{model_type} vs DeepFM: {improvement:+.2f}% LogLoss improvement")
    
    cold_baseline = summary['DeepFM']['cold_logloss_mean']
    for model_type in ['DLRM', 'adaptive_alpha']:
        cold_ll = summary[model_type]['cold_logloss_mean']
        if cold_ll < float('inf') and cold_baseline < float('inf'):
            improvement = (cold_baseline - cold_ll) / cold_baseline * 100
            print(f"{model_type} vs DeepFM (cold-start): {improvement:+.2f}% LogLoss improvement")
    
    # 保存结果
    with open('results/adaptive_alpha_validation.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to results/adaptive_alpha_validation.json")
    
    return summary


if __name__ == '__main__':
    main()
