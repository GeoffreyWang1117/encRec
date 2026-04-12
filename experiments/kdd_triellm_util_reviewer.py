#!/usr/bin/env python3
"""
Reviewer-requested supplementary experiments for KDD 2026

Addresses specific reviewer concerns:
1. Ablation at 1K samples (Table 6)
2. Threshold sensitivity analysis (Table 7)
3. Expert allocation comparison
4. Learned parameter stability across runs/sizes
5. Cardinality analysis (why Criteo is different)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
from sklearn.metrics import roc_auc_score, log_loss
from collections import defaultdict
import json
import warnings
warnings.filterwarnings('ignore')

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class SimpleDeepFM(nn.Module):
    """Simplified DeepFM for fair comparison."""
    def __init__(self, sparse_dims, embedding_dim=32, hidden_dims=[128, 64], dropout=0.1):
        super().__init__()
        self.num_fields = len(sparse_dims)
        self.embeddings = nn.ModuleList([
            nn.Embedding(dim, embedding_dim, padding_idx=0)
            for dim in sparse_dims.values()
        ])
        self.fm_first = nn.ModuleList([
            nn.Embedding(dim, 1, padding_idx=0)
            for dim in sparse_dims.values()
        ])
        
        dnn_input = self.num_fields * embedding_dim
        layers = []
        for hd in hidden_dims:
            layers.extend([nn.Linear(dnn_input, hd), nn.BatchNorm1d(hd), nn.ReLU(), nn.Dropout(dropout)])
            dnn_input = hd
        layers.append(nn.Linear(dnn_input, 1))
        self.dnn = nn.Sequential(*layers)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, sparse):
        first = sum(e(sparse[:, i]).squeeze(-1) for i, e in enumerate(self.fm_first))
        embs = torch.stack([e(sparse[:, i]) for i, e in enumerate(self.embeddings)], dim=1)
        ss = embs.sum(dim=1).pow(2).sum(dim=-1)
        sq = embs.pow(2).sum(dim=1).sum(dim=-1)
        second = 0.5 * (ss - sq)
        dnn_out = self.dnn(embs.view(sparse.shape[0], -1)).squeeze(-1)
        return self.bias + first + second + dnn_out


class SimpleTrieMoE(nn.Module):
    """Simplified Trie-MoE with configurable components."""
    def __init__(self, sparse_dims, embedding_dim=32, num_experts=9, hidden_dims=[128, 64],
                 tier_thresholds=(20, 80), lift_thresholds=(0.8, 1.2),
                 expert_allocation='frequency_aware', use_trie_prior=True, use_adaptive_mixing=True):
        super().__init__()
        self.num_fields = len(sparse_dims)
        self.num_experts = num_experts
        self.tier_thresholds = tier_thresholds
        self.lift_thresholds = lift_thresholds
        self.expert_allocation = expert_allocation
        self.use_trie_prior = use_trie_prior
        self.use_adaptive_mixing = use_adaptive_mixing
        
        self.embeddings = nn.ModuleList([
            nn.Embedding(dim, embedding_dim, padding_idx=0)
            for dim in sparse_dims.values()
        ])
        
        self.register_buffer('token_freqs', None)
        self.register_buffer('token_ctrs', None)
        self.register_buffer('global_ctr', torch.tensor(0.5))
        
        self.alpha_w = nn.Parameter(torch.tensor(-0.5))
        self.alpha_b = nn.Parameter(torch.tensor(0.5))
        
        router_input = self.num_fields * embedding_dim
        self.router = nn.Sequential(nn.Linear(router_input, 64), nn.ReLU(), nn.Linear(64, num_experts))
        
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(router_input, hidden_dims[0]), nn.ReLU(), nn.Linear(hidden_dims[0], hidden_dims[1]))
            for _ in range(num_experts)
        ])
        
        self.output = nn.Linear(hidden_dims[1], 1)
        self.bias = nn.Parameter(torch.zeros(1))

    def compute_stats(self, loader, device):
        fc = defaultdict(lambda: defaultdict(int))
        fp = defaultdict(lambda: defaultdict(int))
        total, pos = 0, 0
        
        for batch in loader:
            sp, lb = batch['sparse'], batch['label']
            for i in range(sp.shape[0]):
                total += 1
                pos += lb[i].item()
                for j in range(sp.shape[1]):
                    t = sp[i, j].item()
                    fc[j][t] += 1
                    fp[j][t] += lb[i].item()
        
        self.global_ctr = torch.tensor(pos / total if total > 0 else 0.5)
        mv = max(max(d.keys()) + 1 for d in fc.values())
        tf = torch.zeros(self.num_fields, mv)
        tc = torch.zeros(self.num_fields, mv)
        
        for j in range(self.num_fields):
            for t, c in fc[j].items():
                tf[j, t] = c
                tc[j, t] = fp[j][t] / c if c > 0 else self.global_ctr.item()
        
        self.token_freqs = tf.to(device)
        self.token_ctrs = tc.to(device)

    def get_tier(self, j, t):
        if self.token_freqs is None:
            return 1, 1
        f = self.token_freqs[j, t].item()
        c = self.token_ctrs[j, t].item()
        
        nz = self.token_freqs[j][self.token_freqs[j] > 0]
        if len(nz) == 0:
            return 1, 1
        
        p_low = torch.quantile(nz.float(), self.tier_thresholds[0]/100).item()
        p_high = torch.quantile(nz.float(), self.tier_thresholds[1]/100).item()
        
        tier = 0 if f > p_high else (1 if f > p_low else 2)
        lift = c / self.global_ctr.item() if self.global_ctr.item() > 0 else 1.0
        lift_tier = 0 if lift > self.lift_thresholds[1] else (2 if lift < self.lift_thresholds[0] else 1)
        return tier, lift_tier

    def get_prior(self, sparse):
        bs = sparse.shape[0]
        prior = torch.zeros(bs, self.num_experts, device=sparse.device)
        
        if not self.use_trie_prior or self.token_freqs is None:
            return torch.ones_like(prior) / self.num_experts
        
        if self.expert_allocation == 'frequency_aware':
            t2e = {0: [0, 1], 1: [2, 3, 4], 2: [5, 6, 7, 8]}
        elif self.expert_allocation == 'uniform':
            t2e = {0: [0, 1, 2], 1: [3, 4, 5], 2: [6, 7, 8]}
        else:
            t2e = {0: [5, 6, 7, 8], 1: [2, 3, 4], 2: [0, 1]}
        
        for i in range(bs):
            tc = defaultdict(int)
            for j in range(sparse.shape[1]):
                tier, _ = self.get_tier(j, sparse[i, j].item())
                tc[tier] += 1
            tot = sum(tc.values())
            for tier, cnt in tc.items():
                w = cnt / tot
                for e in t2e[tier]:
                    prior[i, e] += w / len(t2e[tier])
        
        return prior / (prior.sum(dim=-1, keepdim=True) + 1e-8)

    def get_alpha(self, sparse):
        if not self.use_adaptive_mixing or self.token_freqs is None:
            return torch.ones(sparse.shape[0], device=sparse.device) * 0.5
        
        avg_f = torch.zeros(sparse.shape[0], device=sparse.device)
        for i in range(sparse.shape[0]):
            fs = [self.token_freqs[j, sparse[i, j].item()].item() for j in range(sparse.shape[1])]
            avg_f[i] = np.mean(fs) if fs else 1.0
        
        return torch.sigmoid(self.alpha_w * torch.log(avg_f + 1) + self.alpha_b)

    def forward(self, sparse, return_info=False):
        bs = sparse.shape[0]
        embs = torch.stack([e(sparse[:, i]) for i, e in enumerate(self.embeddings)], dim=1)
        flat = embs.view(bs, -1)
        
        learned = torch.softmax(self.router(flat), dim=-1)
        prior = self.get_prior(sparse)
        alpha = self.get_alpha(sparse)
        
        routing = alpha.unsqueeze(-1) * prior + (1 - alpha.unsqueeze(-1)) * learned
        
        top_w, top_i = torch.topk(routing, 2, dim=-1)
        top_w = top_w / top_w.sum(dim=-1, keepdim=True)
        
        out = torch.zeros(bs, self.experts[0][-1].out_features, device=sparse.device)
        for k in range(2):
            for e in range(self.num_experts):
                m = top_i[:, k] == e
                if m.any():
                    out[m] += top_w[m, k:k+1] * self.experts[e](flat[m])
        
        logits = self.output(out).squeeze(-1) + self.bias
        
        if return_info:
            return logits, {'alpha': alpha, 'w': self.alpha_w.item(), 'b': self.alpha_b.item()}
        return logits


def train_model(model, loader, opt, device, is_moe=False):
    model.train()
    loss_fn = nn.BCEWithLogitsLoss()
    total = 0
    for batch in loader:
        sp = batch['sparse'].to(device)
        lb = batch['label'].to(device)
        opt.zero_grad()
        logits = model(sp)
        loss = loss_fn(logits, lb)
        loss.backward()
        opt.step()
        total += loss.item()
    return total / len(loader)


def eval_model(model, loader, device, is_moe=False):
    model.eval()
    preds, labels, alphas = [], [], []
    with torch.no_grad():
        for batch in loader:
            sp = batch['sparse'].to(device)
            lb = batch['label'].to(device)
            if is_moe:
                logits, info = model(sp, return_info=True)
                alphas.extend(info['alpha'].cpu().tolist())
            else:
                logits = model(sp)
            preds.extend(torch.sigmoid(logits).cpu().tolist())
            labels.extend(lb.cpu().tolist())
    
    auc = roc_auc_score(labels, preds)
    ll = log_loss(labels, np.clip(preds, 1e-7, 1-1e-7))
    res = {'auc': auc, 'logloss': ll}
    if alphas:
        res['alpha_mean'] = np.mean(alphas)
        res['alpha_std'] = np.std(alphas)
    return res


def run_single(train_ds, val_ds, test_ds, ModelClass, kwargs, device, epochs=10, lr=0.001, seed=42, is_moe=False):
    set_seed(seed)
    train_ld = DataLoader(train_ds, batch_size=256, shuffle=True)
    val_ld = DataLoader(val_ds, batch_size=256)
    test_ld = DataLoader(test_ds, batch_size=256)
    
    model = ModelClass(**kwargs).to(device)
    if is_moe:
        model.compute_stats(train_ld, device)
    
    opt = optim.Adam(model.parameters(), lr=lr)
    best_auc, best_res, best_params = 0, None, None
    
    for ep in range(epochs):
        train_model(model, train_ld, opt, device, is_moe)
        val_res = eval_model(model, val_ld, device, is_moe)
        if val_res['auc'] > best_auc:
            best_auc = val_res['auc']
            best_res = eval_model(model, test_ld, device, is_moe)
            if is_moe:
                best_params = {'w': model.alpha_w.item(), 'b': model.alpha_b.item()}
    
    if best_res is None:
        best_res = eval_model(model, test_ld, device, is_moe)
    if is_moe and best_params:
        best_res['w'] = best_params['w']
        best_res['b'] = best_params['b']
    return best_res


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load data
    from src.data.movielens_loader import load_movielens_data
    data_dir = Path('/home/coder-gw/Projects/encRec/data/ml-1m')
    
    if not data_dir.exists():
        print("Downloading MovieLens...")
        import urllib.request, zipfile
        data_dir.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve("https://files.grouplens.org/datasets/movielens/ml-1m.zip", 
                                   data_dir.parent / "ml-1m.zip")
        with zipfile.ZipFile(data_dir.parent / "ml-1m.zip", 'r') as z:
            z.extractall(data_dir.parent)
    
    print("Loading data...")
    train_full, val_ds, test_ds = load_movielens_data(str(data_dir))
    sparse_dims = train_full.get_feature_dims()['sparse_dims']
    print(f"Train: {len(train_full)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    print(f"Sparse dims: {sparse_dims}")
    
    results = {}
    
    # ========== Experiment 1: Ablation at 1K ==========
    print("\n" + "="*60)
    print("Exp 1: Ablation at 1K samples")
    print("="*60)
    
    idx_1k = np.random.choice(len(train_full), 1000, replace=False)
    train_1k = Subset(train_full, idx_1k)
    
    exp1 = {}
    configs = [
        ('DeepFM', SimpleDeepFM, {'sparse_dims': sparse_dims}, False),
        ('MoE_random', SimpleTrieMoE, {'sparse_dims': sparse_dims, 'use_trie_prior': False, 'use_adaptive_mixing': False}, True),
        ('MoE_learned', SimpleTrieMoE, {'sparse_dims': sparse_dims, 'use_trie_prior': False, 'use_adaptive_mixing': True}, True),
        ('Trie_prior', SimpleTrieMoE, {'sparse_dims': sparse_dims, 'use_trie_prior': True, 'use_adaptive_mixing': False}, True),
        ('Trie-MoE_full', SimpleTrieMoE, {'sparse_dims': sparse_dims, 'use_trie_prior': True, 'use_adaptive_mixing': True}, True),
    ]
    
    for name, cls, kw, is_moe in configs:
        print(f"  Running {name}...")
        runs = [run_single(train_1k, val_ds, test_ds, cls, kw, device, seed=s, is_moe=is_moe) 
                for s in [42, 43, 44, 45, 46]]
        exp1[name] = {
            'auc': np.mean([r['auc'] for r in runs]),
            'auc_std': np.std([r['auc'] for r in runs]),
            'logloss': np.mean([r['logloss'] for r in runs]),
            'logloss_std': np.std([r['logloss'] for r in runs])
        }
        print(f"    AUC: {exp1[name]['auc']:.4f} ± {exp1[name]['auc_std']:.4f}")
    results['exp1_ablation_1k'] = exp1
    
    # ========== Experiment 2: Threshold Sensitivity ==========
    print("\n" + "="*60)
    print("Exp 2: Threshold Sensitivity (5K samples)")
    print("="*60)
    
    idx_5k = np.random.choice(len(train_full), 5000, replace=False)
    train_5k = Subset(train_full, idx_5k)
    
    exp2 = {'tier': {}, 'lift': {}}
    
    print("  Tier thresholds:")
    for low, high in [(10, 90), (20, 80), (30, 70), (25, 75)]:
        kw = {'sparse_dims': sparse_dims, 'tier_thresholds': (low, high)}
        runs = [run_single(train_5k, val_ds, test_ds, SimpleTrieMoE, kw, device, seed=s, is_moe=True) 
                for s in [42, 43, 44]]
        exp2['tier'][f'{low}/{high}'] = {'auc': np.mean([r['auc'] for r in runs]), 
                                          'logloss': np.mean([r['logloss'] for r in runs])}
        print(f"    {low}/{high}: AUC={exp2['tier'][f'{low}/{high}']['auc']:.4f}")
    
    print("  Lift thresholds:")
    for low, high in [(0.7, 1.3), (0.8, 1.2), (0.9, 1.1), (0.6, 1.4)]:
        kw = {'sparse_dims': sparse_dims, 'lift_thresholds': (low, high)}
        runs = [run_single(train_5k, val_ds, test_ds, SimpleTrieMoE, kw, device, seed=s, is_moe=True) 
                for s in [42, 43, 44]]
        exp2['lift'][f'{low}/{high}'] = {'auc': np.mean([r['auc'] for r in runs]),
                                          'logloss': np.mean([r['logloss'] for r in runs])}
        print(f"    {low}/{high}: AUC={exp2['lift'][f'{low}/{high}']['auc']:.4f}")
    results['exp2_sensitivity'] = exp2
    
    # ========== Experiment 3: Expert Allocation ==========
    print("\n" + "="*60)
    print("Exp 3: Expert Allocation (5K samples)")
    print("="*60)
    
    exp3 = {}
    for alloc in ['frequency_aware', 'uniform', 'inverse']:
        kw = {'sparse_dims': sparse_dims, 'expert_allocation': alloc}
        runs = [run_single(train_5k, val_ds, test_ds, SimpleTrieMoE, kw, device, seed=s, is_moe=True) 
                for s in [42, 43, 44]]
        exp3[alloc] = {'auc': np.mean([r['auc'] for r in runs]),
                       'auc_std': np.std([r['auc'] for r in runs]),
                       'logloss': np.mean([r['logloss'] for r in runs])}
        print(f"  {alloc}: AUC={exp3[alloc]['auc']:.4f} ± {exp3[alloc]['auc_std']:.4f}")
    results['exp3_allocation'] = exp3
    
    # ========== Experiment 4: Learned Parameter Stability ==========
    print("\n" + "="*60)
    print("Exp 4: Learned Parameter Stability")
    print("="*60)
    
    exp4 = {}
    for n in [1000, 2000, 5000, 10000]:
        idx = np.random.choice(len(train_full), min(n, len(train_full)), replace=False)
        train_n = Subset(train_full, idx)
        runs = [run_single(train_n, val_ds, test_ds, SimpleTrieMoE, 
                          {'sparse_dims': sparse_dims}, device, seed=s, is_moe=True) 
                for s in [42, 43, 44, 45, 46]]
        ws = [r['w'] for r in runs if 'w' in r]
        bs = [r['b'] for r in runs if 'b' in r]
        exp4[n] = {'w_mean': np.mean(ws), 'w_std': np.std(ws),
                   'b_mean': np.mean(bs), 'b_std': np.std(bs),
                   'auc': np.mean([r['auc'] for r in runs])}
        print(f"  n={n}: w={exp4[n]['w_mean']:.3f}±{exp4[n]['w_std']:.3f}, "
              f"b={exp4[n]['b_mean']:.3f}±{exp4[n]['b_std']:.3f}, AUC={exp4[n]['auc']:.4f}")
    results['exp4_stability'] = exp4
    
    # ========== Experiment 5: Cardinality Analysis ==========
    print("\n" + "="*60)
    print("Exp 5: Cardinality Analysis")
    print("="*60)
    
    total_card = sum(sparse_dims.values())
    max_card = max(sparse_dims.values())
    avg_card = total_card / len(sparse_dims)
    
    print(f"  Total cardinality: {total_card}")
    print(f"  Max field cardinality: {max_card}")
    print(f"  Avg field cardinality: {avg_card:.0f}")
    
    # Compare at 5K
    deepfm_runs = [run_single(train_5k, val_ds, test_ds, SimpleDeepFM, 
                              {'sparse_dims': sparse_dims}, device, seed=s) 
                   for s in [42, 43, 44]]
    trie_runs = [run_single(train_5k, val_ds, test_ds, SimpleTrieMoE, 
                            {'sparse_dims': sparse_dims}, device, seed=s, is_moe=True) 
                 for s in [42, 43, 44]]
    
    deepfm_auc = np.mean([r['auc'] for r in deepfm_runs])
    trie_auc = np.mean([r['auc'] for r in trie_runs])
    improvement = (trie_auc - deepfm_auc) / deepfm_auc * 100
    
    results['exp5_cardinality'] = {
        'total': total_card, 'max': max_card, 'avg': avg_card,
        'deepfm_auc': deepfm_auc, 'trie_auc': trie_auc,
        'improvement_pct': improvement
    }
    print(f"  DeepFM AUC: {deepfm_auc:.4f}")
    print(f"  Trie-MoE AUC: {trie_auc:.4f}")
    print(f"  Improvement: {improvement:.2f}%")
    
    # Save results
    out_path = Path('/home/coder-gw/Projects/encRec/paper/kdd2026/reviewer_exp_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
    
    # Print summary for paper
    print("\n" + "="*60)
    print("SUMMARY FOR PAPER")
    print("="*60)
    
    print("\nTable 6: Ablation at 1K samples")
    print("-"*40)
    for name, m in exp1.items():
        print(f"{name:20s} & {m['auc']:.3f}±{m['auc_std']:.3f} & {m['logloss']:.3f} \\\\")
    
    print("\nTable 7: Threshold Sensitivity")
    print("-"*40)
    print("Tier thresholds:")
    for k, v in exp2['tier'].items():
        print(f"  {k}: AUC={v['auc']:.4f}")
    print("Lift thresholds:")
    for k, v in exp2['lift'].items():
        print(f"  {k}: AUC={v['auc']:.4f}")
    
    print("\nLearned Parameters:")
    print("-"*40)
    for n, p in exp4.items():
        print(f"n={n}: w={p['w_mean']:.2f}±{p['w_std']:.2f}, b={p['b_mean']:.2f}±{p['b_std']:.2f}")


if __name__ == '__main__':
    main()
