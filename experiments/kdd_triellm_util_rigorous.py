#!/usr/bin/env python3
"""
Rigorous Paper Experiments for Top-tier Conference Submission.

This script implements:
1. Strict cold-start evaluation protocol (frequency-based holdout)
2. 10-run experiments with statistical significance tests
3. Multiple datasets (Criteo + Amazon)
4. Comprehensive baselines (DeepFM, DLRM, DCNv2, DCNv3, AutoInt, FinalMLP, PLE)
5. Detailed statistical analysis with p-values and confidence intervals

Author: Research Team
Date: January 2026
Target: RecSys 2026 / CIKM 2026
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
from datetime import datetime
from collections import defaultdict, Counter
from scipy import stats
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from src.data.criteo_loader import load_criteo_data, CriteoDataset, SPARSE_COLS, DENSE_COLS
from src.data.amazon_loader import load_amazon_data
from src.trie.builder import TrieBuilder, StatisticalTrie
from src.trie.fast_encoder import FastTrieEncoder
from src.models.backbone import DeepFM, DLRM, DCNv2, DCNv3, AutoInt, FinalMLP
from src.models.moe import TrieMoERecommender
from src.utils.metrics import compute_metrics


# ============================================================================
# COLD-START EVALUATION PROTOCOL
# ============================================================================

class ColdStartEvaluator:
    """
    Strict cold-start evaluation protocol.

    Protocol A: Frequency-based Holdout
    - Compute token frequencies on training set
    - Categorize test samples by max feature frequency
    - Report metrics separately for each frequency bucket

    Protocol B: Zero-shot Items (Unseen tokens)
    - Identify tokens that appear only in test set
    - Evaluate on samples containing at least one unseen token
    """

    def __init__(
        self,
        train_freq: Dict[str, Counter],
        sparse_fields: List[str],
        cold_threshold: int = 5,      # <=5 occurrences = cold
        warm_threshold: int = 50,     # 6-50 = warm
        # >50 = hot
    ):
        self.train_freq = train_freq
        self.sparse_fields = sparse_fields
        self.cold_threshold = cold_threshold
        self.warm_threshold = warm_threshold

    def categorize_sample(self, sparse_indices: np.ndarray, vocab_maps: Dict) -> str:
        """
        Categorize a sample based on feature frequency.
        Returns: 'cold', 'warm', or 'hot'
        """
        max_freq = 0
        min_freq = float('inf')
        has_unseen = False

        for i, field in enumerate(self.sparse_fields):
            if field not in self.train_freq:
                continue

            token_idx = int(sparse_indices[i])
            token_str = vocab_maps.get(field, {}).get(token_idx, str(token_idx))
            freq = self.train_freq[field].get(token_str, 0)

            if freq == 0:
                has_unseen = True
            max_freq = max(max_freq, freq)
            min_freq = min(min_freq, freq) if freq > 0 else min_freq

        # Use minimum frequency for conservative cold-start detection
        effective_freq = min_freq if min_freq != float('inf') else 0

        if has_unseen or effective_freq <= self.cold_threshold:
            return 'cold'
        elif effective_freq <= self.warm_threshold:
            return 'warm'
        else:
            return 'hot'

    def evaluate_by_bucket(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        sparse_data: np.ndarray,
        vocab_maps: Dict,
    ) -> Dict[str, Dict]:
        """
        Evaluate predictions by frequency bucket.

        Returns:
            Dict with 'cold', 'warm', 'hot', 'overall' keys
        """
        buckets = {'cold': [], 'warm': [], 'hot': []}
        bucket_labels = {'cold': [], 'warm': [], 'hot': []}

        for i in range(len(predictions)):
            bucket = self.categorize_sample(sparse_data[i], vocab_maps)
            buckets[bucket].append(predictions[i])
            bucket_labels[bucket].append(labels[i])

        results = {}
        for bucket_name in ['cold', 'warm', 'hot']:
            preds = np.array(buckets[bucket_name])
            lbls = np.array(bucket_labels[bucket_name])

            if len(preds) >= 50:  # Minimum sample size for reliable metrics
                metrics = compute_metrics(preds, lbls)
                results[bucket_name] = {
                    'count': len(preds),
                    'auc': metrics['auc'],
                    'logloss': metrics['logloss'],
                    'positive_rate': float(np.mean(lbls)),
                }
            else:
                results[bucket_name] = {
                    'count': len(preds),
                    'auc': None,
                    'logloss': None,
                    'positive_rate': float(np.mean(lbls)) if len(lbls) > 0 else None,
                }

        # Overall
        overall_metrics = compute_metrics(predictions, labels)
        results['overall'] = {
            'count': len(predictions),
            'auc': overall_metrics['auc'],
            'logloss': overall_metrics['logloss'],
            'positive_rate': float(np.mean(labels)),
        }

        return results


# ============================================================================
# PLE MODEL IMPLEMENTATION
# ============================================================================

class PLELayer(nn.Module):
    """Progressive Layered Extraction (PLE) layer."""

    def __init__(
        self,
        input_dim: int,
        expert_dim: int,
        num_shared_experts: int = 2,
        num_task_experts: int = 2,
        num_tasks: int = 1,
    ):
        super().__init__()
        self.num_tasks = num_tasks
        self.num_shared_experts = num_shared_experts
        self.num_task_experts = num_task_experts

        # Shared experts
        self.shared_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, expert_dim),
                nn.ReLU(),
            )
            for _ in range(num_shared_experts)
        ])

        # Task-specific experts
        self.task_experts = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(
                    nn.Linear(input_dim, expert_dim),
                    nn.ReLU(),
                )
                for _ in range(num_task_experts)
            ])
            for _ in range(num_tasks)
        ])

        # Gating networks
        total_experts = num_shared_experts + num_task_experts
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, total_experts),
                nn.Softmax(dim=-1),
            )
            for _ in range(num_tasks)
        ])

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        # Shared expert outputs
        shared_outputs = [expert(x) for expert in self.shared_experts]

        task_outputs = []
        for task_idx in range(self.num_tasks):
            # Task-specific expert outputs
            task_expert_outputs = [expert(x) for expert in self.task_experts[task_idx]]

            # Combine all expert outputs
            all_outputs = shared_outputs + task_expert_outputs
            expert_stack = torch.stack(all_outputs, dim=1)  # (batch, num_experts, expert_dim)

            # Gating
            gate_weights = self.gates[task_idx](x)  # (batch, num_experts)
            gate_weights = gate_weights.unsqueeze(-1)  # (batch, num_experts, 1)

            # Weighted sum
            task_output = (expert_stack * gate_weights).sum(dim=1)  # (batch, expert_dim)
            task_outputs.append(task_output)

        return task_outputs


class PLE(nn.Module):
    """
    Progressive Layered Extraction Model.

    Reference: Tang et al., "Progressive Layered Extraction (PLE):
    A Novel Multi-Task Learning (MTL) Model for Personalized Recommendations",
    RecSys 2020.
    """

    def __init__(
        self,
        dense_dim: int,
        sparse_dims: Dict[str, int],
        embedding_dim: int = 16,
        num_ple_layers: int = 2,
        expert_dim: int = 64,
        num_shared_experts: int = 2,
        num_task_experts: int = 2,
        tower_dims: List[int] = [64, 32],
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_sparse_fields = len(sparse_dims)

        # Embedding layer
        self.embeddings = nn.ModuleDict({
            field: nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
            for field, vocab_size in sparse_dims.items()
        })
        self.field_names = list(sparse_dims.keys())

        # Dense normalization
        self.dense_bn = nn.BatchNorm1d(dense_dim)

        # Input dimension
        input_dim = dense_dim + self.num_sparse_fields * embedding_dim

        # PLE layers
        self.ple_layers = nn.ModuleList()
        current_dim = input_dim
        for _ in range(num_ple_layers):
            layer = PLELayer(
                input_dim=current_dim,
                expert_dim=expert_dim,
                num_shared_experts=num_shared_experts,
                num_task_experts=num_task_experts,
                num_tasks=1,  # Single task (CTR)
            )
            self.ple_layers.append(layer)
            current_dim = expert_dim

        # Task tower
        tower_layers = []
        prev_dim = expert_dim
        for hidden_dim in tower_dims:
            tower_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        tower_layers.append(nn.Linear(prev_dim, 1))
        self.tower = nn.Sequential(*tower_layers)

    def forward(self, dense: torch.Tensor, sparse: torch.Tensor, **kwargs) -> Dict:
        batch_size = dense.shape[0]

        # Dense features
        dense = self.dense_bn(dense)

        # Sparse embeddings
        emb_list = []
        for i, field in enumerate(self.field_names):
            emb_list.append(self.embeddings[field](sparse[:, i]))
        sparse_emb = torch.cat(emb_list, dim=1)  # (batch, num_fields * emb_dim)

        # Concatenate
        x = torch.cat([dense, sparse_emb], dim=1)

        # PLE layers
        for ple_layer in self.ple_layers:
            outputs = ple_layer(x)
            x = outputs[0]  # Single task

        # Tower
        logits = self.tower(x).squeeze(-1)

        return {'logits': logits}


# ============================================================================
# TRAINING AND EVALUATION
# ============================================================================

def build_trie(dataset, sparse_fields, num_experts=8, strategy='frequency_aware'):
    """Build Trie with specified parameters."""
    builder = TrieBuilder(
        hierarchy_config=['frequency', 'info'],
        num_experts=num_experts,
        expert_strategy=strategy,
    )
    for field in sparse_fields:
        if field in dataset.freq_stats:
            freq_dict = dataset.freq_stats[field]
            for token, count in freq_dict.items():
                for _ in range(min(count, 100)):
                    builder.statistics.update(field, token, label=1)
    builder.statistics.compute_statistics()
    for field in sparse_fields:
        if field in builder.statistics.statistics:
            trie = StatisticalTrie(field_name=field, hierarchy_config=['frequency', 'info'])
            trie.build(builder.statistics.statistics[field])
            trie.assign_experts(num_experts, strategy)
            builder.tries[field] = trie
    return builder


def train_epoch(model, train_loader, optimizer, criterion, device,
                trie_encoder=None, sparse_fields=None, vocab_maps=None):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0

    for batch in train_loader:
        optimizer.zero_grad()
        dense = batch['dense'].to(device)
        sparse = batch['sparse'].to(device)
        labels = batch['label'].to(device)

        if trie_encoder is not None:
            trie_features = trie_encoder(sparse, sparse_fields, vocab_maps)
            output_dict = model(dense, sparse, trie_routing_vec=trie_features)
        else:
            output_dict = model(dense, sparse)

        outputs = output_dict['logits'] if isinstance(output_dict, dict) else output_dict
        loss = criterion(outputs.squeeze(), labels.float())
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def evaluate(model, data_loader, device, trie_encoder=None,
             sparse_fields=None, vocab_maps=None):
    """Evaluate model on a dataset."""
    model.eval()
    all_preds = []
    all_labels = []
    all_sparse = []

    with torch.no_grad():
        for batch in data_loader:
            dense = batch['dense'].to(device)
            sparse = batch['sparse'].to(device)

            if trie_encoder is not None:
                trie_features = trie_encoder(sparse, sparse_fields, vocab_maps)
                output_dict = model(dense, sparse, trie_routing_vec=trie_features)
            else:
                output_dict = model(dense, sparse)

            outputs = output_dict['logits'] if isinstance(output_dict, dict) else output_dict
            preds = torch.sigmoid(outputs).cpu().numpy()

            all_preds.extend(preds.flatten())
            all_labels.extend(batch['label'].numpy().flatten())
            all_sparse.append(batch['sparse'].numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_sparse = np.vstack(all_sparse)

    metrics = compute_metrics(all_preds, all_labels)

    return {
        'auc': metrics['auc'],
        'logloss': metrics['logloss'],
        'predictions': all_preds,
        'labels': all_labels,
        'sparse_data': all_sparse,
    }


def run_single_experiment(
    model_class,
    model_kwargs,
    train_loader,
    val_loader,
    test_loader,
    device,
    epochs=10,
    lr=0.001,
    is_moe=False,
    trie_builder=None,
    sparse_fields=None,
    vocab_maps=None,
    vocab_sizes=None,
):
    """Run a single training experiment."""
    # Create model
    model = model_class(**model_kwargs).to(device)

    # Setup Trie encoder for MoE
    trie_encoder = None
    if is_moe and trie_builder is not None:
        trie_encoder = FastTrieEncoder(
            tries=trie_builder.tries,
            vocab_sizes=vocab_sizes,
        ).to(device)

    # Training setup
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_auc = 0
    best_model_state = None
    train_start = time.time()

    for epoch in range(epochs):
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device,
            trie_encoder, sparse_fields, vocab_maps
        )

        val_result = evaluate(
            model, val_loader, device,
            trie_encoder, sparse_fields, vocab_maps
        )

        if val_result['auc'] > best_val_auc:
            best_val_auc = val_result['auc']
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        scheduler.step()

    train_time = time.time() - train_start

    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)

    # Final test evaluation
    test_result = evaluate(
        model, test_loader, device,
        trie_encoder, sparse_fields, vocab_maps
    )

    # Measure inference latency
    model.eval()
    sample_batch = next(iter(test_loader))
    dense = sample_batch['dense'][:100].to(device)
    sparse = sample_batch['sparse'][:100].to(device)

    # Warmup
    for _ in range(10):
        with torch.no_grad():
            if trie_encoder:
                trie_features = trie_encoder(sparse, sparse_fields, vocab_maps)
                model(dense, sparse, trie_routing_vec=trie_features)
            else:
                model(dense, sparse)

    # Measure
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    latency_start = time.time()
    for _ in range(100):
        with torch.no_grad():
            if trie_encoder:
                trie_features = trie_encoder(sparse, sparse_fields, vocab_maps)
                model(dense, sparse, trie_routing_vec=trie_features)
            else:
                model(dense, sparse)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    inference_latency = (time.time() - latency_start) / 100 * 1000  # ms per batch

    return {
        'test_auc': test_result['auc'],
        'test_logloss': test_result['logloss'],
        'best_val_auc': best_val_auc,
        'train_time': train_time,
        'inference_latency_ms': inference_latency,
        'predictions': test_result['predictions'],
        'labels': test_result['labels'],
        'sparse_data': test_result['sparse_data'],
    }


# ============================================================================
# STATISTICAL ANALYSIS
# ============================================================================

def compute_statistical_tests(all_results: Dict, baseline='DeepFM', alpha=0.05):
    """
    Compute comprehensive statistical tests.

    Tests:
    1. Paired t-test (parametric)
    2. Wilcoxon signed-rank test (non-parametric)
    3. Effect size (Cohen's d)
    4. 95% confidence interval
    """
    if baseline not in all_results:
        baseline = list(all_results.keys())[0]

    baseline_aucs = all_results[baseline]['all_aucs']
    baseline_losses = all_results[baseline]['all_losses']

    stat_results = {}

    for model_name, results in all_results.items():
        if model_name == baseline:
            continue

        model_aucs = results['all_aucs']
        model_losses = results['all_losses']

        # AUC comparison
        auc_diff = np.array(model_aucs) - np.array(baseline_aucs)

        # Paired t-test
        t_stat_auc, p_value_auc = stats.ttest_rel(model_aucs, baseline_aucs)

        # Wilcoxon signed-rank test
        try:
            w_stat_auc, p_wilcox_auc = stats.wilcoxon(model_aucs, baseline_aucs)
        except ValueError:
            w_stat_auc, p_wilcox_auc = None, None

        # Cohen's d (effect size)
        pooled_std = np.sqrt((np.std(model_aucs)**2 + np.std(baseline_aucs)**2) / 2)
        cohens_d_auc = (np.mean(model_aucs) - np.mean(baseline_aucs)) / pooled_std if pooled_std > 0 else 0

        # 95% CI for difference
        ci_auc = stats.t.interval(0.95, len(auc_diff)-1, loc=np.mean(auc_diff), scale=stats.sem(auc_diff))

        # LogLoss comparison
        loss_diff = np.array(model_losses) - np.array(baseline_losses)
        t_stat_loss, p_value_loss = stats.ttest_rel(model_losses, baseline_losses)

        stat_results[model_name] = {
            'auc': {
                'mean_diff': float(np.mean(auc_diff)),
                't_statistic': float(t_stat_auc),
                'p_value_ttest': float(p_value_auc),
                'p_value_wilcoxon': float(p_wilcox_auc) if p_wilcox_auc else None,
                'cohens_d': float(cohens_d_auc),
                'ci_95_lower': float(ci_auc[0]),
                'ci_95_upper': float(ci_auc[1]),
                'significant_ttest': p_value_auc < alpha,
                'significant_wilcoxon': p_wilcox_auc < alpha if p_wilcox_auc else None,
            },
            'logloss': {
                'mean_diff': float(np.mean(loss_diff)),
                't_statistic': float(t_stat_loss),
                'p_value_ttest': float(p_value_loss),
                'significant': p_value_loss < alpha,
            }
        }

    return stat_results


def generate_latex_table(all_results: Dict, stat_results: Dict, baseline='DeepFM'):
    """Generate publication-ready LaTeX table."""
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Model comparison results. Best results in \\textbf{bold}. $\\dagger$ indicates statistically significant improvement over baseline (p<0.05).}",
        "\\label{tab:main_results}",
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        "Model & AUC & LogLoss & Train(s) & Latency(ms) \\\\",
        "\\midrule",
    ]

    # Find best AUC and LogLoss
    best_auc = max(r['test_auc_mean'] for r in all_results.values())
    best_loss = min(r['test_logloss_mean'] for r in all_results.values())

    for model_name, results in all_results.items():
        auc_mean = results['test_auc_mean']
        auc_std = results['test_auc_std']
        loss_mean = results['test_logloss_mean']
        train_time = results.get('train_time_mean', 0)
        latency = results.get('inference_latency_mean', 0)

        # Check significance
        sig_marker = ""
        if model_name in stat_results:
            if stat_results[model_name]['auc']['significant_ttest']:
                if stat_results[model_name]['auc']['mean_diff'] > 0:
                    sig_marker = "$^\\dagger$"

        # Format with bold for best
        auc_str = f"{auc_mean:.4f}$\\pm${auc_std:.4f}{sig_marker}"
        if abs(auc_mean - best_auc) < 0.0001:
            auc_str = f"\\textbf{{{auc_str}}}"

        loss_str = f"{loss_mean:.4f}"
        if abs(loss_mean - best_loss) < 0.0001:
            loss_str = f"\\textbf{{{loss_str}}}"

        lines.append(f"{model_name} & {auc_str} & {loss_str} & {train_time:.1f} & {latency:.2f} \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])

    return "\n".join(lines)


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Rigorous Paper Experiments')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_runs', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--embed_dim', type=int, default=32)
    parser.add_argument('--dataset', type=str, default='criteo', choices=['criteo', 'amazon', 'both'])
    parser.add_argument('--sample_size', type=int, default=None, help='Sample size for quick testing')
    parser.add_argument('--output_dir', type=str, default='results/rigorous_experiments')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    print("=" * 70)
    print("RIGOROUS PAPER EXPERIMENTS FOR TOP-TIER CONFERENCE")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Number of runs: {args.num_runs}")
    print(f"Epochs per run: {args.epochs}")
    print(f"Dataset: {args.dataset}")
    print(f"Output: {output_dir}")
    print("=" * 70)

    # Define models
    models_config = {
        'DeepFM': {'class': DeepFM, 'is_moe': False},
        'DLRM': {'class': DLRM, 'is_moe': False},
        'DCNv2': {'class': DCNv2, 'is_moe': False},
        'DCNv3': {'class': DCNv3, 'is_moe': False},
        'AutoInt': {'class': AutoInt, 'is_moe': False},
        'FinalMLP': {'class': FinalMLP, 'is_moe': False},
        'PLE': {'class': PLE, 'is_moe': False},
        'Trie-MoE': {'class': TrieMoERecommender, 'is_moe': True},
    }

    datasets_to_run = []
    if args.dataset in ['criteo', 'both']:
        datasets_to_run.append('criteo')
    if args.dataset in ['amazon', 'both']:
        datasets_to_run.append('amazon')

    all_dataset_results = {}

    for dataset_name in datasets_to_run:
        print(f"\n{'='*70}")
        print(f"DATASET: {dataset_name.upper()}")
        print(f"{'='*70}")

        # Load data
        if dataset_name == 'criteo':
            data_path = 'data/criteo/criteo_5m.parquet'
            train_dataset, val_dataset, test_dataset = load_criteo_data(
                data_path,
                sample_size=args.sample_size,
            )
            sparse_fields = SPARSE_COLS
        else:  # amazon
            train_dataset, val_dataset, test_dataset = load_amazon_data(
                reviews_path='data/amazon/electronics_encrypted.parquet',
                mode='encrypted',
            )
            sparse_fields = train_dataset.actual_sparse_cols

        # Prepare data loaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True
        )

        # Feature dimensions
        feature_dims = train_dataset.get_feature_dims()
        vocab_maps = {f: {i: t for t, i in v.items()} for f, v in train_dataset.vocab.items()}
        vocab_sizes = {f: len(v) for f, v in train_dataset.vocab.items()}

        # Build Trie for MoE
        print("Building Trie structure...")
        trie_builder = build_trie(train_dataset, sparse_fields, num_experts=8)

        # Cold-start evaluator
        cold_start_evaluator = ColdStartEvaluator(
            train_freq=train_dataset.freq_stats,
            sparse_fields=sparse_fields,
        )

        # Results storage
        all_results = {}
        cold_start_results = {}

        # Run experiments for each model
        for model_name, config in models_config.items():
            print(f"\n{'='*50}")
            print(f"Model: {model_name}")
            print(f"{'='*50}")

            model_aucs = []
            model_losses = []
            train_times = []
            latencies = []
            all_predictions = []
            all_labels = []
            all_sparse_data = []

            for run_idx in range(args.num_runs):
                # Set seed for reproducibility
                seed = 42 + run_idx
                torch.manual_seed(seed)
                np.random.seed(seed)

                # Model kwargs
                model_kwargs = {
                    'dense_dim': feature_dims['dense_dim'],
                    'sparse_dims': feature_dims['sparse_dims'],
                    'embedding_dim': args.embed_dim,
                    'dropout': 0.1,
                }

                # Model-specific kwargs
                if config['is_moe']:
                    model_kwargs.update({
                        'num_experts': 8,
                        'top_k': 1,
                        'routing_mode': 'learned_only',
                    })

                # Run experiment
                result = run_single_experiment(
                    model_class=config['class'],
                    model_kwargs=model_kwargs,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    test_loader=test_loader,
                    device=device,
                    epochs=args.epochs,
                    lr=args.lr,
                    is_moe=config['is_moe'],
                    trie_builder=trie_builder if config['is_moe'] else None,
                    sparse_fields=sparse_fields if config['is_moe'] else None,
                    vocab_maps=vocab_maps if config['is_moe'] else None,
                    vocab_sizes=vocab_sizes if config['is_moe'] else None,
                )

                model_aucs.append(result['test_auc'])
                model_losses.append(result['test_logloss'])
                train_times.append(result['train_time'])
                latencies.append(result['inference_latency_ms'])

                if run_idx == 0:  # Save predictions from first run for cold-start analysis
                    all_predictions = result['predictions']
                    all_labels = result['labels']
                    all_sparse_data = result['sparse_data']

                print(f"  Run {run_idx+1}/{args.num_runs}: AUC={result['test_auc']:.4f}, "
                      f"LogLoss={result['test_logloss']:.4f}, Time={result['train_time']:.1f}s")

            # Aggregate results
            all_results[model_name] = {
                'all_aucs': model_aucs,
                'all_losses': model_losses,
                'test_auc_mean': float(np.mean(model_aucs)),
                'test_auc_std': float(np.std(model_aucs)),
                'test_logloss_mean': float(np.mean(model_losses)),
                'test_logloss_std': float(np.std(model_losses)),
                'train_time_mean': float(np.mean(train_times)),
                'train_time_std': float(np.std(train_times)),
                'inference_latency_mean': float(np.mean(latencies)),
                'inference_latency_std': float(np.std(latencies)),
            }

            # Cold-start evaluation (using first run)
            if len(all_predictions) > 0:
                cold_start_results[model_name] = cold_start_evaluator.evaluate_by_bucket(
                    all_predictions, all_labels, all_sparse_data, vocab_maps
                )

            print(f"\n  Summary: AUC={all_results[model_name]['test_auc_mean']:.4f}±"
                  f"{all_results[model_name]['test_auc_std']:.4f}")

        # Statistical tests
        print(f"\n{'='*50}")
        print("STATISTICAL SIGNIFICANCE TESTS")
        print(f"{'='*50}")

        stat_results = compute_statistical_tests(all_results, baseline='DeepFM')

        for model_name, stats_dict in stat_results.items():
            print(f"\n{model_name} vs DeepFM:")
            print(f"  AUC diff: {stats_dict['auc']['mean_diff']:.4f} "
                  f"(p={stats_dict['auc']['p_value_ttest']:.4f}, "
                  f"Cohen's d={stats_dict['auc']['cohens_d']:.4f})")
            print(f"  95% CI: [{stats_dict['auc']['ci_95_lower']:.4f}, "
                  f"{stats_dict['auc']['ci_95_upper']:.4f}]")
            print(f"  Significant: {stats_dict['auc']['significant_ttest']}")

        # Cold-start results
        print(f"\n{'='*50}")
        print("COLD-START EVALUATION")
        print(f"{'='*50}")

        for model_name, cs_results in cold_start_results.items():
            print(f"\n{model_name}:")
            for bucket in ['cold', 'warm', 'hot', 'overall']:
                if bucket in cs_results and cs_results[bucket]['auc'] is not None:
                    print(f"  {bucket:8s}: n={cs_results[bucket]['count']:6d}, "
                          f"AUC={cs_results[bucket]['auc']:.4f}, "
                          f"LogLoss={cs_results[bucket]['logloss']:.4f}")

        # Save results
        all_dataset_results[dataset_name] = {
            'results': {k: {kk: vv for kk, vv in v.items() if not kk.startswith('all_')}
                       for k, v in all_results.items()},
            'statistical_tests': stat_results,
            'cold_start': cold_start_results,
        }

        # Generate LaTeX table
        latex_table = generate_latex_table(all_results, stat_results)
        with open(output_dir / f'{dataset_name}_table_{timestamp}.tex', 'w') as f:
            f.write(latex_table)

    # Save comprehensive results
    final_output = {
        'timestamp': timestamp,
        'config': {
            'num_runs': args.num_runs,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'embed_dim': args.embed_dim,
            'device': str(device),
        },
        'datasets': all_dataset_results,
    }

    with open(output_dir / f'full_results_{timestamp}.json', 'w') as f:
        json.dump(final_output, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*70}")
    print(f"Results saved to: {output_dir}")
    print(f"  - full_results_{timestamp}.json")
    for ds in datasets_to_run:
        print(f"  - {ds}_table_{timestamp}.tex")


if __name__ == '__main__':
    main()
