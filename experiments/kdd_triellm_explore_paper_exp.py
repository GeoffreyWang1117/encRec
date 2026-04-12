#!/usr/bin/env python3
"""
Comprehensive Paper Experiments for Trie-MoE.

Includes:
1. Multi-dataset evaluation (Criteo + Amazon)
2. Statistical significance tests
3. Expert utilization analysis
4. Cold-start performance
5. Scalability analysis
6. Ablation studies
7. Visualization generation
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
from collections import defaultdict
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

from src.data.amazon_loader import load_amazon_data
from src.data.criteo_loader import load_criteo_data
from src.trie.builder import TrieBuilder, StatisticalTrie
from src.trie.fast_encoder import FastTrieEncoder
from src.models.backbone import DeepFM, DLRM, DCNv2, AutoInt
from src.models.moe import TrieMoERecommender
from src.utils.metrics import compute_metrics, AUCMeter

# Set style for paper figures
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['figure.figsize'] = (8, 6)
plt.rcParams['figure.dpi'] = 150


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


def train_and_evaluate(model, train_loader, val_loader, test_loader, device,
                       epochs=5, lr=0.0005, trie_encoder=None, sparse_fields=None,
                       vocab_maps=None, track_experts=False):
    """Train and evaluate model, optionally tracking expert utilization."""
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    best_val_auc = 0
    best_model_state = None
    expert_usage = defaultdict(int) if track_experts else None

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            dense = batch['dense'].to(device)
            sparse = batch['sparse'].to(device)
            labels = batch['label'].to(device)

            if trie_encoder is not None:
                trie_features = trie_encoder(sparse, sparse_fields, vocab_maps)
                output_dict = model(dense, sparse, trie_routing_vec=trie_features)

                # Track expert usage
                if track_experts and hasattr(model, 'last_expert_indices'):
                    for idx in model.last_expert_indices.cpu().numpy().flatten():
                        expert_usage[int(idx)] += 1
            else:
                output_dict = model(dense, sparse)

            outputs = output_dict['logits'] if isinstance(output_dict, dict) else output_dict
            loss = criterion(outputs.squeeze(), labels.float())
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                dense = batch['dense'].to(device)
                sparse = batch['sparse'].to(device)
                if trie_encoder is not None:
                    trie_features = trie_encoder(sparse, sparse_fields, vocab_maps)
                    output_dict = model(dense, sparse, trie_routing_vec=trie_features)
                else:
                    output_dict = model(dense, sparse)
                outputs = output_dict['logits'] if isinstance(output_dict, dict) else output_dict
                val_preds.extend(torch.sigmoid(outputs).cpu().numpy())
                val_labels.extend(batch['label'].numpy())

        val_metrics = compute_metrics(np.array(val_preds), np.array(val_labels))
        if val_metrics['auc'] > best_val_auc:
            best_val_auc = val_metrics['auc']
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Load best model and test
    if best_model_state:
        model.load_state_dict(best_model_state)

    model.eval()
    test_preds, test_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            dense = batch['dense'].to(device)
            sparse = batch['sparse'].to(device)
            if trie_encoder is not None:
                trie_features = trie_encoder(sparse, sparse_fields, vocab_maps)
                output_dict = model(dense, sparse, trie_routing_vec=trie_features)
            else:
                output_dict = model(dense, sparse)
            outputs = output_dict['logits'] if isinstance(output_dict, dict) else output_dict
            test_preds.extend(torch.sigmoid(outputs).cpu().numpy())
            test_labels.extend(batch['label'].numpy())

    test_metrics = compute_metrics(np.array(test_preds), np.array(test_labels))

    return {
        'test_auc': test_metrics['auc'],
        'test_logloss': test_metrics['logloss'],
        'best_val_auc': best_val_auc,
        'expert_usage': dict(expert_usage) if expert_usage else None,
        'predictions': np.array(test_preds),
        'labels': np.array(test_labels),
    }


def run_statistical_tests(results_dict, baseline='DeepFM', alpha=0.05):
    """Run statistical significance tests comparing models."""
    baseline_aucs = results_dict[baseline]['all_aucs']

    test_results = {}
    for model_name, results in results_dict.items():
        if model_name == baseline:
            continue

        model_aucs = results['all_aucs']

        # Paired t-test
        t_stat, p_value = stats.ttest_rel(baseline_aucs, model_aucs)

        # Effect size (Cohen's d)
        diff = np.array(baseline_aucs) - np.array(model_aucs)
        cohens_d = np.mean(diff) / np.std(diff) if np.std(diff) > 0 else 0

        # 95% CI for difference
        ci = stats.t.interval(0.95, len(diff)-1, loc=np.mean(diff), scale=stats.sem(diff))

        test_results[model_name] = {
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < alpha,
            'cohens_d': cohens_d,
            'ci_lower': ci[0],
            'ci_upper': ci[1],
            'mean_diff': np.mean(diff),
        }

    return test_results


def analyze_cold_start(model, test_loader, trie_builder, device, trie_encoder=None,
                       sparse_fields=None, vocab_maps=None, freq_threshold=10):
    """Analyze model performance on cold-start (rare) items."""
    model.eval()

    # Categorize samples by frequency
    head_preds, head_labels = [], []
    mid_preds, mid_labels = [], []
    tail_preds, tail_labels = [], []

    with torch.no_grad():
        for batch in test_loader:
            dense = batch['dense'].to(device)
            sparse = batch['sparse'].to(device)
            labels = batch['label'].numpy()

            if trie_encoder is not None:
                trie_features = trie_encoder(sparse, sparse_fields, vocab_maps)
                outputs = model(dense, sparse, trie_routing_vec=trie_features)
            else:
                outputs = model(dense, sparse)

            preds = torch.sigmoid(outputs).cpu().numpy()

            # Categorize each sample
            for i in range(len(labels)):
                # Check frequency of sparse features
                is_cold = False
                max_freq = 0

                for j, field in enumerate(sparse_fields):
                    if field in trie_builder.tries:
                        trie = trie_builder.tries[field]
                        token_str = str(sparse[i, j].item())
                        node = trie.token_to_node.get(token_str)
                        if node and hasattr(node, 'stats'):
                            freq = node.stats.get('count', 0)
                            max_freq = max(max_freq, freq)

                if max_freq < freq_threshold:
                    tail_preds.append(preds[i])
                    tail_labels.append(labels[i])
                elif max_freq < freq_threshold * 10:
                    mid_preds.append(preds[i])
                    mid_labels.append(labels[i])
                else:
                    head_preds.append(preds[i])
                    head_labels.append(labels[i])

    results = {}
    for name, preds, labels in [('head', head_preds, head_labels),
                                 ('mid', mid_preds, mid_labels),
                                 ('tail', tail_preds, tail_labels)]:
        if len(preds) > 0:
            metrics = compute_metrics(np.array(preds), np.array(labels))
            results[name] = {
                'count': len(preds),
                'auc': metrics['auc'],
                'logloss': metrics['logloss'],
            }
        else:
            results[name] = {'count': 0, 'auc': None, 'logloss': None}

    return results


def plot_expert_utilization(expert_usage, num_experts, save_path):
    """Plot expert utilization distribution."""
    fig, ax = plt.subplots(figsize=(10, 6))

    experts = list(range(num_experts))
    counts = [expert_usage.get(e, 0) for e in experts]
    total = sum(counts)
    percentages = [c / total * 100 if total > 0 else 0 for c in counts]

    bars = ax.bar(experts, percentages, color='steelblue', edgecolor='black')

    # Add value labels
    for bar, pct in zip(bars, percentages):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{pct:.1f}%', ha='center', va='bottom', fontsize=10)

    ax.set_xlabel('Expert ID')
    ax.set_ylabel('Utilization (%)')
    ax.set_title('Expert Utilization Distribution')
    ax.set_xticks(experts)

    # Add ideal uniform line
    ideal = 100 / num_experts
    ax.axhline(y=ideal, color='red', linestyle='--', label=f'Ideal ({ideal:.1f}%)')
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Calculate entropy (load balance metric)
    probs = np.array(percentages) / 100
    probs = probs[probs > 0]
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    max_entropy = np.log(num_experts)

    return entropy / max_entropy  # Normalized entropy (1.0 = perfect balance)


def plot_model_comparison(results_dict, metric='auc', save_path=None):
    """Plot model comparison bar chart with error bars."""
    fig, ax = plt.subplots(figsize=(10, 6))

    models = list(results_dict.keys())
    means = [results_dict[m][f'test_{metric}_mean'] for m in models]
    stds = [results_dict[m][f'test_{metric}_std'] for m in models]

    colors = ['#2ecc71' if m == 'Trie-MoE' else '#3498db' for m in models]

    bars = ax.bar(models, means, yerr=stds, capsize=5, color=colors, edgecolor='black')

    ax.set_ylabel(f'Test {metric.upper()}')
    ax.set_title(f'Model Comparison - {metric.upper()}')

    # Add value labels
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.001,
                f'{mean:.4f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_cold_start_analysis(cold_start_results, save_path):
    """Plot cold-start performance comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    models = list(cold_start_results.keys())
    buckets = ['head', 'mid', 'tail']

    # AUC plot
    x = np.arange(len(buckets))
    width = 0.15

    for i, model in enumerate(models):
        aucs = [cold_start_results[model].get(b, {}).get('auc', 0) or 0 for b in buckets]
        axes[0].bar(x + i * width, aucs, width, label=model)

    axes[0].set_xlabel('Frequency Bucket')
    axes[0].set_ylabel('AUC')
    axes[0].set_title('AUC by Frequency Bucket (Cold-Start Analysis)')
    axes[0].set_xticks(x + width * (len(models) - 1) / 2)
    axes[0].set_xticklabels(['Head\n(frequent)', 'Mid', 'Tail\n(rare)'])
    axes[0].legend()

    # Sample count
    for i, model in enumerate(models):
        counts = [cold_start_results[model].get(b, {}).get('count', 0) for b in buckets]
        axes[1].bar(x + i * width, counts, width, label=model)

    axes[1].set_xlabel('Frequency Bucket')
    axes[1].set_ylabel('Sample Count')
    axes[1].set_title('Sample Distribution by Bucket')
    axes[1].set_xticks(x + width * (len(models) - 1) / 2)
    axes[1].set_xticklabels(['Head', 'Mid', 'Tail'])
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def run_scalability_experiment(dataset_loader, device, sample_sizes=[10000, 50000, 100000, 200000]):
    """Run scalability experiment with varying data sizes."""
    results = []

    for size in sample_sizes:
        print(f"\nRunning with {size} samples...")

        train_dataset, val_dataset, test_dataset = dataset_loader(sample_size=size)

        # Quick training run
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1024, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1024)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1024)

        sparse_fields = train_dataset.actual_sparse_cols
        feature_dims = train_dataset.get_feature_dims()

        # Build Trie
        trie_builder = build_trie(train_dataset, sparse_fields)

        # Create model
        model = TrieMoERecommender(
            dense_dim=feature_dims['dense_dim'],
            sparse_dims=feature_dims['sparse_dims'],
            embedding_dim=32,
            num_experts=8,
            top_k=1,
        ).to(device)

        vocab_maps = {f: {i: t for t, i in v.items()} for f, v in train_dataset.vocab.items()}
        vocab_sizes = {f: len(v) for f, v in train_dataset.vocab.items()}

        trie_encoder = FastTrieEncoder(
            tries=trie_builder.tries,
            vocab_sizes=vocab_sizes,
        ).to(device)

        # Measure training time
        start_time = time.time()
        result = train_and_evaluate(
            model, train_loader, val_loader, test_loader, device,
            epochs=3, trie_encoder=trie_encoder, sparse_fields=sparse_fields,
            vocab_maps=vocab_maps
        )
        train_time = time.time() - start_time

        results.append({
            'sample_size': size,
            'train_time': train_time,
            'test_auc': result['test_auc'],
            'test_logloss': result['test_logloss'],
        })

    return results


def generate_latex_table(results_dict, caption="Model Comparison Results"):
    """Generate LaTeX table for paper."""
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Model & AUC & LogLoss & Train Time (s) & Latency (ms) \\\\",
        "\\midrule",
    ]

    for model, results in results_dict.items():
        auc = f"{results['test_auc_mean']:.4f}$\\pm${results['test_auc_std']:.4f}"
        logloss = f"{results['test_logloss_mean']:.4f}"
        train_time = f"{results.get('train_time_mean', 0):.1f}"
        latency = f"{results.get('inference_latency_mean', 0):.2f}"

        if model == 'Trie-MoE':
            lines.append(f"\\textbf{{{model}}} & {auc} & {logloss} & {train_time} & {latency} \\\\")
        else:
            lines.append(f"{model} & {auc} & {logloss} & {train_time} & {latency} \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\label{tab:comparison}",
        "\\end{table}",
    ])

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description='Paper Experiments')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_runs', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--output_dir', type=str, default='logs/paper_experiments')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Output: {output_dir}")

    # ========== Experiment 1: Multi-run comparison on Amazon ==========
    print("\n" + "="*60)
    print("Experiment 1: Multi-run Baseline Comparison (Amazon)")
    print("="*60)

    train_dataset, val_dataset, test_dataset = load_amazon_data(
        reviews_path='data/amazon/electronics_encrypted.parquet',
        mode='encrypted',
    )

    sparse_fields = train_dataset.actual_sparse_cols
    feature_dims = train_dataset.get_feature_dims()
    vocab_maps = {f: {i: t for t, i in v.items()} for f, v in train_dataset.vocab.items()}
    vocab_sizes = {f: len(v) for f, v in train_dataset.vocab.items()}

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1024, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1024, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1024, num_workers=4)

    trie_builder = build_trie(train_dataset, sparse_fields, num_experts=8)

    models_config = {
        'DeepFM': {'class': DeepFM, 'is_moe': False},
        'DLRM': {'class': DLRM, 'is_moe': False},
        'DCNv2': {'class': DCNv2, 'is_moe': False},
        'Trie-MoE': {'class': TrieMoERecommender, 'is_moe': True},
    }

    all_results = {}
    expert_usage_all = {}

    for model_name, config in models_config.items():
        print(f"\nTraining {model_name}...")
        model_aucs = []
        model_losses = []
        train_times = []

        for run in range(args.num_runs):
            torch.manual_seed(run)
            np.random.seed(run)

            if config['is_moe']:
                model = TrieMoERecommender(
                    dense_dim=feature_dims['dense_dim'],
                    sparse_dims=feature_dims['sparse_dims'],
                    embedding_dim=32,
                    num_experts=8,
                    top_k=1,
                    routing_mode='learned_only',
                    dropout=0.1,
                ).to(device)

                trie_encoder = FastTrieEncoder(
                    tries=trie_builder.tries,
                    vocab_sizes=vocab_sizes,
                ).to(device)
            else:
                if model_name == 'DeepFM':
                    model = DeepFM(
                        dense_dim=feature_dims['dense_dim'],
                        sparse_dims=feature_dims['sparse_dims'],
                        embedding_dim=32,
                        dropout=0.1,
                    ).to(device)
                elif model_name == 'DLRM':
                    model = DLRM(
                        dense_dim=feature_dims['dense_dim'],
                        sparse_dims=feature_dims['sparse_dims'],
                        embedding_dim=32,
                        dropout=0.1,
                    ).to(device)
                elif model_name == 'DCNv2':
                    model = DCNv2(
                        dense_dim=feature_dims['dense_dim'],
                        sparse_dims=feature_dims['sparse_dims'],
                        embedding_dim=32,
                        dropout=0.1,
                    ).to(device)
                trie_encoder = None

            start_time = time.time()
            result = train_and_evaluate(
                model, train_loader, val_loader, test_loader, device,
                epochs=args.epochs,
                trie_encoder=trie_encoder,
                sparse_fields=sparse_fields if config['is_moe'] else None,
                vocab_maps=vocab_maps if config['is_moe'] else None,
                track_experts=config['is_moe'] and run == 0,
            )
            train_time = time.time() - start_time

            model_aucs.append(result['test_auc'])
            model_losses.append(result['test_logloss'])
            train_times.append(train_time)

            if config['is_moe'] and run == 0 and result['expert_usage']:
                expert_usage_all[model_name] = result['expert_usage']

            print(f"  Run {run+1}: AUC={result['test_auc']:.4f}, LogLoss={result['test_logloss']:.4f}")

        all_results[model_name] = {
            'all_aucs': model_aucs,
            'all_losses': model_losses,
            'test_auc_mean': np.mean(model_aucs),
            'test_auc_std': np.std(model_aucs),
            'test_logloss_mean': np.mean(model_losses),
            'test_logloss_std': np.std(model_losses),
            'train_time_mean': np.mean(train_times),
        }

    # ========== Experiment 2: Statistical Significance ==========
    print("\n" + "="*60)
    print("Experiment 2: Statistical Significance Tests")
    print("="*60)

    stat_tests = run_statistical_tests(all_results, baseline='DeepFM')

    for model, results in stat_tests.items():
        print(f"\n{model} vs DeepFM:")
        print(f"  t-statistic: {results['t_statistic']:.4f}")
        print(f"  p-value: {results['p_value']:.4f}")
        print(f"  Significant (α=0.05): {results['significant']}")
        print(f"  Cohen's d: {results['cohens_d']:.4f}")
        print(f"  95% CI: [{results['ci_lower']:.4f}, {results['ci_upper']:.4f}]")

    # ========== Experiment 3: Expert Utilization ==========
    print("\n" + "="*60)
    print("Experiment 3: Expert Utilization Analysis")
    print("="*60)

    if 'Trie-MoE' in expert_usage_all:
        balance_score = plot_expert_utilization(
            expert_usage_all['Trie-MoE'],
            num_experts=8,
            save_path=output_dir / 'expert_utilization.png'
        )
        print(f"Expert Load Balance Score: {balance_score:.4f} (1.0 = perfect)")

    # ========== Experiment 4: Model Comparison Plots ==========
    print("\n" + "="*60)
    print("Experiment 4: Generating Comparison Plots")
    print("="*60)

    plot_model_comparison(all_results, metric='auc', save_path=output_dir / 'auc_comparison.png')
    plot_model_comparison(all_results, metric='logloss', save_path=output_dir / 'logloss_comparison.png')
    print("Plots saved.")

    # ========== Save Results ==========
    print("\n" + "="*60)
    print("Saving Results")
    print("="*60)

    # Summary JSON
    summary = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'num_runs': args.num_runs,
            'epochs': args.epochs,
            'device': str(device),
        },
        'results': {k: {kk: vv for kk, vv in v.items() if kk != 'all_aucs' and kk != 'all_losses'}
                   for k, v in all_results.items()},
        'statistical_tests': stat_tests,
    }

    with open(output_dir / 'experiment_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    # LaTeX table
    latex_table = generate_latex_table(all_results)
    with open(output_dir / 'results_table.tex', 'w') as f:
        f.write(latex_table)

    print(f"\nAll results saved to {output_dir}")

    # Print final summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"\n{'Model':<15} {'AUC (mean±std)':<20} {'LogLoss':<15}")
    print("-"*50)
    for model, results in all_results.items():
        auc_str = f"{results['test_auc_mean']:.4f}±{results['test_auc_std']:.4f}"
        print(f"{model:<15} {auc_str:<20} {results['test_logloss_mean']:.4f}")


if __name__ == '__main__':
    main()
