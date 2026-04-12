#!/usr/bin/env python3
"""
Generate publication-quality figures and LaTeX tables for Trie-MoE paper.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Set publication-quality defaults
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})


def load_experiment_results():
    """Load all experiment results."""
    results = {}
    logs_dir = Path('logs')

    # Load Criteo experiments
    criteo_files = list(logs_dir.glob('criteo_experiments_*.json'))
    if criteo_files:
        with open(sorted(criteo_files)[-1]) as f:
            results['criteo'] = json.load(f)

    # Load scalability results
    scalability_file = logs_dir / 'scalability' / 'scalability_results_20260107_175942.json'
    if scalability_file.exists():
        with open(scalability_file) as f:
            results['scalability'] = json.load(f)

    # Load cold-start results
    cold_start_file = logs_dir / 'cold_start' / 'cold_start_results_20260107_180214.json'
    if cold_start_file.exists():
        with open(cold_start_file) as f:
            results['cold_start'] = json.load(f)

    # Load paper experiments
    paper_files = list(logs_dir.glob('paper_experiments_*.json'))
    if paper_files:
        with open(sorted(paper_files)[-1]) as f:
            results['paper'] = json.load(f)

    return results


def generate_main_comparison_table(results):
    """Generate LaTeX table for main model comparison."""
    print("\n" + "="*80)
    print("TABLE 1: Main Model Comparison (Amazon Electronics Dataset)")
    print("="*80)

    if 'paper' not in results:
        print("Paper experiment results not found")
        return

    paper = results['paper']

    # LaTeX table header
    latex = r"""
\begin{table}[h]
\centering
\caption{Model Performance Comparison on Amazon Electronics Dataset}
\label{tab:main_comparison}
\begin{tabular}{lcccc}
\toprule
Model & AUC & LogLoss & Train Time (s) \\
\midrule
"""

    print(f"\n{'Model':<12} {'AUC':<20} {'LogLoss':<15} {'Train Time':<12}")
    print("-"*60)

    for model_name in ['DeepFM', 'DLRM', 'DCNv2', 'Trie-MoE']:
        if model_name in paper:
            m = paper[model_name]
            auc_str = f"{m['overall_auc_mean']:.4f}±{m['overall_auc_std']:.4f}"
            logloss_str = f"{m['logloss_mean']:.4f}±{m['logloss_std']:.4f}"
            time_str = f"{m['train_time_mean']:.1f}"

            print(f"{model_name:<12} {auc_str:<20} {logloss_str:<15} {time_str:<12}")
            latex += f"{model_name} & ${m['overall_auc_mean']:.4f} \\pm {m['overall_auc_std']:.4f}$ & ${m['logloss_mean']:.4f} \\pm {m['logloss_std']:.4f}$ & {m['train_time_mean']:.1f} \\\\\n"

    latex += r"""
\bottomrule
\end{tabular}
\end{table}
"""
    return latex


def generate_criteo_table(results):
    """Generate LaTeX table for Criteo dataset results."""
    print("\n" + "="*80)
    print("TABLE 2: Criteo Dataset Results (Synthetic 1M)")
    print("="*80)

    if 'criteo' not in results:
        print("Criteo experiment results not found")
        return

    criteo = results['criteo']

    print(f"\n{'Model':<12} {'AUC':<20} {'LogLoss':<15}")
    print("-"*50)

    for model_name in ['DeepFM', 'DLRM', 'Trie-MoE']:
        if model_name in criteo:
            m = criteo[model_name]
            auc_str = f"{m['overall_auc_mean']:.4f}±{m['overall_auc_std']:.4f}"

            # Calculate logloss mean from runs
            logloss_mean = np.mean([r['overall']['logloss'] for r in m['runs']])
            print(f"{model_name:<12} {auc_str:<20} {logloss_mean:.4f}")


def generate_cold_start_table(results):
    """Generate LaTeX table for cold-start analysis."""
    print("\n" + "="*80)
    print("TABLE 3: Cold-Start Performance Analysis")
    print("="*80)

    if 'cold_start' not in results:
        print("Cold-start experiment results not found")
        return

    cs = results['cold_start']['summary']

    print(f"\n{'Model':<12} {'Cold AUC':<15} {'Warm AUC':<15} {'Hot AUC':<15} {'Cold LogLoss':<15}")
    print("-"*75)

    latex = r"""
\begin{table}[h]
\centering
\caption{Cold-Start Performance Analysis}
\label{tab:cold_start}
\begin{tabular}{lcccc}
\toprule
Model & Cold AUC & Warm AUC & Hot AUC & Cold LogLoss \\
\midrule
"""

    for model_name in ['DeepFM', 'DLRM', 'Trie-MoE']:
        if model_name in cs:
            m = cs[model_name]
            cold_auc = f"{m['cold']['auc_mean']:.4f}"
            warm_auc = f"{m['warm']['auc_mean']:.4f}"
            hot_auc = f"{m['hot']['auc_mean']:.4f}"
            cold_logloss = f"{m['cold']['logloss_mean']:.4f}"

            print(f"{model_name:<12} {cold_auc:<15} {warm_auc:<15} {hot_auc:<15} {cold_logloss:<15}")
            latex += f"{model_name} & {cold_auc} & {warm_auc} & {hot_auc} & {cold_logloss} \\\\\n"

    latex += r"""
\bottomrule
\end{tabular}
\end{table}
"""

    # Calculate improvements
    if 'Trie-MoE' in cs and 'DeepFM' in cs:
        trie_cold_ll = cs['Trie-MoE']['cold']['logloss_mean']
        deepfm_cold_ll = cs['DeepFM']['cold']['logloss_mean']
        improvement = (deepfm_cold_ll - trie_cold_ll) / deepfm_cold_ll * 100
        print(f"\nTrie-MoE cold-start LogLoss improvement over DeepFM: {improvement:.1f}%")

    return latex


def generate_scalability_figure(results):
    """Generate scalability analysis figure."""
    print("\n" + "="*80)
    print("FIGURE 1: Scalability Analysis")
    print("="*80)

    if 'scalability' not in results:
        print("Scalability experiment results not found")
        return

    scale = results['scalability']

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    colors = {'DeepFM': '#1f77b4', 'DLRM': '#ff7f0e', 'Trie-MoE': '#2ca02c'}
    markers = {'DeepFM': 'o', 'DLRM': 's', 'Trie-MoE': '^'}

    # Plot 1: AUC vs Data Size
    ax1 = axes[0]
    for model_name in ['DeepFM', 'DLRM', 'Trie-MoE']:
        if model_name in scale:
            fracs = [r['data_fraction'] for r in scale[model_name]]
            aucs = [r['auc'] for r in scale[model_name]]
            ax1.plot(fracs, aucs, marker=markers[model_name],
                    label=model_name, color=colors[model_name],
                    linewidth=2, markersize=8)

    ax1.set_xlabel('Data Fraction')
    ax1.set_ylabel('Test AUC')
    ax1.set_title('(a) Model Performance vs Data Size')
    ax1.legend(loc='lower right')
    ax1.set_xlim([0.05, 1.05])

    # Plot 2: Training Time vs Data Size
    ax2 = axes[1]
    for model_name in ['DeepFM', 'DLRM', 'Trie-MoE']:
        if model_name in scale:
            fracs = [r['data_fraction'] for r in scale[model_name]]
            times = [r['train_time'] for r in scale[model_name]]
            ax2.plot(fracs, times, marker=markers[model_name],
                    label=model_name, color=colors[model_name],
                    linewidth=2, markersize=8)

    ax2.set_xlabel('Data Fraction')
    ax2.set_ylabel('Training Time (s)')
    ax2.set_title('(b) Training Time Scalability')
    ax2.legend(loc='upper left')
    ax2.set_xlim([0.05, 1.05])

    # Plot 3: Throughput vs Data Size
    ax3 = axes[2]
    for model_name in ['DeepFM', 'DLRM', 'Trie-MoE']:
        if model_name in scale:
            fracs = [r['data_fraction'] for r in scale[model_name]]
            throughputs = [r['throughput'] / 1000 for r in scale[model_name]]  # K samples/sec
            ax3.plot(fracs, throughputs, marker=markers[model_name],
                    label=model_name, color=colors[model_name],
                    linewidth=2, markersize=8)

    ax3.set_xlabel('Data Fraction')
    ax3.set_ylabel('Throughput (K samples/s)')
    ax3.set_title('(c) Inference Throughput')
    ax3.legend(loc='upper right')
    ax3.set_xlim([0.05, 1.05])

    plt.tight_layout()

    output_dir = Path('logs/paper_figures')
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_dir / 'scalability_analysis.pdf')
    plt.savefig(output_dir / 'scalability_analysis.png', dpi=300)
    plt.close()

    print(f"Scalability figure saved to {output_dir}")


def generate_cold_start_figure(results):
    """Generate cold-start analysis figure."""
    print("\n" + "="*80)
    print("FIGURE 2: Cold-Start Analysis")
    print("="*80)

    if 'cold_start' not in results:
        print("Cold-start experiment results not found")
        return

    cs = results['cold_start']['summary']

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    models = ['DeepFM', 'DLRM', 'Trie-MoE']
    categories = ['cold', 'warm', 'hot']
    colors = ['#e74c3c', '#f39c12', '#27ae60']

    # Plot 1: AUC by category
    ax1 = axes[0]
    x = np.arange(len(models))
    width = 0.25

    for i, cat in enumerate(categories):
        values = [cs[m][cat]['auc_mean'] for m in models]
        bars = ax1.bar(x + i*width, values, width, label=cat.capitalize(), color=colors[i])

    ax1.set_xlabel('Model')
    ax1.set_ylabel('AUC')
    ax1.set_title('(a) AUC by Interaction Frequency')
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(models)
    ax1.legend()
    ax1.set_ylim([0.6, 0.85])

    # Plot 2: LogLoss by category
    ax2 = axes[1]

    for i, cat in enumerate(categories):
        values = [cs[m][cat]['logloss_mean'] for m in models]
        bars = ax2.bar(x + i*width, values, width, label=cat.capitalize(), color=colors[i])

    ax2.set_xlabel('Model')
    ax2.set_ylabel('LogLoss')
    ax2.set_title('(b) LogLoss by Interaction Frequency')
    ax2.set_xticks(x + width)
    ax2.set_xticklabels(models)
    ax2.legend()

    plt.tight_layout()

    output_dir = Path('logs/paper_figures')
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_dir / 'cold_start_analysis.pdf')
    plt.savefig(output_dir / 'cold_start_analysis.png', dpi=300)
    plt.close()

    print(f"Cold-start figure saved to {output_dir}")


def generate_model_comparison_figure(results):
    """Generate main model comparison bar chart."""
    print("\n" + "="*80)
    print("FIGURE 3: Model Comparison")
    print("="*80)

    if 'paper' not in results and 'criteo' not in results:
        print("No comparison results found")
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # Use Criteo results if available, otherwise paper results
    source = results.get('criteo', results.get('paper', {}))

    models = ['DeepFM', 'DLRM', 'Trie-MoE']

    # Extract AUC values
    aucs = []
    auc_stds = []
    for m in models:
        if m in source:
            aucs.append(source[m]['overall_auc_mean'])
            auc_stds.append(source[m]['overall_auc_std'])
        else:
            aucs.append(0)
            auc_stds.append(0)

    # Plot AUC
    ax1 = axes[0]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    x = np.arange(len(models))
    bars = ax1.bar(x, aucs, yerr=auc_stds, capsize=5, color=colors, edgecolor='black', linewidth=1.2)

    ax1.set_xlabel('Model')
    ax1.set_ylabel('Test AUC')
    ax1.set_title('(a) Model AUC Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)

    # Add value labels
    for bar, auc in zip(bars, aucs):
        height = bar.get_height()
        ax1.annotate(f'{auc:.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

    # Plot training time
    ax2 = axes[1]
    times = []
    for m in models:
        if m in source:
            times.append(source[m]['train_time_mean'])
        else:
            times.append(0)

    bars = ax2.bar(x, times, color=colors, edgecolor='black', linewidth=1.2)

    ax2.set_xlabel('Model')
    ax2.set_ylabel('Training Time (s)')
    ax2.set_title('(b) Training Time Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models)

    for bar, t in zip(bars, times):
        height = bar.get_height()
        ax2.annotate(f'{t:.1f}s',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

    plt.tight_layout()

    output_dir = Path('logs/paper_figures')
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_dir / 'model_comparison.pdf')
    plt.savefig(output_dir / 'model_comparison.png', dpi=300)
    plt.close()

    print(f"Model comparison figure saved to {output_dir}")


def generate_summary_stats(results):
    """Generate summary statistics for the paper."""
    print("\n" + "="*80)
    print("PAPER SUMMARY STATISTICS")
    print("="*80)

    if 'criteo' in results:
        criteo = results['criteo']
        if 'Trie-MoE' in criteo and 'DeepFM' in criteo:
            trie_auc = criteo['Trie-MoE']['overall_auc_mean']
            deepfm_auc = criteo['DeepFM']['overall_auc_mean']
            improvement = (trie_auc - deepfm_auc) * 100
            print(f"\nCriteo Dataset:")
            print(f"  Trie-MoE AUC improvement over DeepFM: +{improvement:.2f}%")

    if 'cold_start' in results:
        cs = results['cold_start']['summary']
        if 'Trie-MoE' in cs and 'DeepFM' in cs:
            trie_cold_ll = cs['Trie-MoE']['cold']['logloss_mean']
            deepfm_cold_ll = cs['DeepFM']['cold']['logloss_mean']
            improvement = (deepfm_cold_ll - trie_cold_ll) / deepfm_cold_ll * 100
            print(f"\nCold-Start Analysis:")
            print(f"  Trie-MoE cold LogLoss improvement: {improvement:.1f}%")
            print(f"  Cold samples: n={cs['DeepFM']['cold']['count']}")

    if 'scalability' in results:
        scale = results['scalability']
        print(f"\nScalability:")
        for model in ['DeepFM', 'DLRM', 'Trie-MoE']:
            if model in scale:
                full_auc = scale[model][-1]['auc']
                full_throughput = scale[model][-1]['throughput'] / 1000
                print(f"  {model}: Full AUC={full_auc:.4f}, Throughput={full_throughput:.1f}K samples/s")


def main():
    print("="*80)
    print("GENERATING PAPER-READY FIGURES AND TABLES")
    print("Trie-MoE: Mixture-of-Experts with Statistical Trie Routing")
    print("="*80)

    # Load all results
    results = load_experiment_results()

    print(f"\nLoaded experiments: {list(results.keys())}")

    # Generate tables
    generate_main_comparison_table(results)
    generate_criteo_table(results)
    generate_cold_start_table(results)

    # Generate figures
    generate_scalability_figure(results)
    generate_cold_start_figure(results)
    generate_model_comparison_figure(results)

    # Summary statistics
    generate_summary_stats(results)

    print("\n" + "="*80)
    print("All figures and tables generated successfully!")
    print("Output directory: logs/paper_figures/")
    print("="*80)


if __name__ == '__main__':
    main()
