#!/usr/bin/env python3
"""
Visualization for data scale ablation results.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load results
results_dir = Path('results/data_scale_ablation')
results_files = list(results_dir.glob('results_*.json'))
latest_file = max(results_files)

with open(latest_file) as f:
    results = json.load(f)

# Prepare data
sizes = sorted([int(s) for s in results.keys()])
deepfm_means = [results[str(s)]['DeepFM']['mean'] for s in sizes]
deepfm_stds = [results[str(s)]['DeepFM']['std'] for s in sizes]
trie_means = [results[str(s)]['Trie-MoE']['mean'] for s in sizes]
trie_stds = [results[str(s)]['Trie-MoE']['std'] for s in sizes]
advantages = [results[str(s)]['advantage'] for s in sizes]

# Create figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: AUC vs Data Size
ax1 = axes[0]
sizes_k = [s / 1000 for s in sizes]

ax1.errorbar(sizes_k, deepfm_means, yerr=deepfm_stds,
             marker='o', capsize=3, label='DeepFM', linewidth=2)
ax1.errorbar(sizes_k, trie_means, yerr=trie_stds,
             marker='s', capsize=3, label='Trie-MoE', linewidth=2)

# Mark crossover region
ax1.axvspan(1000, 2000, alpha=0.2, color='gray', label='Crossover Region')

ax1.set_xlabel('Training Data Size (K samples)', fontsize=12)
ax1.set_ylabel('Test AUC', fontsize=12)
ax1.set_title('AUC vs Data Size', fontsize=14)
ax1.legend(loc='lower right')
ax1.grid(True, alpha=0.3)
ax1.set_xscale('log')

# Plot 2: Trie-MoE Advantage
ax2 = axes[1]

colors = ['green' if a > 0 else 'red' for a in advantages]
bars = ax2.bar([f'{s//1000}K' for s in sizes], [a * 100 for a in advantages], color=colors)

# Add zero line
ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

# Add crossover annotation
ax2.axvline(x=3.5, color='gray', linestyle='--', alpha=0.7)
ax2.annotate('Crossover\nPoint', xy=(3.5, 1), fontsize=10, ha='center')

ax2.set_xlabel('Training Data Size', fontsize=12)
ax2.set_ylabel('Trie-MoE Advantage (%)', fontsize=12)
ax2.set_title('Trie-MoE Relative Advantage vs DeepFM', fontsize=14)
ax2.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar, adv in zip(bars, advantages):
    height = bar.get_height()
    ax2.annotate(f'{adv*100:.2f}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3 if height > 0 else -10),
                textcoords="offset points",
                ha='center', va='bottom' if height > 0 else 'top',
                fontsize=9)

plt.tight_layout()

# Save
output_path = 'results/data_scale_ablation/data_scale_comparison.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Figure saved to: {output_path}")

# Also save as PDF for paper
output_pdf = 'results/data_scale_ablation/data_scale_comparison.pdf'
plt.savefig(output_pdf, bbox_inches='tight')
print(f"PDF saved to: {output_pdf}")

plt.close()

# Create a summary table for LaTeX
print("\n" + "=" * 60)
print("LaTeX Table for Paper")
print("=" * 60)

print(r"""
\begin{table}[h]
\centering
\caption{Trie-MoE vs DeepFM across different training data sizes on Criteo.}
\label{tab:data_scale}
\begin{tabular}{lcccc}
\toprule
Data Size & DeepFM & Trie-MoE & $\Delta$ AUC & Winner \\
\midrule""")

for s in sizes:
    r = results[str(s)]
    winner = "\\textbf{Trie-MoE}" if r['advantage'] > 0 else "DeepFM"
    diff = f"+{r['advantage']:.4f}" if r['advantage'] > 0 else f"{r['advantage']:.4f}"
    print(f"{s//1000}K & {r['DeepFM']['mean']:.4f}$\\pm${r['DeepFM']['std']:.4f} & "
          f"{r['Trie-MoE']['mean']:.4f}$\\pm${r['Trie-MoE']['std']:.4f} & {diff} & {winner} \\\\")

print(r"""\midrule
\multicolumn{5}{l}{\textit{Crossover point: $\sim$1.5M samples}} \\
\bottomrule
\end{tabular}
\end{table}
""")
