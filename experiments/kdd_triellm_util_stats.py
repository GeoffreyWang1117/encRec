#!/usr/bin/env python3
"""
Statistical significance tests for paper experiments.

Provides:
1. Paired t-test
2. Wilcoxon signed-rank test
3. Bootstrap confidence intervals
4. Effect size (Cohen's d)
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Tuple
import json
from pathlib import Path


def paired_ttest(baseline: List[float], proposed: List[float]) -> Dict:
    """
    Paired t-test for comparing two models.

    Args:
        baseline: List of metric values from baseline model
        proposed: List of metric values from proposed model

    Returns:
        Dict with test statistics and p-value
    """
    t_stat, p_value = stats.ttest_rel(proposed, baseline)
    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'significant_005': p_value < 0.05,
        'significant_001': p_value < 0.01,
    }


def wilcoxon_test(baseline: List[float], proposed: List[float]) -> Dict:
    """
    Wilcoxon signed-rank test (non-parametric alternative).

    Args:
        baseline: List of metric values from baseline model
        proposed: List of metric values from proposed model

    Returns:
        Dict with test statistics and p-value
    """
    try:
        w_stat, p_value = stats.wilcoxon(proposed, baseline)
        return {
            'w_statistic': w_stat,
            'p_value': p_value,
            'significant_005': p_value < 0.05,
            'significant_001': p_value < 0.01,
        }
    except ValueError as e:
        # Wilcoxon fails if all differences are zero
        return {
            'w_statistic': None,
            'p_value': 1.0,
            'error': str(e),
        }


def cohens_d(baseline: List[float], proposed: List[float]) -> float:
    """
    Calculate Cohen's d effect size.

    Returns:
        Effect size (small: 0.2, medium: 0.5, large: 0.8)
    """
    diff = np.array(proposed) - np.array(baseline)
    if np.std(diff) == 0:
        return 0.0
    return np.mean(diff) / np.std(diff)


def bootstrap_ci(
    baseline: List[float],
    proposed: List[float],
    n_bootstrap: int = 10000,
    confidence: float = 0.95
) -> Dict:
    """
    Bootstrap confidence interval for the difference.

    Args:
        baseline: Baseline metric values
        proposed: Proposed model metric values
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level

    Returns:
        Dict with mean difference and CI bounds
    """
    baseline = np.array(baseline)
    proposed = np.array(proposed)
    n = len(baseline)

    # Bootstrap differences
    diff_samples = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        diff = np.mean(proposed[idx]) - np.mean(baseline[idx])
        diff_samples.append(diff)

    diff_samples = np.array(diff_samples)

    # Compute percentiles
    alpha = 1 - confidence
    lower = np.percentile(diff_samples, alpha / 2 * 100)
    upper = np.percentile(diff_samples, (1 - alpha / 2) * 100)

    return {
        'mean_diff': np.mean(proposed) - np.mean(baseline),
        'ci_lower': lower,
        'ci_upper': upper,
        'confidence': confidence,
        'significant': not (lower <= 0 <= upper),  # CI doesn't include 0
    }


def comprehensive_test(
    baseline: List[float],
    proposed: List[float],
    metric_name: str = 'metric',
    higher_is_better: bool = True
) -> Dict:
    """
    Run all statistical tests and return comprehensive results.

    Args:
        baseline: Baseline metric values
        proposed: Proposed model metric values
        metric_name: Name of the metric being compared
        higher_is_better: Whether higher values are better

    Returns:
        Comprehensive test results
    """
    baseline = np.array(baseline)
    proposed = np.array(proposed)

    # Flip sign if lower is better (e.g., LogLoss)
    if not higher_is_better:
        baseline_test = -baseline
        proposed_test = -proposed
    else:
        baseline_test = baseline
        proposed_test = proposed

    results = {
        'metric': metric_name,
        'higher_is_better': higher_is_better,
        'n_samples': len(baseline),
        'baseline_mean': float(np.mean(baseline)),
        'baseline_std': float(np.std(baseline)),
        'proposed_mean': float(np.mean(proposed)),
        'proposed_std': float(np.std(proposed)),
        'improvement': float(np.mean(proposed) - np.mean(baseline)),
        'improvement_pct': float((np.mean(proposed) - np.mean(baseline)) / np.mean(baseline) * 100),
        'paired_ttest': paired_ttest(baseline_test.tolist(), proposed_test.tolist()),
        'wilcoxon': wilcoxon_test(baseline_test.tolist(), proposed_test.tolist()),
        'cohens_d': float(cohens_d(baseline_test.tolist(), proposed_test.tolist())),
        'bootstrap_ci': bootstrap_ci(baseline.tolist(), proposed.tolist()),
    }

    # Interpret effect size
    d = abs(results['cohens_d'])
    if d < 0.2:
        results['effect_size_interpretation'] = 'negligible'
    elif d < 0.5:
        results['effect_size_interpretation'] = 'small'
    elif d < 0.8:
        results['effect_size_interpretation'] = 'medium'
    else:
        results['effect_size_interpretation'] = 'large'

    return results


def format_latex_table(results: Dict[str, Dict]) -> str:
    """
    Format results as LaTeX table.

    Args:
        results: Dict mapping comparison_name -> test results

    Returns:
        LaTeX table string
    """
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Statistical Significance Tests}",
        r"\begin{tabular}{lccccl}",
        r"\toprule",
        r"Comparison & $\Delta$ & t-test p & Wilcoxon p & Cohen's d & Sig. \\",
        r"\midrule",
    ]

    for name, res in results.items():
        delta = res['improvement']
        if res['metric'] == 'logloss':
            delta_str = f"{delta:.4f}"
        else:
            delta_str = f"{delta:+.4f}"

        t_p = res['paired_ttest']['p_value']
        w_p = res['wilcoxon']['p_value']
        d = res['cohens_d']

        # Significance markers
        sig = ""
        if t_p < 0.001:
            sig = "***"
        elif t_p < 0.01:
            sig = "**"
        elif t_p < 0.05:
            sig = "*"

        lines.append(
            f"{name} & {delta_str} & {t_p:.4f} & {w_p:.4f} & {d:.2f} & {sig} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\multicolumn{6}{l}{\small * p<0.05, ** p<0.01, *** p<0.001} \\",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)


def main():
    """Run statistical tests on saved experiment results."""
    import argparse

    parser = argparse.ArgumentParser(description='Statistical significance tests')
    parser.add_argument('--results_file', type=str, required=True,
                       help='Path to JSON results file')
    parser.add_argument('--baseline', type=str, default='dlrm',
                       help='Baseline model name')
    parser.add_argument('--proposed', type=str, default='trie_moe',
                       help='Proposed model name')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file for results')
    args = parser.parse_args()

    # Load results
    with open(args.results_file, 'r') as f:
        all_results = json.load(f)

    if args.baseline not in all_results:
        print(f"Baseline '{args.baseline}' not found in results")
        return
    if args.proposed not in all_results:
        print(f"Proposed '{args.proposed}' not found in results")
        return

    baseline_runs = all_results[args.baseline]
    proposed_runs = all_results[args.proposed]

    # Extract metrics
    baseline_auc = [r['auc'] for r in baseline_runs]
    proposed_auc = [r['auc'] for r in proposed_runs]
    baseline_logloss = [r['logloss'] for r in baseline_runs]
    proposed_logloss = [r['logloss'] for r in proposed_runs]

    # Run tests
    results = {}

    print("\n" + "="*60)
    print(f"Statistical Tests: {args.proposed} vs {args.baseline}")
    print("="*60)

    # AUC test
    auc_results = comprehensive_test(
        baseline_auc, proposed_auc,
        metric_name='auc', higher_is_better=True
    )
    results['AUC'] = auc_results

    print(f"\nAUC:")
    print(f"  Baseline: {auc_results['baseline_mean']:.4f} ± {auc_results['baseline_std']:.4f}")
    print(f"  Proposed: {auc_results['proposed_mean']:.4f} ± {auc_results['proposed_std']:.4f}")
    print(f"  Improvement: {auc_results['improvement']:+.4f} ({auc_results['improvement_pct']:+.2f}%)")
    print(f"  t-test p-value: {auc_results['paired_ttest']['p_value']:.4f}")
    print(f"  Wilcoxon p-value: {auc_results['wilcoxon']['p_value']:.4f}")
    print(f"  Cohen's d: {auc_results['cohens_d']:.3f} ({auc_results['effect_size_interpretation']})")
    print(f"  Significant (p<0.05): {auc_results['paired_ttest']['significant_005']}")

    # LogLoss test
    logloss_results = comprehensive_test(
        baseline_logloss, proposed_logloss,
        metric_name='logloss', higher_is_better=False
    )
    results['LogLoss'] = logloss_results

    print(f"\nLogLoss:")
    print(f"  Baseline: {logloss_results['baseline_mean']:.4f} ± {logloss_results['baseline_std']:.4f}")
    print(f"  Proposed: {logloss_results['proposed_mean']:.4f} ± {logloss_results['proposed_std']:.4f}")
    print(f"  Improvement: {logloss_results['improvement']:+.4f} ({logloss_results['improvement_pct']:+.2f}%)")
    print(f"  t-test p-value: {logloss_results['paired_ttest']['p_value']:.4f}")
    print(f"  Wilcoxon p-value: {logloss_results['wilcoxon']['p_value']:.4f}")
    print(f"  Cohen's d: {logloss_results['cohens_d']:.3f} ({logloss_results['effect_size_interpretation']})")
    print(f"  Significant (p<0.05): {logloss_results['paired_ttest']['significant_005']}")

    # Print LaTeX table
    print("\n" + "="*60)
    print("LaTeX Table:")
    print("="*60)
    print(format_latex_table(results))

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert numpy/bool types to Python native types
        def convert_to_serializable(obj):
            if isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_serializable(v) for v in obj]
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            return obj

        with open(output_path, 'w') as f:
            json.dump(convert_to_serializable(results), f, indent=2)
        print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
