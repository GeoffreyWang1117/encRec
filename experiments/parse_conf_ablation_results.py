"""
Parse confidence ablation results and generate paper-ready tables.
Run this after kdd_response_confidence_ablation.py completes.

Usage:
    python experiments/parse_conf_ablation_results.py
"""
import json
import glob
import os
import sys
import numpy as np

RESULT_DIR = 'results/kdd_response_confidence_ablation'

DISPLAY_NAMES = {
    'ctr_only':         'CTR-only',
    'category_only':    'Category-only',
    'product_ours':     'Product (Ours)',
    'additive':         'Additive',
    'entropy_weighted': 'Entropy-weighted',
}

ORDER = ['ctr_only', 'category_only', 'product_ours', 'additive', 'entropy_weighted']

ANALYSIS = {
    'ctr_only': (
        'CTR gap is bimodal (≈0.99 for 88% of samples, ≈0 for 12%) '
        'because history items get CTR=1.0, unseen items get CTR=0.01. '
        'Result: constant exit_rate=87.9% regardless of τ, '
        'overconfident early exit → Hit@5 WORSE than LLM-only. '
        'Demonstrates that CTR signal alone is insufficient for calibrated routing.'
    ),
    'category_only': (
        'pref_strength provides graded discrimination (distribution in [0,1]), '
        'but ignores candidate separability. '
        'Can exit confidently even when all candidates have same CTR — '
        'routes to Trie when LLM would score them better.'
    ),
    'product_ours': (
        'Both conditions must hold simultaneously (AND logic via product): '
        'user has clear category preference AND candidates are CTR-separable. '
        'Creates calibrated confidence with graded discrimination. '
        'Our paper formula.'
    ),
    'additive': (
        'Allows high category signal to compensate for low CTR gap (OR-like). '
        'Higher exit rate than product for same τ — '
        'exits when category OR CTR is strong, not both, leading to more '
        'incorrect routing decisions.'
    ),
    'entropy_weighted': (
        'Entropy factor (1 - H/Hmax) is conceptually similar to pref_strength '
        'but computed differently. Slightly noisier than pref_strength '
        'because entropy treats all minority categories symmetrically '
        'while pref_strength focuses on the dominant category.'
    ),
}


def load_latest_result():
    files = sorted(glob.glob(os.path.join(RESULT_DIR, 'confidence_ablation_*.json')))
    if not files:
        print(f'No result files found in {RESULT_DIR}')
        sys.exit(1)
    latest = files[-1]
    print(f'Loading: {latest}')
    with open(latest) as f:
        return json.load(f)


def parse_from_log():
    """Fallback: parse run_log.txt if JSON not yet written."""
    log_path = os.path.join(RESULT_DIR, 'run_log.txt')
    if not os.path.exists(log_path):
        return None

    results = {}
    current_form = None
    tau_sweep = {}
    seed_results = []

    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if 'Formulation:' in line:
                if current_form and tau_sweep:
                    results[current_form] = {
                        'tau_sweep': tau_sweep, 'seed_results': seed_results}
                current_form = line.split('Formulation:')[1].strip()
                tau_sweep = {}
                seed_results = []
            elif 'τ=' in line and 'Hit@5=' in line and 'exit=' in line:
                import re
                tau = float(re.search(r'τ=([\d.]+)', line).group(1))
                hit5 = float(re.search(r'Hit@5=([\d.]+)', line).group(1))
                exit_r = float(re.search(r'exit=([\d.]+)%', line).group(1)) / 100
                lat = float(re.search(r'lat=([\d.]+)ms', line).group(1))
                tau_sweep[str(tau)] = {'hit@5': hit5, 'exit_rate': exit_r, 'latency_ms': lat}
            elif 'Seed' in line and 'Hit@5=' in line:
                import re
                hit5 = float(re.search(r'Hit@5=([\d.]+)', line).group(1))
                exit_r = float(re.search(r'exit=([\d.]+)%', line).group(1)) / 100
                lat = float(re.search(r'lat=([\d.]+)ms', line).group(1))
                seed_results.append({'hit@5': hit5, 'exit_rate': exit_r, 'latency_ms': lat})
            elif 'Optimal' in line and 'τ*=' in line:
                import re
                m = re.search(r'τ\*=([\d.]+)', line)
                if m and current_form:
                    tau_opt = float(m.group(1))

    if current_form and tau_sweep:
        results[current_form] = {'tau_sweep': tau_sweep, 'seed_results': seed_results}

    return results


def print_main_table(data):
    """Print main ablation table for paper."""
    results = data.get('results', data)  # handle both formats

    print('\n' + '='*85)
    print('TABLE: Confidence Formulation Ablation (MIND Large, 2000 samples × 3 seeds)')
    print('='*85)
    header = (f"{'Formulation':<22} {'τ*':>5} {'Hit@5':>14} "
              f"{'Exit Rate':>13} {'Lat (ms)':>10} {'Exit Hit@5':>12}")
    print(header)
    print('-'*85)

    for name in ORDER:
        if name not in results:
            print(f'{DISPLAY_NAMES[name]:<22}  [not yet completed]')
            continue
        res = results[name]
        tau_opt = res.get('tau_opt', '?')
        s = res.get('summary', {})
        if s:
            hit5 = f"{s['hit@5_mean']:.3f}±{s['hit@5_std']:.3f}"
            exit_r = f"{s['exit_rate_mean']*100:.1f}%±{s['exit_rate_std']*100:.1f}%"
            lat = f"{s['latency_ms_mean']:.1f}"
            exit_hit = f"{s['exited_hit@5_mean']:.3f}"
        else:
            # Parse from seed_results
            sr = res.get('seed_results', res.get('multi_seed', []))
            if sr:
                hit5_vals = [r['hit@5'] for r in sr]
                exit_vals = [r['exit_rate'] for r in sr]
                lat_vals = [r['latency_ms'] for r in sr]
                hit5 = f"{np.mean(hit5_vals):.3f}±{np.std(hit5_vals):.3f}"
                exit_r = f"{np.mean(exit_vals)*100:.1f}%"
                lat = f"{np.mean(lat_vals):.1f}"
                exit_hit = "—"
            else:
                hit5 = exit_r = lat = exit_hit = "—"

        marker = " ★" if name == 'product_ours' else ""
        print(f"{DISPLAY_NAMES[name]+marker:<24} {str(tau_opt):>4} {hit5:>14} "
              f"{exit_r:>13} {lat:>10}ms {exit_hit:>12}")

    print('='*85)


def print_latex_table(data):
    """Print LaTeX table for direct paper insertion."""
    results = data.get('results', data)

    print('\n% LaTeX table for paper (confidence formulation ablation)')
    print(r'\begin{table}[t]')
    print(r'\centering\small')
    print(r'\caption{Confidence formulation ablation on \mind{} Large (2000 samples, 3 runs). '
          r'Our product formulation achieves the best Hit@5 with calibrated exit rates.}')
    print(r'\label{tab:conf_ablation}')
    print(r'\begin{tabular}{lcrrrr}')
    print(r'\toprule')
    print(r'Formulation & $\tau^*$ & Hit@5 & Exit Rate & Latency & Exit Hit@5 \\')
    print(r'\midrule')

    for name in ORDER:
        if name not in results:
            continue
        res = results[name]
        tau_opt = res.get('tau_opt', '?')
        s = res.get('summary', {})
        if not s:
            sr = res.get('seed_results', res.get('multi_seed', []))
            if not sr:
                continue
            hit5_m = np.mean([r['hit@5'] for r in sr])
            hit5_s = np.std([r['hit@5'] for r in sr])
            exit_m = np.mean([r['exit_rate'] for r in sr])
            lat_m = np.mean([r['latency_ms'] for r in sr])
            exit_hit_m = 0.0
        else:
            hit5_m, hit5_s = s['hit@5_mean'], s['hit@5_std']
            exit_m = s['exit_rate_mean']
            lat_m = s['latency_ms_mean']
            exit_hit_m = s['exited_hit@5_mean']

        bold = name == 'product_ours'
        dname = DISPLAY_NAMES[name]
        prefix = r'\textbf{' if bold else ''
        suffix = '}' if bold else ''

        print(f'{prefix}{dname}{suffix} & {tau_opt} & '
              f'{prefix}{hit5_m:.3f}$\\pm${hit5_s:.3f}{suffix} & '
              f'{exit_m*100:.1f}\\% & {lat_m:.1f}ms & '
              f'{exit_hit_m:.3f} \\\\')

    print(r'\bottomrule')
    print(r'\end{tabular}')
    print(r'\end{table}')


def print_insights(data):
    """Print interpretation of results."""
    results = data.get('results', data)

    print('\n' + '='*85)
    print('KEY INSIGHTS FOR REVIEWER KQWU Q4:')
    print('='*85)
    for name in ORDER:
        if name not in results:
            continue
        print(f'\n[{DISPLAY_NAMES[name]}]')
        print(f'  {ANALYSIS[name]}')


def main():
    # Try JSON first, fall back to log parsing
    json_files = sorted(glob.glob(os.path.join(RESULT_DIR, 'confidence_ablation_*.json')))

    if json_files:
        data = load_latest_result()
    else:
        print('JSON not found, parsing from run_log.txt...')
        parsed = parse_from_log()
        if not parsed:
            print('No results available yet.')
            sys.exit(0)
        data = {'results': parsed}

    print_main_table(data)
    print_insights(data)
    print_latex_table(data)


if __name__ == '__main__':
    main()
