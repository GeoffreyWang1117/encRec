"""
KDD 2026 Reviewer Response: Confidence Formulation Ablation.

Addresses Reviewer KQWU Q4:
  "The confidence score and routing rule, while intuitive, are heuristically motivated
   and lack ablation showing why this specific formulation is preferable to alternative
   statistical confidence measures."

Actual confidence formula in code (EarlyExitChecker):
  pref_strength = cat_counts[top_cat] / len(history[-20:])   # top-cat dominance
  CTR_gap       = (avg_top_ctr - avg_rest_ctr) / avg_top_ctr  # normalised gap
  conf (ours)   = pref_strength × CTR_gap                     ← PRODUCT formulation

Tests 5 confidence formulations at the same threshold τ (swept per formulation):
  1. CTR-only:        CTR_gap only, no category signal
  2. Category-only:   pref_strength only, no CTR signal
  3. Product (ours):  pref_strength × CTR_gap          ← current code
  4. Additive:        (pref_strength + CTR_gap) / 2
  5. Entropy-weighted: (1 - cat_entropy/log2(N)) × CTR_gap

For each: Hit@5, early exit rate, latency (ms), and Hit@5 of exited queries.
All formulations evaluated on MIND Large, 2000 samples × 3 seeds.

Usage:
    python experiments/kdd_response_confidence_ablation.py \
        --samples 2000 --device cuda:0 --seeds 42,43,44

Results saved to: results/kdd_response_confidence_ablation/
"""

import os
import sys
import json
import time
import logging
import argparse
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict, Counter
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import importlib.util
_spec = importlib.util.spec_from_file_location(
    "ablation", str(Path(__file__).parent / "kdd_triellm_01_ablation.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
ExperimentConfig = _mod.ExperimentConfig
LocalLLMRanker = _mod.LocalLLMRanker
TrieStatistics = _mod.TrieStatistics
evaluate_recommendations = _mod.evaluate_recommendations


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _ctr_gap(history, candidates, trie_stats, k=5):
    """Normalised CTR gap: (avg_top_k_ctr - avg_rest_ctr) / avg_top_k_ctr.

    Scores candidates by raw CTR and measures how separable the top-k are.
    """
    ctrs = [(cid, trie_stats.get_ctr(cid)) for cid in candidates]
    ctrs.sort(key=lambda x: x[1], reverse=True)
    top_ctrs = [c for _, c in ctrs[:k]]
    rest_ctrs = [c for _, c in ctrs[k:k+10]]
    mean_top = np.mean(top_ctrs) if top_ctrs else 0.0
    mean_rest = np.mean(rest_ctrs) if rest_ctrs else 0.0
    if mean_top <= 1e-9:
        return 0.0
    return float(np.clip((mean_top - mean_rest) / mean_top, 0.0, 1.0))


def _pref_strength(history, trie_stats):
    """Top-category dominance in user history (last 20 items).

    pref_strength = count(top_cat) / len(history[-20:])
    High → user has clear category preference → Trie can handle.
    """
    cat_counts = defaultdict(int)
    window = history[-20:]
    for item_id in window:
        cat = trie_stats.get_category(item_id)
        cat_counts[cat] += 1
    if not cat_counts or not window:
        return 0.0
    top_count = max(cat_counts.values())
    return float(top_count / len(window))


def _cat_entropy_factor(history, trie_stats, n_categories=18):
    """Entropy-based diversity measure: (1 - H / H_max).

    High entropy (diverse) → lower factor → less confident.
    """
    cat_counts = defaultdict(int)
    window = history[-20:]
    for item_id in window:
        cat = trie_stats.get_category(item_id)
        cat_counts[cat] += 1
    total = sum(cat_counts.values())
    if total == 0:
        return 0.0
    cat_entropy = -sum((c / total) * np.log2(c / total)
                       for c in cat_counts.values() if c > 0)
    max_entropy = np.log2(max(n_categories, 2))
    return float(np.clip(1.0 - cat_entropy / max_entropy, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Five confidence formulations
# ---------------------------------------------------------------------------

def compute_ctr_only(history, candidates, trie_stats, k=5):
    """Formulation 1: CTR gap only — ignores category signal entirely.

    conf = (avg_top_k_ctr - avg_rest_ctr) / avg_top_k_ctr
    """
    return _ctr_gap(history, candidates, trie_stats, k=k)


def compute_category_only(history, candidates, trie_stats, k=5):
    """Formulation 2: Category preference strength only — ignores CTR gap.

    conf = count(top_cat_in_history) / len(history[-20:])
    """
    return _pref_strength(history, trie_stats)


def compute_product_ours(history, candidates, trie_stats, k=5):
    """Formulation 3 (OURS): pref_strength × CTR_gap — current code formula.

    Both signals must be strong simultaneously (AND logic via product).
    """
    return float(_pref_strength(history, trie_stats) *
                 _ctr_gap(history, candidates, trie_stats, k=k))


def compute_additive(history, candidates, trie_stats, k=5):
    """Formulation 4: Arithmetic mean of pref_strength and CTR_gap.

    conf = (pref_strength + CTR_gap) / 2
    Allows one strong signal to compensate for a weak other (OR-like logic).
    """
    return float((_pref_strength(history, trie_stats) +
                  _ctr_gap(history, candidates, trie_stats, k=k)) / 2.0)


def compute_entropy_weighted(history, candidates, trie_stats, k=5):
    """Formulation 5: Entropy-attenuated CTR gap.

    conf = (1 - H/H_max) × CTR_gap
    Penalises diverse histories regardless of top-category count.
    """
    return float(_cat_entropy_factor(history, trie_stats) *
                 _ctr_gap(history, candidates, trie_stats, k=k))


CONF_FORMULATIONS = {
    'ctr_only':         compute_ctr_only,
    'category_only':    compute_category_only,
    'product_ours':     compute_product_ours,
    'additive':         compute_additive,
    'entropy_weighted': compute_entropy_weighted,
}


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def trie_recommend(history, candidates, trie_stats, k=5):
    """Pure Trie recommendation — mirrors EarlyExitChecker scoring."""
    cat_counts = defaultdict(int)
    for h in history[-20:]:
        cat_counts[trie_stats.get_category(h)] += 1
    top_cat = max(cat_counts, key=cat_counts.get) if cat_counts else None

    scores = []
    for cid in candidates:
        ctr = trie_stats.get_ctr(cid)
        cat = trie_stats.get_category(cid)
        cat_match = 1.0 if cat == top_cat else 0.3
        scores.append((cid, cat_match * ctr))
    scores.sort(key=lambda x: x[1], reverse=True)
    return [cid for cid, _ in scores[:k]]


def evaluate_routing_formulation(
        samples, news_items, all_items, llm_ranker, trie_stats,
        conf_fn, tau, pool_size=20, k=5, seed=42):
    """Evaluate one (formulation, threshold) pair.

    Returns dict with Hit@5, exit_rate, latency_ms, exit_hit@5.
    """
    np.random.seed(seed)

    hits_all = []
    hits_exited = []   # quality of early-exit queries
    hits_llm = []      # quality of LLM-invoked queries
    n_exited = 0
    latency_ms_sum = 0.0

    for sample in samples:
        gt = sample['ground_truth']
        history = sample['history']
        neg_items = [it for it in all_items
                     if it != gt and it not in history]
        n_neg = min(pool_size - 1, len(neg_items))
        candidates = [gt] + list(np.random.choice(neg_items, size=n_neg, replace=False))
        np.random.shuffle(candidates)

        # Compute confidence with this formulation
        conf = conf_fn(history, candidates, trie_stats, k=k)

        if conf >= tau:
            # Early exit — Trie recommendation
            t0 = time.perf_counter()
            recs = trie_recommend(history, candidates, trie_stats, k=k)
            latency_ms_sum += (time.perf_counter() - t0) * 1000 + 0.2
            hit = 1 if gt in recs else 0
            hits_exited.append(hit)
            n_exited += 1
        else:
            # Full LLM path
            t0 = time.perf_counter()
            history_text = " | ".join(
                news_items.get(h, {}).get('title', '')[:30]
                for h in history[-5:])
            cand_texts = [(cid, news_items.get(cid, {}).get('title', '')[:50])
                          for cid in candidates]
            llm_scores = llm_ranker.score_candidates(history_text, cand_texts)
            latency_ms_sum += (time.perf_counter() - t0) * 1000
            llm_scores.sort(key=lambda x: x[1], reverse=True)
            recs = [cid for cid, _ in llm_scores[:k]]
            hit = 1 if gt in recs else 0
            hits_llm.append(hit)

        hits_all.append(hit)

    n_total = len(samples)
    exit_rate = n_exited / n_total if n_total > 0 else 0.0
    return {
        'hit@5': float(np.mean(hits_all)),
        'exit_rate': float(exit_rate),
        'latency_ms': float(latency_ms_sum / n_total),
        'exited_hit@5': float(np.mean(hits_exited)) if hits_exited else 0.0,
        'llm_hit@5': float(np.mean(hits_llm)) if hits_llm else 0.0,
        'n_exited': n_exited,
        'n_total': n_total,
    }


def run_ablation(args):
    """Main ablation loop: 5 formulations × 3 seeds × τ sweep."""
    from src.data.mind_loader import load_mind_for_trie_experiment  # type: ignore

    logger.info("Loading MIND dataset...")
    samples, news_items, all_items = load_mind_for_trie_experiment(
        n_samples=args.samples, seed=42)
    all_items = list(all_items)

    logger.info(f"Loaded {len(news_items)} news, {len(samples)} samples")

    trie_stats = TrieStatistics(news_items, samples)
    llm_ranker = LocalLLMRanker(device=args.device)

    seeds = [int(s) for s in args.seeds.split(',')]

    # Threshold sweep values
    tau_grid = [0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]

    all_results = {}

    for name, conf_fn in CONF_FORMULATIONS.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Formulation: {name}")
        logger.info(f"{'='*60}")

        # First: sweep τ on seed=42 to find optimal threshold
        tau_sweep = {}
        logger.info(f"  Sweeping τ ∈ {tau_grid} ...")
        for tau in tau_grid:
            res = evaluate_routing_formulation(
                samples, news_items, all_items, llm_ranker, trie_stats,
                conf_fn, tau, pool_size=args.pool_size, seed=42)
            tau_sweep[tau] = res
            logger.info(f"    τ={tau:.2f}: Hit@5={res['hit@5']:.3f}, "
                        f"exit={res['exit_rate']*100:.1f}%, lat={res['latency_ms']:.1f}ms")

        # Find τ* = best Hit@5
        tau_opt = max(tau_sweep, key=lambda t: tau_sweep[t]['hit@5'])
        logger.info(f"  → Optimal τ*={tau_opt} → Hit@5={tau_sweep[tau_opt]['hit@5']:.3f}")

        # Multi-seed evaluation at τ*
        seed_results = []
        for seed in seeds:
            res = evaluate_routing_formulation(
                samples, news_items, all_items, llm_ranker, trie_stats,
                conf_fn, tau_opt, pool_size=args.pool_size, seed=seed)
            seed_results.append(res)
            logger.info(f"  Seed {seed}: Hit@5={res['hit@5']:.3f}, "
                        f"exit={res['exit_rate']*100:.1f}%, lat={res['latency_ms']:.1f}ms")

        hit5_vals = [r['hit@5'] for r in seed_results]
        exit_vals = [r['exit_rate'] for r in seed_results]
        lat_vals = [r['latency_ms'] for r in seed_results]
        exit_hit_vals = [r['exited_hit@5'] for r in seed_results if r['n_exited'] > 0]

        all_results[name] = {
            'tau_opt': tau_opt,
            'tau_sweep': {str(k): v for k, v in tau_sweep.items()},
            'multi_seed': seed_results,
            'summary': {
                'hit@5_mean': float(np.mean(hit5_vals)),
                'hit@5_std': float(np.std(hit5_vals)),
                'exit_rate_mean': float(np.mean(exit_vals)),
                'exit_rate_std': float(np.std(exit_vals)),
                'latency_ms_mean': float(np.mean(lat_vals)),
                'exited_hit@5_mean': float(np.mean(exit_hit_vals)) if exit_hit_vals else 0.0,
            }
        }

    return all_results


def print_summary_table(results: Dict):
    """Print a LaTeX-ready summary table."""
    logger.info("\n" + "="*80)
    logger.info("CONFIDENCE FORMULATION ABLATION SUMMARY")
    logger.info("="*80)
    header = f"{'Formulation':<22} {'τ*':>5} {'Hit@5':>12} {'Exit Rate':>12} {'Lat (ms)':>12} {'Exit Hit@5':>12}"
    logger.info(header)
    logger.info("-" * 80)

    display_names = {
        'ctr_only':        'CTR-only',
        'category_only':   'Category-only',
        'product_ours':    'Product (Ours) ★',
        'additive':        'Additive',
        'entropy_weighted': 'Entropy-weighted',
    }
    for name, res in results.items():
        s = res['summary']
        dname = display_names.get(name, name)
        logger.info(
            f"{dname:<22} {res['tau_opt']:>5.2f} "
            f"{s['hit@5_mean']:>6.3f}±{s['hit@5_std']:.3f} "
            f"{s['exit_rate_mean']*100:>8.1f}%±{s['exit_rate_std']*100:.1f}% "
            f"{s['latency_ms_mean']:>10.1f}ms "
            f"{s['exited_hit@5_mean']:>10.3f}")

    logger.info("\nLaTeX table row format (for paper):")
    logger.info("\\midrule")
    for name, res in results.items():
        s = res['summary']
        dname = display_names.get(name, name).replace("★", "").strip()
        bold = name == 'product_ours'
        prefix = "\\textbf{" if bold else ""
        suffix = "}" if bold else ""
        logger.info(
            f"{prefix}{dname}{suffix} & {res['tau_opt']:.2f} & "
            f"{prefix}{s['hit@5_mean']:.3f}$\\pm${s['hit@5_std']:.3f}{suffix} & "
            f"{s['exit_rate_mean']*100:.1f}\\% & "
            f"{s['latency_ms_mean']:.1f}ms & "
            f"{s['exited_hit@5_mean']:.3f} \\\\")


def main():
    parser = argparse.ArgumentParser(description='Confidence formulation ablation for KDD rebuttal')
    parser.add_argument('--data_dir', default='data/mind/MINDlarge_train',
                        help='Path to MIND dataset directory')
    parser.add_argument('--samples', type=int, default=2000,
                        help='Number of evaluation samples per run')
    parser.add_argument('--pool_size', type=int, default=20,
                        help='Candidate pool size')
    parser.add_argument('--device', default='cuda:0',
                        help='Device for LLM inference')
    parser.add_argument('--seeds', default='42,43,44',
                        help='Comma-separated random seeds')
    parser.add_argument('--output_dir', default='results/kdd_response_confidence_ablation',
                        help='Output directory for results')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Confidence Ablation | samples={args.samples} | seeds={args.seeds}")
    logger.info(f"Device: {args.device}")

    results = run_ablation(args)

    print_summary_table(results)

    # Save full results
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = os.path.join(args.output_dir, f'confidence_ablation_{ts}.json')
    with open(out_path, 'w') as f:
        json.dump({
            'timestamp': ts,
            'args': vars(args),
            'results': results,
        }, f, indent=2)
    logger.info(f"\nResults saved to: {out_path}")


if __name__ == '__main__':
    main()
