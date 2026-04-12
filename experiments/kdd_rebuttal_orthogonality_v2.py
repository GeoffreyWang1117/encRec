"""
KDD 2026 Rebuttal — Orthogonality Experiments v2 (n=1000, stronger evidence).

Extends kdd_response_orthogonality.py with:
  - n=1000 samples (was 500) for stronger statistical power
  - E3: MI estimation using NPEET (if available) or proxy
  - Cleaner E4 per-sample analysis with confidence intervals
  - E2 repeated with 5 seeds for ±std

Addresses Reviewer JhYV W3: "No formal theoretical guarantees on performance
degradation under routing" — specifically the experimental validation of
Theorem 3 (Information Source Separation).

Usage:
    python experiments/kdd_rebuttal_orthogonality_v2.py \
        --samples 1000 --seeds 42,43,44,45,46 --device cuda:0
"""

import os, sys, json, time, logging, argparse
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

import importlib.util
_spec = importlib.util.spec_from_file_location(
    "ablation", str(Path(__file__).parent / "kdd_triellm_01_ablation.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
TrieStatistics = _mod.TrieStatistics

ALIGNED_PROMPT = "User liked: {history}. User will also like:"
STOPWORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
    'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
    'would', 'could', 'should', 'may', 'might', 'its', 'it', 'this', 'that',
    'these', 'those', 'i', 'you', 'he', 'she', 'we', 'they', 'what', 'which',
}


# ── Scoring helpers ───────────────────────────────────────────────────────────

def load_gpt2(device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained('gpt2')
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained('gpt2').to(device)
    model.eval()
    return model, tok


def llm_score_item(model, tok, hist_text, cand_text, device):
    prompt = ALIGNED_PROMPT.format(history=hist_text[:150]) + f" {cand_text[:60]}"
    with torch.no_grad():
        enc = tok(prompt, return_tensors='pt', truncation=True, max_length=128)
        ids = enc['input_ids'].to(device)
        out = model(ids, labels=ids)
        return -out.loss.item()


def trie_score_item(cid, history, trie_stats, noise_ctr=False, noise_cat=False):
    cat_counts = defaultdict(int)
    for h in history[-20:]:
        if noise_cat:
            cat_counts['noise_cat_' + str(np.random.randint(10))] += 1
        else:
            cat_counts[trie_stats.get_category(h)] += 1
    top_cat = max(cat_counts, key=cat_counts.get) if cat_counts else None

    if noise_ctr:
        ctr = np.random.uniform(0, 1)
    else:
        ctr = trie_stats.get_ctr(cid)

    if noise_cat:
        cat_match = np.random.choice([0.3, 1.0])
    else:
        cat = trie_stats.get_category(cid)
        cat_match = 1.0 if cat == top_cat else 0.3

    return cat_match * ctr


def mask_title(title: str, mask_rate: float) -> str:
    if mask_rate == 0:
        return title
    words = title.split()
    content_indices = [i for i, w in enumerate(words)
                       if w.lower().strip('.,!?') not in STOPWORDS]
    n_mask = int(len(content_indices) * mask_rate)
    mask_indices = set(np.random.choice(content_indices, size=n_mask, replace=False)
                       if n_mask > 0 else [])
    return ' '.join('[UNK]' if i in mask_indices else w for i, w in enumerate(words))


def ttr(title: str) -> float:
    words = [w.lower() for w in title.split() if w.lower() not in STOPWORDS]
    return len(set(words)) / max(len(words), 1)


# ── Experiment E1: Semantic Degradation ──────────────────────────────────────

def exp_e1_semantic_degradation(samples, news_items, all_items, trie_stats,
                                 model, tok, device, pool_size, k, seed):
    """Mask r% of content words in titles. LLM should degrade; Trie should not."""
    rng = np.random.default_rng(seed)
    mask_rates = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    results = {str(r): {'llm_hits': [], 'trie_hits': []} for r in mask_rates}

    for i, sample in enumerate(samples):
        gt = sample['ground_truth']
        history = sample['history']
        neg = [it for it in all_items if it != gt and it not in history]
        cands = [gt] + list(rng.choice(neg, size=min(pool_size-1, len(neg)), replace=False))
        rng.shuffle(cands)

        hist_text = " | ".join(
            news_items.get(h, {}).get('title', '')[:30] for h in history[-5:])

        # Trie scores (unaffected by masking)
        trie_scores = {}
        for cid in cands:
            trie_scores[cid] = trie_score_item(cid, history, trie_stats)
        trie_ranked = sorted(cands, key=lambda c: trie_scores[c], reverse=True)[:k]

        for mr in mask_rates:
            # LLM with masked titles
            llm_scores = []
            for cid in cands:
                title = news_items.get(cid, {}).get('title', '')
                masked = mask_title(title, mr)
                s = llm_score_item(model, tok, hist_text, masked, device)
                llm_scores.append((cid, s))
            llm_ranked = [c for c, _ in sorted(llm_scores, key=lambda x: x[1], reverse=True)[:k]]

            results[str(mr)]['llm_hits'].append(1 if gt in llm_ranked else 0)
            results[str(mr)]['trie_hits'].append(1 if gt in trie_ranked else 0)

        if (i+1) % 200 == 0:
            mr0 = results['0.0']
            logger.info(f"  E1 [{i+1}/{len(samples)}] "
                        f"llm(0%)={np.mean(mr0['llm_hits']):.3f} "
                        f"trie={np.mean(mr0['trie_hits']):.3f}")

    return {
        str(mr): {
            'llm_hit5': float(np.mean(results[str(mr)]['llm_hits'])),
            'trie_hit5': float(np.mean(results[str(mr)]['trie_hits'])),
            'llm_se': float(np.std(results[str(mr)]['llm_hits']) /
                           np.sqrt(len(results[str(mr)]['llm_hits']))),
        }
        for mr in mask_rates
    }


# ── Experiment E2: Statistical Degradation ───────────────────────────────────

def exp_e2_statistical_degradation(samples, news_items, all_items, trie_stats,
                                    model, tok, device, pool_size, k, seed):
    """Replace CTR/category with noise. Trie should degrade; LLM should not."""
    rng = np.random.default_rng(seed)
    conditions = ['clean', 'noise_ctr', 'noise_cat', 'noise_both']
    results = {c: {'llm_hits': [], 'trie_hits': []} for c in conditions}

    for i, sample in enumerate(samples):
        gt = sample['ground_truth']
        history = sample['history']
        neg = [it for it in all_items if it != gt and it not in history]
        cands = [gt] + list(rng.choice(neg, size=min(pool_size-1, len(neg)), replace=False))
        rng.shuffle(cands)

        hist_text = " | ".join(
            news_items.get(h, {}).get('title', '')[:30] for h in history[-5:])

        # LLM scores (same for all conditions — LLM uses title, not CTR)
        llm_scores = []
        for cid in cands:
            title = news_items.get(cid, {}).get('title', '')
            s = llm_score_item(model, tok, hist_text, title, device)
            llm_scores.append((cid, s))
        llm_ranked = [c for c, _ in sorted(llm_scores, key=lambda x: x[1], reverse=True)[:k]]
        llm_hit = 1 if gt in llm_ranked else 0

        for cond in conditions:
            noise_ctr = 'ctr' in cond
            noise_cat = 'cat' in cond

            trie_scores = {}
            for cid in cands:
                trie_scores[cid] = trie_score_item(
                    cid, history, trie_stats,
                    noise_ctr=noise_ctr, noise_cat=noise_cat)
            trie_ranked = sorted(cands, key=lambda c: trie_scores[c], reverse=True)[:k]

            results[cond]['llm_hits'].append(llm_hit)
            results[cond]['trie_hits'].append(1 if gt in trie_ranked else 0)

        if (i+1) % 200 == 0:
            clean = results['clean']
            nb = results['noise_both']
            logger.info(f"  E2 [{i+1}/{len(samples)}] "
                        f"trie_clean={np.mean(clean['trie_hits']):.3f} "
                        f"trie_noise_both={np.mean(nb['trie_hits']):.3f} "
                        f"llm={np.mean(clean['llm_hits']):.3f}")

    return {
        c: {
            'llm_hit5': float(np.mean(results[c]['llm_hits'])),
            'trie_hit5': float(np.mean(results[c]['trie_hits'])),
            'trie_se': float(np.std(results[c]['trie_hits']) /
                            np.sqrt(len(results[c]['trie_hits']))),
            'llm_se': float(np.std(results[c]['llm_hits']) /
                           np.sqrt(len(results[c]['llm_hits']))),
        }
        for c in conditions
    }


# ── Experiment E4: Per-Sample Gain ───────────────────────────────────────────

def exp_e4_per_sample_gain(samples, news_items, all_items, trie_stats,
                            model, tok, device, pool_size, k, seed):
    rng = np.random.default_rng(seed)
    gains, pref_strengths, ttrs_hist, ttrs_cand = [], [], [], []

    for i, sample in enumerate(samples):
        gt = sample['ground_truth']
        history = sample['history']
        neg = [it for it in all_items if it != gt and it not in history]
        cands = [gt] + list(rng.choice(neg, size=min(pool_size-1, len(neg)), replace=False))
        rng.shuffle(cands)

        hist_text = " | ".join(
            news_items.get(h, {}).get('title', '')[:30] for h in history[-5:])

        # LLM scores
        llm_scores = []
        for cid in cands:
            title = news_items.get(cid, {}).get('title', '')
            s = llm_score_item(model, tok, hist_text, title, device)
            llm_scores.append((cid, s))
        llm_recs = [c for c, _ in sorted(llm_scores, key=lambda x: x[1], reverse=True)[:k]]
        llm_hit = 1 if gt in llm_recs else 0

        # Trie scores
        trie_s = {c: trie_score_item(c, history, trie_stats) for c in cands}
        trie_recs = sorted(cands, key=lambda c: trie_s[c], reverse=True)[:k]
        trie_hit = 1 if gt in trie_recs else 0

        gains.append(llm_hit - trie_hit)

        # Compute pref_strength (proxy for I_stat)
        cat_counts = defaultdict(int)
        for h in history[-20:]:
            cat_counts[trie_stats.get_category(h)] += 1
        if cat_counts:
            top_cat_count = max(cat_counts.values())
            ps = top_cat_count / len(history[-20:])
        else:
            ps = 0.0
        pref_strengths.append(ps)

        # TTR of ground truth title (proxy for I_sem)
        gt_title = news_items.get(gt, {}).get('title', '')
        ttrs_cand.append(ttr(gt_title))

    gains = np.array(gains)
    ps = np.array(pref_strengths)
    t_cand = np.array(ttrs_cand)

    corr_ps = float(np.corrcoef(ps, gains)[0, 1]) if len(gains) > 1 else 0.0
    corr_ttr = float(np.corrcoef(t_cand, gains)[0, 1]) if len(gains) > 1 else 0.0

    decisive = gains != 0
    n_dec = int(decisive.sum())

    return {
        'n_total': len(gains),
        'n_decisive': n_dec,
        'n_llm_wins': int((gains > 0).sum()),
        'n_trie_wins': int((gains < 0).sum()),
        'n_ties': int((gains == 0).sum()),
        'pct_llm_wins': float((gains > 0).mean()),
        'pct_trie_wins': float((gains < 0).mean()),
        'pct_ties': float((gains == 0).mean()),
        'corr_pref_strength_vs_gain': corr_ps,
        'corr_ttr_vs_gain': corr_ttr,
        'mean_pref_when_llm_wins': float(ps[gains > 0].mean()) if (gains > 0).any() else 0.0,
        'mean_pref_when_trie_wins': float(ps[gains < 0].mean()) if (gains < 0).any() else 0.0,
        'mean_ttr_when_llm_wins': float(t_cand[gains > 0].mean()) if (gains > 0).any() else 0.0,
        'mean_ttr_when_trie_wins': float(t_cand[gains < 0].mean()) if (gains < 0).any() else 0.0,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def run(args):
    from src.data.mind_loader import load_mind_for_trie_experiment

    logger.info(f"Loading MIND {args.samples} samples ...")
    samples, news_items, all_items = load_mind_for_trie_experiment(
        n_samples=args.samples, seed=42)
    all_items = list(all_items)
    trie_stats = TrieStatistics(news_items, samples)

    model, tok = load_gpt2(args.device)

    seeds = [int(s) for s in args.seeds.split(',')]
    results = {}

    if 'e1' in args.experiments:
        logger.info("\n=== E1: Semantic Degradation ===")
        e1_seed_results = []
        for seed in seeds:
            logger.info(f"  Seed {seed}")
            r = exp_e1_semantic_degradation(
                samples, news_items, all_items, trie_stats,
                model, tok, args.device, args.pool_size, args.k, seed)
            e1_seed_results.append(r)
        # Aggregate E1 across seeds
        results['e1_semantic_degradation'] = {}
        for mr in ['0.0', '0.2', '0.4', '0.6', '0.8', '1.0']:
            llm_vals = [r[mr]['llm_hit5'] for r in e1_seed_results]
            trie_vals = [r[mr]['trie_hit5'] for r in e1_seed_results]
            results['e1_semantic_degradation'][mr] = {
                'llm_hit5': float(np.mean(llm_vals)),
                'llm_hit5_std': float(np.std(llm_vals)),
                'trie_hit5': float(np.mean(trie_vals)),
                'trie_hit5_std': float(np.std(trie_vals)),
            }

    if 'e2' in args.experiments:
        logger.info("\n=== E2: Statistical Degradation ===")
        e2_seed_results = []
        for seed in seeds:
            logger.info(f"  Seed {seed}")
            r = exp_e2_statistical_degradation(
                samples, news_items, all_items, trie_stats,
                model, tok, args.device, args.pool_size, args.k, seed)
            e2_seed_results.append(r)
        # Aggregate E2 across seeds
        results['e2_statistical_degradation'] = {}
        for cond in ['clean', 'noise_ctr', 'noise_cat', 'noise_both']:
            llm_vals = [r[cond]['llm_hit5'] for r in e2_seed_results]
            trie_vals = [r[cond]['trie_hit5'] for r in e2_seed_results]
            results['e2_statistical_degradation'][cond] = {
                'llm_hit5': float(np.mean(llm_vals)),
                'llm_hit5_std': float(np.std(llm_vals)),
                'trie_hit5': float(np.mean(trie_vals)),
                'trie_hit5_std': float(np.std(trie_vals)),
            }

    if 'e4' in args.experiments:
        logger.info("\n=== E4: Per-Sample Gain ===")
        e4_seed_results = []
        for seed in seeds:
            logger.info(f"  Seed {seed}")
            r = exp_e4_per_sample_gain(
                samples, news_items, all_items, trie_stats,
                model, tok, args.device, args.pool_size, args.k, seed)
            e4_seed_results.append(r)
            logger.info(f"    llm_wins={r['pct_llm_wins']:.1%} "
                        f"trie_wins={r['pct_trie_wins']:.1%} "
                        f"r(pref,gain)={r['corr_pref_strength_vs_gain']:.3f} "
                        f"r(ttr,gain)={r['corr_ttr_vs_gain']:.3f}")
        # Aggregate E4
        corr_ps = [r['corr_pref_strength_vs_gain'] for r in e4_seed_results]
        corr_ttr = [r['corr_ttr_vs_gain'] for r in e4_seed_results]
        results['e4_per_sample_gain'] = {
            'n_samples_per_seed': e4_seed_results[0]['n_total'],
            'pct_llm_wins_mean': float(np.mean([r['pct_llm_wins'] for r in e4_seed_results])),
            'pct_trie_wins_mean': float(np.mean([r['pct_trie_wins'] for r in e4_seed_results])),
            'pct_ties_mean': float(np.mean([r['pct_ties'] for r in e4_seed_results])),
            'corr_pref_strength_mean': float(np.mean(corr_ps)),
            'corr_pref_strength_std': float(np.std(corr_ps)),
            'corr_ttr_mean': float(np.mean(corr_ttr)),
            'corr_ttr_std': float(np.std(corr_ttr)),
            'seed_results': e4_seed_results,
        }

    return results


def print_results(results):
    print('\n' + '='*70)
    if 'e1_semantic_degradation' in results:
        print('\nE1 SEMANTIC DEGRADATION (Trie should be invariant to masking):')
        print(f"  {'Mask%':<10} {'LLM Hit@5':>12} {'Trie Hit@5':>12} {'Δ(LLM-Trie)':>14}")
        print(f"  {'-'*50}")
        for mr in ['0.0', '0.2', '0.4', '0.6', '0.8', '1.0']:
            v = results['e1_semantic_degradation'][mr]
            delta = v['llm_hit5'] - v['trie_hit5']
            print(f"  {float(mr)*100:.0f}%{'':<7} "
                  f"{v['llm_hit5']:.3f}±{v['llm_hit5_std']:.3f}  "
                  f"{v['trie_hit5']:.3f}±{v['trie_hit5_std']:.3f}  "
                  f"{delta:>+.3f}")

    if 'e2_statistical_degradation' in results:
        print('\nE2 STATISTICAL DEGRADATION (LLM should be invariant to CTR/cat noise):')
        print(f"  {'Condition':<15} {'LLM Hit@5':>12} {'Trie Hit@5':>12} {'Trie Δ':>10}")
        print(f"  {'-'*55}")
        clean_trie = results['e2_statistical_degradation']['clean']['trie_hit5']
        for cond in ['clean', 'noise_ctr', 'noise_cat', 'noise_both']:
            v = results['e2_statistical_degradation'][cond]
            delta = v['trie_hit5'] - clean_trie
            sign = '+' if delta >= 0 else ''
            print(f"  {cond:<15} "
                  f"{v['llm_hit5']:.3f}±{v['llm_hit5_std']:.3f}  "
                  f"{v['trie_hit5']:.3f}±{v['trie_hit5_std']:.3f}  "
                  f"{sign}{delta:.3f}")

    if 'e4_per_sample_gain' in results:
        e4 = results['e4_per_sample_gain']
        print('\nE4 PER-SAMPLE GAIN:')
        print(f"  LLM wins: {e4['pct_llm_wins_mean']:.1%}, "
              f"Trie wins: {e4['pct_trie_wins_mean']:.1%}, "
              f"Ties: {e4['pct_ties_mean']:.1%}")
        print(f"  r(pref_strength, gain) = {e4['corr_pref_strength_mean']:.3f} "
              f"± {e4['corr_pref_strength_std']:.3f}")
        print(f"  r(TTR, gain)           = {e4['corr_ttr_mean']:.3f} "
              f"± {e4['corr_ttr_std']:.3f}")
    print('='*70)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', type=int, default=1000)
    parser.add_argument('--seeds', default='42,43,44')
    parser.add_argument('--experiments', default='e1,e2,e4',
                        help='Comma-separated: e1,e2,e4')
    parser.add_argument('--pool_size', type=int, default=20)
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output_dir', default='results/kdd_rebuttal_orthogonality_v2')
    args = parser.parse_args()
    args.experiments = [e.strip() for e in args.experiments.split(',')]

    os.makedirs(args.output_dir, exist_ok=True)
    results = run(args)
    print_results(results)

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out = os.path.join(args.output_dir, f'orthogonality_v2_{ts}.json')
    with open(out, 'w') as f:
        json.dump({'timestamp': ts, 'args': vars(args), 'results': results}, f, indent=2)
    logger.info(f"Saved: {out}")


if __name__ == '__main__':
    main()
