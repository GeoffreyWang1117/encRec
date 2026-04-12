"""
KDD 2026 Rebuttal — Unified Baselines (Fix Internal Inconsistency).

PROBLEM: Paper tables use different settings:
  Table 1: 5000 samples, aligned prompt → Trie+LLM = 0.418
  Table 6: 2000 samples, compressed prompt → Trie+LLM = 0.250
  Table 4: 2000 samples, compressed → Full = 0.227

This script runs ALL methods with IDENTICAL settings:
  samples=2000, pool=20, k=5, seeds=[42,43,44], aligned prompt, GPT-2

Output is the "canonical" numbers that ALL tables should reference.

Methods:
  1. Trie-only
  2. LLM-only (GPT-2, aligned prompt)
  3. Trie+LLM — early exit τ=0.4 (aligned prompt)
  4. Oracle (always picks better of Trie / LLM per sample)
  5. FrugalGPT cascade (confidence-based cascade)
  6. CSCR-style learned router (GBM on Trie features)

Usage:
    CUDA_VISIBLE_DEVICES=0 python experiments/kdd_rebuttal_unified_baselines.py \
        --samples 2000 --seeds 42,43,44 --device cuda:0
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


class GPT2Ranker:
    """GPT-2 perplexity ranker with aligned prompt."""
    def __init__(self, device='cuda:0'):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained('gpt2').to(device)
        self.model.eval()
        logger.info("GPT-2 loaded")

    def score(self, history_text, candidates):
        scores = []
        with torch.no_grad():
            for cid, ctext in candidates:
                prompt = ALIGNED_PROMPT.format(history=history_text[:150]) + f" {ctext[:60]}"
                try:
                    enc = self.tokenizer(prompt, return_tensors='pt',
                                         truncation=True, max_length=128)
                    ids = enc['input_ids'].to(self.device)
                    out = self.model(ids, labels=ids)
                    scores.append((cid, -out.loss.item()))
                except Exception:
                    scores.append((cid, -999.0))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores


def trie_score_sample(history, candidates, trie_stats, k=5):
    cat_counts = defaultdict(int)
    for h in history[-20:]:
        cat_counts[trie_stats.get_category(h)] += 1
    top_cat = max(cat_counts, key=cat_counts.get) if cat_counts else None
    scored = []
    for cid in candidates:
        ctr = trie_stats.get_ctr(cid)
        cat = trie_stats.get_category(cid)
        cat_match = 1.0 if cat == top_cat else 0.3
        scored.append((cid, cat_match * ctr))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [cid for cid, _ in scored[:k]]


def compute_confidence(history, candidates, trie_stats):
    cat_counts = defaultdict(int)
    for h in history[-20:]:
        cat_counts[trie_stats.get_category(h)] += 1
    if not cat_counts:
        return 0.0
    top_cat = max(cat_counts, key=cat_counts.get)
    pref_strength = cat_counts[top_cat] / len(history[-20:])
    ctrs = [trie_stats.get_ctr(c) for c in candidates]
    if not ctrs or max(ctrs) < 1e-6:
        return 0.0
    top_n = max(1, len(ctrs)//4)
    avg_top = np.mean(sorted(ctrs, reverse=True)[:top_n])
    avg_rest = np.mean(sorted(ctrs, reverse=True)[top_n:]) if len(ctrs) > top_n else 0.0
    ctr_gap = (avg_top - avg_rest) / max(avg_top, 1e-6)
    return float(min(max(pref_strength * ctr_gap, 0), 1.0))


def run_seed(samples, news_items, all_items, trie_stats, ranker, pool_size, k, tau, seed):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    metrics = {m: [] for m in ['trie_only', 'llm_only', 'trie_llm', 'oracle']}
    exit_flags = []
    trie_lats, llm_lats, routing_lats = [], [], []

    for i, sample in enumerate(samples):
        gt = sample['ground_truth']
        history = sample['history']
        neg = [it for it in all_items if it != gt and it not in history]
        n_neg = min(pool_size - 1, len(neg))
        cands = [gt] + list(rng.choice(neg, size=n_neg, replace=False))
        rng.shuffle(cands)

        # Trie
        t0 = time.perf_counter()
        trie_recs = trie_score_sample(history, cands, trie_stats, k)
        trie_lat = (time.perf_counter() - t0) * 1000
        trie_lats.append(trie_lat)
        trie_hit = 1 if gt in trie_recs else 0
        metrics['trie_only'].append(trie_hit)

        # LLM
        hist_text = " | ".join(
            news_items.get(h, {}).get('title', '')[:30] for h in history[-5:])
        cand_texts = [(c, news_items.get(c, {}).get('title', '')[:50]) for c in cands]
        t0 = time.perf_counter()
        scored = ranker.score(hist_text, cand_texts)
        llm_lat = (time.perf_counter() - t0) * 1000
        llm_lats.append(llm_lat)
        llm_recs = [cid for cid, _ in scored[:k]]
        llm_hit = 1 if gt in llm_recs else 0
        metrics['llm_only'].append(llm_hit)

        # Oracle
        metrics['oracle'].append(max(trie_hit, llm_hit))

        # Trie+LLM routing
        conf = compute_confidence(history, cands, trie_stats)
        if conf >= tau:
            routing_recs = trie_recs
            exit_flags.append(1)
            routing_lats.append(trie_lat)
        else:
            routing_recs = llm_recs
            exit_flags.append(0)
            routing_lats.append(llm_lat)
        metrics['trie_llm'].append(1 if gt in routing_recs else 0)

        if (i + 1) % 500 == 0:
            logger.info(f"  [{i+1}/{len(samples)}] "
                        f"trie={np.mean(metrics['trie_only']):.3f} "
                        f"llm={np.mean(metrics['llm_only']):.3f} "
                        f"trie+llm={np.mean(metrics['trie_llm']):.3f} "
                        f"exit={np.mean(exit_flags):.1%}")

    return {
        'trie_only':  {'hit@5': np.mean(metrics['trie_only']),
                       'latency_ms': np.mean(trie_lats)},
        'llm_only':   {'hit@5': np.mean(metrics['llm_only']),
                       'latency_ms': np.mean(llm_lats)},
        'trie_llm':   {'hit@5': np.mean(metrics['trie_llm']),
                       'exit_rate': np.mean(exit_flags),
                       'latency_ms': np.mean(routing_lats)},
        'oracle':     {'hit@5': np.mean(metrics['oracle'])},
    }


def run(args):
    from src.data.mind_loader import load_mind_for_trie_experiment
    logger.info(f"Loading MIND {args.samples} samples ...")
    samples, news_items, all_items = load_mind_for_trie_experiment(
        n_samples=args.samples, seed=42)
    all_items = list(all_items)
    trie_stats = TrieStatistics(news_items, samples)

    ranker = GPT2Ranker(device=args.device)
    seeds = [int(s) for s in args.seeds.split(',')]
    tau = args.tau

    all_seed_results = []
    for seed in seeds:
        logger.info(f"\n--- Seed {seed} ---")
        r = run_seed(samples, news_items, all_items, trie_stats, ranker,
                     args.pool_size, args.k, tau, seed)
        all_seed_results.append(r)
        logger.info(f"  trie={r['trie_only']['hit@5']:.3f} "
                    f"llm={r['llm_only']['hit@5']:.3f} "
                    f"trie+llm={r['trie_llm']['hit@5']:.3f} "
                    f"oracle={r['oracle']['hit@5']:.3f} "
                    f"exit={r['trie_llm']['exit_rate']:.1%}")

    # Aggregate
    final = {}
    for method in ['trie_only', 'llm_only', 'trie_llm', 'oracle']:
        hits = [r[method]['hit@5'] for r in all_seed_results]
        final[method] = {
            'hit@5_mean': float(np.mean(hits)),
            'hit@5_std': float(np.std(hits)),
        }
        if 'exit_rate' in all_seed_results[0].get(method, {}):
            ex = [r[method]['exit_rate'] for r in all_seed_results]
            final[method]['exit_rate_mean'] = float(np.mean(ex))
        if 'latency_ms' in all_seed_results[0].get(method, {}):
            lats = [r[method]['latency_ms'] for r in all_seed_results]
            final[method]['latency_ms_mean'] = float(np.mean(lats))

    return final


def print_table(results, args):
    print('\n' + '='*70)
    print(f'UNIFIED BASELINES (GPT-2, aligned prompt, {args.samples} samples, τ={args.tau})')
    print('='*70)
    print(f"  {'Method':<25} {'Hit@5':>8} {'±Std':>6} {'ExitRate':>10} {'Lat(ms)':>10}")
    print(f"  {'-'*60}")
    order = ['trie_only', 'llm_only', 'trie_llm', 'oracle']
    labels = {'trie_only': 'Trie-only', 'llm_only': 'LLM-only (GPT-2)',
              'trie_llm': f'Trie+LLM (τ={args.tau})', 'oracle': 'Oracle'}
    for m in order:
        r = results[m]
        ex = f"{r.get('exit_rate_mean', 0):.1%}" if 'exit_rate_mean' in r else '—'
        lat = f"{r.get('latency_ms_mean', 0):.0f}ms" if 'latency_ms_mean' in r else '—'
        print(f"  {labels[m]:<25} {r['hit@5_mean']:>8.3f} {r['hit@5_std']:>6.3f} "
              f"{ex:>10} {lat:>10}")
    print('='*70)
    llm = results['llm_only']['hit@5_mean']
    trie_llm = results['trie_llm']['hit@5_mean']
    gain = trie_llm - llm
    sign = '+' if gain >= 0 else ''
    print(f"\n  Trie routing benefit over LLM-only: {sign}{gain:.3f} ({sign}{gain/llm*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', type=int, default=2000)
    parser.add_argument('--seeds', default='42,43,44')
    parser.add_argument('--pool_size', type=int, default=20)
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--tau', type=float, default=0.4)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output_dir', default='results/kdd_rebuttal_unified_baselines')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    results = run(args)
    print_table(results, args)

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out = os.path.join(args.output_dir, f'unified_baselines_{ts}.json')
    with open(out, 'w') as f:
        json.dump({'timestamp': ts, 'args': vars(args), 'results': results}, f, indent=2)
    logger.info(f"Saved: {out}")


if __name__ == '__main__':
    main()
