"""
Verification: reproduce the original Feb 7 model_scale "0.418" number at 5K×3.

The Feb 7 model_scale_20260207_171916.json reported GPT-2 LLM-only Hit@5 with
seed 456 = 0.418 at 500 samples. The other two seeds (42, 123) gave 0.436 and
0.386. This script repeats the *exact* original pipeline (LocalLLMRanker.score_
candidates from kdd_triellm_01_ablation, which uses the prompt
   "User interests: {history}. Recommend: {candidate}"
not the "User liked / User will also like" aligned prompt) at 5000 samples ×
3 seeds to test:

  H1: 0.418 reproduces at scale → 500-sample number is genuine
  H2: 0.418 collapses to ~0.25 at scale → 500-sample number was outlier noise

Either result definitively settles whether the paper's 0.418 claim is sound.

Usage:
    CUDA_VISIBLE_DEVICES=1 python experiments/kdd_rebuttal_verify_0418.py
"""

import os, sys, json, time, logging, argparse
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

import importlib.util
_spec = importlib.util.spec_from_file_location(
    "ablation", str(Path(__file__).parent / "kdd_triellm_01_ablation.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
LocalLLMRanker = _mod.LocalLLMRanker
TrieStatistics = _mod.TrieStatistics


def build_pool(sample, all_items, pool_size, rng):
    gt = sample['ground_truth']
    history = sample['history']
    neg = [it for it in all_items if it != gt and it not in history]
    n_neg = min(pool_size - 1, len(neg))
    cands = [gt] + list(rng.choice(neg, size=n_neg, replace=False))
    rng.shuffle(cands)
    return cands


def fmt_history(history, news_items):
    items = history[-10:]
    parts = []
    for h in items:
        item = news_items.get(h, {})
        title = item.get('title', '')
        cat = item.get('category', '')
        parts.append(f"[{cat}]{title[:30]}")
    return " -> ".join(parts)


def fmt_candidates(cands, news_items):
    out = []
    for cid in cands:
        item = news_items.get(cid, {})
        title = item.get('title', '')
        cat = item.get('category', '')
        out.append((cid, f"[{cat}]{title[:40]}"))
    return out


def evaluate_seed(samples, news_items, all_items, ranker, pool_size, k, seed):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    hits = []
    for i, sample in enumerate(samples):
        gt = sample['ground_truth']
        cands = build_pool(sample, all_items, pool_size, rng)
        hist_text = fmt_history(sample['history'], news_items)
        cand_texts = fmt_candidates(cands, news_items)
        # Original LocalLLMRanker uses use_compression=True by default (line 100):
        # prompt = "User interests: {hist[:100]}. Recommend: {cand[:50]}"
        scores = ranker.score_candidates(hist_text, cand_texts, use_compression=True)
        scores.sort(key=lambda x: x[1], reverse=True)
        recs = [cid for cid, _ in scores[:k]]
        hits.append(int(gt in recs))
        if (i + 1) % 500 == 0:
            logger.info(f"  [{i+1}/{len(samples)}] hit@5={np.mean(hits):.4f}")
    return float(np.mean(hits))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', type=int, default=5000)
    parser.add_argument('--seeds', default='42,123,456')
    parser.add_argument('--pool_size', type=int, default=20)
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output_dir', default='results/kdd_rebuttal_verify_0418')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    from src.data.mind_loader import load_mind_for_trie_experiment
    logger.info(f"Loading MIND {args.samples} samples ...")
    samples, news_items, all_items = load_mind_for_trie_experiment(
        n_samples=args.samples, seed=42)
    all_items = list(all_items)

    ranker = LocalLLMRanker(model_name='gpt2', device=args.device)

    seeds = [int(s) for s in args.seeds.split(',')]
    seed_hits = {}
    for seed in seeds:
        logger.info(f"\n--- Seed {seed} ---")
        h = evaluate_seed(samples, news_items, all_items, ranker,
                          args.pool_size, args.k, seed)
        seed_hits[seed] = h
        logger.info(f"  Seed {seed} → Hit@5 = {h:.4f}")

    vals = list(seed_hits.values())
    summary = {
        'samples': args.samples,
        'seeds': seeds,
        'per_seed': seed_hits,
        'mean': float(np.mean(vals)),
        'std': float(np.std(vals)),
        'note': ('Reproduces original kdd_triellm_11_model_scale GPT-2 LLM-only '
                 'pipeline ("User interests: ... Recommend: ..." prompt) at 5K×3 '
                 'to verify whether Feb 7 single-seed 0.418 was signal or noise.')
    }
    print('\n' + '='*60)
    print('VERIFICATION OF 0.418 (GPT-2 LLM-only, "interests/recommend" prompt)')
    print('='*60)
    for s, h in seed_hits.items():
        print(f"  seed {s}: Hit@5 = {h:.4f}")
    print(f"  mean = {summary['mean']:.4f} ± {summary['std']:.4f}")
    print('='*60)

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out = os.path.join(args.output_dir, f'verify_0418_{ts}.json')
    with open(out, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved: {out}")


if __name__ == '__main__':
    main()
