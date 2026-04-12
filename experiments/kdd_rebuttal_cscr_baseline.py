"""
RecSys 2026: CSCR-inspired Learned Router Baseline.

Implements the core idea of Cost-Spectrum Contrastive Routing (NeurIPS'25)
adapted for recommendation: train a lightweight router that learns which
queries benefit from LLM reasoning vs statistical methods.

Key idea from CSCR: embed queries into a space where distance to expert
centroids predicts the best cost-quality tradeoff.

Our adaptation:
- Expert 0: Trie-only (cost=0.1ms)
- Expert 1: Trie+LLM (cost=212ms)
- Router: XGBoost on Trie-derived features → binary classification
- Features: category_entropy, ctr_gap, history_diversity, max_ctr, etc.

Comparison with our zero-training Trie router.

Usage:
    python experiments/kdd_rebuttal_cscr_baseline.py --samples 2000 --device cuda:0
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
from typing import Dict, List
from collections import defaultdict, Counter
from datetime import datetime
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score

sys.path.insert(0, str(Path(__file__).parent.parent))
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
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


def extract_routing_features(sample, trie_stats, news_items, candidates):
    """Extract features for the learned router.

    These features capture query difficulty — whether Trie alone suffices
    or LLM reasoning is needed.
    """
    history = sample['history']

    # 1. User category diversity (high → harder, may need LLM)
    user_cats = Counter()
    for h in history[-15:]:
        cat = trie_stats.get_category(h)
        user_cats[cat] += 1
    n_cats = len(user_cats)
    cat_entropy = -sum((c / len(history[-15:])) * np.log2(c / len(history[-15:]))
                       for c in user_cats.values() if c > 0) if user_cats else 0

    # 2. Top category dominance (high → easy, Trie can handle)
    if user_cats:
        top_cat_frac = max(user_cats.values()) / sum(user_cats.values())
    else:
        top_cat_frac = 0

    # 3. Candidate CTR statistics
    ctrs = [trie_stats.get_ctr(c) for c in candidates]
    mean_ctr = np.mean(ctrs) if ctrs else 0
    max_ctr = max(ctrs) if ctrs else 0
    ctr_std = np.std(ctrs) if ctrs else 0

    # 4. CTR gap between top candidates
    sorted_ctrs = sorted(ctrs, reverse=True)
    ctr_gap = (sorted_ctrs[0] - sorted_ctrs[1]) if len(sorted_ctrs) >= 2 else 0

    # 5. Category match between history and candidates
    cand_cats = Counter(trie_stats.get_category(c) for c in candidates)
    cat_overlap = sum(min(user_cats.get(c, 0), cand_cats.get(c, 0))
                      for c in set(user_cats) | set(cand_cats))
    cat_overlap_frac = cat_overlap / max(len(candidates), 1)

    # 6. History length
    hist_len = len(history)

    # 7. Title length diversity in candidates (proxy for semantic richness)
    title_lens = [len(news_items.get(c, {}).get('title', '').split())
                  for c in candidates]
    avg_title_len = np.mean(title_lens) if title_lens else 0

    return np.array([
        n_cats,              # user category diversity
        cat_entropy,         # category entropy
        top_cat_frac,        # top category dominance
        mean_ctr,            # mean candidate CTR
        max_ctr,             # max candidate CTR
        ctr_std,             # CTR variance
        ctr_gap,             # gap between top-1 and top-2 CTR
        cat_overlap_frac,    # category overlap
        hist_len,            # history length
        avg_title_len,       # candidate title richness
    ])


FEATURE_NAMES = [
    'n_user_cats', 'cat_entropy', 'top_cat_dominance',
    'mean_ctr', 'max_ctr', 'ctr_std', 'ctr_gap',
    'cat_overlap', 'hist_len', 'avg_title_len',
]


def run_cscr_experiment(samples, news_items, all_items, device, seed, pool_size=20):
    """Phase 1: Collect oracle labels. Phase 2: Train router. Phase 3: Evaluate."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    config = ExperimentConfig(
        n_samples=len(samples), candidate_pool_size=pool_size,
        device=device, early_exit_threshold=0.5, history_length=15)
    trie_stats = TrieStatistics(news_items, samples)
    llm_ranker = LocalLLMRanker(device=device)

    # ---- Phase 1: Collect data for both experts ----
    logger.info("Phase 1: Collecting expert performance data...")
    features_list = []
    labels_list = []  # 1 = LLM better, 0 = Trie better
    trie_results = []
    llm_results = []

    for i, sample in enumerate(samples):
        neg_items = [it for it in all_items if it != sample['ground_truth'] and it not in sample['history']]
        n_neg = min(pool_size - 1, len(neg_items))
        candidates = [sample['ground_truth']] + list(np.random.choice(neg_items, size=n_neg, replace=False))
        np.random.shuffle(candidates)

        gt = sample['ground_truth']

        # Extract features
        feats = extract_routing_features(sample, trie_stats, news_items, candidates)
        features_list.append(feats)

        # Trie-only result
        user_cats = defaultdict(int)
        for h in sample['history'][-15:]:
            cat = trie_stats.get_category(h)
            user_cats[cat] += 1
        trie_scores = []
        for cid in candidates:
            ctr = trie_stats.get_ctr(cid)
            cat = trie_stats.get_category(cid)
            cat_match = 1.0 + user_cats.get(cat, 0) * 0.2
            trie_scores.append((cid, ctr * cat_match))
        trie_scores.sort(key=lambda x: x[1], reverse=True)
        trie_recs = [cid for cid, _ in trie_scores[:10]]
        trie_hit = 1 if gt in trie_recs[:5] else 0
        trie_results.append(trie_hit)

        # LLM result
        history_text = " | ".join([news_items.get(h, {}).get('title', '')[:30]
                                   for h in sample['history'][-5:]])
        cand_texts = [(cid, news_items.get(cid, {}).get('title', '')[:50])
                      for cid in candidates]
        llm_scores = llm_ranker.score_candidates(history_text, cand_texts)
        llm_scores.sort(key=lambda x: x[1], reverse=True)
        llm_recs = [cid for cid, _ in llm_scores[:10]]
        llm_hit = 1 if gt in llm_recs[:5] else 0
        llm_results.append(llm_hit)

        # Label: 1 if LLM is better (or both correct), 0 if Trie is better
        labels_list.append(1 if llm_hit > trie_hit else 0)

        if (i + 1) % 500 == 0:
            logger.info(f"  [{i+1}/{len(samples)}] Trie Hit@5={np.mean(trie_results):.3f}, "
                         f"LLM Hit@5={np.mean(llm_results):.3f}")

    X = np.array(features_list)
    y = np.array(labels_list)
    trie_hits = np.array(trie_results)
    llm_hits = np.array(llm_results)

    logger.info(f"\nOracle stats: Trie better={np.sum(y==0)}, LLM better={np.sum(y==1)}, "
                f"Both correct={np.sum((trie_hits==1)&(llm_hits==1))}")

    # ---- Phase 2: Train learned router (CSCR-style) ----
    logger.info("\nPhase 2: Training learned router...")

    # Split: 60% train, 40% test (like CSCR's evaluation protocol)
    split = int(len(X) * 0.6)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    trie_test = trie_hits[split:]
    llm_test = llm_hits[split:]

    # Train GBM router (lightweight, like CSCR's contrastive encoder)
    router = GradientBoostingClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.1, random_state=seed)
    router.fit(X_train, y_train)

    # Cross-val accuracy
    cv_scores = cross_val_score(router, X_train, y_train, cv=3, scoring='accuracy')
    logger.info(f"  Router CV accuracy: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")

    # Feature importance
    importances = sorted(zip(FEATURE_NAMES, router.feature_importances_),
                         key=lambda x: x[1], reverse=True)
    logger.info("  Feature importances:")
    for name, imp in importances[:5]:
        logger.info(f"    {name}: {imp:.3f}")

    # ---- Phase 3: Evaluate routing strategies ----
    logger.info("\nPhase 3: Evaluating routing strategies on test set...")

    predictions = router.predict(X_test)
    pred_proba = router.predict_proba(X_test)[:, 1]

    # Compute Hit@5 for each strategy
    results = {}

    # 1. Trie-only
    results['trie_only'] = {'hit@5': float(np.mean(trie_test)), 'latency_ms': 0.1,
                            'llm_calls_pct': 0.0}

    # 2. LLM-only
    results['llm_only'] = {'hit@5': float(np.mean(llm_test)), 'latency_ms': 350,
                           'llm_calls_pct': 100.0}

    # 3. Oracle
    oracle_hits = np.maximum(trie_test, llm_test)
    results['oracle'] = {'hit@5': float(np.mean(oracle_hits)), 'latency_ms': 200,
                         'llm_calls_pct': float(np.mean(y_test) * 100)}

    # 4. Our Trie router (threshold-based)
    # Use CTR gap as confidence proxy
    trie_confidence = X_test[:, 6]  # ctr_gap feature
    trie_threshold = np.percentile(trie_confidence, 50)  # median split
    trie_route = trie_confidence >= trie_threshold
    trie_router_hits = np.where(trie_route, trie_test, llm_test)
    trie_llm_rate = float(np.mean(~trie_route))
    results['trie_router'] = {
        'hit@5': float(np.mean(trie_router_hits)),
        'latency_ms': round(0.1 * np.mean(trie_route) + 350 * np.mean(~trie_route), 1),
        'llm_calls_pct': round(trie_llm_rate * 100, 1),
    }

    # 5. CSCR-style learned router
    cscr_route_to_llm = predictions == 1
    cscr_hits = np.where(cscr_route_to_llm, llm_test, trie_test)
    cscr_llm_rate = float(np.mean(cscr_route_to_llm))
    results['cscr_learned'] = {
        'hit@5': float(np.mean(cscr_hits)),
        'latency_ms': round(0.1 * np.mean(~cscr_route_to_llm) + 350 * cscr_llm_rate + 0.01, 1),
        # +0.01ms for router inference (CSCR uses FAISS k-NN, ~microseconds)
        'llm_calls_pct': round(cscr_llm_rate * 100, 1),
        'router_accuracy': float(np.mean(predictions == y_test)),
    }

    # 6. CSCR with cost-aware threshold (trade quality for cost)
    for threshold in [0.3, 0.5, 0.7]:
        cscr_route = pred_proba >= threshold
        hits = np.where(cscr_route, llm_test, trie_test)
        llm_rate = float(np.mean(cscr_route))
        results[f'cscr_t{threshold}'] = {
            'hit@5': float(np.mean(hits)),
            'latency_ms': round(0.1 * (1 - llm_rate) + 350 * llm_rate, 1),
            'llm_calls_pct': round(llm_rate * 100, 1),
        }

    # Print results
    logger.info(f"\n{'Method':<25} {'Hit@5':>8} {'Latency':>10} {'LLM%':>8}")
    logger.info("-" * 55)
    for method, data in sorted(results.items(), key=lambda x: x[1]['hit@5'], reverse=True):
        logger.info(f"{method:<25} {data['hit@5']:>8.4f} {data['latency_ms']:>8.1f}ms {data['llm_calls_pct']:>7.1f}%")

    del llm_ranker
    torch.cuda.empty_cache()

    return results, {'importances': importances, 'cv_accuracy': float(np.mean(cv_scores)),
                     'n_train': split, 'n_test': len(X_test)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", default="results/kdd_rebuttal_cscr")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("CSCR-STYLE LEARNED ROUTER vs TRIE ROUTER")
    logger.info("=" * 60)

    from src.data.mind_loader import load_mind_for_trie_experiment
    samples, news_items, all_items = load_mind_for_trie_experiment(args.samples, seed=42)

    results, router_info = run_cscr_experiment(
        samples, news_items, all_items, args.device, seed=42)

    final = {
        'timestamp': datetime.now().isoformat(),
        'n_samples': args.samples,
        'results': results,
        'router_info': {k: v for k, v in router_info.items() if k != 'importances'},
        'feature_importances': {name: round(float(imp), 4)
                                for name, imp in router_info['importances']},
    }

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"cscr_comparison_{ts}.json"
    with open(out_path, 'w') as f:
        json.dump(final, f, indent=2)
    logger.info(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
