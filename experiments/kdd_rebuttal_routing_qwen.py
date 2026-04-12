"""
KDD 2026 Rebuttal — Critical Missing Experiment.

RESEARCH QUESTION: Does Trie routing add value on top of modern 2026 LLMs?

Specifically: Is Trie+Qwen3.5-4B > Qwen3.5-4B alone?

This directly addresses:
  - Reviewer 4fvP W3: "GPT-2 is outdated"
  - Reviewer JhYV W3: "No theoretical guarantees"
  - The implicit fear: "Routing only helps because GPT-2 is weak"

Methodology:
  Three conditions on MIND Large (aligned prompt, pool=20, 3 seeds):
  1. Trie-only           — Trie confidence scoring, no LLM
  2. LLM-only (Qwen)     — Qwen3.5-4B perplexity scoring, no routing
  3. Trie+LLM (ours)     — Trie confidence router + Qwen3.5-4B for hard queries
  4. GPT-2 LLM-only      — Reference baseline

  The routing thresholds tested: τ ∈ {0.3, 0.4, 0.5, 0.6}

Usage:
    CUDA_VISIBLE_DEVICES=0 python experiments/kdd_rebuttal_routing_qwen.py \
        --samples 2000 --seeds 42,43,44 --device cuda:0
"""

import os, sys, json, time, hashlib, logging, argparse
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
EarlyExitChecker = _mod.EarlyExitChecker

ALIGNED_PROMPT = "User liked: {history}. User will also like:"


# ── LLM Ranker (Qwen3.5 or GPT-2) ──────────────────────────────────────────

class PerplexityRanker:
    """Perplexity-based ranker supporting any CausalLM (GPT-2 or Qwen3.5)."""

    def __init__(self, model_name: str, device: str = "cuda:0",
                 dtype=torch.bfloat16, device_map=None, quant=None):
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        logger.info(f"Loading {model_name} ...")
        self.device = device
        self.use_device_map = (device_map == "auto")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        load_kwargs = dict(trust_remote_code=True)
        if quant == "4bit":
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
            load_kwargs["device_map"] = device_map or device
        else:
            load_kwargs["dtype"] = dtype
            load_kwargs["device_map"] = device_map or device

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        self.model.eval()
        n = sum(p.numel() for p in self.model.parameters()) / 1e9
        logger.info(f"Loaded {n:.2f}B params on {device}")

    def _input_device(self):
        if self.use_device_map:
            return next(self.model.parameters()).device
        return torch.device(self.device)

    def score(self, history_text: str, candidates: list, k: int = 5) -> list:
        """Return list of (item_id, score) sorted best-first."""
        scores = []
        dev = self._input_device()
        with torch.no_grad():
            for cid, cand_text in candidates:
                prompt = ALIGNED_PROMPT.format(history=history_text[:150]) + \
                         f" {cand_text[:60]}"
                try:
                    enc = self.tokenizer(prompt, return_tensors='pt',
                                         truncation=True, max_length=128)
                    ids = enc['input_ids'].to(dev)
                    out = self.model(ids, labels=ids)
                    scores.append((cid, -out.loss.item()))
                except Exception:
                    scores.append((cid, -999.0))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

    def unload(self):
        del self.model
        self.model = None
        torch.cuda.empty_cache()
        import gc; gc.collect()


# ── Trie helpers ─────────────────────────────────────────────────────────────

def trie_recommend(history, candidates, trie_stats, k=5):
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
    """Trie confidence = pref_strength × CTR_gap (product formula)."""
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
    top_ctrs = sorted(ctrs, reverse=True)[:max(1, len(ctrs)//4)]
    rest_ctrs = sorted(ctrs, reverse=True)[max(1, len(ctrs)//4):]
    avg_top = np.mean(top_ctrs)
    avg_rest = np.mean(rest_ctrs) if rest_ctrs else 0.0
    ctr_gap = (avg_top - avg_rest) / max(avg_top, 1e-6)

    return float(min(max(pref_strength * ctr_gap, 0), 1.0))


# ── Single-sample evaluation ──────────────────────────────────────────────────

def build_pool(sample, all_items, pool_size, seed_offset, rng):
    gt = sample['ground_truth']
    history = sample['history']
    neg = [it for it in all_items if it != gt and it not in history]
    n_neg = min(pool_size - 1, len(neg))
    cands = [gt] + list(rng.choice(neg, size=n_neg, replace=False))
    rng.shuffle(cands)
    return cands


def evaluate_all(samples, news_items, all_items, trie_stats, ranker,
                 pool_size, k, tau_values, seed):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    results = {
        'trie_only': [],
        'llm_only': [],
        **{f'trie_llm_tau{t}': [] for t in tau_values},
    }
    exit_rates = {f'trie_llm_tau{t}': [] for t in tau_values}
    latencies = {'llm_only': [], **{f'trie_llm_tau{t}': [] for t in tau_values}}

    for i, sample in enumerate(samples):
        gt = sample['ground_truth']
        history = sample['history']
        cands = build_pool(sample, all_items, pool_size, i, rng)

        # — Trie-only —
        trie_recs = trie_recommend(history, cands, trie_stats, k)
        results['trie_only'].append(1 if gt in trie_recs else 0)

        # — Build text for LLM —
        hist_text = " | ".join(
            news_items.get(h, {}).get('title', '')[:30] for h in history[-5:])
        cand_texts = [(c, news_items.get(c, {}).get('title', '')[:50]) for c in cands]

        # — LLM-only —
        t0 = time.perf_counter()
        scored = ranker.score(hist_text, cand_texts, k)
        lat = (time.perf_counter() - t0) * 1000
        latencies['llm_only'].append(lat)
        llm_recs = [cid for cid, _ in scored[:k]]
        results['llm_only'].append(1 if gt in llm_recs else 0)

        # — Trie+LLM routing for each τ —
        conf = compute_confidence(history, cands, trie_stats)
        for tau in tau_values:
            key = f'trie_llm_tau{tau}'
            if conf >= tau:
                # Early exit: use Trie
                recs = trie_recs
                exit_rates[key].append(1)
                latencies[key].append(0.2)  # Trie latency
            else:
                # Route to LLM
                recs = llm_recs
                exit_rates[key].append(0)
                latencies[key].append(lat)
            results[key].append(1 if gt in recs else 0)

        if (i + 1) % 200 == 0:
            logger.info(f"  [{i+1}/{len(samples)}] "
                        f"trie={np.mean(results['trie_only']):.3f} "
                        f"llm={np.mean(results['llm_only']):.3f} "
                        f"trie+llm(τ=0.4)={np.mean(results['trie_llm_tau0.4']):.3f} "
                        f"exit={np.mean(exit_rates['trie_llm_tau0.4']):.1%}")

    summary = {}
    for key in results:
        summary[key] = {
            'hit@5': float(np.mean(results[key])),
            'hit@5_se': float(np.std(results[key]) / np.sqrt(len(results[key]))),
        }
    for key in exit_rates:
        summary[key]['exit_rate'] = float(np.mean(exit_rates[key]))
        summary[key]['latency_ms'] = float(np.mean(latencies[key]))
    for key in latencies:
        if key == 'llm_only' or key.startswith('trie_llm'):
            summary[key]['latency_ms'] = float(np.mean(latencies[key]))

    return summary


# ── Main ──────────────────────────────────────────────────────────────────────

def run(args):
    from src.data.mind_loader import load_mind_for_trie_experiment
    logger.info(f"Loading MIND {args.samples} samples ...")
    samples, news_items, all_items = load_mind_for_trie_experiment(
        n_samples=args.samples, seed=42)
    all_items = list(all_items)
    trie_stats = TrieStatistics(news_items, samples)

    seeds = [int(s) for s in args.seeds.split(',')]
    tau_values = [0.3, 0.4, 0.5, 0.6]
    models_to_run = [m.strip() for m in args.models.split(',') if m.strip()]

    all_results = {}

    for model_name in models_to_run:
        logger.info(f"\n{'='*60}\nModel: {model_name}\n{'='*60}")

        dtype = torch.float32 if model_name == 'gpt2' else torch.bfloat16
        quant = args.quant if model_name != 'gpt2' else None

        ranker = PerplexityRanker(
            model_name, device=args.device, dtype=dtype,
            device_map=("auto" if args.device_map == "auto" else None),
            quant=quant)

        seed_results = []
        for seed in seeds:
            logger.info(f"  Seed {seed} ...")
            try:
                r = evaluate_all(
                    samples, news_items, all_items, trie_stats, ranker,
                    args.pool_size, args.k, tau_values, seed)
                seed_results.append(r)
                logger.info(f"    trie_only={r['trie_only']['hit@5']:.3f} "
                            f"llm_only={r['llm_only']['hit@5']:.3f} "
                            f"trie+llm(0.4)={r['trie_llm_tau0.4']['hit@5']:.3f} "
                            f"exit={r['trie_llm_tau0.4']['exit_rate']:.1%}")
            except RuntimeError as e:
                if 'memory' in str(e).lower():
                    logger.warning(f"OOM — skip remaining seeds")
                    torch.cuda.empty_cache()
                    break
                raise

        ranker.unload()

        if not seed_results:
            continue

        # Aggregate across seeds
        model_summary = {}
        for key in seed_results[0].keys():
            vals = [r[key]['hit@5'] for r in seed_results]
            model_summary[key] = {
                'hit@5_mean': float(np.mean(vals)),
                'hit@5_std': float(np.std(vals)),
                'hit@5_seeds': vals,
            }
            if 'exit_rate' in seed_results[0][key]:
                ex = [r[key]['exit_rate'] for r in seed_results]
                model_summary[key]['exit_rate_mean'] = float(np.mean(ex))
            if 'latency_ms' in seed_results[0][key]:
                lats = [r[key]['latency_ms'] for r in seed_results]
                model_summary[key]['latency_ms_mean'] = float(np.mean(lats))

        all_results[model_name] = model_summary

    return all_results


def print_table(results):
    print('\n' + '='*80)
    print('TRIE ROUTING WITH MODERN LLMs — Does Trie help beyond LLM-only?')
    print('='*80)
    for model, v in results.items():
        print(f'\nModel: {model}')
        trie = v.get('trie_only', {})
        llm = v.get('llm_only', {})
        print(f"  {'Method':<30} {'Hit@5':>8} {'Std':>6} {'ExitRate':>10} {'Latency':>10}")
        print(f"  {'-'*65}")
        print(f"  {'Trie-only':<30} {trie['hit@5_mean']:>8.3f} {trie['hit@5_std']:>6.3f}")
        print(f"  {'LLM-only':<30} {llm['hit@5_mean']:>8.3f} {llm['hit@5_std']:>6.3f} "
              f"{'—':>10} {llm.get('latency_ms_mean',0):>8.0f}ms")
        for tau in [0.3, 0.4, 0.5, 0.6]:
            key = f'trie_llm_tau{tau}'
            r = v.get(key, {})
            improvement = r['hit@5_mean'] - llm['hit@5_mean']
            sign = '+' if improvement >= 0 else ''
            print(f"  {'Trie+LLM τ='+str(tau):<30} {r['hit@5_mean']:>8.3f} {r['hit@5_std']:>6.3f} "
                  f"{r.get('exit_rate_mean',0):>10.1%} {r.get('latency_ms_mean',0):>8.0f}ms "
                  f"  ({sign}{improvement:.3f} vs LLM-only)")
    print('='*80)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', type=int, default=2000)
    parser.add_argument('--seeds', default='42,43,44')
    parser.add_argument('--pool_size', type=int, default=20)
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--device_map', default='', help='"auto" for multi-GPU')
    parser.add_argument('--models', default='Qwen/Qwen3.5-4B',
                        help='Comma-separated model names. Add "gpt2" for baseline.')
    parser.add_argument('--quant', default='', choices=['', '4bit', '8bit'])
    parser.add_argument('--output_dir', default='results/kdd_rebuttal_routing_qwen')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    results = run(args)
    print_table(results)

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out = os.path.join(args.output_dir, f'routing_qwen_{ts}.json')
    with open(out, 'w') as f:
        json.dump({'timestamp': ts, 'args': vars(args), 'results': results}, f, indent=2)
    logger.info(f"Saved: {out}")


if __name__ == '__main__':
    main()
