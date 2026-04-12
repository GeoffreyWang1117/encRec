"""
KDD 2026 Reviewer Response: 2026 Model Scale Experiment.

Addresses Reviewer 4fvP W3:
  "Relying on GPT-2 from 2019 is inappropriate in the current time."

Tests latest 2026 models available locally:
  Qwen3.5-2B   2.0B  2026  Qwen/Qwen3.5-2B        ← Qwen3.5 small
  Qwen3.5-9B   9.0B  2026  Qwen/Qwen3.5-9B        ← Qwen3.5 large (device_map=auto)

Note: Qwen3.5 uses model_type=qwen3_5 (transformers 5.5.0 required).
Both are multimodal-capable but used text-only here (perplexity scoring).

Usage:
    # 2B model on GPU1
    CUDA_VISIBLE_DEVICES=1 python experiments/kdd_response_model_scale_2026.py \
        --models Qwen3.5-2B --samples 1000 --device cuda:0

    # 9B model across both GPUs
    python experiments/kdd_response_model_scale_2026.py \
        --models Qwen3.5-9B --samples 1000 --device_map auto

Results saved to: results/kdd_response_model_scale_2025/
"""

import os, sys, json, time, logging, argparse
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import importlib.util
_spec = importlib.util.spec_from_file_location(
    "ablation", str(Path(__file__).parent / "kdd_triellm_01_ablation.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
TrieStatistics = _mod.TrieStatistics

MODEL_REGISTRY_2026 = {
    "Gemma-4-E2B (2026)": {
        "hf_name": "google/gemma-4-E2B-it",
        "params": "1.0B",
        "year": 2026,
        "dtype": torch.bfloat16,
        "device_map": None,         # tiny — single GPU
        "scoring": "perplexity",
        "text_only": True,          # load language_model submodule only
    },
    "Qwen3.5-2B (2026)": {
        "hf_name": "Qwen/Qwen3.5-2B",
        "params": "2.0B",
        "year": 2026,
        "dtype": torch.bfloat16,   # model uses bfloat16
        "device_map": None,         # single GPU
        "scoring": "perplexity",
    },
    "Qwen3.5-4B (2026)": {
        "hf_name": "Qwen/Qwen3.5-4B",
        "params": "4.0B",
        "year": 2026,
        "dtype": torch.bfloat16,
        "device_map": None,         # single GPU (fits in ~8GB bf16)
        "scoring": "perplexity",
    },
    "Qwen3.5-9B (2026)": {
        "hf_name": "Qwen/Qwen3.5-9B",
        "params": "9.0B",
        "year": 2026,
        "dtype": torch.bfloat16,
        "device_map": "auto",       # splits across both 3090s
        "scoring": "perplexity",
    },
}

ALIGNED_PROMPT = "User liked: {history}. User will also like:"


class Qwen35Ranker:
    """Perplexity ranker for Qwen3.5 (qwen3_5 architecture, transformers 5.5)."""

    def __init__(self, model_name: str, device: str = "cuda:0",
                 dtype=torch.bfloat16, device_map=None, quant=None,
                 text_only=False):
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        logger.info(f"  Loading {model_name} (quant={quant}, text_only={text_only}) ...")
        self.device = device
        self.use_device_map = (device_map == "auto")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        load_kwargs = dict(trust_remote_code=True)

        if quant == "4bit":
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            load_kwargs["device_map"] = device_map if device_map == "auto" else device
        elif quant == "8bit":
            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            load_kwargs["device_map"] = device_map if device_map == "auto" else device
        else:
            load_kwargs["dtype"] = dtype
            if device_map == "auto":
                load_kwargs["device_map"] = "auto"
            else:
                load_kwargs["device_map"] = device

        full_model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)

        # For multimodal models (Gemma4ForConditionalGeneration), extract text LM
        if text_only and hasattr(full_model, 'language_model'):
            logger.info("  Extracting language_model submodule (text-only mode)")
            self.model = full_model.language_model
            # Move to device if not already placed by device_map
            if device_map != "auto" and not quant:
                self.model = self.model.to(device)
            # Free the vision/audio modules
            del full_model
            torch.cuda.empty_cache()
        else:
            self.model = full_model

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.eval()

        n_params = sum(p.numel() for p in self.model.parameters()) / 1e9
        logger.info(f"  Loaded — {n_params:.2f}B parameters")
        if device_map == "auto":
            # Log device placement
            try:
                devices = set(str(p.device) for p in self.model.parameters())
                logger.info(f"  Devices: {devices}")
            except Exception:
                pass

    def _get_input_device(self):
        """Get the device to send inputs to."""
        if self.use_device_map:
            # For device_map=auto, find the first parameter's device
            try:
                return next(self.model.parameters()).device
            except Exception:
                return torch.device("cuda:0")
        return torch.device(self.device)

    def score_perplexity(self, history_text: str, candidates: list,
                         pool_size: int = 20, k: int = 5) -> list:
        """Perplexity scoring: lower loss = better candidate."""
        short_hist = history_text[:150]
        scores = []
        input_device = self._get_input_device()

        with torch.no_grad():
            for cid, cand_text in candidates:
                prompt = ALIGNED_PROMPT.format(history=short_hist) + \
                         f" {cand_text[:60]}"
                try:
                    enc = self.tokenizer(
                        prompt, return_tensors='pt',
                        truncation=True, max_length=128)
                    input_ids = enc['input_ids'].to(input_device)
                    out = self.model(input_ids, labels=input_ids)
                    scores.append((cid, -out.loss.item()))
                except Exception as e:
                    logger.debug(f"Scoring error: {e}")
                    scores.append((cid, -999.0))
        return scores

    def unload(self):
        del self.model
        self.model = None
        torch.cuda.empty_cache()
        import gc; gc.collect()


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


def evaluate_model(cfg, samples, news_items, all_items,
                   trie_stats, device, seed, pool_size=20, k=5, quant=None):
    np.random.seed(seed)
    torch.manual_seed(seed)

    ranker = Qwen35Ranker(
        cfg['hf_name'], device=device,
        dtype=cfg['dtype'], device_map=cfg['device_map'], quant=quant,
        text_only=cfg.get('text_only', False))

    hits, trie_hits, latencies = [], [], []

    for i, sample in enumerate(samples):
        gt = sample['ground_truth']
        history = sample['history']
        neg = [it for it in all_items if it != gt and it not in history]
        n_neg = min(pool_size - 1, len(neg))
        cands = [gt] + list(np.random.choice(neg, size=n_neg, replace=False))
        np.random.shuffle(cands)

        hist_text = " | ".join(
            news_items.get(h, {}).get('title', '')[:30] for h in history[-5:])
        cand_texts = [(cid, news_items.get(cid, {}).get('title', '')[:50])
                      for cid in cands]

        t0 = time.perf_counter()
        scored = ranker.score_perplexity(hist_text, cand_texts, pool_size, k)
        lat_ms = (time.perf_counter() - t0) * 1000
        latencies.append(lat_ms)

        scored.sort(key=lambda x: x[1], reverse=True)
        recs = [cid for cid, _ in scored[:k]]
        hits.append(1 if gt in recs else 0)

        trie_recs = trie_recommend(history, cands, trie_stats, k=k)
        trie_hits.append(1 if gt in trie_recs else 0)

        if (i + 1) % 100 == 0:
            logger.info(f"    [{i+1}/{len(samples)}] "
                        f"running Hit@5={np.mean(hits):.3f}, "
                        f"lat={np.mean(latencies):.0f}ms")

    ranker.unload()
    return {
        'hit@5': float(np.mean(hits)),
        'trie_hit@5': float(np.mean(trie_hits)),
        'latency_ms_mean': float(np.mean(latencies)),
        'latency_ms_p50': float(np.percentile(latencies, 50)),
        'n_samples': len(samples),
    }


def run(args):
    from src.data.mind_loader import load_mind_for_trie_experiment
    logger.info(f"Loading MIND {args.samples} samples ...")
    samples, news_items, all_items = load_mind_for_trie_experiment(
        n_samples=args.samples, seed=42)
    all_items = list(all_items)
    trie_stats = TrieStatistics(news_items, samples)

    seeds = [int(s) for s in args.seeds.split(',')]
    results = {}

    model_subset = [m for m in MODEL_REGISTRY_2026
                    if not args.models or any(k in m for k in args.models.split(','))]
    logger.info(f"Models to evaluate: {model_subset}")

    for model_display in model_subset:
        cfg = MODEL_REGISTRY_2026[model_display]
        # Override device_map from args if provided
        if args.device_map:
            cfg = dict(cfg)
            cfg['device_map'] = args.device_map if args.device_map != 'single' else None

        logger.info(f"\n{'='*60}\n{model_display}\n{'='*60}")
        seed_results = []

        for seed in seeds:
            logger.info(f"  Seed {seed} ...")
            try:
                r = evaluate_model(
                    cfg, samples, news_items, all_items,
                    trie_stats, args.device, seed,
                    pool_size=args.pool_size,
                    quant=args.quant if args.quant else None)
                seed_results.append(r)
                logger.info(f"    Hit@5={r['hit@5']:.3f}, "
                            f"Trie={r['trie_hit@5']:.3f}, "
                            f"lat={r['latency_ms_mean']:.0f}ms")
            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    logger.warning(f"  OOM on {model_display} — skipping")
                    torch.cuda.empty_cache()
                    break
                raise

        if not seed_results:
            continue

        h5 = [r['hit@5'] for r in seed_results]
        lat = [r['latency_ms_mean'] for r in seed_results]
        results[model_display] = {
            'hit@5_mean': float(np.mean(h5)),
            'hit@5_std': float(np.std(h5)),
            'latency_ms_mean': float(np.mean(lat)),
            'year': cfg['year'],
            'params': cfg['params'],
            'seed_results': seed_results,
        }

    return results


def print_table(results):
    print('\n' + '='*75)
    print('2026 MODEL SCALE — Hit@5 Comparison (MIND Large, perplexity scoring)')
    print('='*75)
    print(f"{'Model':<26} {'Year':>5} {'Params':>7} {'Hit@5':>14} {'Lat(ms)':>10}")
    print('-'*75)
    for model_name, v in results.items():
        print(f"  {model_name:<24} {v['year']:>5} {v['params']:>7} "
              f"{v['hit@5_mean']:>6.3f}±{v['hit@5_std']:.3f} "
              f"{v['latency_ms_mean']:>8.0f}ms")
    print('='*75)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', type=int, default=1000)
    parser.add_argument('--seeds', default='42,43,44')
    parser.add_argument('--pool_size', type=int, default=20)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--device_map', default='',
                        help='Override device_map: "auto" or "single" or ""')
    parser.add_argument('--models', default='',
                        help='Comma-separated model name substrings to filter')
    parser.add_argument('--quant', default='',
                        choices=['', '4bit', '8bit'],
                        help='Quantization: 4bit or 8bit (uses bitsandbytes)')
    parser.add_argument('--output_dir',
                        default='results/kdd_response_model_scale_2025')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Device: {args.device} | device_map: {args.device_map or 'per-model'}")

    results = run(args)
    print_table(results)

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out = os.path.join(args.output_dir, f'model_scale_2026_{ts}.json')
    with open(out, 'w') as f:
        json.dump({'timestamp': ts, 'args': vars(args), 'results': results}, f, indent=2)
    logger.info(f"\nSaved: {out}")


if __name__ == '__main__':
    main()
