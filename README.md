# Supplementary Materials — KDD 2026 Submission #4023

**Title**: Cost-Aware Conditional Computation for LLM Recommendation: A Deterministic Routing Framework

This repository contains experiment code, results, and figures for reproducibility.

## Repository Structure

```
src/                          Core framework implementation
  trie/                       Statistical Trie (retrieval_trie.py, advanced_trie.py)
  llm/                        LLM client and Trie-augmented recommender
  data/                       Dataset loaders (MIND, MovieLens, Amazon, Criteo)
  metrics/                    Evaluation metrics (Hit, NDCG, MRR, Coverage, etc.)

experiments/                  All experiment scripts (self-contained, reproducible)
  kdd_triellm_01_ablation.py              Core framework (imported by all scripts)
  kdd_triellm_09_large_scale.py           Main MIND 5K×3 benchmark
  kdd_triellm_02_sota_baselines.py        NRMS/NAML/PLM-NR/TALLRec/Prompt4NR
  kdd_rebuttal_multi_dataset.py           Cross-domain (Amazon×2, MovieLens)
  kdd_rebuttal_cache_fix.py               Score Cache vs Rank Cache
  kdd_rebuttal_frugal_baseline.py         FrugalGPT + Oracle comparison
  kdd_rebuttal_cscr_baseline.py           CSCR learned router (GBM)
  kdd_rebuttal_sri_beyond_carbon.py       SRI + carbon footprint
  kdd_rebuttal_llama_hf.py               Llama3.1 8B via HuggingFace
  kdd_rebuttal_aligned_ablation.py        Unified 5K×3 aligned-prompt ablation
  kdd_rebuttal_routing_qwen.py            Qwen3.5-4B routing experiment
  kdd_rebuttal_verify_0418.py             Original prompt verification
  kdd_rebuttal_orthogonality_v2.py        Information orthogonality (E1/E2/E4)
  kdd_rebuttal_unified_baselines.py       Unified baselines (Trie/LLM/Oracle)
  kdd_response_confidence_ablation.py     Confidence formulation ablation
  kdd_response_model_scale_2025.py        Model scale (2024-2025 models)
  kdd_response_model_scale_2026.py        Model scale (2026 SOTA)

results/                      All experiment outputs (JSON + logs)
  kdd_rebuttal_aligned_ablation/          Unified 5K×3 ablation (Apr 8)
  kdd_rebuttal_verify_0418/               Prompt verification (Apr 8)
  kdd_rebuttal_unified_baselines/         Unified baselines (Apr 7)
  kdd_rebuttal_routing_qwen/              Qwen3.5-4B routing (Apr 7)
  kdd_rebuttal_orthogonality_v2/          Orthogonality experiments (Apr 7)
  kdd_rebuttal_*/                         All other rebuttal experiments
  kdd_response_*/                         Reviewer response experiments
  kdd_triellm_*/                          Original submission experiments

figures/                      Publication-quality figures (PDF + PNG)
```

## Key Results (5 datasets, 3 runs each, GPT-2, pool=20)

| Dataset | SRI | Trie-only Hit@5 | LLM-only Hit@5 | LLM Benefit |
|---------|-----|-----------------|-----------------|-------------|
| MIND (news) | 0.56 | 0.289±.002 | **0.423±.005** | +46% |
| MovieLens 1M | 0.41 | **0.483±.006** | 0.467±.001 | -3% |
| Amazon Movies | 0.19 | **0.692±.006** | 0.618±.003 | -11% |
| Amazon Electronics | 0.38 | **0.632±.006** | 0.314±.007 | -50% |
| Criteo (encrypted) | 0.26 | **0.472±.003** | 0.254±.003 | -46% |

## Aligned-Prompt Ablation (MIND, 5000×3 seeds, GPT-2)

| Configuration | Hit@5 | Notes |
|---|---|---|
| LLM-only (aligned) | 0.423±.005 | Quality ceiling on high-SRI domain |
| Trie+LLM, no-CTR (τ=0.4) | 0.383±.003 | 36% latency reduction, -9.5% quality |
| Trie+LLM, full (τ=0.4) | 0.331±.002 | With CTR fusion |
| Trie-only | 0.289±.002 | Zero-parameter baseline |

## Requirements

```
python >= 3.10
torch >= 2.1
transformers >= 4.40
numpy, scikit-learn, lightgbm
```

## Reproduction

```bash
# Core ablation (requires GPU, ~65 min)
CUDA_VISIBLE_DEVICES=0 python experiments/kdd_rebuttal_aligned_ablation.py \
    --samples 5000 --seeds 42,43,44 --device cuda:0

# Cross-domain (requires GPU, ~2h per dataset)
python experiments/kdd_rebuttal_multi_dataset.py

# SRI + carbon (CPU only, ~5 min)
python experiments/kdd_rebuttal_sri_beyond_carbon.py
```

## Datasets

MIND Large and other datasets are not included due to size. Download instructions:
- MIND: https://msnews.github.io/
- MovieLens 1M: https://grouplens.org/datasets/movielens/1m/
- Amazon Reviews: https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/
- Criteo: https://www.kaggle.com/c/criteo-display-ad-challenge
