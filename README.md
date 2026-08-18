> [!WARNING]
> **This artifact is superseded — see [ERRATUM.md](ERRATUM.md) before using anything here.**
>
> Three defects were found by the authors after publication:
> 1. The headline `Trie+LLM (aligned) = 0.418` below is **mislabeled**. It is an
>    **LLM-only** result from a single seed; the actual hybrid pipeline scores
>    **0.331 ± .002** at 5000 samples x 3 seeds.
> 2. The Trie's "CTR" is computed as `clicks/impressions` after incrementing both
>    counters identically, so it is **identically 1.0** — a binary seen/unseen flag,
>    not a CTR.
> 3. The cross-domain comparison is affected by **uniform negative sampling**. Under
>    popularity-matched negatives, Goodreads Trie-only falls from 0.874 to 0.232 —
>    below the random floor.
>
> The code and numbers below are left exactly as they were run, so that the published
> results stay reproducible and the defects independently checkable.

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
  kdd_rebuttal_multi_dataset.py           Cross-domain (Amazon, MovieLens, Criteo)
  kdd_rebuttal_cache_fix.py               Score Cache vs Rank Cache
  kdd_rebuttal_frugal_baseline.py         FrugalGPT + Oracle comparison
  kdd_rebuttal_cscr_baseline.py           CSCR learned router (GBM)
  kdd_rebuttal_sri_beyond_carbon.py       SRI + carbon footprint
  kdd_rebuttal_llama_hf.py               Llama3.1 8B via HuggingFace
  kdd_rebuttal_routing_qwen.py            Qwen3.5-4B routing experiment
  kdd_response_confidence_ablation.py     Confidence formulation ablation
  kdd_response_model_scale_2025.py        Model scale (2024-2025 models)
  kdd_response_model_scale_2026.py        Model scale (2026 SOTA)

results/                      All experiment outputs (JSON + logs)

paper/kdd2026_rebuttal/figures/   Publication-quality figures (PDF + PNG)
```

## Key Results

### Main Results (MIND Large, 5000 samples, Table 1 in paper)

| Method | Type | Hit@5 | NDCG@5 | MRR |
|--------|------|-------|--------|-----|
| Prompt4NR | LLM | 0.421 | 0.264 | 0.212 |
| Trie-only | Stat | 0.207 | 0.119 | 0.091 |
| Trie+LLM (aligned) | Hybrid | **0.418** | **0.269** | **0.218** |

### Cross-Domain Validation (supplementary experiments, 5000 samples, 3 runs)

| Dataset | SRI | Trie-only | LLM | Benefit |
|---------|-----|-----------|-----|---------|
| MIND (news) | 0.56 | 0.207 | **0.418** | +102% |
| MovieLens 1M | 0.41 | **0.483** | 0.467 | -3% |
| Amazon Movies | 0.19 | **0.692** | 0.618 | -11% |
| Amazon Electronics | 0.38 | **0.632** | 0.314 | -50% |
| Criteo (encrypted) | 0.26 | **0.472** | 0.254 | -46% |

**SRI Regression**: Pearson r=0.60, providing decision rule: SRI < 0.45 -> skip LLM.

### Efficiency (Table 2 in paper)

| Method | Latency | Tokens | Cost/1K |
|--------|---------|--------|---------|
| Trie-only | 0.12ms | 0 | $0.00 |
| Pure LLM | 165ms | 354 | $3.54 |
| **Trie+LLM** | **69ms** | **189** | **$0.69** |

81.8% cost reduction through conditional computation.

## Requirements

```
python >= 3.10
torch >= 2.1
transformers >= 4.35
numpy, scikit-learn, lightgbm
```

## Reproduction

```bash
# Main experiment (requires GPU, ~2h)
python experiments/kdd_triellm_09_large_scale.py --samples 5000 --runs 3

# Cross-domain validation (requires GPU, ~2h per dataset)
python experiments/kdd_rebuttal_multi_dataset.py

# SRI + carbon (CPU only, ~5 min)
python experiments/kdd_rebuttal_sri_beyond_carbon.py

# Confidence ablation (requires GPU, ~1h)
python experiments/kdd_response_confidence_ablation.py
```

## Datasets

Datasets not included due to size. Download instructions:
- MIND: https://msnews.github.io/
- MovieLens 1M: https://grouplens.org/datasets/movielens/1m/
- Amazon Reviews: https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/
- Criteo: https://www.kaggle.com/c/criteo-display-ad-challenge
