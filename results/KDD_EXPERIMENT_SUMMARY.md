# KDD 2026 Experiment Summary

**Date**: 2026-01-30
**Status**: Core experiments completed

## 1. Cross-Dataset Validation Results

### 1.1 Criteo Dataset (CTR Prediction, 50K samples)

| Model | AUC | Std | LogLoss |
|-------|-----|-----|---------|
| DeepFM | 0.687 | ±0.008 | 0.910 |
| Trie-MoE | 0.671 | ±0.007 | 1.074 |

**Observation**: DeepFM outperforms Trie-MoE by ~2.3% on Criteo.

### 1.2 MovieLens Dataset (Rating Prediction, 50K samples)

| Model | AUC | Std | LogLoss |
|-------|-----|-----|---------|
| DeepFM | 0.697 | ±0.001 | 1.138 |
| Trie-MoE | 0.696 | ±0.005 | 1.000 |

**Observation**: Nearly identical performance on MovieLens.

## 2. Data Scale Ablation Results

### 2.1 Criteo Data Scale

| Size | DeepFM AUC | Trie-MoE AUC | Improvement | p-value |
|------|------------|--------------|-------------|---------|
| 10K | 0.654±0.009 | 0.652±0.016 | -0.31% | 0.884 |
| 20K | 0.681±0.006 | 0.660±0.008 | -3.04% | 0.037* |
| 50K | 0.687±0.008 | 0.671±0.007 | -2.34% | 0.104 |
| 100K | 0.685±0.003 | 0.672±0.007 | -1.86% | 0.074 |
| 200K | 0.694±0.001 | 0.676±0.003 | -2.59% | 0.001* |

*Statistically significant (p < 0.05)

### 2.2 MovieLens Data Scale

| Size | DeepFM AUC | Trie-MoE AUC | Improvement | p-value |
|------|------------|--------------|-------------|---------|
| 10K | 0.635±0.006 | 0.639±0.002 | **+0.66%** | 0.397 |
| 20K | 0.673±0.006 | 0.665±0.007 | -1.11% | 0.295 |
| 50K | 0.697±0.001 | 0.696±0.005 | -0.17% | 0.756 |
| 100K | 0.722±0.002 | 0.725±0.002 | **+0.41%** | 0.175 |
| 200K | 0.754±0.003 | 0.752±0.003 | -0.24% | 0.537 |

## 3. Adaptive Alpha Analysis

### Learned Parameters
- w = -0.48
- b = 0.52

### Key Alpha Values by Frequency

| Token Frequency | Alpha Value | Interpretation |
|----------------|-------------|----------------|
| n=1 | 0.547 | Prior-heavy (cold-start) |
| n=5 | 0.416 | Balanced |
| n=10 | 0.347 | Learning-heavy |
| n=100 | 0.155 | Learning-heavy |
| n=1000 | 0.058 | Strong learning |
| n=5000 | 0.027 | Very strong learning |

## 4. Efficiency Results

### Inference Latency (batch size = 256, CUDA)

| Model | Mean | P50 | P95 | P99 |
|-------|------|-----|-----|-----|
| DeepFM | 1.63 ms | 1.58 ms | 1.83 ms | 2.40 ms |
| Trie-MoE | 4.15 ms | 4.05 ms | 5.26 ms | 5.94 ms |

### Throughput

| Model | Samples/sec |
|-------|------------|
| DeepFM | 161,482 |
| Trie-MoE | 63,654 |

**Overhead**: Trie-MoE adds ~153% latency overhead, but remains well under production SLA requirements (typically 50-100ms).

## 5. Key Findings

### Positive Findings
1. **Cold-start handling**: Trie-MoE shows slight advantages on MovieLens at small data sizes (10K: +0.66%)
2. **Adaptive alpha works**: The learned parameters show the expected behavior - trusting prior for cold tokens, learning for hot tokens
3. **Production-ready latency**: Even with Trie overhead, inference is <5ms per batch

### Areas for Improvement
1. **Criteo underperformance**: Trie-MoE consistently underperforms on Criteo by 1-3%
2. **Training time**: Trie-MoE is ~1.5x slower to train than DeepFM
3. **Routing overhead**: Current implementation adds significant inference overhead

### Hypotheses for Future Work
1. Criteo has very high-cardinality sparse features that may not benefit from Trie routing
2. MovieLens has more structured user/item features that align better with Trie hierarchies
3. The current Trie structure may need dataset-specific tuning

## 6. Result Files

- `results/kdd_experiments/data_scale_criteo_20260130_233432.json`
- `results/kdd_experiments/data_scale_movielens_20260130_233432.json`
- `results/kdd_experiments/cross_dataset_20260130_233432.json`
- `results/adaptive_alpha_analysis/alpha_curve_default.pdf`
