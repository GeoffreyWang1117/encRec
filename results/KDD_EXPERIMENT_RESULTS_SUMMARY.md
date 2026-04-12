# KDD 2026 Experiment Results Summary

**Date**: 2026-02-05
**Status**: Complete (MIND Small + MIND Large + MovieLens 1M)

---

## MIND Large Dataset Results (Industrial-Scale News Recommendation)

**Configuration**: 101,527 news articles, 1000 samples, 3 runs, K=5

### Main Results

| Method | Hit@5 | NDCG@5 | MRR | Precision@5 | Recall@5 | Latency (ms) |
|--------|-------|--------|-----|-------------|----------|--------------|
| **Trie+LLM** | **0.473** | **0.293** | **0.282** | 0.105 | 0.407 | 5608 |
| Trie-only | 0.442 | 0.262 | 0.242 | 0.096 | 0.382 | 0.12 |
| NeuMF | 0.411 | 0.240 | 0.215 | 0.087 | 0.358 | 5.77 |
| BPR | 0.401 | 0.240 | 0.219 | 0.084 | 0.351 | 6.13 |

### Key Findings

1. **Trie+LLM achieves the best performance** on industrial-scale news
   - +7.0% Hit@5 improvement over Trie-only
   - +11.8% NDCG@5 improvement over Trie-only
   - +17.9% Hit@5 improvement over BPR baseline

2. **Results consistent with MIND Small**
   - Validates scalability of the approach
   - Content-aware methods excel on news recommendation

3. **Trie-only remains highly competitive**
   - 46,000x faster than LLM (0.12ms vs 5608ms)
   - Strong early-exit candidate

### Diversity & Beyond-Accuracy Metrics

| Method | Diversity (ILD) | Novelty | Long-tail Coverage |
|--------|-----------------|---------|-------------------|
| Trie+LLM | 0.572 | 9.61 | 0.049 |
| Trie-only | 0.568 | 9.65 | 0.047 |
| NeuMF | 0.774 | 10.10 | 0.146 |
| BPR | 0.769 | 10.09 | 0.138 |

---

## MIND Small Dataset Results (News Recommendation)

**Configuration**: 1000 samples, 3 runs, K=5

### Main Results

| Method | Hit@5 | NDCG@5 | MRR | Precision@5 | Recall@5 | Latency (ms) |
|--------|-------|--------|-----|-------------|----------|--------------|
| **Trie+LLM** | **0.463** | **0.279** | **0.267** | 0.097 | 0.418 | 4904 |
| Trie-only | 0.452 | 0.273 | 0.253 | 0.095 | 0.409 | 0.15 |
| NeuMF | 0.403 | 0.243 | 0.212 | 0.084 | 0.368 | 5.56 |
| BPR | 0.387 | 0.235 | 0.207 | 0.081 | 0.355 | 5.18 |

### Key Findings

1. **Trie+LLM achieves the best performance** across all accuracy metrics
   - +2.4% Hit@5 improvement over Trie-only
   - +2.2% NDCG@5 improvement over Trie-only
   - +19.6% Hit@5 improvement over BPR baseline

2. **Trie-only is highly competitive**
   - Outperforms both CF baselines (BPR, NeuMF)
   - 33,000x faster than LLM (0.15ms vs 4904ms)
   - Strong baseline for early exit

3. **Traditional CF methods (BPR, NeuMF) underperform**
   - Pure collaborative signals insufficient for news recommendation
   - Lack of content understanding hurts performance

### Optimization Statistics (Trie+LLM)

| Metric | Value |
|--------|-------|
| Early Exit Rate | 2.1% |
| Cache Hit Rate | 0.0% (cold cache) |
| LLM Call Rate | 97.9% |

### Diversity & Beyond-Accuracy Metrics

| Method | Diversity (ILD) | Novelty | Long-tail Coverage |
|--------|-----------------|---------|-------------------|
| Trie+LLM | 0.600 | 9.87 | 0.103 |
| Trie-only | 0.570 | 9.63 | 0.091 |
| NeuMF | 0.752 | 10.06 | 0.211 |
| BPR | 0.765 | 10.06 | 0.207 |

---

## MovieLens 1M Dataset Results (Movie Recommendation)

**Configuration**: 1000 samples, 3 runs, K=5

### Main Results

| Method | Hit@5 | NDCG@5 | MRR | Precision@5 | Recall@5 | Latency (ms) |
|--------|-------|--------|-----|-------------|----------|--------------|
| **BPR** | **0.834** | **0.427** | **0.583** | 0.399 | 0.251 | 6.76 |
| NeuMF | 0.764 | 0.372 | 0.509 | 0.347 | 0.249 | 6.95 |
| Trie-only | 0.744 | 0.283 | 0.472 | 0.260 | 0.138 | 0.47 |
| Trie+LLM | 0.724 | 0.278 | 0.460 | 0.257 | 0.138 | 4849 |

### Key Findings

1. **BPR dominates on MovieLens** (+15.2% Hit@5 over Trie+LLM)
   - Strong collaborative signals in movie domain
   - User-item interactions are highly predictive

2. **Trie methods underperform on movies**
   - Genre-based Trie less effective than user-user similarity
   - Movie preferences more complex than news categories

3. **LLM does not improve over Trie-only**
   - t-statistic: -1.72, p-value: 0.23 (not significant)
   - Content-based reasoning less valuable for movie recommendation

### Diversity & Beyond-Accuracy Metrics

| Method | Diversity (ILD) | Novelty | Long-tail Coverage |
|--------|-----------------|---------|-------------------|
| BPR | 0.569 | 9.40 | 0.019 |
| NeuMF | 0.566 | 10.08 | 0.149 |
| Trie-only | 0.647 | 10.19 | 0.150 |
| Trie+LLM | 0.665 | 10.17 | 0.154 |

**Insight**: Trie methods show higher diversity but lower accuracy on movies.

---

## Cross-Dataset Comparison

| Dataset | Best Method | Hit@5 | Key Insight |
|---------|-------------|-------|-------------|
| MIND Large (News) | **Trie+LLM** | 0.473 | Industrial-scale validation |
| MIND Small (News) | **Trie+LLM** | 0.463 | Content understanding matters |
| MovieLens 1M (Movies) | **BPR** | 0.834 | Collaborative signals dominate |

### Domain-Specific Analysis

**Why Trie+LLM wins on News:**
1. News articles require semantic understanding
2. Category hierarchies are highly predictive
3. LLM can understand article content and context
4. User preferences are content-driven

**Why BPR wins on Movies:**
1. Movie preferences are strongly collaborative
2. User-user similarity outweighs genre matching
3. Long-tail movie preferences need learned embeddings
4. Genre-based Trie is too coarse for nuanced tastes

---

## Efficiency Analysis

| Method | Training Time | Inference/req | Tokens/req | Cost/1K req |
|--------|--------------|---------------|------------|-------------|
| Trie-only | 0 (no training) | 0.15-0.47ms | 0 | $0 |
| BPR | ~6s | 5-7ms | 0 | $0 |
| NeuMF | ~25s | 6-7ms | 0 | $0 |
| Trie+LLM | 0 | 4849-4904ms | ~38-80 | ~$0.40-0.80 |

---

## Statistical Significance

### MIND Small (Trie+LLM vs Trie-only)
- Δ Hit@5 = +0.011 (+2.4%)
- Effect: Positive improvement

### MovieLens (Trie+LLM vs Trie-only)
- Δ Hit@5 = -0.019 (-2.6%)
- t-statistic: -1.72
- p-value: 0.23 (not significant)
- Cohen's d: -1.22 (large negative effect)

---

## Conclusions (Final)

1. **Domain matters**: Trie+LLM excels on content-heavy domains (news), CF methods excel on collaborative domains (movies)

2. **MIND Small (News)**:
   - Trie+LLM achieves **+19.6%** Hit@5 over BPR
   - Content-aware methods outperform pure CF

3. **MovieLens 1M (Movies)**:
   - BPR achieves **+15.2%** Hit@5 over Trie+LLM
   - Collaborative signals dominate genre-based filtering

4. **Practical Implications**:
   - For news/articles: Use Trie+LLM with early exit optimization
   - For movies/products: Use collaborative filtering (BPR/NeuMF)
   - Hybrid approach recommended for mixed-content platforms

---

## Next Steps

1. ~~**Run MIND Large** for industrial-scale validation on news~~ ✅ DONE
2. **Ablation studies**:
   - Early exit threshold tuning
   - Compression ratio impact
   - History length sensitivity
3. **Additional baselines**:
   - LLMLingua (prompt compression)
   - NRMS (news recommendation SOTA)
4. **Hybrid model**: Combine Trie+LLM with CF for cross-domain recommendation

---

## Summary Table (All Datasets)

| Dataset | Articles | Trie+LLM Hit@5 | vs BPR | vs Trie-only |
|---------|----------|----------------|--------|--------------|
| MIND Large | 101,527 | **0.473** | +17.9% | +7.0% |
| MIND Small | ~15,000 | **0.463** | +19.6% | +2.4% |
| MovieLens 1M | 3,883 | 0.724 | -13.2% | -2.6% |

**Conclusion**: Trie+LLM is highly effective for content-heavy domains (news) at industrial scale.

---

*Results updated: 2026-02-05 20:00*
