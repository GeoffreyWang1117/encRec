# ERRATUM — corrections to this supplementary artifact

**Status: superseded. Do not cite the numbers below as reported.**

This repository is the supplementary artifact for a KDD 2026 submission. Three defects
have since been found by the authors. They are listed here rather than silently patched,
so that anyone who already read or reproduced this artifact can see exactly what changed.

---

## 1. `Trie+LLM (aligned) = 0.418` is mislabeled — it is **LLM-only** (found 2026-04-08)

`README.md` and `SUPPLEMENTARY_README.md` report

| Method | Type | Hit@5 |
|---|---|---|
| Trie+LLM (aligned) | Hybrid | **0.418** |

The 0.418 figure does not come from the hybrid Trie+LLM pipeline. It is a **single seed
(456) of a 500-sample GPT-2 LLM-only run**. Verification at 5000 samples × 3 seeds gives:

| Configuration | Hit@5 |
|---|---|
| LLM-only, aligned prompt | 0.423 ± .005 |
| LLM-only, original "interests/Recommend" prompt | 0.402 ± .001 |
| **Trie+LLM, full pipeline (τ=0.4, with CTR fusion)** | **0.331 ± .002** |
| Trie+LLM, no CTR fusion | 0.383 ± .003 |
| Trie-only | 0.289 ± .002 |

So the hybrid pipeline **does not** improve quality over LLM-only on MIND; it trades
quality for latency. The "+102% over Trie-only" claim compares LLM-only against
Trie-only, not the hybrid against anything.

## 2. The Trie's "CTR" is a binary seen/unseen flag, not a CTR (found 2026-08-17)

In `experiments/kdd_triellm_01_ablation.py`, `TrieStatistics._build_statistics`:

```python
for item_id in session.get('history', []):
    clicks[item_id] += 1
    impressions[item_id] += 1          # incremented identically
ctr_scores[item_id] = clicks[item_id] / impressions[item_id]     # == 1.0, always
```

Both counters receive the same increments, so the ratio is identically 1.0. Verified on
MIND (5000 samples): `set(ctr_scores.values()) == {1.0}` — exactly one distinct value
across 14 848 items, with `get_ctr` returning the 0.01 default for everything else.

Every result described as using "Trie-indexed CTR" therefore used a **binary
has-this-item-appeared-in-any-history flag**. The datasets' real click labels (e.g. MIND's
impression logs) were never used, and the raw appearance count — a usable popularity
signal — is divided away by the ratio.

Consequence on MIND, stratifying by the ground truth's appearance count:

| GT frequency | share of queries | Trie Hit@5 | LM Hit@5 |
|---|---|---|---|
| 0 (unseen) | 87.4% | **0.178** | 0.418 |
| ≥1 (seen) | 12.6% | **0.96–0.98** | 0.39–0.51 |

The Trie is near-perfect when the answer already appears in the training histories, and
**below the 5/20 = 0.25 random floor** when it does not.

## 3. The cross-domain comparison is affected by the negative-sampling protocol (found 2026-08-17)

Candidate pools are built by drawing negatives uniformly from the full catalogue
(`build_pool`, and `random.sample(pool, 19)` in the Goodreads loader). Catalogues are
overwhelmingly composed of never-interacted items, so whenever the positive happens to be
a "seen" item, the binary flag of defect #2 separates it from 19 unseen negatives.

Re-running Trie-only under three candidate protocols (5000 samples × 3 seeds, pool = 20,
random floor 0.25). Each domain first reproduces its published number under `uniform`:

| Domain | gt seen | neg seen | `uniform` (as published) | `popularity` | `matched` |
|---|---|---|---|---|---|
| MIND | 0.126 | 0.145 | 0.288 | 0.041 | 0.323 |
| **Goodreads** | **0.846** | **0.064** | **0.874** | 0.692 | **0.232** |
| MovieLens | 0.996 | 0.821 | 0.485 | 0.416 | 0.417 |

where `popularity` draws negatives with probability proportional to appearance count and
`matched` draws them from the positive's own popularity stratum, so that popularity
carries no information about which candidate is the positive.

Goodreads' Trie-only score falls from **0.874 to 0.232 — below the random floor** — and
becomes flat across every popularity stratum. MIND is unaffected (it has no
positive/negative popularity contrast to exploit); MovieLens is mildly affected.

**Not yet measured:** the language model must be re-scored under each protocol as well,
since its candidate pools change too. Until that is done, no claim is made here about
whether the sign of the LM-vs-Trie comparison flips. The published LM numbers were all
measured under `uniform` and should not be carried over.

## Also worth noting

Fixing defect #2 — replacing the binary flag with `log1p(appearance count)` — *improves*
the Trie substantially (MovieLens `uniform` 0.485 → 0.771). The baseline as published is
under-powered by the bug.

---

## What this means for reuse

- Do not cite `0.418` as a hybrid Trie+LLM result.
- Do not treat the cross-domain table as a domain-level finding without re-running it
  under a popularity-controlled candidate protocol.
- The code is left as it was run, bug included, so that the published numbers remain
  reproducible and the defects independently checkable.

A corrected artifact is in preparation.
