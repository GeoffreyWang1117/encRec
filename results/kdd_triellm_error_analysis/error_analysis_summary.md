# Error Analysis

**Date**: 2026-02-07 00:35
**Samples**: 2000
**Overall Hit@5**: 0.2405

## Performance by Category

| Category | Hit@5 | Count |
|----------|-------|-------|
| movies | 0.5833 | 48 |
| foodanddrink | 0.4216 | 102 |
| entertainment | 0.4154 | 65 |
| lifestyle | 0.2826 | 230 |
| health | 0.2632 | 95 |
| finance | 0.2558 | 172 |
| music | 0.2481 | 133 |
| tv | 0.2180 | 133 |
| autos | 0.2105 | 57 |
| news | 0.2098 | 591 |

## Performance by History Length

| History Length | Hit@5 | Count |
|----------------|-------|-------|
| 1-3 | 0.2923 | 65 |
| 4-6 | 0.2771 | 231 |
| 7-10 | 0.2424 | 198 |
| 11-15 | 0.2714 | 210 |
| 16-20 | 0.2941 | 136 |
| 21+ | 0.2181 | 1160 |

## Performance by Item Popularity

| Popularity | Hit@5 | Count |
|------------|-------|-------|
| cold (0) | 0.2363 | 1765 |
| rare (1-p25) | 0.2432 | 37 |
| moderate (p25-p50) | 0.1282 | 39 |
| popular (p50-p75) | 0.4146 | 41 |
| very popular (>p75) | 0.2797 | 118 |

## Category Match Analysis

| Type | Hit@5 | Count |
|------|-------|-------|
| category_match | 0.2668 | 611 |
| category_mismatch | 0.2289 | 1389 |
