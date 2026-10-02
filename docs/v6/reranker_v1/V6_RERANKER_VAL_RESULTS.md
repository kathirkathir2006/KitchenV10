# V6 candidate reranker — VAL results (TEST not opened)

**Verdict: the pre-registered gate is formally PASSED, but the improvement is not material.**
Raw top-1 rises from 0.8190 to 0.8319, which is +3 of 232 known VAL queries. That comes from 4 fixed and
1 broken; a McNemar test on 4 vs 1 is not significant. The gain shows up in only 1 of 6 configs, and that
config was picked on the same VAL split. Accepted-correct, coverage, unknown rejection and false confirmation
are all **unchanged**, so at E's operating point the reranker changes no confirmed answer.

## Setup
- Base model: frozen Adapter E (config `ab26699426b4…`). The candidates are E's top-5 under the unchanged V5
  scorer, whose source sha is `ec2dd8a4…`. The embeddings are the frozen DINOv2 cache from `kiq-v5-adapter-e`
  v1, checked against `reference_library_v1` with max abs diff < 1e-5.
- Fit: TRAIN, leave-one-image-out. 524 self-references were removed by sha256, giving 3955 rows.
  Selection: VAL, 1410 rows. TEST was never opened.
- Model: PyTorch logistic regression (LBFGS, L2 ∈ {1e-4, 1e-3, 1e-2}). Features are standardised on TRAIN
  statistics. There are 23 E-evidence features (rank, score, gap, separation, prototype cosine, ref
  top-1/top-3 mean/std, max/mean/median/std, support count, ref consistency, and family/state
  score/separation/agreement). An optional second feature set adds 4 frozen-DINOv2 features. Query
  ground-truth family and state are never used as features.
- Decision: E's frozen thresholds. E's own top-5 score values are permuted onto the reranked order, so the
  thresholds are not changed.

## Oracle control (VAL, 232 known queries)
| | Value |
|---|---|
| Adapter E top-1 | 0.8190 |
| Top-5 oracle ceiling | 0.9483 |
| Maximum possible gain | +0.1293 (30 queries) |
| E errors with correct identity in top-5 | 30 |
| E errors with correct identity absent | 12 |

## VAL comparison
| Metric | Baseline | C | D | E | E + reranker |
|---|---|---|---|---|---|
| Identity top-1 raw | 0.836 | **0.858** | 0.836 | 0.819 | 0.832 |
| Identity top-5 raw | **0.974** | 0.940 | 0.948 | 0.948 | 0.948 |
| Family top-1 | 0.892 | **0.901** | 0.888 | 0.871 | 0.892 |
| State top-1 | 0.935 | **0.944** | 0.918 | 0.910 | 0.927 |
| Accepted-correct | 0.470 | 0.655 | **0.681** | 0.591 | 0.591 |
| Identity coverage | 0.483 | 0.672 | **0.707** | 0.612 | 0.612 |
| Unknown rejection | 0.96 | 0.96 | 0.96 | 0.96 | 0.96 |
| False confirmation | **0.0177** | 0.0213 | 0.0284 | 0.0248 | 0.0248 |
| Objective | 1.483 | 1.663 | **1.677** | 1.637 | 1.641 |

Grid (VAL raw top-1): with E evidence only, 0.8319 / 0.8147 / 0.8190 for L2 1e-4 / 1e-3 / 1e-2. With the frozen
DINOv2 features added, 0.8190 / 0.8147 / 0.8190. Selected: E evidence only, L2 1e-4.

## Why it barely moves: the training rows are not representative
Adapter E was trained on these same TRAIN images. On TRAIN queries the correct identity is in the top-5 in
99.87% of cases, and the mean top-1 score is 0.942, against 0.910 on VAL. Almost every TRAIN query is already
right at rank 1, so the reranker mostly learns "keep rank 1". Leave-one-image-out removes self-references,
but it cannot remove the adapter's memorisation of its own training images.

## Failure diagnosis (VAL)
- **A. The correct identity is usually in the top-5: yes.** 30 of 42 E errors have it there. The evidence
  available to the reranker cannot separate those candidates (4 fixed, 1 broken).
- **B. The correct identity is absent from the top-5: 12 of 42 errors.** These need better representation or
  reference data; reranking cannot fix them.
- **C. Errors cluster in a few pairs.** The top 10 pairs account for 45% of E errors:
  packaged_cheese→cheese, biryani↔fried_rice, salad→caesar_salad, cream↔ice_cream, guacamole→hummus,
  pad_thai→omelette, bibimbap→rice. These are fine-grained same-family or prepared-dish confusions that need
  specialist discrimination or more data.

Also note that on VAL raw top-1, both the frozen baseline (0.836) and Adapter C (0.858) beat E + reranker (0.832).

## Frozen artifacts (test_used = false)
- `v6_reranker_final.pt`: sha256 `600ea2ff098f771b843527ce8c9e508801345330b3df680791598dcbb0f8f21b`
- `v6_reranker_frozen_config.json`: sha256 `187e53edd8f3904f550ad4d7ec59519463b8759806fc9c06b4486b27122280c5`
- `v6_val_results.json`, `v6_val_failure_analysis.json`
- Code: `kaggle/v6_src/kiq_v6_reranker.py`
