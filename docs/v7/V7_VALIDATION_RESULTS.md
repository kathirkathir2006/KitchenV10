# V7 pairwise specialists — VAL results

**V7_STATUS = SUCCESS (on VAL; provisional).** Raw top-1 rises from 0.819 to 0.845 (+0.0258 = +6 net queries:
7 corrected, 1 broken). Open-set safety is unchanged or better. **TEST was not opened.**

## Pipeline
Frozen Adapter E produces the top-5. A specialist for A ↔ B is active only if **both** A and B are in the top-5.
It swaps A and B only if its preference disagrees with E's order and |P(A) − P(B)| ≥ margin. E's frozen
thresholds are applied to E's own top-5 score values, permuted onto the new order, so no threshold is changed.

## Selection (automatic, from VAL errors; `specialist_selection.json`)
The rule: at least 2 recoverable errors, at least 10 TRAIN images per class, at least 3 VAL images per class,
and at most 5 specialists. Exactly 5 pairs qualified: cream ↔ ice_cream, biryani ↔ fried_rice,
cheese ↔ packaged_cheese, caesar_salad ↔ salad, guacamole ↔ hummus.

## Global VAL comparison

| Metric | Adapter E | E + V7 |
|---|---|---|
| Identity top-1 raw | 0.8190 | **0.8448** |
| Identity top-5 raw | 0.9483 | 0.9483 |
| Family top-1 | 0.8707 | 0.8793 |
| State top-1 | 0.9095 | 0.9181 |
| Accepted-correct | 0.5905 | 0.5948 |
| Identity coverage | 0.6121 | 0.6121 (unchanged, so no extra acceptance) |
| Unknown rejection | 0.96 | 0.96 |
| False confirmation | 0.0248 | 0.0213 |
| Objective | 1.6366 | 1.6502 |

| Success check | Result |
|---|---|
| Top-1 gain ≥ 0.02 | 0.0258 ✔ |
| Corrections ≥ 5 | 7 ✔ |
| Regressions ≤ 2 | 1 ✔ |
| Unknown rejection ≥ E | 0.96 ≥ 0.96 ✔ |
| False confirmation ≤ E + 0.005 | 0.0213 ≤ 0.0298 ✔ |

## Combination safety
Each specialist was tested individually and then all 5 together. The individual net gains sum to +6, and the
combined net is also +6. No query had two specialists fire, so there is no interaction loss. Every specialist
had a positive individual net, so the "only individually positive" set equals the full set.

## Caveats (read before trusting the number)
1. **Circular by design.** The pairs were chosen *because* E failed on these VAL images, and the corrections
   are counted on the same images. Each specialist's epoch, its linear-vs-MLP choice and its margin were also
   picked on VAL. This VAL gain is therefore an upper bound.
2. **The pass is thin.** The bar was 5 net corrections and V7 has 6. One flip less and it would still pass;
   two less and it would fail.
3. Only the independent TEST split can confirm this. It has not been run.

## Frozen artifacts (test_used = false)
- `v7_specialists_final.pt`: sha256 `eeae5fbeaf84969af199b263bde13ebb8a69895fcb20f7fdbd0dfbc0fc29eb25`
- `v7_specialists_frozen_config.json`: sha256 `c29079096ed20ecd78d90bf0b29d86201241f705757d152f1dcef915c1b297b5`
- `specialist_results.json`: sha256 `e93cd3f3cd1923eba427aff91322fe5cdca57e5044a16ba7a6e406b65291d2e6`
- Code: `kaggle/v7_src/kiq_v7_specialists.py`
