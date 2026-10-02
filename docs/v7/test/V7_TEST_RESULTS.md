# V7 one-shot TEST — NOT CONFIRMED

The frozen V7 (config `c29079096ed2…`, checkpoint `eeae5fbeaf84…`) was evaluated once on real_photo_set_v1
TEST (284 images: 234 known, 50 open-set) and on the frozen 41-scene benchmark.
- Frozen DINOv2 test embeddings came from Kaggle `kiq-v7-test-embed` v1, with every image hash-verified.
- Nothing was fitted or re-thresholded.
- Baseline, C, D and E reproduce their historical frozen test numbers exactly.
- One earlier run of this script exited silently before writing any output, and its numbers were never seen.
  The results below are from the only completed run.

**V7_STATUS = NOT CONFIRMED.** The +2.6-point VAL gain does not hold on TEST. TEST shows +0.43 points:
3 fixed, 2 broken, a net of +1 image.

| TEST | Adapter E | E + V7 |
|---|---|---|
| Identity top-1 raw | 0.8376 | 0.8419 |
| Identity top-5 raw | 0.9487 | 0.9487 |
| Family top-1 | 0.8889 | 0.8846 |
| State top-1 | 0.9188 | 0.9103 |
| Accepted-correct | 0.6111 | 0.6068 |
| Identity coverage | 0.6282 | 0.6282 |
| Unknown rejection | 0.92 | 0.92 |
| False confirmation | 0.0282 | 0.0317 |
| Objective | 1.6073 | 1.5959 |

| Check (same bar as VAL) | Result |
|---|---|
| Top-1 gain ≥ 0.02 | +0.0043 ✘ |
| Fixed ≥ 5 | 3 ✘ |
| Broken ≤ 2 | 2 ✔ |
| Unknown rejection ≥ E | 0.92 ✔ |
| False confirmation ≤ E + 0.005 | 0.0317 ≤ 0.0332 ✔ |

| Specialist | Fired | Fixed | Broken | Fired on unknowns |
|---|---|---|---|---|
| cream ↔ ice_cream | 3 | 0 | 0 | 0 |
| biryani ↔ fried_rice | 10 | 1 | 0 | 2 |
| cheese ↔ packaged_cheese | 5 | 1 | **2** | 0 |
| caesar_salad ↔ salad | 2 | 1 | 0 | 0 |
| guacamole ↔ hummus | 1 | 0 | 0 | 1 |

Open-set: E and E + V7 are identical, with 46 of 50 rejected and 4 identity false accepts.
41-scene benchmark: no change at all (photo 6/22, synthetic 2/19, all 8/41 scene-complete for both).

## Reading
- The VAL result was circular. The pairs, epochs, model types and margins were all chosen on the same VAL
  errors used to score them. On independent images the specialists rarely fire on a fixable case, and when
  they do they are near coin-flip.
- cheese ↔ packaged_cheese was the data-starved pair (18 train images, flat learning curve), and it is net
  negative on TEST.
- Adapter E stays the production model. V7 should not be deployed. Its artifacts stay frozen as a historical
  record.
- The limit is data, not modelling. Fine-grained pairs need more, and more diverse, real images per class
  (see `../V7_FAILURE_ANALYSIS.md`). About 5% of errors (12 of 42 on VAL) never reach the top-5.
