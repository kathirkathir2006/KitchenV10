# V5 Adapter C — results (frozen DINOv2 baseline vs Adapter C)

Identical conditions: same frozen DINOv2 ViT-B/14 backbone, same frozen reference
library, same scorer, thresholds calibrated on val (282) only for each model,
test (284) evaluated once after the Adapter C config was frozen.

## Versions

| Item | Value |
|---|---|
| Dataset | real_photo_set_v1, manifest sha `89ea9b20effdf40983ce670722f43fce42e44bd4378cc5810853dbba94c656e0` (train 791 / val 282 / test 284) |
| Reference library | reference_library_v1, sha `7e95ee5636af0a14b2839b1a242699bc02635b5d0956c6d6e89a5b9019fc0a02` (524 refs, 27 labels, train only) |
| Adapter C config | sha `2d1121dd428e9e8fcb06c4438988ea86cb1b8f0464b38896d2b082c18c3e561c` (cfg0: dim 128, dropout 0.1, lr 1e-3, epoch 30 of 80) |
| Checkpoint | `adapter_c_final.pt`, sha `8eac5dfd319bbc1d9ba6f76c28cd709628168dab875d8652135863d7d4b1eb5e` |
| 41-scene catalog | sha `38fc3eaee75c76fa5fd1dc3d4177af4abde2cfeb3e410d290fb524f9e30a1523` (unmodified) |
| Kaggle runs | `kiq-v5-real-photo-v1` v3, `kiq-v5-reflib-v1` v1 (commit `1e712e3`), `kiq-v5-adapter-c` v1 (commit `df40459`), `kiq-v5-adapter-c-test` v1 (commit `ea0834a`) |

## Held-out test (284 images, evaluated once)

| Metric | Baseline | Adapter C |
|---|---|---|
| Known accepted-correct | 0.5085 | **0.6538** |
| Identity top-1 (raw) | 0.8248 | 0.8547 |
| Identity top-5 (raw) | **0.9615** | 0.9487 |
| Family top-1 | 0.8718 | 0.8932 |
| State top-1 | 0.9188 | 0.9231 |
| Unknown rejection (n=50) | **0.92** | 0.86 |
| False confirmation | **0.0317** | 0.0352 |
| Objective | 1.4485 | 1.5225 |

Per item: fixed 41, newly failing 10, both correct 155, both failing 78.

Earliest divergence (baseline → Adapter C): open-set over-abstain 74 → 47,
FAMILY 28 → 23, IDENTITY 11 → 9, FOOD_NONFOOD 2 → 2, open-set false accept 4 → 7.

Open-set: rejected 46 → 43 of 50; family back-off 6 → 18.

**Reading:** most of the gain comes from better confidence separation, so fewer
correct answers are abstained on. Raw ranking moves only a little (+3 pts
top-1), and top-5 drops. **Regressions:** unknown rejection is worse (−6 pts),
open-set false accepts rise from 4 to 7, and false confirmation rises slightly.

## 41-scene benchmark (ground-truth regions, diagnostic, detector excluded)

| Split | Metric | Baseline | Adapter C |
|---|---|---|---|
| Photo (22 scenes, 37 regions) | Identity top-1 raw | 0.6486 | 0.8649 |
| | Accepted-correct | 0.3243 | 0.3784 |
| | False confirmation | 0.027 | 0.0 |
| | Scene-complete | 4/22 | 4/22 |
| | Fixed / newly failing | | 3 / 1 |
| Synthetic (19 scenes, 34 regions) | Identity top-1 raw | 0.0 | 0.0312 |
| | Accepted-correct | 0.0 | 0.0 |
| | Scene-complete | 2/19 | 2/19 |
| All (41 scenes, 71 regions) | Objective | 1.2472 | 1.2899 |
| | Scene-complete | 6/41 | 6/41 |

Photo divergence: IDENTITY 9 → 1, open-set over-abstain 12 → 18. Adapter C now
ranks correctly but still abstains. Synthetic drawings are still out of domain
(26/34 diverge at FAMILY), and no amount of adapter training fixes that.

## Decision

Adapter C v1 is a real but modest improvement on real photos, and it costs
open-set safety. Do not promote it as-is. Next: D/E with train-only hard
negatives and an open-set term, selected on val only. Test stays used once.

Raw outputs: `adapter_c_v1_eval/` (test_results.json, test_per_item.json,
41scene_results.json, 41scene_per_region.json).
