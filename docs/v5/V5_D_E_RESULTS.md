# V5 Adapters D and E — results

Everything was run under the same conditions as Adapter C. DINOv2 ViT-B/14 stays frozen, with the same
224 preprocessing, `reference_library_v1`, `real_photo_set_v1`, scorer (0.5·prototype + 0.5·top-3) and
identity → family → unknown decision. Thresholds are calibrated on VAL only. The scorer source hash
`ec2dd8a4…` is identical in every kernel. TEST was opened once (`kiq-v5-adapter-de-test` v1), after D and
E were frozen. Adapter C's test numbers reproduce its historical run exactly.

## What D and E are

- **Adapter D**: LayerNorm → Linear(768→512) → GELU → Dropout → Linear(→dim), plus a linear skip, L2-normalised.
  It has no class vectors. The loss is SupCon (T=0.07) over image-level batches of 32 images × 8 views
  (8 labels × 4 images), plus λ_hard=0.5 × a margin loss (0.20) on 5 mined train-only DINOv2 cross-label hard
  negatives per anchor. The grid has 8 configs, 60 epochs each, validated every 5 epochs.
- **Adapter E**: the same architecture and the same D-winner hyper-parameters, plus λ_open × an episodic
  pseudo-unknown loss. Each step holds out one whole TRAIN label, builds prototypes from all other train
  labels, and applies relu(max_known_cos − 0.10). λ_open is one of {0.10, 0.25, 0.50}.
- **E safety gate** (on VAL): unknown_rejection ≥ D's VAL value (0.96) and false_confirmation ≤ D's VAL value
  (0.0284). Among candidates that pass, the one with the highest known_score wins.

## Versions

| Item | Value |
|---|---|
| Photo set | `89ea9b20effdf40983ce670722f43fce42e44bd4378cc5810853dbba94c656e0` |
| Reference library | `7e95ee5636af0a14b2839b1a242699bc02635b5d0956c6d6e89a5b9019fc0a02` |
| Hard negatives (train only, 3955 pairs) | `9b724cf046136493ad81672f21756bdbef3187ce71bd5013e548730aa409ad60` |
| Adapter C config | `2d1121dd428e9e8fcb06c4438988ea86cb1b8f0464b38896d2b082c18c3e561c` |
| Adapter D config / checkpoint | `987c29889356db4ff8055ebe8f39e95a4be9a0bd778e86e3463e8de62a0e4951` / `5a1af335cd919a0e6a62b221914d53333f12ce146e60cc10a64288c33779a761` |
| Adapter E config / checkpoint | `ab26699426b4eb71b52166c99b38c0d2376098bde8a43dcb5b45d075e49020cc` / `40ff977e1b90866436f663bd9c2014b26d1fd55bc72d65649ab427d8afa9b331` |
| D winner | cfg7: dim 256, dropout 0.3, lr 3e-4, epoch 5 |
| E winner | cfg2: λ_open 0.50, epoch 5; safety gate passed (29/36 candidates) |
| Kaggle runs | `kiq-v5-adapter-d` v1, `kiq-v5-adapter-e` v1, `kiq-v5-adapter-de-test` v1 (commits in `RUN_REGISTRY.log`) |

## VAL (selection split; selection-biased)

| Metric | Baseline | C | D | E |
|---|---|---|---|---|
| Identity top-1 raw | 0.836 | 0.858 | 0.836 | 0.819 |
| Identity top-5 raw | 0.974 | 0.940 | 0.948 | 0.948 |
| Accepted-correct | 0.470 | 0.655 | 0.681 | 0.591 |
| Identity coverage | 0.483 | 0.672 | 0.707 | 0.612 |
| Known score | 0.558 | 0.746 | 0.774 | 0.726 |
| Unknown rejection | 0.96 | 0.96 | 0.96 | 0.96 |
| False confirmation | 0.0177 | 0.0213 | 0.0284 | 0.0248 |
| Objective | 1.483 | 1.663 | 1.677 | 1.637 |

## TEST (284 images, evaluated once)

| Metric | Baseline | C | D | E |
|---|---|---|---|---|
| Identity top-1 raw | 0.825 | **0.855** | 0.833 | 0.838 |
| Identity top-5 raw | **0.962** | 0.949 | **0.962** | 0.949 |
| Family top-1 | 0.872 | **0.893** | **0.893** | 0.889 |
| State top-1 | 0.919 | **0.923** | 0.919 | 0.919 |
| Accepted-correct | 0.509 | 0.654 | **0.658** | 0.611 |
| Identity coverage | 0.530 | 0.667 | **0.701** | 0.628 |
| Unknown rejection | **0.92** | 0.86 | **0.92** | **0.92** |
| False confirmation | 0.0317 | 0.0352 | 0.0493 | **0.0282** |
| Objective | 1.449 | 1.523 | 1.582 | **1.607** |

### Change counts (correct means the earliest divergence is OK)

| Transition | Fixed | Newly failing | Both OK | Both fail |
|---|---|---|---|---|
| baseline → D | 43 | 8 | 157 | 76 |
| baseline → E | 38 | 14 | 151 | 81 |
| C → D | 18 | 14 | 182 | 70 |
| C → E | 11 | 18 | 178 | 77 |
| D → E | 9 | 20 | 180 | 75 |

### Open-set (50 unknown test items, 5 held-out identities)

| | Baseline | C | D | E |
|---|---|---|---|---|
| Rejected (not identity) | 46 | 43 | 46 | 46 |
| Identity false accepts | 4 | 7 | 4 | 4 |
| Family back-offs | 6 | 18 | 40 | 37 |

### Earliest divergence

| Category | Baseline | C | D | E |
|---|---|---|---|---|
| OK | 165 | 196 | **200** | 189 |
| OPEN_SET_OVER_ABSTAIN | 74 | 47 | **41** | 53 |
| FAMILY | 28 | 23 | 23 | 23 |
| IDENTITY | 11 | **9** | 14 | 12 |
| FOOD_NONFOOD | 2 | 2 | 2 | 3 |
| OPEN_SET_FALSE_ACCEPT | **4** | 7 | **4** | **4** |

## 41-scene benchmark (frozen; ground-truth boxes; evaluated after freeze; not used for selection)

| Split | Metric | Baseline | C | D | E |
|---|---|---|---|---|---|
| Photo (22 scenes, 37 regions) | Identity top-1 raw | 0.649 | **0.865** | 0.757 | 0.757 |
| | Accepted-correct | 0.324 | 0.378 | **0.459** | **0.459** |
| | False confirmation | 0.027 | **0.0** | 0.054 | **0.0** |
| | Scene-complete | 4/22 | 4/22 | **6/22** | **6/22** |
| Synthetic (19 scenes, 34 regions) | Accepted-correct | 0.0 | 0.0 | 0.0 | 0.0 |
| | Scene-complete | 2/19 | 2/19 | 2/19 | 2/19 |
| All 41 (71 regions) | Accepted-correct | 0.174 | 0.203 | **0.246** | **0.246** |
| | Scene-complete | 6/41 | 6/41 | **8/41** | **8/41** |

Photo-scene changes: baseline → D fixed 6 and broke 1; baseline → E fixed 5 and broke 0; C → E fixed 5 and broke 2.
Synthetic scenes do not change for any adapter: 32 of 34 regions fail at FAMILY or FOOD_NONFOOD.

## Reading

1. **D (metric learning) recovers C's open-set loss.** Unknown rejection is back to 0.92 and false accepts are
   back to 4. D keeps C's accepted-correct (0.658) and coverage is the highest (0.701). The cost is confident
   errors on known items: false confirmation is 0.049, the worst of the four, and IDENTITY divergences rise from
   9 to 14.
2. **E (D + pseudo-unknown episodes) is the safest model.** Its false confirmation (0.028) is the lowest of all
   four, below the frozen baseline. It matches baseline open-set behaviour and has the best test objective
   (1.607). The cost is accepted-correct: 0.611, down from D's 0.658, because it backs off to family more often.
3. **The ranking itself barely moves.** Top-1 is 0.825–0.855 and top-5 is 0.949–0.962 for every model. The
   adapters mainly reshape confidence, so they decide when to accept, not what ranks first. C still has the
   best raw top-1.
4. **Both D and E picked epoch 5 out of 60.** Validation performance peaks early, which suggests the SupCon
   objective overfits 791 train images fast.
5. **The E safety gate is weak evidence.** VAL unknown rejection is 0.96 for every model, so it carries almost
   no signal. The gate effectively discriminated only on false confirmation.
6. **Synthetic scenes stay out of domain.** No adapter changes them.

Raw outputs are in `adapter_de_v1_eval/`. Frozen artifacts are in `adapter_d_v1/` and `adapter_e_v1/`.
