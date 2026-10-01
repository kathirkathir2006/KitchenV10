# KitchenIQ V5 — Experiment Readiness

Repo: `C:\Projects\KitchenIQ-V10-AI` (KitchenV10), HEAD `92a38b7` at inspection. KitchenIQ-OS read-only.
Machine-readable evidence: `docs/v5/readiness_probe.json`,
`backend/instance/dev_experiments/v10-vision-v5/v5_exp_AB_retrieval_001/`.

## 1. Verified baseline

| System | 41-scene strict completeness | Source |
|---|---|---|
| Immutable original | 0.2439 | prior acceptance reports |
| Identity-coverage specialist (current best, production reference) | **0.2927** (12/41) | `v10-vision-recovery/acceptance/identity_coverage_v1_acceptance.json` |
| V4 open-world engine, no retrieval | 0.2195 | `docs/v4/v4_benchmark_results.json` |
| V4 open-world engine + 108-item retrieval | 0.2195 | `v10-vision-v4/acceptance/candidate_v4_retrieval/` |

Corrections to the brief: the best verified baseline is 0.2927, not 0.2439. "207" is the size of
the required identity vocabulary (`docs/vision/required_identity_vocabulary_v1.json`), not 207
audited labels. The production training manifest has 20 trainable identities, 8 excluded as
taxonomy gaps, and 310 production-eligible images in total.
Production gate: FAIL (substantial target 0.4390).

## 2. Repository architecture

- `src/kitcheniq_v10/specialist/`: specialist ensemble, fusion, 41-scene acceptance runner (`acceptance.py`, official matcher from OS backend, read-only import).
- `src/kitcheniq_v10/v3/`, `src/kitcheniq_v10/v4/`: V3 evidence graph, V4 `VisionV4Engine` (detector → crop → DINOv2 specialist + retrieval → fusion → open-set).
- `orchestration/vision_v4/`: forensics, root-cause matrix, benchmark/finalize.
- `kaggle/kernels/`: GPU kernels (Kaggle is the only GPU; laptop is CPU-only).
- `docs/v4/`: V4 forensics, root-cause matrix, manifests. `docs/v5/`: this cycle.

## 3. Existing V4/V10 components reused

Faster R-CNN detector from the OS backend (read-only, via a temporary DB copy); FoodVisionV2 heads on
DINOv2 ViT-B/14 (`identity_coverage_best.pt`, sha `e1efaa4a…`); `ReferenceLibrary` (cosine top-k);
`run_acceptance` scorer; V4 open-set and food-gate logic.

## 4. Benchmark location

Catalog: `backend/instance/dev_experiments/v10-specialist-ensemble/acceptance/fixtures/catalog/fixture_catalog.json`
(sha `38fc3eae…a1523`). Images: `KitchenIQ-OS/backend/instance/dev_experiments/fv-real-world-acceptance-v2/fixtures/generated/`
(41 scenes, 71 required instances, 28 distinct labels).

## 5. Benchmark protection status

Catalog hash is asserted at the start of every V5 experiment. The V5 scripts open the images
read-only and write nothing there. None of the 108 library `source_hash` values match a fixture
hash, so no leakage was detected. The benchmark is used only for scoring, never for fitting,
thresholds or selection.

## 6. Current DINOv2 pipeline

Input is resized to 224×224 (bilinear), ImageNet-normalised, and passed through DINOv2 ViT-B/14.
The embedding is `x_norm_clstoken`, L2-normalised. The specialist checkpoint's backbone is
bit-identical to the pretrained weights (175/175 tensors, max diff 0.0), so the backbone is frozen
in practice and library/query embeddings share one space. Embedding mismatch is ruled out.

## 7. Current retrieval pipeline

Current retrieval (V4) is a flat cosine top-5 over 108 references, added into specialist scores
(weight 0.85), followed by open-set thresholds. There is no hierarchy, no prototypes and no
calibration.

## 8. Reference-library coverage

The library has 108 items over 11 labels (about 10 each): biryani, cheese, cream, fried_rice,
garlic, ginger, packaged_cheese, rice, tomato, tomato_paste, yogurt. All are Openverse CC0/CC-BY
images marked PRODUCTION_ELIGIBLE.

**Library quality is poor.** Mean within-class similarity is 0.06–0.39 (cheese 0.06, ginger 0.06,
yogurt 0.07). Leave-one-out top-1 accuracy within the library is only 0.48, and family-level
top-1 is 0.61. These are keyword-search images with noisy labels.

Coverage limits what retrieval can achieve. Only 32 of the 71 required benchmark regions are
library labels, and at most 18 of 41 scenes (0.439) contain only library labels or non-food.
Even perfect retrieval with this library cannot exceed about 0.44.

## 9. Current failure categories (V4 first-stage matrix)

Scenes blocked by first failing stage: OPEN_SET 10, FOOD_NONFOOD 9, DETECTOR 7, IDENTITY 3.

**New V5 finding: benchmark domain split.** An automatic check (`fixture_domain.json`) classifies
19 of 41 scenes as **synthetic drawings** (flat shapes, for example a red disc labelled "tomato")
and 22 as photographs.

| Domain | Scenes | Required regions | Library-covered | Frozen-DINOv2 A top-1 correct |
|---|---|---|---|---|
| Synthetic drawing | 19 | 34 | 27 | **1/27** |
| Photo | 22 | 37 | 5 | **4/5** |

Most of the identities retrieval could fix are drawn shapes. Most real photos contain identities
the library does not have (salad 9, pizza 5, club_sandwich 3, lasagna, pasta, and others).

## 10. Available training data

There are 310 production-eligible images over 20 trainable identities. They live on Kaggle
(manifest `docs/vision/production_identity_training_manifest_v1.json`, sha `9da443fa…`); only 7
identities are SUFFICIENT. Openverse/Wikimedia CC images are bulk-acquirable on Kaggle. No GPU
locally.

## 11. Positive-pair availability

Positives exist only as same-label pairs (about 10 per label in the library, up to about 40 per
label in the 310-image set). There are no same-entity, multi-view or temporal pairs, and noisy
labels make many same-label pairs semantically wrong.

## 12. Hard-negative availability

Verified pairs inside the library: tomato↔tomato_paste, rice↔fried_rice↔biryani,
cheese↔packaged_cheese, yogurt↔cream. Missing: red pepper, flatbread, stew, dry rice packet,
sliced cheese.

## 13. Leakage risks

1. Tuning anything on 41-scene drawings.
2. Adding drawn shapes to training, which would game the 19 synthetic scenes.
3. Selecting C/D/E by 41-scene score.
4. Building references from fixture images.

Every V5 script guards against these by hash checks and library-only calibration.

## 14. Available compute

Laptop: CPU only (6 logical cores, 7.9 GB RAM, 5.6 GB free disk, torch 2.14 CPU). It does not
persist when the lid is closed. Kaggle: free GPU, persistent, checkpointed outputs. Training
(C–E) must run on Kaggle.

## 15. Experiments A–F

- **A:** frozen DINOv2 plus the current flat top-1 retrieval, with an open-set threshold
  calibrated on the library.
- **B:** A plus hierarchical retrieval. Prototype + top-3 kNN score, then identity → family →
  unknown back-off, with thresholds from library leave-one-out and leave-one-label-out.
- **C:** a lightweight projection adapter (768→256) trained with supervised cross-entropy on the
  310-image set, on Kaggle.
- **D:** C trained with a supervised-contrastive loss plus the verified hard negatives above.
- **E:** a multi-task adapter with identity, family and state heads and a hard-negative margin
  loss.
- **F:** E plus ontology, OCR, barcode, entity memory and scene/state compatibility reranking.

## 16. Metrics

- Covered-region identity top-1 and top-5, and family top-1.
- Unknown rejection on regions whose labels are not in the library.
- False-confirmation rate.
- Diagnostic oracle-region scene completeness, using ground-truth boxes. This is not the official
  number.
- Official 41-scene strict completeness, using the detector, for final candidates only.
- Every result is split into synthetic and photo scenes.
- Validation metrics come from the library or a held-out split, never the benchmark.

## 17. Success gates

Official strict completeness of at least 0.4390, with no regression on the 12 scenes currently
complete. On top of that:

- false-confirmation rate no higher than A's;
- unknown rejection of at least 0.9;
- better family and state results than A;
- all reported separately for photo and synthetic scenes.

## 18. Missing data

1. A clean reference set. Current labels are noisy (leave-one-out 0.48).
2. Reference images for the photo-scene identities: salad, pizza, club_sandwich, lasagna, pasta,
   omelette, guacamole, hummus, donuts, ice_cream, pad_thai, spring_rolls, bibimbap, caesar_salad,
   french_fries.
3. A real-photo validation set separate from the 41 scenes.
4. Hard negatives: red pepper, flatbread, stew, sliced cheese.

## 19. First experiment run (A + B), and result

`v5_exp_AB_retrieval_001`: CPU, no training, 2 minutes.

| Metric (71 GT regions) | A flat | B hierarchical |
|---|---|---|
| Covered identity top-1 (raw) | 0.156 | 0.094 |
| Covered identity accepted and correct | 0.000 | 0.000 |
| Covered top-5 | — | 0.844 |
| Covered family top-1 | 0.25 | 0.25 |
| Unknown rejection (uncovered) | 1.00 | 1.00 |
| False confirmation | 0.00 | 0.00 |
| Oracle-region scene completeness | 0.098 | 0.098 |

Library-only calibration chose high thresholds (A 0.72, B 0.64), because the noisy library cannot
separate known from unknown at lower values. Query similarities to the benchmark top out at 0.64
(mean 0.35), so nothing is confirmed: it is safe, but nothing is recognised. B's top-5 hit rate
(0.84) shows that the correct label is usually a candidate. The ranking fails because synthetic
crops all collapse onto `cream` (35 of 71 regions).

**Earliest divergence.** The first failure is not the representation. It is two things:

- **Evaluation domain.** 27 of the 32 recognisable regions are drawings.
- **Reference quality and coverage.** The library is noisy, and the photo scenes need identities
  it does not have.

On the photo regions it does cover, frozen DINOv2 is already 4 of 5 correct at top-1.

**Decision.** Training adapters C, D or E now would not answer the V5 question. Real-photo
training cannot be validated against drawn shapes, and adding drawings to training would be
benchmark leakage or gaming. C is therefore not launched; that would waste GPU on an experiment
known to be uninformative.

## 20. Command to reproduce

```
cd C:\Projects\KitchenIQ-V10-AI
python orchestration\vision_v5\exp_ab_retrieval.py
python orchestration\vision_v5\diag_hubness.py
python orchestration\vision_v5\diag_fixture_domain.py
```

Next runnable experiment (on Kaggle, after the evaluation decision): `kaggle/kernels/kiq-v10-vision-v5-adapter`
(not yet created). It would clean the references, train adapter C, and validate on a held-out
real-photo split.
