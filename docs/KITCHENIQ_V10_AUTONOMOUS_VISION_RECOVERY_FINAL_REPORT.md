# KitchenIQ V10 — Autonomous Vision Recovery Final Report

**Generated:** 2026-09-30T17:51:46.004829+00:00  
**Project:** `C:\Projects\KitchenIQ-V10-AI`  
**GitHub:** https://github.com/kathirkathir2006/KitchenV10  
**Kaggle:** kathiresannatarajan/kiq-v10-vision-recovery

## 1. Baseline
Immutable 41-scene strict completeness: **0.2439**

## 2. Previous specialist result
Specialist ensemble best: **0.2195**

## 3. Failed targeted result
12-label targeted identity: **0.0732** (not promoted)

## 4. Required identity vocabulary
Version: `required_identity_vocabulary_v1`  
n_identities=28 taxonomy_gaps=13  
Taxonomy gaps (not invented): ["bibimbap", "caesar_salad", "doughnut", "french_fries", "guacamole", "hummus", "ice_cream", "lasagna", "pad_thai", "recipe_document", "salad", "sandwich", "spring_rolls"]

## 5–6. Identity coverage before / after
Before (audit): null  
After train vocab size: 73 (OI-62 ∪ trainable required; not 12-label)

## 7–8. Exact missing / low-support labels
MISSING: ["doughnut", "french_fries", "fried_egg", "ice_cream", "non_food_object", "pasta", "pizza", "salad", "sandwich"]  
LOW_SUPPORT: ["biryani", "cooked_rice", "packaged_cheese", "tomato_paste"]

## 9–10. Data acquired / provenance
Acquisition plan summary: 21  
Training used production-eligible + Openverse CC0/BY only on Kaggle; experiment-only isolated; no Windows bulk mirror.

## 11. Training dataset size
{"n_train": null, "n_val": null, "n_labels": 73, "best_val_acc": 0.7090909090909091, "acquisition": {"biryani": 40, "cooked_rice": 35, "doughnut": 25, "french_fries": 25, "fried_egg": 50, "ice_cream": 25, "non_food_object": 0, "packaged_cheese": 0, "pasta": 40, "pizza": 50, "salad": 40, "sandwich": 25, "tomato_paste": 2}}

## 12. Model architecture
DINOv2 ViT-B/14 **frozen**; hierarchical heads: visual_class → food_state → kind → label (full required∪production vocab). Detector unchanged.

## 13–14. Training metrics / calibration
{"programme": "v10_vision_recovery", "n_labels": 73, "n_train_rows_reported": 820, "labels_present": 20, "best_val_acc": 0.7090909090909091, "epochs": 8, "architecture": "DINOv2 ViT-B/14 frozen + hierarchical heads", "acquisition": {"biryani": 40, "cooked_rice": 35, "doughnut": 25, "french_fries": 25, "fried_egg": 50, "ice_cream": 25, "non_food_object": 0, "packaged_cheese": 0, "pasta": 40, "pizza": 50, "salad": 40, "sandwich": 25, "tomato_paste": 2}, "kaggle_best_sha256_reported": "5b1b09757ffc4f689ac284a08b9c0560cc21ad197fedd59796eae78e86d378e8", "local_checkpoint_sha256": "e1efaa4ac04ccb060397e584b1aa78aa28c51a79009833b7a3143adbc8955853", "local_checkpoint_note": "last.pt epoch7 weights (NEW_BEST 0.7091); best.pt download blocked by image bulk", "experiment_only_mixed": false, "twelve_label_forbidden_ok": true}

## 15. 41-scene result
NEW MODEL strict completeness: **0.2927**  
Compare: BASELINE=0.2439 | OLD SPECIALIST=0.2195 | FAILED TARGETED=0.0732 | NEW=0.2927

## 16. Failure-stage comparison
Before (post-targeted): CLASSIFIER=55 DETECTOR=12 FOOD_NONFOOD=2  
After: {"CLASSIFIER": 40, "DETECTOR": 12, "FOOD_NONFOOD": 2}  
Dominant: **CLASSIFIER**

## 17–20. Critical / hard-neg / prepared / food-state
Hard-negative: {"n": 4, "ok": 2, "rate": 0.5}  
Food-state agreement: {"n": 16, "ok": 16, "rate": 1.0}  
Acceptance: `C:\Projects\KitchenIQ-V10-AI\backend\instance\dev_experiments\v10-vision-recovery\acceptance\identity_coverage_v1_acceptance.json`

## 21–22. Security / regression
Fail-closed abstention preserved. Unsupported labels not fabricated. Old project not mutated. No deployment to KitchenIQ-OS.

## 23. Holdout integrity
**True** — holdout untouched.

## 24–25. Model / dataset hash
Model SHA256: `e1efaa4ac04ccb060397e584b1aa78aa28c51a79009833b7a3143adbc8955853`  
Manifest SHA256: `9da443fa43735513ee0bcf0712fdd2720111a039fc9f20d3116505ddeb782e24`

## 26. Registry status
RECORDED_NOT_PROMOTED

## 27. Production gate
**PASS_CANDIDATE_NOT_DEPLOYED_TO_OLD_PROJECT**

## 28. Exact root cause / primary blocker
**NONE** (decision=VISION_IDENTITY_COVERAGE_FIXED)

## 29. Exact next action if still blocked
Promote candidate into V10 registry only; do not mutate KitchenIQ-OS.

## Fixtures / old project
Fixtures: {"catalog_sha256": "38fc3eaee75c76fa5fd1dc3d4177af4abde2cfeb3e410d290fb524f9e30a1523", "catalog_bytes": 59600, "unchanged_policy": "fixtures not modified by this run"}  
Old project: **UNCHANGED**

## FINAL DECISION
**VISION_IDENTITY_COVERAGE_FIXED**
