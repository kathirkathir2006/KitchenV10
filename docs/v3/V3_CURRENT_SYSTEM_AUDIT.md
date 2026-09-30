# KitchenIQ Vision V3 — Current System Audit

**Generated:** 2026-09-30T19:04:32.611719+00:00  
**Project:** `C:\Projects\KitchenIQ-V10-AI` (GitHub KitchenV10 only)  
**Old project:** READ-ONLY (`C:\Projects\KitchenIQ-OS`)

## Current detector
- Production KitchenIQ Faster R-CNN via `select_object_detector()` (old backend, read-only registry DB copy).
- Wired in `src/kitcheniq_v10/specialist/pipeline.py` and recovery ensemble.
- COCO torchvision FRCNN is fallback only — domain-wrong for 41-scene GT.
- **Must keep as baseline** for V3 Phase 3; RT-DETR is challenger only.

## Current classifier
- Baseline production: FoodVisionV2 + **DINOv2 ViT-B/14 frozen** multi-head (`visual_class`, `food_state`, `kind`, `label`), OI-62 vocab.
- Identity-coverage recovery candidate: DINOv2 ViT-B/14 frozen + **73-label** head (OI-62 ∪ trainable required).
  - Artifact: `C:\Projects\KitchenIQ-V10-AI\backend\instance\dev_experiments\v10-vision-recovery\models\identity_coverage_best.pt`
  - SHA256: `e1efaa4ac04ccb060397e584b1aa78aa28c51a79009833b7a3143adbc8955853`
  - 41-scene strict: **0.2927** (12/41)
- Inference path: detect → crop → softmax top-1 label → specialist fusion → official observation schema.

## Current data flow
IMAGE → Faster R-CNN regions → FoodVisionV2 heads → `fuse_specialists` (fail-closed / abstain) → `obs_to_official` → `match_instances` / `scene_completeness`.

## Current APIs / harness
- Acceptance: `kitcheniq_v10.specialist.acceptance.run_acceptance`
- Official metrics imported from old project (read-only): `app.services.food_vision.real_world_acceptance.metrics`
- Fixtures: `backend/instance/dev_experiments/v10-specialist-ensemble/acceptance/fixtures/` (IMMUTABLE)
- Catalog SHA256: `38fc3eaee75c76fa5fd1dc3d4177af4abde2cfeb3e410d290fb524f9e30a1523`

## Current failure attribution (0.2927 run)
- CLASSIFIER = 40
- DETECTOR = 12
- FOOD_NONFOOD = 2
- Dominant blocker for incremental softmax heads remains **identity / open-set / food-gate**, not detector alone.

## Reusable components
- KIQ Faster R-CNN detector + registry load pattern
- Acceptance harness + official matchers (do not modify fixtures/scoring)
- Fusion fail-closed / prepared-meal protection patterns
- Production-eligible manifests + required identity vocabulary v1
- Canonical taxonomy CSV / entity interfaces (read-only from old where needed)

## Components that must be replaced for V3
- Single softmax “pick one label” as final identity
- Flat label space without hierarchical retrieval
- Missing evidence graph / competing hypotheses
- Missing calibrated open-set decisions (KNOWN / UNKNOWN_* / CONFLICTING / INSUFFICIENT)
- Missing cross-object relationship reasoning
- DINOv2-only representation → **DINOv3 primary frozen backbone** (controlled compare vs DINOv2 baseline)
- Optional RT-DETR region discovery (only if 41-scene region metrics improve)

## Security boundary (preserve)
- Fail-closed abstention, uncertainty bands, evidence provenance, holdout untouched, no KitchenState mutation, no allergen-safe claims, experiment-only isolation.
