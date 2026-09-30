# KitchenIQ — Vision Engine V2 Implementation Final Report

**Document type:** Implementation final report (measured)
**Contract:** `docs/VISION_ENGINE_V2_IMPLEMENTATION_CONTRACT.md`
**Algorithm version:** `vision-engine-v2`
**Acceptance run:** `backend/instance/dev_experiments/fv-real-world-acceptance-v2/`
**Generated:** 2026-09-26
**Production status:** BLOCKED

---

## IMPLEMENTED

- Executable Vision Engine V2 pipeline (`run_vision_engine_v2`)
- Stages: secure validation → scene → region/instance discovery (NMS) → food/non-food gate → hierarchical identity → food-state → OCR → barcode → quantity/instance → evidence fusion → confidence/abstention → canonicalisation hints → entity-match hints → VisualObservation
- Open-world outcomes: KNOWN | UNKNOWN_FOOD | UNKNOWN_NONFOOD | UNCERTAIN | CONFLICTING_EVIDENCE | REJECTED
- Null identity when evidence insufficient (no forced closed-set class)
- Hierarchical stop-at-evidence (prepared-meal non-decomposition)
- Explicit food/non-food gate with hard-negative hints
- Independent multi-attribute food-state prediction (V9.5 attributes)
- Real OCR provider (`KitchenIqOcrProvider` / pytesseract) — fail-closed
- Real barcode decoder (`pyzbar`) — absence is not verification
- Evidence fusion with per-channel available|unavailable|failed|conflicting + provenance
- Entity-match hints interface only (Vision does not mutate KitchenState)
- Additive ProviderObservation / validator fields (backwards compatible)
- Registry provider + real-world acceptance harness wired to V2
- Automated contract tests (24 scenario families)

## REUSED

- Secure Input Boundary
- Vision API / session / orchestrator
- Faster R-CNN object detector (`food-vision-detector-v1@detector-v1.0.0`)
- DINOv2 ViT-B/14 backbone + FoodVisionV2 heads (`food-vision-classifier-v1@classifier-prod-v1.0.0`)
- Model registry / integrity / select_servable
- Confidence bands HIGH≥0.85 / MEDIUM≥0.55 (unchanged)
- Calibration tables
- Canonicalisation + localization
- Prepared-meal protection
- Evidence fusion core (`cook_domain.evidence_fusion`)
- Output validation / security path
- v9.4 `run_production_algorithm` (retained, callable)
- Protected holdout (untouched)

## REPLACED

- Production OCR stub → `food_vision.v2.ocr.KitchenIqOcrProvider` (factory default)
- `KitchenIqRegistryVisionProvider` path → `run_vision_engine_v2` (was `run_production_algorithm`)
- Acceptance runner production call → V2 pipeline

**Not replaced:** DINOv2 backbone, Faster R-CNN detector, production weights, holdout, registry history.

## TESTS

| Suite | Result |
|---|---|
| `backend/tests/test_vision_engine_v2.py` | **35 passed** |
| `backend/tests/test_production_algorithm_v94.py` + `test_vision_stage3.py` | **55 passed** |
| Real-world acceptance V2 (`fv-real-world-acceptance-v2`) | **executed** (41 scenes) |

Covered V2 scenarios: single food, multiple food, dense kitchen, food+non-food, packaged, prepared meal, multiple prepared meals, raw/cooked, plated, leftover, unknown food, uncertain identity, hard negative, duplicate instances, quantity uncertainty, OCR success, OCR failure, barcode success, barcode failure, conflicting evidence, prepared-meal non-decomposition, entity-match hints, secure malformed input, fail-closed.

## REAL-WORLD ACCEPTANCE (before → after)

- Before (v1, 2026-09-25): strict_complete_rate = **0.2439**, production = **BLOCKED**, overall = **FAIL**
- After (v2, 2026-09-26): strict_complete_rate = **0.2439**, production = **BLOCKED**, overall = **FAIL**
- Interpretation: V2 correctly adds open-world/OCR/barcode/fusion behaviour; **identity completeness did not improve** because classifier artifacts were not retrained (by contract).

### Per-category

| Category | Before status | Before mean completeness | Before strict | After status | After mean completeness | After strict |
|---|---|---|---|---|---|---|
| 01_single_raw_ingredient | FAIL | 0.0 | 0.0 | FAIL | 0.0 | 0.0 |
| 02_multiple_raw_ingredients | FAIL | 0.0 | 0.0 | FAIL | 0.0 | 0.0 |
| 03_dense_kitchen_scene | PARTIAL | 0.5 | 0.0 | PARTIAL | 0.5 | 0.0 |
| 04_ingredient_plus_packaged_food | FAIL | 0.0 | 0.0 | FAIL | 0.0 | 0.0 |
| 05_ingredient_plus_prepared_meal | PARTIAL | 0.5 | 0.0 | PARTIAL | 0.5 | 0.0 |
| 06_multiple_prepared_components | FAIL | 0.3333 | 0.0 | FAIL | 0.3333 | 0.0 |
| 07_packaged_food_scene | FAIL | 0.0 | 0.0 | FAIL | 0.0 | 0.0 |
| 08_fruit_collection | FAIL | 0.0 | 0.0 | FAIL | 0.0 | 0.0 |
| 09_snack_ready_to_eat_scene | FAIL | 0.0 | 0.0 | FAIL | 0.0 | 0.0 |
| 10_raw_vs_partially_prepared | FAIL | 0.0 | 0.0 | FAIL | 0.0 | 0.0 |
| 11_cooked_vs_plated_vs_leftover | FAIL | 0.0 | 0.0 | FAIL | 0.0 | 0.0 |
| 12_recipe_document | FAIL | 0.0 | 0.0 | FAIL | 0.0 | 0.0 |
| 13_non_food | FAIL | 0.25 | 0.25 | FAIL | 0.25 | 0.25 |
| 14_occlusion | PASS | 1.0 | 1.0 | PASS | 1.0 | 1.0 |
| 15_overlap | PASS | 1.0 | 1.0 | PASS | 1.0 | 1.0 |
| 16_small_objects | FAIL | 0.0 | 0.0 | FAIL | 0.0 | 0.0 |
| 17_partial_visibility | PASS | 1.0 | 1.0 | PASS | 1.0 | 1.0 |
| 18_low_light | PASS | 1.0 | 1.0 | PASS | 1.0 | 1.0 |
| 19_motion_blur | PASS | 1.0 | 1.0 | PASS | 1.0 | 1.0 |
| 20_perspective_distortion | FAIL | 0.0 | 0.0 | FAIL | 0.0 | 0.0 |
| 21_duplicate_instances | FAIL | 0.0 | 0.0 | FAIL | 0.0 | 0.0 |
| 22_quantity_variation | FAIL | 0.0 | 0.0 | FAIL | 0.0 | 0.0 |
| 23_hard_negative_food_pairs | FAIL | 0.0 | 0.0 | FAIL | 0.0 | 0.0 |
| 24_mixed_food_and_nonfood | PARTIAL | 0.5 | 0.0 | PARTIAL | 0.5 | 0.0 |
| 25_complex_real_world_kitchen | PARTIAL | 0.5833 | 0.5 | PARTIAL | 0.5833 | 0.5 |

### Per-scene (key measured fields)

| Scene | Category | Before status | Before completeness | Before pred | After status | After completeness | After pred |
|---|---|---|---|---|---|---|---|
| 01_tomato_pilot | 01_single_raw_ingredient | FAIL | 0.0 | 'unable_to_determine' | FAIL | 0.0 | '' |
| 02_multi_raw_composite | 02_multiple_raw_ingredients | FAIL | 0.0 | 'mixed_scene' | FAIL | 0.0 | 'mixed_scene' |
| 03_dense_kitchen | 03_dense_kitchen_scene | PARTIAL | 0.5 | 'mixed_scene' | PARTIAL | 0.5 | 'mixed_scene' |
| 04_ingredient_packaged | 04_ingredient_plus_packaged_food | FAIL | 0.0 | 'mixed_scene' | FAIL | 0.0 | 'mixed_scene' |
| 05_ingredient_prepared | 05_ingredient_plus_prepared_meal | PARTIAL | 0.5 | 'mixed_scene' | PARTIAL | 0.5 | 'mixed_scene' |
| 06_multi_prepared | 06_multiple_prepared_components | PARTIAL | 0.3333 | 'mixed_scene' | PARTIAL | 0.3333 | 'mixed_scene' |
| 07_packaged | 07_packaged_food_scene | FAIL | 0.0 | 'beer' | FAIL | 0.0 | 'beer' |
| 08_fruit_proxy | 08_fruit_collection | FAIL | 0.0 | 'mixed_scene' | FAIL | 0.0 | 'mixed_scene' |
| 09_snack_rte | 09_snack_ready_to_eat_scene | FAIL | 0.0 | 'mixed_scene' | FAIL | 0.0 | 'mixed_scene' |
| 10_raw_vs_partial | 10_raw_vs_partially_prepared | FAIL | 0.0 | 'mixed_scene' | FAIL | 0.0 | 'mixed_scene' |
| 11_state_triad | 11_cooked_vs_plated_vs_leftover | FAIL | 0.0 | 'mixed_scene' | FAIL | 0.0 | 'mixed_scene' |
| 12_nutrition_label | 12_recipe_document | FAIL | 0.0 | 'unable_to_determine' | FAIL | 0.0 | '' |
| 12_recipe_document | 12_recipe_document | FAIL | 0.0 | 'unable_to_determine' | FAIL | 0.0 | '' |
| 13_empty_plate | 13_non_food | FAIL | 0.0 | 'shellfish' | FAIL | 0.0 | 'shellfish' |
| 13_human_hand | 13_non_food | PASS | 0.0 | 'unable_to_determine' | PASS | 0.0 | '' |
| 13_kitchen_mood | 13_non_food | PASS | 1.0 | 'mixed_scene' | PASS | 1.0 | 'mixed_scene' |
| 13_pantry_empty | 13_non_food | FAIL | 0.0 | 'mixed_scene' | FAIL | 0.0 | 'mixed_scene' |
| 14_occlusion | 14_occlusion | PASS | 1.0 | 'mixed_scene' | PASS | 1.0 | 'mixed_scene' |
| 15_overlap | 15_overlap | PASS | 1.0 | 'mixed_scene' | PASS | 1.0 | 'mixed_scene' |
| 16_small_objects | 16_small_objects | FAIL | 0.0 | None | FAIL | 0.0 | None |
| 17_partial_visibility | 17_partial_visibility | PASS | 1.0 | 'mixed_scene' | PASS | 1.0 | 'mixed_scene' |
| 18_low_light | 18_low_light | PASS | 1.0 | 'mixed_scene' | PASS | 1.0 | 'mixed_scene' |
| 19_motion_blur | 19_motion_blur | PASS | 1.0 | 'mixed_scene' | PASS | 1.0 | 'mixed_scene' |
| 20_perspective_distortion | 20_perspective_distortion | FAIL | 0.0 | None | FAIL | 0.0 | None |
| 21_duplicate_tomatoes | 21_duplicate_instances | FAIL | 0.0 | 'mixed_scene' | FAIL | 0.0 | 'mixed_scene' |
| 22_qty_four | 22_quantity_variation | FAIL | 0.0 | 'mixed_scene' | FAIL | 0.0 | 'mixed_scene' |
| 22_qty_one | 22_quantity_variation | FAIL | 0.0 | 'mixed_scene' | FAIL | 0.0 | 'mixed_scene' |
| 23_cheese_vs_packaged_cheese | 23_hard_negative_food_pairs | FAIL | 0.0 | 'beer' | FAIL | 0.0 | 'beer' |
| 23_fried_rice_not_raw | 23_hard_negative_food_pairs | FAIL | 0.0 | 'mixed_scene' | FAIL | 0.0 | 'mixed_scene' |
| 23_rice_vs_biryani | 23_hard_negative_food_pairs | FAIL | 0.0 | 'unable_to_determine' | FAIL | 0.0 | '' |
| 23_tomato_vs_tomato_paste | 23_hard_negative_food_pairs | FAIL | 0.0 | 'mixed_scene' | FAIL | 0.0 | 'mixed_scene' |
| 23_yogurt_vs_cream | 23_hard_negative_food_pairs | FAIL | 0.0 | 'mixed_scene' | FAIL | 0.0 | 'mixed_scene' |
| 24_food_nonfood | 24_mixed_food_and_nonfood | PARTIAL | 0.5 | 'mixed_scene' | PARTIAL | 0.5 | 'mixed_scene' |
| 25_complex_mixed | 25_complex_real_world_kitchen | PARTIAL | 0.6667 | 'mixed_scene' | PARTIAL | 0.6667 | 'mixed_scene' |
| 25_salad_bowl | 25_complex_real_world_kitchen | PASS | 1.0 | 'mixed_scene' | PASS | 1.0 | 'mixed_scene' |
| 25_tomato_pasta | 25_complex_real_world_kitchen | PASS | 1.0 | 'mixed_scene' | PASS | 1.0 | 'mixed_scene' |
| prepared_bibimbap | 25_complex_real_world_kitchen | FAIL | 0.0 | 'mixed_scene' | FAIL | 0.0 | 'mixed_scene' |
| prepared_club_sandwich | 25_complex_real_world_kitchen | PASS | 1.0 | 'mixed_scene' | PASS | 1.0 | 'mixed_scene' |
| prepared_fried_rice | 25_complex_real_world_kitchen | FAIL | 0.0 | 'mixed_scene' | FAIL | 0.0 | 'mixed_scene' |
| prepared_lasagna | 25_complex_real_world_kitchen | FAIL | 0.0 | 'food' | FAIL | 0.0 | 'food' |
| prepared_pizza | 25_complex_real_world_kitchen | PASS | 1.0 | 'pizza' | PASS | 1.0 | 'pizza' |

### Notable behaviour deltas

- **tomato / low-confidence:** before `unable_to_determine` → after **empty/null identity** (open-world abstention)
- **human hand:** PASS → PASS
- **salad bowl:** PASS → PASS
- **empty plate → shellfish:** still FAIL (high-confidence false positive; thresholds not lowered)
- Failure taxonomy primary stages (after): {"CLASSIFIER": 40, "DETECTOR": 30, "FOOD_STATE": 8, "NON_FOOD_REJECTION": 3}

## REMAINING FAILURES

- critical_category_21_duplicate_instances=FAIL
- critical_category_12_recipe_document=FAIL
- critical_category_06_multiple_prepared_components=FAIL
- critical_category_11_cooked_vs_plated_vs_leftover=FAIL
- critical_category_02_multiple_raw_ingredients=FAIL
- critical_category_23_hard_negative_food_pairs=FAIL
- critical_category_13_non_food=FAIL
- Dense/multi-object identity still classifier-dominant
- High-confidence mislabels not abstained (by design: do not lower thresholds)
- Packaged / fruit / recipe / small-object / perspective / duplicates / quantity categories still incomplete
- Acceptance summary still lists legacy OCR/barcode limitation text in `remaining_limitations` from report template; actual OCR/barcode status below supersedes that

## OCR

- Status: **UNAVAILABLE**
- Implementation: `food_vision.v2.ocr.KitchenIqOcrProvider (pytesseract/tesseract-local)`
- Model version: `tesseract-local-v1`
- Note: Provider wired; runtime binary unavailable (tesseract_binary_unavailable:TesseractNotFoundError)
- Fail-closed: yes (no fabricated text)

## BARCODE

- Status: **AVAILABLE**
- Implementation: `food_vision.v2.barcode.decode_barcodes (pyzbar-local)`
- Model version: `pyzbar-local-v1`
- Absence is not verification: yes
- Fail-closed: yes

## MODEL ARTIFACTS

- Detector PRODUCTION: `food-vision-detector-v1@detector-v1.0.0` (servable; unchanged)
- Classifier PRODUCTION: `food-vision-classifier-v1@classifier-prod-v1.0.0` (servable; unchanged)
- DINOv2 hub backbone required at serve (heads-only artifact)
- No new trained hierarchy/open-world head weights created in this pass
- Protected holdout: untouched
- Production registry history / weights: untouched

## PRODUCTION

**BLOCKED**

Measured real-world acceptance does not yet prove robust dense kitchen scene completeness across critical categories; production remains BLOCKED for real-world vision completeness even if prior component PRODUCTION_GATE=PASS.

## BLOCKERS

1. Tesseract OCR system binary not installed on host (Python `pytesseract` present; channel correctly UNAVAILABLE)
2. Existing classifier label-space/calibration insufficient for dense real-world identity completeness (requires future trained heads/data — not fabricated here)
3. Critical acceptance categories still FAIL (see blockers list above)

## FILES CHANGED

- `backend/app/services/vision/contracts.py`
- `backend/app/services/vision/validator.py`
- `backend/app/services/vision/factory.py`
- `backend/app/services/vision/ocr.py`
- `backend/app/services/vision/kitcheniq_registry_provider.py`
- `backend/app/services/food_vision/real_world_acceptance/runner.py`
- `backend/tests/test_production_algorithm_v94.py`
- `backend/requirements.txt`

## FILES CREATED

- `backend/app/services/food_vision/v2/__init__.py`
- `backend/app/services/food_vision/v2/types.py`
- `backend/app/services/food_vision/v2/scene.py`
- `backend/app/services/food_vision/v2/regions.py`
- `backend/app/services/food_vision/v2/food_gate.py`
- `backend/app/services/food_vision/v2/hierarchy.py`
- `backend/app/services/food_vision/v2/food_state.py`
- `backend/app/services/food_vision/v2/ocr.py`
- `backend/app/services/food_vision/v2/barcode.py`
- `backend/app/services/food_vision/v2/open_world.py`
- `backend/app/services/food_vision/v2/quantity.py`
- `backend/app/services/food_vision/v2/entity_match_hints.py`
- `backend/app/services/food_vision/v2/fusion.py`
- `backend/app/services/food_vision/v2/assembly.py`
- `backend/app/services/food_vision/v2/pipeline.py`
- `backend/tests/test_vision_engine_v2.py`
- `backend/instance/dev_experiments/fv-real-world-acceptance-v2/reports/*`
- `docs/VISION_ENGINE_V2_IMPLEMENTATION_FINAL_REPORT.md` (this document)

---

VISION ENGINE V2 IMPLEMENTATION COMPLETE — PRODUCTION STATUS: BLOCKED
