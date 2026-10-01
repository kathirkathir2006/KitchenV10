# Current Pipeline Forensics (0.2927 system)

Generated: 2026-09-30T21:26:32.162608+00:00

Weights: `C:\Projects\KitchenIQ-V10-AI\backend\instance\dev_experiments\v10-vision-recovery\models\identity_coverage_best.pt`
SHA256: `e1efaa4ac04ccb060397e584b1aa78aa28c51a79009833b7a3143adbc8955853`

## Stage trace

### input
- **model**: None
- **weights**: None
- **behaviour**: PIL RGB image from fixture path; JPEG bytes for detector
- **threshold**: None
- **fail_closed**: False
- **notes**: No secure upload boundary in local acceptance path

### secure_boundary
- **model**: None
- **weights**: None
- **behaviour**: Acceptance harness bypasses Flask secure input; production path uses OS food_vision routes
- **threshold**: None
- **fail_closed**: True
- **notes**: DEAD in local 41-scene eval — not exercised

### image_preprocessing
- **model**: resize 224 + ImageNet normalize
- **weights**: None
- **behaviour**: Crop → RGB → 224x224 bilinear → mean/std
- **threshold**: None
- **fail_closed**: False
- **notes**: Detector uses full-res JPEG bytes; classifier uses 224 crop

### detector
- **model**: KitchenIQ production Faster R-CNN via select_object_detector()
- **weights**: OS registry_final.db → food-vision detector artifact (read-only copy)
- **behaviour**: Top-12 boxes by confidence; empty → whole-image fallback box
- **threshold**: implicit detector score ranking; COCO FRCNN fallback score>=0.4 if KIQ load fails
- **fail_closed**: False
- **fallback**: torchvision fasterrcnn_resnet50_fpn COCO (DOMAIN WRONG)
- **notes**: Detector NOT retrained; 12 DETECTOR-stage failures at 0.2927

### crop_region_generation
- **model**: None
- **behaviour**: Axis-aligned crop from bbox; no segmentation; no NMS beyond detector
- **fail_closed**: False
- **notes**: Overlapping/small objects poorly served — no mask/proposal refinement

### food_nonfood
- **model**: FoodVisionV2 visual_class head + identity top1 heuristics in SpecialistEnsemble
- **weights**: C:\Projects\KitchenIQ-V10-AI\backend\instance\dev_experiments\v10-vision-recovery\models\identity_coverage_best.pt
- **behaviour**: visual_class==non_food OR top1 in {non_food,empty_plate,hand} → non_food
- **threshold**: hardneg_strict mode raises non_food; coverage ensemble uses fusion rules
- **fail_closed**: Aggressive non_food can veto true food (tomato→non_food observed)
- **notes**: Acceptance often MISLABELS these as CLASSIFIER; first divergence is FOOD_NONFOOD

### classifier_identity
- **model**: FoodVisionV2 fc_label over 73-class vocab (OI-62 ∪ trainable)
- **weights**: C:\Projects\KitchenIQ-V10-AI\backend\instance\dev_experiments\v10-vision-recovery\models\identity_coverage_best.pt
- **behaviour**: softmax top-1; OOV → unknown in coverage path
- **threshold**: temperature from blob calibration or 1.0; fusion MEDIUM/HIGH bands
- **calibration**: temperature scalar if present in checkpoint
- **fail_closed**: unsupported label → unknown (coverage)
- **notes**: Closed-set softmax remains primary identity authority in 0.2927 system

### food_state
- **model**: FoodVisionV2 fc_state + prepared specialist remaps
- **weights**: C:\Projects\KitchenIQ-V10-AI\backend\instance\dev_experiments\v10-vision-recovery\models\identity_coverage_best.pt
- **behaviour**: raw/prepared/packaged/processed/unknown; prepared meals mapped toward plated
- **fail_closed**: False
- **notes**: At 0.2927 food_state_agreement=1.0 on measured subset; V3 broke this to 0.0

### evidence_fusion
- **model**: fuse_specialists deterministic rules
- **weights**: None
- **behaviour**: food vs nonfood conflict; prepared labels; abstain bands
- **ocr**: accepted as optional dict but unused in coverage acceptance path
- **barcode**: accepted as optional dict but unused in coverage acceptance path
- **fail_closed**: True
- **notes**: OCR/barcode are DEAD fields in current 41-scene path

### canonicalisation
- **model**: None
- **behaviour**: obs_to_official maps FusedObservation → acceptance schema
- **fail_closed**: False
- **notes**: No Food Entity canonicalisation engine in V10 src

### entity_matching
- **model**: None
- **behaviour**: ABSENT in V10 — no runtime entity continuity
- **fail_closed**: False
- **notes**: DEAD / not implemented

### final_observation
- **model**: match_instances + scene_completeness (OS metrics, read-only)
- **behaviour**: IoU match GT vs obs; failure_stage from matcher
- **fail_closed**: False
- **notes**: Matcher may attribute FOOD_NONFOOD failures as CLASSIFIER when pred=null

## Duplicated authorities
- visual_class head AND identity label both gate food/non-food
- prepared specialist AND identity PREPARED_MEAL_LABELS set
- OS production classifier weights vs V10 coverage weights — only coverage used for 0.2927

## Dead / unused
- OCR evidence in fusion (None in acceptance)
- Barcode evidence in fusion (None in acceptance)
- Food Entity matching
- Secure input boundary in local eval
- Retrieval / evidence graph (V3 built but library n=0; not in 0.2927 path)

## Stale / fallback risks
- COCO Faster R-CNN fallback if KIQ detector load fails
- Whole-image box when detector returns empty — creates false 'regions'