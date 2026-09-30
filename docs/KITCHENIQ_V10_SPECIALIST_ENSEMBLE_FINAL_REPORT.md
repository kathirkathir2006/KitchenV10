# KitchenIQ V10 Specialist Ensemble — Final Report

**Generated:** 2026-09-30T13:25:42.658112+00:00  
**Project:** `C:\Projects\KitchenIQ-V10-AI`

## 1. Baseline
- Strict scene completeness: **0.2439**
- Prior classifier-only challenger: 0.2195 (do not repeat)
- Immutable record: `backend/instance/dev_experiments/v10-specialist-ensemble/baseline/v10-specialist-baseline.json`

## 2. Architecture implemented
B_detector_specialist_ensemble (Faster R-CNN + food/nonfood + identity + prepared-meal + food-state + fusion/abstention)

## 3. Specialists implemented
- food/non-food specialist (visual_class + hardneg-strict mode)
- identity specialist (hierarchical stop via fusion statuses)
- prepared-meal specialist (no raw decomposition)
- food-state specialist (independent confidence)
- evidence fusion + conflict resolution + abstention

## 4–5. Datasets / production vs experiment
- 41-scene fixtures copied into V10 acceptance (eval only)
- Remediation production-eligible manifests referenced read-only from old project
- No experiment-only data mixed into production promotion
- No 207×250 manufacture

## 6–7. Training / candidates
- No blind classifier-only retrain
- Candidate A: specialist fusion default
- Candidate B: hard-negative strict
- Candidate C: hardneg-strict + prepared dominance; Kaggle GPU confirmation kernel

Candidate scores: {
  "candidate_a": {
    "strict": 0.2195,
    "delta": -0.0244,
    "stages": {
      "CLASSIFIER": 43,
      "DETECTOR": 12,
      "FOOD_NONFOOD": 4
    }
  },
  "candidate_b": {
    "strict": 0.2195,
    "delta": -0.0244,
    "stages": {
      "CLASSIFIER": 43,
      "DETECTOR": 12,
      "FOOD_NONFOOD": 4
    }
  },
  "candidate_c": {
    "strict": 0.2195,
    "delta": -0.0244,
    "stages": {
      "CLASSIFIER": 43,
      "DETECTOR": 12,
      "FOOD_NONFOOD": 4
    }
  }
}

## 8. Calibration
Bands preserved: HIGH≥0.85, MEDIUM≥0.55, LOW<0.55. Abstention statuses used.

## 9–10. 41-scene / baseline comparison
**0.2439 → 0.2195** (best=candidate_a)

## 11. Failure attribution
{
  "CLASSIFIER": 43,
  "DETECTOR": 12,
  "FOOD_NONFOOD": 4
}
Dominant stage: CLASSIFIER
Root cause hypothesis: DATA

## 12–17. Hard-neg / multi-object / prepared / state / OCR / barcode / qty
See per-scene acceptance JSON under `acceptance/`. OCR remains fail-closed if unavailable; barcode not the primary lever this cycle.

## 18. Downstream entity/state
Ensemble does not mutate KitchenState; contract check recorded in `reports/17_end_to_end.json`.

## 19. Production gate
**FAIL**

## 20. Final decision
**B. SPECIALIST ENSEMBLE NEEDS TARGETED DATA REMEDIATION**

## 21. Dominant blocker
Best specialist candidate strict=0.2195 vs baseline 0.2439 (delta=-0.0244). Dominant failure stage=CLASSIFIER.

## Old project integrity
**UNCHANGED**
