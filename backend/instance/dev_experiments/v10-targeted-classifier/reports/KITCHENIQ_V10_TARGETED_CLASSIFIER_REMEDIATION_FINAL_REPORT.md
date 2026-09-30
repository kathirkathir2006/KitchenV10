# KitchenIQ V10 — Targeted Classifier Remediation Final Report

**Generated:** 2026-09-30T16:27:15.683722+00:00  
**Project:** `C:\Projects\KitchenIQ-V10-AI`  
**GitHub:** https://github.com/kathirkathir2006/KitchenV10  
**Kaggle:** kathiresannatarajan/kiq-v10-targeted-clf

## 1. Baseline
Immutable 41-scene strict completeness: **0.2439**

## 2. Previous specialist result
Specialist ensemble best: **0.2195** (delta -0.0244)

## 3–4. New candidate result / delta
Candidate strict: **0.0732**  
Delta vs baseline: **-0.1707**  
Delta vs specialist: **-0.1463**

## 5. Whether the classifier improved
Vs specialist ensemble: **False**

## 6. Whether 41-scene completeness improved
Vs immutable baseline 0.2439: **False**

## 7. Failure-stage breakdown
Before (specialist candidate_a): CLASSIFIER=43, DETECTOR=12, FOOD_NONFOOD=4  
After (targeted): {"CLASSIFIER": 55, "DETECTOR": 12, "FOOD_NONFOOD": 2}

## 8–11. Critical / hard-neg / prepared / food-state
Hard-negative: {"n": 4, "ok": 2, "rate": 0.5}  
Food-state agreement: {"n": 0, "ok": 0, "rate": 0.0}  
Per-scene details: `backend/instance/dev_experiments/v10-targeted-classifier/acceptance/targeted_identity_v1_acceptance.json`

## 12. Detector unchanged
**True** — production Faster R-CNN / KIQ detector only; not retrained.

## 13. Holdout untouched
**True**

## 14–16. Dataset provenance / size / model hash
Production-eligible manifests only (experiment-only isolated, not mixed).  
Training summary: 371  
Model: `C:\Projects\KitchenIQ-V10-AI\backend\instance\dev_experiments\v10-targeted-classifier\models\identity_specialist_best.pt`  
SHA256: `8105b63ba049cc2e4a633518de8e2a05b069667809edcd53dcc5110ca771cea0`

## 17. Registry status
RECORDED_NOT_PROMOTED (old project not modified)

## 18. Security / regression
No old-project mutation. Fail-closed abstention preserved in fusion. Candidate not deployed to production registry of old project.

## 19. Production gate
**FAIL**

## 20. Root cause if blocked
Dominant stage: **CLASSIFIER**  
Decision: **TARGETED_CLASSIFIER_FAILED_DATA_LIMITATION**

## 21. Recommended next direction if blocked
Expand production-eligible coverage for CLASSIFIER-failed labels that remain under-supported (especially prepared meals outside the remediation set and multi-object identity), without another architecture cycle; keep detector frozen until detector-stage dominates.

## Final decision
**TARGETED_CLASSIFIER_FAILED_DATA_LIMITATION**

## Old project integrity
**UNCHANGED** (sha match=True)
