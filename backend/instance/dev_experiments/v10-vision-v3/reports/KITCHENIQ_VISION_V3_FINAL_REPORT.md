# KitchenIQ Vision V3 — Final Report

**Generated:** 2026-09-30T20:55:51.728771+00:00
**Repo:** https://github.com/kathirkathir2006/KitchenV10 (KitchenIQ-V10-AI only; KitchenIQ-OS READ-ONLY)

## 1. Baseline
0.2927 (identity-coverage recovery)

## 2. Substantial-improvement threshold (frozen before eval)
min_strict=**0.4390** (18/41), max_classifier_failures=**20**, marginal band 0.30–0.34 rejected.

## 3. Candidates
| Cand | Strict | Delta | Notes |
|------|--------|-------|-------|
| A | 0.2927 | 0 | existing reference |
| B | 0.2439 | -0.0488 | det+retrieval+specialist evidence |
| E | 0.2439 | -0.0488 | +evidence graph + open-set |
| C/D | skipped | — | failure matrix not detector-dominant |

## 4. Best V3 result
**0.2927** (Candidate A). No V3 candidate beat baseline; B/E regressed.

## 5. Failure-stage comparison
Before: CLASSIFIER=40 DETECTOR=12 FOOD_NONFOOD=2
After B/E: CLASSIFIER=42 DETECTOR=12 FOOD_STATE=16 FOOD_NONFOOD=2

## 6. DINOv3
Attempted dinov3_vitb16 pretrained — **HTTP 403**. Refused untrained weights. Fell back to frozen DINOv2 ViT-B/14 with V3 algorithm intact (documented).

## 7. Reference library / data
Kaggle v1 library **n=0**. Kernel v2 pushed with image_url/	arget_class fix. Production-eligible provenance retained; experiment-only isolated; no Windows bulk mirror.

## 8. Open-set / state / hard-neg
Hard-neg rate 0.5. Food-state agreement on B/E = **0.0** (state specialist mapping regression).

## 9. Production gate
**FAIL** — substantial gate not met.

## 10. Final decision
**V3_BLOCKED_DATA**

## 11. Exact next action
1) Obtain DINOv3 pretrained access OR document permanent DINOv2-primary with Meta license path. 2) Complete Kaggle reference_library build from image_url production-eligible manifests (kernel v2). 3) Re-run Candidate E only after library n_items>>0 and food_state specialist calibrated on validation (not 41-scene). 4) Keep substantial gate frozen at 0.439 / classifier<=20.

## Integrity
Old project: UNCHANGED
Fixtures SHA: 38fc3eaee75c76fa5fd1dc3d4177af4abde2cfeb3e410d290fb524f9e30a1523
Holdout untouched: true
