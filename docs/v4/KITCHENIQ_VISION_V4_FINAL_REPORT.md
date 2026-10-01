# KitchenIQ Vision V4 — Final Report

**Generated:** 2026-10-01T01:47:42.083170+00:00  
**Repo:** KitchenV10 / `C:\Projects\KitchenIQ-V10-AI`  
**Old project:** UNCHANGED

## 1. Baseline
Current best verified: **0.2927**

## 2. Previous results
Immutable 0.2439 | specialist 0.2195 | targeted 0.0732 | recovery 0.2927 | V3 B/E 0.2439 | V4 no-retrieval 0.2195 | V4+retrieval(n=108) 0.2195

## 3–5. Root causes / bottlenecks
PRIMARY: **OPEN_SET** (10 scenes blocked)  
SECONDARY: **FOOD_NONFOOD** (9 scenes)  
TERTIARY: **DETECTOR** (7 scenes)  
IDENTITY: 3 scenes

## 6–7. Architecture
**V4_OPEN_WORLD_SCENE_ENGINE** — DINOv2 retained; top-k candidates; evidence fusion; open-set; food-gate repair; scene context; hierarchical retrieval with assert n>0.  
Rejected: blind DINOv3, blind RT-DETR, small softmax retrain, empty-library retrieval.

## 8–12. Data / model / hashes
Retrieval library PRESENT n=108 sha=`72db59a963781d5752239d6374d64393fa06661867c4d472c68234311e21c91a`  
Model SHA256: `e1efaa4ac04ccb060397e584b1aa78aa28c51a79009833b7a3143adbc8955853`  
Dataset manifest SHA256: `9da443fa43735513ee0bcf0712fdd2720111a039fc9f20d3116505ddeb782e24`  
Fixture catalog SHA256: `38fc3eaee75c76fa5fd1dc3d4177af4abde2cfeb3e410d290fb524f9e30a1523`

## 13–15. Validation / 41-scene / failure stages
V4+retrieval strict: **0.2195** (delta -0.0732)  
Best remains A: **0.2927**  
Substantial target: **0.439**  
Stages after V4: CLASSIFIER=43 DETECTOR=12 FOOD_NONFOOD=2  
Before (A): CLASSIFIER=40 DETECTOR=12 FOOD_NONFOOD=2

## 16–20. Secondary metrics
Hard-neg rate: 0.5 (2/4)  
Food-state agreement: 1.0 (16/16)  
Open-set over-abstention dominated (tomato→unknown_food even with retrieval top containing tomato at low score)

## 21–25. Security / regression / holdout / latency / GPU
Fail-closed preserved. Holdout untouched. KitchenIQ-OS UNCHANGED.  
Kaggle: kathiresannatarajan/kiq-v10-vision-v4 (library build). Local CPU 41-scene eval.

## 26–28. Gate / decision / next
Production gate: **FAIL**  
Final decision: **V4_BLOCKED_OPEN_WORLD**  
Exact blocker: V4 open-world scene engine (top-k + fusion + open-set + food-gate + retrieval n=108) regressed to 0.2195 vs baseline 0.2927; over-abstention / weak identity evidence remains the scene-blocking failure mode (PRIMARY=OPEN_SET). Retrieval library present but does not lift 41-scene strict completeness.  
Exact next: Do not launch another flat softmax retrain or empty-library retrieval. Required next: (1) production-eligible hard-negative + calibrated open-set thresholds tuned on identity_coverage failures without sacrificing the 12 currently-complete scenes; (2) for the measured misses — tomato(4), fried_rice(3), omelette(2), lasagna(2), non_food(2) — either raise specialist recall above the open-set gate with >=80 verified production images each for fried_rice/omelette (currently 0) and fill taxonomy gaps lasagna/guacamole/hummus, OR replace the crop identity path with an open-vocab detector+segmenter that emits regions before open-set (DETECTOR blocks 7 scenes); (3) keep A=0.2927 as production reference until a candidate >=0.4390.

## Feasibility (precise)
{
  "remaining_gap_to_substantial": 0.2195,
  "scenes_needed_for_substantial": 9,
  "precise_identity_requirements": [
    {
      "canonical_identity": "tomato",
      "failures_addressed_approx": 4,
      "production_image_count": 40,
      "status": "SUFFICIENT",
      "taxonomy_gap": false,
      "additional_production_images_needed": 40,
      "acceptable_sources": [
        "openverse_cc0_cc_by",
        "wikimedia_cc"
      ],
      "priority": "HIGH"
    },
    {
      "canonical_identity": "fried_rice",
      "failures_addressed_approx": 3,
      "production_image_count": 0,
      "status": null,
      "taxonomy_gap": null,
      "additional_production_images_needed": 80,
      "acceptable_sources": [
        "openverse_cc0_cc_by",
        "wikimedia_cc"
      ],
      "priority": "HIGH"
    },
    {
      "canonical_identity": "omelette",
      "failures_addressed_approx": 2,
      "production_image_count": 0,
      "status": null,
      "taxonomy_gap": null,
      "additional_production_images_needed": 80,
      "acceptable_sources": [
        "openverse_cc0_cc_by",
        "wikimedia_cc"
      ],
      "priority": "MEDIUM"
    },
    {
      "canonical_identity": "lasagna",
      "failures_addressed_approx": 2,
      "production_image_count": 0,
      "status": "TAXONOMY_GAP",
      "taxonomy_gap": true,
      "additional_production_images_needed": null,
      "acceptable_sources": [
        "openverse_cc0_cc_by",
        "wikimedia_cc"
      ],
      "priority": "MEDIUM"
    },
    {
      "canonical_identity": "non_food",
      "failures_addressed_approx": 2,
      "production_image_count": 0,
      "status": null,
      "taxonomy_gap": null,
      "additional_production_images_needed": 80,
      "acceptable_sources": [
        "openverse_cc0_cc_by",
        "wikimedia_cc"
      ],
      "priority": "MEDIUM"
    },
    {
      "canonical_identity": "garlic",
      "failures_addressed_approx": 1,
      "production_image_count": 36,
      "status": "SUFFICIENT",
      "taxonomy_gap": false,
      "additional_production_images_needed": 44,
      "acceptable_sources": [
        "openverse_cc0_cc_by",
        "wikimedia_cc"
      ],
      "priority": "MEDIUM"
    },
    {
      "canonical_identity": "ginger",
      "failures_addressed_approx": 1,
      "production_image_count": 33,
      "status": "SUFFICIENT",
      "taxonomy_gap": false,
      "additional_production_images_needed": 47,
      "acceptable_sources": [
        "openverse_cc0_cc_by",
        "wikimedia_cc"
      ],
      "priority": "MEDIUM"
    },
    {
      "canonical_identity": "rice",
      "failures_addressed_approx": 1,
      "production_image_count": 40,
      "status": "SUFFICIENT",
      "taxonomy_gap": false,
      "additional_production_images_needed": 40,
      "acceptable_sources": [
        "openverse_cc0_cc_by",
        "wikimedia_cc"
      ],
      "priority": "MEDIUM"
    },
    {
      "canonical_identity": "packaged_cheese",
      "failures_addressed_approx": 1,
      "production_image_count": 9,
      "status": "LOW_SUPPORT",
      "taxonomy_gap": false,
      "additional_production_images_needed": 71,
      "acceptable_sources": [
        "openverse_cc0_cc_by",
        "wikimedia_cc"
      ],
      "priority": "MEDIUM"
    },
    {
      "canonical_identity": "guacamole",
      "failures_addressed_approx": 1,
      "production_image_count": 0,
      "status": "TAXONOMY_GAP",
      "taxonomy_gap": true,
      "additional_production_images_needed": null,
      "acceptable_sources": [
        "openverse_cc0_cc_by",
        "wikimedia_cc"
      ],
      "priority": "MEDIUM"
    },
    {
      "canonical_identity": "hummus",
      "failures_addressed_approx": 1,
      "production_image_count": 0,
      "status": "TAXONOMY_GAP",
      "taxonomy_gap": true,
      "additional_production_images_needed": null,
      "acceptable_sources": [
        "openverse_cc0_cc_by",
        "wikimedia_cc"
      ],
      "priority": "MEDIUM"
    },
    {
 
