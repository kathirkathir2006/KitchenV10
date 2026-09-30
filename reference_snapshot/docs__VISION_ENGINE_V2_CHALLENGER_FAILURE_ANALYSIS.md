# Vision Engine V2 Challenger — Failure Analysis

**Generated:** 2026-09-29T11:57:23.413873+00:00
**Scope:** Diagnostic only — no retrain, no acquisition, no production/holdout changes.

## Executive conclusion

Strict complete-scene rate fell solely because one scene flipped: 13_kitchen_mood (prod strict OK -> challenger fail). Same detector boxes; challenger became more confident that non-food kitchen-mood crops were tomato/vegetable (e.g. tomato 0.978->0.998), adding a CLASSIFIER failure stage alongside NON_FOOD_REJECTION. Detector stage totals unchanged (30->30). FOOD_STATE stage failures rose 8->17 (hurts partial completeness) but did not flip additional strict scenes. OI crop Top-1 up does not transfer to kitchen completeness: shared missing critical labels (garlic/ginger/rice/biryani/yogurt/...) are absent from BOTH models (53.6% of required identities) and therefore do not explain the prod->challenger delta.

**Dominant failure source:** `E_MULTIPLE_BLOCKERS`

**Challenger failed-instance attribution (official acceptance stages):** detector 33.0% · classifier 45.1% · state 18.7% · non-food 3.3% · other 0.0%

**Strict delta cause:** 1 scene `13_kitchen_mood` — non-food overconfidence / classifier label+calibration regression (not missing garlic/ginger/rice labels; those are absent from both models).

**Missing-label impact:** 53.6% of acceptance-required identities absent from challenger (15/28); absent from both: 15; absent from challenger only: []; pure absent-label fail events observed: 21.

**Another classifier-only training cycle justified?** **NO** — Classifier-only retraining on more OI food crops is NOT justified as the next step: the measured strict drop is a non-food overconfidence regression on 1/41 scenes, detector stage events remain ~33% of challenger failures, and absent critical labels are already shared with production (absent_from_challenger_only=[]). A future training cycle is only justified after acquiring PRODUCTION_ELIGIBLE coverage for absent critical identities PLUS hard non-food/empty negatives — not another OI-only food expansion.

**Exact next engineering action:** `5_acquire_hard_nonfood_negatives_and_absent_critical_labels_then_retrain; do_not_run_classifier_only_OI_cycle; detector_improvement_secondary`

## Production vs challenger comparison

- Strict: **0.2439 → 0.2195** (Δ -0.0244 = exactly 1/41 scenes)
- Top-1 val/test: **0.6177/0.5554 → 0.6155/0.5714**
- Failure stage totals prod→ch: {'CLASSIFIER': 40, 'DETECTOR': 30, 'FOOD_STATE': 8, 'NON_FOOD_REJECTION': 3} → {'CLASSIFIER': 41, 'DETECTOR': 30, 'FOOD_STATE': 17, 'NON_FOOD_REJECTION': 3}

## Exact scene regressions (prod strict OK, challenger not)

- **13_kitchen_mood** (13_non_food): reasons=['classifier_label_regression', 'confidence_regression', 'threshold_calibration_regression', 'state_regression']; prod_stages={'NON_FOOD_REJECTION': 1}; ch_stages={'CLASSIFIER': 1, 'NON_FOOD_REJECTION': 1}

## Critical-class top-5 comparison

| Scene | Actual | In prod vocab | In ch vocab | Prod top-1 (conf) | Ch top-1 (conf) | Prod top-5 | Ch top-5 |
|---|---|---|---|---|---|---|---|
| 01_tomato_pilot | tomato | True | True | pancake (0.1966) | potato (0.3165) | ['pancake', 'lemon', 'grapefruit', 'shellfish', 'beer'] | ['potato', 'shellfish', 'pancake', 'fruit', 'non_food'] |
| 02_multi_raw_composite | tomato | True | True | grapefruit (0.2764) | potato (0.3357) | ['grapefruit', 'pancake', 'beer', 'shellfish', 'food'] | ['potato', 'shellfish', 'pancake', 'non_food', 'beer'] |
| 02_multi_raw_composite | garlic | False | False | shellfish (0.5535) | shellfish (0.5337) | ['shellfish', 'beer', 'pancake', 'lemon', 'grapefruit'] | ['shellfish', 'pancake', 'potato', 'lemon', 'beer'] |
| 02_multi_raw_composite | ginger | False | False | beer (0.3627) | shellfish (0.4371) | ['beer', 'shellfish', 'grapefruit', 'pancake', 'lemon'] | ['shellfish', 'pancake', 'potato', 'beer', 'lemon'] |
| 02_multi_raw_composite | rice | False | False | grape (0.2171) | pancake (0.3217) | ['grape', 'food', 'broccoli', 'shellfish', 'cake'] | ['pancake', 'broccoli', 'shellfish', 'non_food', 'banana'] |
| 03_dense_kitchen | fried_rice | False | False | food (0.9824) | food (0.9731) | ['food', 'fast_food', 'vegetable', 'pasta', 'sushi'] | ['food', 'vegetable', 'fast_food', 'bread', 'pasta'] |
| 04_ingredient_packaged | tomato | True | True | beer (0.4875) | potato (0.2558) | ['beer', 'grapefruit', 'pancake', 'lemon', 'food'] | ['potato', 'non_food', 'beer', 'vegetable', 'shellfish'] |
| 04_ingredient_packaged | packaged_cheese | True | True | beer (0.8168) | cheese (0.1505) | ['beer', 'fast_food', 'shellfish', 'grapefruit', 'lemon'] | ['cheese', 'grapefruit', 'beer', 'shellfish', 'lemon'] |
| 05_ingredient_prepared | tomato | True | True | beer (0.4314) | potato (0.2319) | ['beer', 'pancake', 'food', 'grapefruit', 'lemon'] | ['potato', 'non_food', 'pancake', 'shellfish', 'beer'] |
| 06_multi_prepared | fried_rice | False | False | food (0.9645) | food (0.9498) | ['food', 'fast_food', 'vegetable', 'pasta', 'fruit'] | ['food', 'vegetable', 'fast_food', 'bread', 'pasta'] |
| 07_packaged | packaged_cheese | True | True | beer (0.8222) | cheese (0.2325) | ['beer', 'fast_food', 'shellfish', 'cheese', 'lemon'] | ['cheese', 'beer', 'shellfish', 'non_food', 'food'] |
| 10_raw_vs_partial | tomato | True | True | beer (0.4559) | potato (0.2445) | ['beer', 'pancake', 'lemon', 'food', 'grapefruit'] | ['potato', 'pancake', 'beer', 'shellfish', 'non_food'] |
| 10_raw_vs_partial | tomato_paste | True | True | beer (0.7652) | shellfish (0.1922) | ['beer', 'shellfish', 'cheese', 'grapefruit', 'tomato'] | ['shellfish', 'cheese', 'fish', 'tomato', 'food'] |
| 13_empty_plate | non_food | True | True | shellfish (0.686) | shellfish (0.4106) | ['shellfish', 'grape', 'starfish', 'grapefruit', 'beer'] | ['shellfish', 'potato', 'grapefruit', 'cucumber', 'pancake'] |
| 13_human_hand | non_food | True | True | grapefruit (0.2929) | shellfish (0.5061) | ['grapefruit', 'shellfish', 'waffle', 'coffee_cup', 'food'] | ['shellfish', 'food', 'egg', 'pancake', 'wine_glass'] |
| 13_pantry_empty | non_food | True | True | tomato (0.5992) | tomato (0.9718) | ['tomato', 'fruit', 'broccoli', 'food', 'vegetable'] | ['tomato', 'fruit', 'vegetable', 'salad', 'broccoli'] |
| 13_kitchen_mood | non_food | True | True | tomato (0.9779) | tomato (0.9982) | ['tomato', 'fruit', 'food', 'grapefruit', 'egg'] | ['tomato', 'fruit', 'egg', 'vegetable', 'grapefruit'] |
| 21_duplicate_tomatoes | tomato | True | True | grapefruit (0.2577) | potato (0.3102) | ['grapefruit', 'pancake', 'beer', 'lemon', 'shellfish'] | ['potato', 'shellfish', 'pancake', 'beer', 'non_food'] |
| 21_duplicate_tomatoes | tomato | True | True | beer (0.3572) | potato (0.4114) | ['beer', 'grapefruit', 'pancake', 'lemon', 'shellfish'] | ['potato', 'shellfish', 'beer', 'pancake', 'non_food'] |
| 21_duplicate_tomatoes | tomato | True | True | grapefruit (0.2632) | potato (0.3232) | ['grapefruit', 'beer', 'pancake', 'lemon', 'shellfish'] | ['potato', 'shellfish', 'pancake', 'beer', 'non_food'] |
| 21_duplicate_tomatoes | tomato | True | True | beer (0.3343) | potato (0.2895) | ['beer', 'pancake', 'food', 'grapefruit', 'lemon'] | ['potato', 'shellfish', 'pancake', 'non_food', 'beer'] |
| 22_qty_one | tomato | True | True | beer (0.3621) | potato (0.353) | ['beer', 'grapefruit', 'pancake', 'food', 'shellfish'] | ['potato', 'shellfish', 'beer', 'non_food', 'pancake'] |
| 22_qty_four | tomato | True | True | grapefruit (0.2577) | potato (0.3102) | ['grapefruit', 'pancake', 'beer', 'lemon', 'shellfish'] | ['potato', 'shellfish', 'pancake', 'beer', 'non_food'] |
| 22_qty_four | tomato | True | True | beer (0.3572) | potato (0.4114) | ['beer', 'grapefruit', 'pancake', 'lemon', 'shellfish'] | ['potato', 'shellfish', 'beer', 'pancake', 'non_food'] |
| 22_qty_four | tomato | True | True | grapefruit (0.2632) | potato (0.3232) | ['grapefruit', 'beer', 'pancake', 'lemon', 'shellfish'] | ['potato', 'shellfish', 'pancake', 'beer', 'non_food'] |
| 22_qty_four | tomato | True | True | beer (0.3343) | potato (0.2895) | ['beer', 'pancake', 'food', 'grapefruit', 'lemon'] | ['potato', 'shellfish', 'pancake', 'non_food', 'beer'] |
| 23_tomato_vs_tomato_paste | tomato | True | True | beer (0.3353) | potato (0.299) | ['beer', 'grapefruit', 'pancake', 'lemon', 'shellfish'] | ['potato', 'shellfish', 'non_food', 'beer', 'pancake'] |
| 23_tomato_vs_tomato_paste | tomato_paste | True | True | beer (0.6212) | cheese (0.2102) | ['beer', 'tomato', 'grapefruit', 'shellfish', 'strawberry'] | ['cheese', 'shellfish', 'tomato', 'food', 'fish'] |
| 23_yogurt_vs_cream | yogurt | False | False | shellfish (0.6001) | shellfish (0.6046) | ['shellfish', 'beer', 'grapefruit', 'pancake', 'lemon'] | ['shellfish', 'pancake', 'non_food', 'beer', 'banana'] |
| 23_yogurt_vs_cream | cream | True | True | shellfish (0.5637) | shellfish (0.5985) | ['shellfish', 'beer', 'pancake', 'lemon', 'grapefruit'] | ['shellfish', 'pancake', 'beer', 'non_food', 'lemon'] |
| 23_cheese_vs_packaged_cheese | cheese | True | True | beer (0.6583) | shellfish (0.2133) | ['beer', 'grapefruit', 'lemon', 'pancake', 'shellfish'] | ['shellfish', 'beer', 'pancake', 'egg', 'lemon'] |
| 23_cheese_vs_packaged_cheese | packaged_cheese | True | True | beer (0.6583) | shellfish (0.2133) | ['beer', 'grapefruit', 'lemon', 'pancake', 'shellfish'] | ['shellfish', 'beer', 'pancake', 'egg', 'lemon'] |
| 23_rice_vs_biryani | rice | False | False | grape (0.1775) | shellfish (0.2182) | ['grape', 'shellfish', 'fruit', 'pancake', 'food'] | ['shellfish', 'pancake', 'fruit', 'broccoli', 'banana'] |
| 23_rice_vs_biryani | biryani | False | False | grape (0.2228) | shellfish (0.2933) | ['grape', 'grapefruit', 'shellfish', 'fruit', 'beer'] | ['shellfish', 'fruit', 'pancake', 'broccoli', 'potato'] |
| 23_fried_rice_not_raw | fried_rice | False | False | food (0.9871) | food (0.9647) | ['food', 'fast_food', 'vegetable', 'sushi', 'hamburger'] | ['food', 'vegetable', 'sushi', 'fast_food', 'bread'] |
| 24_food_nonfood | non_food | True | True | shellfish (0.3314) | shellfish (0.6319) | ['shellfish', 'grapefruit', 'food', 'coffee_cup', 'waffle'] | ['shellfish', 'food', 'egg', 'pancake', 'wine_glass'] |
| prepared_fried_rice | fried_rice | False | False | food (0.9871) | food (0.9647) | ['food', 'fast_food', 'vegetable', 'sushi', 'hamburger'] | ['food', 'vegetable', 'sushi', 'fast_food', 'bread'] |

## Detector / classifier attribution

### Official acceptance stage totals (primary)
- Production %: {'detector_pct': 37.0, 'classifier_pct': 49.4, 'state_pct': 9.9, 'non_food_pct': 3.7, 'other_pct': 0.0, 'stage_counts': {'CLASSIFIER': 40, 'DETECTOR': 30, 'FOOD_STATE': 8, 'NON_FOOD_REJECTION': 3}, 'n_stage_events': 81}
- Challenger %: {'detector_pct': 33.0, 'classifier_pct': 45.1, 'state_pct': 18.7, 'non_food_pct': 3.3, 'other_pct': 0.0, 'stage_counts': {'CLASSIFIER': 41, 'DETECTOR': 30, 'FOOD_STATE': 17, 'NON_FOOD_REJECTION': 3}, 'n_stage_events': 91}

### Crop-IoU re-instrumentation (secondary)
- Production: `{'B_detected_misclassified': 54, 'D_correct_identity_wrong_state': 13, 'A_never_detected': 3, 'OK': 1}`
- Challenger: `{'B_detected_misclassified': 56, 'D_correct_identity_wrong_state': 12, 'A_never_detected': 3}`

## Dataset support (challenger train)

- n_train=12712 n_labels=55
- Absent critical labels from train: ['garlic', 'ginger', 'rice', 'fried_rice', 'biryani', 'yogurt', 'cream', 'packaged_cheese', 'tomato_paste', 'empty_plate', 'hand']
- Labels with train_count < 30: ['watermelon', 'banana', 'pineapple', 'pretzel', 'waffle', 'milk', 'popcorn', 'hot_dog']

## Label-space mismatch

```json
{
  "acceptance_required_identities": 28,
  "in_production": 13,
  "in_challenger": 13,
  "absent_from_challenger": 15,
  "absent_from_both": 15,
  "absent_from_challenger_only": [],
  "pure_absent_label_fail_events_observed": 21,
  "pct_required_absent_challenger": 53.6
}
```

## Calibration comparison

```json
{
  "production_reported_ece": {
    "n": 1601,
    "top1": 0.6177,
    "top5": 0.935,
    "top1_pct": 61.77,
    "top5_pct": 93.5,
    "macro_recall": 0.605,
    "n_classes_eval": 57
  },
  "challenger_val_ece": 0.1788,
  "challenger_temperature": 1.165985107421875,
  "production_temperature": 3.0,
  "confidence_distribution_on_41scene_crops": {
    "production": {
      "n": 177,
      "mean": 0.6605,
      "p50": 0.686,
      "p90": 0.9747,
      "frac_high": 0.3277,
      "frac_medium": 0.3164,
      "frac_low": 0.3559
    },
    "challenger": {
      "n": 177,
      "mean": 0.6464,
      "p50": 0.6439,
      "p90": 0.9805,
      "frac_high": 0.3107,
      "frac_medium": 0.3277,
      "frac_low": 0.3616
    }
  },
  "band_counts_on_41scene_crops": {
    "production": {
      "LOW": 63,
      "MEDIUM": 56,
      "HIGH": 58
    },
    "challenger": {
      "LOW": 64,
      "HIGH": 55,
      "MEDIUM": 58
    }
  },
  "abstention_low_or_empty_count": {
    "production": 59,
    "challenger": 59
  },
  "high_confidence_false_positive_crop_events": {
    "production": 6,
    "challenger": 5
  },
  "challenger_more_confident_when_wrong": false
}
```

## Failure percentages (challenger, official stages)

- Detector: **33.0%**
- Classifier: **45.1%**
- State: **18.7%**
- Non-food: **3.3%**
- Other: **0.0%**

## Exact next engineering action

Classifier-only retraining on more OI food crops is NOT justified as the next step: the measured strict drop is a non-food overconfidence regression on 1/41 scenes, detector stage events remain ~33% of challenger failures, and absent critical labels are already shared with production (absent_from_challenger_only=[]). A future training cycle is only justified after acquiring PRODUCTION_ELIGIBLE coverage for absent critical identities PLUS hard non-food/empty negatives — not another OI-only food expansion.

Recommended code: `5_acquire_hard_nonfood_negatives_and_absent_critical_labels_then_retrain; do_not_run_classifier_only_OI_cycle; detector_improvement_secondary`
