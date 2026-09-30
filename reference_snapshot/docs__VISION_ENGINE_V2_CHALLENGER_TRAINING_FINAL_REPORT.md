# Vision Engine V2 Challenger Training — Final Report

**Generated:** 2026-09-29T11:26:11.664247+00:00
**Experiment:** `fv-v2-challenger` / Kaggle `kathiresannatarajan/kitcheniq-fv-v2-challenger`
**Auto-promote:** FALSE

---

## 1. Executive summary

- Challenger trained: **YES**
- GPU: **Tesla T4**
- Production strict scene rate: **0.2439**
- Challenger strict scene rate: **0.2195**
- Production val Top-1: **0.6177** / test Top-1: **0.5554**
- Challenger val Top-1: **0.6155** / test Top-1: **0.5714**
- Production promotion: **False**
- Holdout untouched: **YES** (sha `fed84f55d4a101a1280563a291229d8646eecc778562b2706ff692fb15b9134b`)

## 2. Exact current-production baseline

```json
{
  "model_id": "food-vision-classifier-v1",
  "model_version": "classifier-prod-v1.0.0",
  "architecture": "FoodVisionV2",
  "backbone": "DINOv2 ViT-B/14",
  "n_labels": 62,
  "label_vocab": [
    "apple",
    "bagel",
    "banana",
    "beer",
    "bell_pepper",
    "bread",
    "broccoli",
    "burrito",
    "cake",
    "carrot",
    "cheese",
    "coffee",
    "coffee_cup",
    "coffee_table",
    "coffeemaker",
    "cookie",
    "cucumber",
    "doughnut",
    "egg",
    "fast_food",
    "fish",
    "food",
    "food_processor",
    "french_fries",
    "fruit",
    "goldfish",
    "grape",
    "grapefruit",
    "hamburger",
    "hot_dog",
    "ice_cream",
    "jellyfish",
    "lemon",
    "milk",
    "mushroom",
    "orange",
    "pancake",
    "pasta",
    "pineapple",
    "pizza",
    "popcorn",
    "potato",
    "pretzel",
    "salad",
    "salt_and_pepper_shakers",
    "sandwich",
    "seafood",
    "shellfish",
    "shrimp",
    "starfish",
    "strawberry",
    "submarine_sandwich",
    "sushi",
    "taco",
    "tomato",
    "vegetable",
    "waffle",
    "waffle_iron",
    "watermelon",
    "wine",
    "wine_glass",
    "wine_rack"
  ],
  "train_strategy": "frozen_backbone_heads_only",
  "val_top1": 0.6177,
  "test_top1": 0.5554,
  "n_train_crops": 12712,
  "n_val_crops": 1601,
  "weights_sha256_heads": "665c96fc0ad3ed759633054f0fbf3f62cbaabfe488874ef64d0ae618bd48464c",
  "weights_full_sha256_kaggle": "2b911b9b4d0a0d953d126ee730682239e440ddfbc3c78ce6875926c3eab2861e"
}
```

Acceptance baseline (V2 suite): strict=0.2439 status=BLOCKED

## 3. Challenger dataset composition

- n_train=12712 n_val=1602 n_test=1743 n_labels=55
- zone_counts={'PRODUCTION_ELIGIBLE': 12712}
- source_counts={'open_images_ccby': 12712}
- missing critical still: ['garlic', 'ginger', 'rice', 'biryani', 'yogurt', 'cream']

## 4. Dataset provenance/licensing status

- Open Images CC-BY via `kiq-fv-detector-meta` → PRODUCTION_ELIGIBLE crops
- Food101 via `kiq-fv-food101b` → **EXPERIMENT_ONLY** expansion (not production-eligible)
- Domain-gap local 645 records available but not silently promoted
- Protected holdout: not used for train/val/calibration/selection

## 5. Critical failure coverage

```json
{
  "raw": {
    "tomato": "IN_PROD_VOCAB",
    "garlic": "MISSING_FROM_PROD_VOCAB",
    "ginger": "MISSING_FROM_PROD_VOCAB",
    "rice": "MISSING_FROM_PROD_VOCAB",
    "carrot": "IN_PROD_VOCAB",
    "potato": "IN_PROD_VOCAB",
    "egg": "IN_PROD_VOCAB",
    "mushroom": "IN_PROD_VOCAB"
  },
  "prepared": {
    "fried_rice": "MISSING_FROM_PROD_VOCAB",
    "omelette": "MISSING_FROM_PROD_VOCAB",
    "pizza": "IN_PROD_VOCAB",
    "lasagna": "MISSING_FROM_PROD_VOCAB",
    "biryani": "MISSING_FROM_PROD_VOCAB",
    "club_sandwich": "MISSING_FROM_PROD_VOCAB",
    "salad": "IN_PROD_VOCAB",
    "sandwich": "IN_PROD_VOCAB",
    "pasta": "IN_PROD_VOCAB",
    "hamburger": "IN_PROD_VOCAB"
  },
  "packaged": {
    "cheese": "IN_PROD_VOCAB",
    "beer": "IN_PROD_VOCAB",
    "milk": "IN_PROD_VOCAB",
    "wine": "IN_PROD_VOCAB"
  },
  "hardneg": {
    "cheese": "IN_PROD_VOCAB",
    "rice": "MISSING_FROM_PROD_VOCAB",
    "fried_rice": "MISSING_FROM_PROD_VOCAB",
    "tomato": "IN_PROD_VOCAB",
    "yogurt": "MISSING_FROM_PROD_VOCAB",
    "cream": "MISSING_FROM_PROD_VOCAB"
  }
}
```

Taxonomy limitations: ['Production OI-62 vocab lacks garlic, ginger, rice, fried_rice, lasagna, biryani, omelette, club_sandwich, yogurt, cream, explicit non_food/plate/hand', 'Cannot invent these labels into production taxonomy silently', 'Challenger may add EXPERIMENT_ONLY Food101-mapped labels for evaluation; production promotion requires separate provenance gate']

## 6. Kaggle training configuration

- epochs=8 batch=16 lr=0.0008 wd=0.0001 seed=42
- optimizer=AdamW augmentation=['hflip_p0.5', 'brightness_jitter_p0.3', 'resize_224', 'imagenet_norm']
- device=cuda gpu=Tesla T4

## 7. Model architecture

- FoodVisionV2 / DINOv2 ViT-B/14 frozen backbone + multi-heads (fc_label/visual/state/kind)
- train_strategy: frozen_backbone_heads_only
- Detector unchanged (Faster R-CNN production)

## 8. Training results

```json
{
  "history": [
    {
      "epoch": 1,
      "avg_loss": 1.585803,
      "val_top1": 0.5999,
      "sec": 206.3
    },
    {
      "epoch": 2,
      "avg_loss": 1.052829,
      "val_top1": 0.6017,
      "sec": 220.9
    },
    {
      "epoch": 3,
      "avg_loss": 0.948118,
      "val_top1": 0.6142,
      "sec": 220.8
    },
    {
      "epoch": 4,
      "avg_loss": 0.902614,
      "val_top1": 0.6155,
      "sec": 220.1
    },
    {
      "epoch": 5,
      "avg_loss": 0.85406,
      "val_top1": 0.6042,
      "sec": 220.6
    },
    {
      "epoch": 6,
      "avg_loss": 0.815934,
      "val_top1": 0.6005,
      "sec": 220.7
    },
    {
      "epoch": 7,
      "avg_loss": 0.792122,
      "val_top1": 0.5855,
      "sec": 220.5
    },
    {
      "epoch": 8,
      "avg_loss": 0.783254,
      "val_top1": 0.6024,
      "sec": 220.8
    }
  ],
  "n_train": 12712,
  "n_labels": 55
}
```

## 9. Validation/test results

```json
{
  "val": {
    "n": 1602,
    "top1": 0.6155,
    "top5": 0.927,
    "macro_recall": 0.6225,
    "macro_precision": 0.5419,
    "macro_f1": 0.5497,
    "ece": 0.1788,
    "reliability": [
      {
        "bin": 1,
        "n": 7,
        "acc": 0.1429,
        "mean_conf": 0.165
      },
      {
        "bin": 2,
        "n": 19,
        "acc": 0.2632,
        "mean_conf": 0.271
      },
      {
        "bin": 3,
        "n": 61,
        "acc": 0.377,
        "mean_conf": 0.3466
      },
      {
        "bin": 4,
        "n": 114,
        "acc": 0.3684,
        "mean_conf": 0.4568
      },
      {
        "bin": 5,
        "n": 147,
        "acc": 0.4014,
        "mean_conf": 0.5472
      },
      {
        "bin": 6,
        "n": 150,
        "acc": 0.4133,
        "mean_conf": 0.6499
      },
      {
        "bin": 7,
        "n": 171,
        "acc": 0.5789,
        "mean_conf": 0.7485
      },
      {
        "bin": 8,
        "n": 195,
        "acc": 0.5846,
        "mean_conf": 0.8522
      },
      {
        "bin": 9,
        "n": 738,
        "acc": 0.7873,
        "mean_conf": 0.9717
      }
    ],
    "per_class": {
      "0": {
        "precision": 0.75,
        "recall": 0.5357,
        "f1": 0.625,
        "support": 28
      },
      "1": {
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "support": 3
      },
      "2": {
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "support": 1
      },
      "3": {
        "precision": 0.48,
        "recall": 0.8571,
        "f1": 0.6154,
        "support": 14
      },
      "4": {
        "precision": 1.0,
        "recall": 0.5,
        "f1": 0.6667,
        "support": 2
      },
      "5": {
        "precision": 0.6757,
        "recall": 0.5682,
        "f1": 0.6173,
        "support": 44
      },
      "6": {
        "precision": 0.6,
        "recall": 0.6,
        "f1": 0.6,
        "support": 5
      },
      "7": {
        "precision": 0.3333,
        "recall": 1.0,
        "f1": 0.5,
        "support": 2
      },
      "8": {
        "precision": 0.7551,
        "recall": 0.8605,
        "f1": 0.8043,
        "support": 43
      },
      "9": {
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "support": 1
      },
      "10": {
        "precision": 0.8571,
        "recall": 0.5,
        "f1": 0.6316,
        "support": 12
      },
      "11": {
        "precision": 0.4444,
        "recall": 0.5,
        "f1": 0.4706,
        "support": 16
      },
      "12": {
        "precision": 0.8,
        "recall": 0.75,
        "f1": 0.7742,
        "support": 16
      },
      "13": {
        "precision": 0.8,
        "recall": 0.9091,
        "f1": 0.8511,
        "support": 22
      },
      "14": {
        "precision": 0.7097,
        "recall": 0.6111,
        "f1": 0.6567,
        "support": 36
      },
      "15": {
        "precision": 0.4,
        "recall": 0.3333,
        "f1": 0.3636,
        "support": 6
      },
      "16": {
        "precision": 0.5,
        "recall": 0.5,
        "f1": 0.5,
        "support": 6
      },
      "17": {
        "precision": 0.8421,
        "recall": 0.8889,
        "f1": 0.8649,
        "support": 18
      },
      "18": {
        "precision": 0.6299,
        "recall": 0.5026,
        "f1": 0.5591,
        "support": 193
      },
      "19": {
        "precision": 0.7654,
        "recall": 0.8267,
        "f1": 0.7949,
        "support": 75
      },
      "20": {
        "precision": 0.5928,
        "recall": 0.4323,
        "f1": 0.5,
        "support": 266
      },
      "21": {
        "precision": 0.7333,
        "recall": 0.7857,
        "f1": 0.7586,
        "support": 14
      },
      "22": {
        "precision": 0.75,
        "recall": 0.7818,
        "f1": 0.7656,
        "support": 307
      },
      "23": {
        "precision": 0.8286,
        "recall": 0.8056,
        "f1": 0.8169,
        "support": 36
      },
      "24": {
        "prec
```

## 10. Domain-gap results

- Food101 experiment expansion used as proxy domain-gap prepared-meal coverage on Kaggle.
- Local domain-gap remediation set remains experiment-only (645 images); not merged into production candidate.

## 11. 41-scene real-world results

- Challenger strict_complete_rate=0.2195
- Challenger blockers=['critical_category_23_hard_negative_food_pairs=FAIL', 'critical_category_21_duplicate_instances=FAIL', 'critical_category_12_recipe_document=FAIL', 'critical_category_02_multiple_raw_ingredients=FAIL', 'critical_category_13_non_food=FAIL', 'critical_category_11_cooked_vs_plated_vs_leftover=FAIL', 'critical_category_06_multiple_prepared_components=FAIL']

## 12. BEFORE vs AFTER comparison

- delta_strict=-0.0244
- improved categories=[]
- still failing=['01_single_raw_ingredient', '02_multiple_raw_ingredients', '04_ingredient_plus_packaged_food', '06_multiple_prepared_components', '07_packaged_food_scene', '08_fruit_collection', '09_snack_ready_to_eat_scene', '10_raw_vs_partially_prepared', '11_cooked_vs_plated_vs_leftover', '12_recipe_document', '13_non_food', '16_small_objects', '20_perspective_distortion', '21_duplicate_instances', '22_quantity_variation', '23_hard_negative_food_pairs']

## 13. Per-category comparison

| Category | Prod status | Prod mean completeness | Challenger status | Challenger mean completeness |
|---|---|---|---|---|
| 01_single_raw_ingredient | FAIL | 0.0 | FAIL | 0.0 |
| 02_multiple_raw_ingredients | FAIL | 0.0 | FAIL | 0.0 |
| 03_dense_kitchen_scene | PARTIAL | 0.5 | PARTIAL | 0.5 |
| 04_ingredient_plus_packaged_food | FAIL | 0.0 | FAIL | 0.0 |
| 05_ingredient_plus_prepared_meal | PARTIAL | 0.5 | PARTIAL | 0.5 |
| 06_multiple_prepared_components | FAIL | 0.3333 | FAIL | 0.3333 |
| 07_packaged_food_scene | FAIL | 0.0 | FAIL | 0.0 |
| 08_fruit_collection | FAIL | 0.0 | FAIL | 0.0 |
| 09_snack_ready_to_eat_scene | FAIL | 0.0 | FAIL | 0.0 |
| 10_raw_vs_partially_prepared | FAIL | 0.0 | FAIL | 0.0 |
| 11_cooked_vs_plated_vs_leftover | FAIL | 0.0 | FAIL | 0.0 |
| 12_recipe_document | FAIL | 0.0 | FAIL | 0.0 |
| 13_non_food | FAIL | 0.25 | FAIL | 0.0 |
| 14_occlusion | PASS | 1.0 | PASS | 1.0 |
| 15_overlap | PASS | 1.0 | PASS | 1.0 |
| 16_small_objects | FAIL | 0.0 | FAIL | 0.0 |
| 17_partial_visibility | PASS | 1.0 | PASS | 1.0 |
| 18_low_light | PASS | 1.0 | PASS | 1.0 |
| 19_motion_blur | PASS | 1.0 | PASS | 1.0 |
| 20_perspective_distortion | FAIL | 0.0 | FAIL | 0.0 |
| 21_duplicate_instances | FAIL | 0.0 | FAIL | 0.0 |
| 22_quantity_variation | FAIL | 0.0 | FAIL | 0.0 |
| 23_hard_negative_food_pairs | FAIL | 0.0 | FAIL | 0.0 |
| 24_mixed_food_and_nonfood | PARTIAL | 0.5 | PARTIAL | 0.5 |
| 25_complex_real_world_kitchen | PARTIAL | 0.5833 | PARTIAL | 0.5833 |

## 14. Failure attribution

```json
{
  "primary_stage_totals": {
    "CLASSIFIER": 41,
    "DETECTOR": 30,
    "FOOD_STATE": 17,
    "NON_FOOD_REJECTION": 3
  },
  "note": "Attributed from acceptance diagnosis stages on challenger run",
  "by_category_scenes": {
    "01_single_raw_ingredient": [
      {
        "scene_id": "01_tomato_pilot",
        "primary": "CLASSIFIER",
        "counts": {
          "CLASSIFIER": 1
        }
      }
    ],
    "02_multiple_raw_ingredients": [
      {
        "scene_id": "02_multi_raw_composite",
        "primary": "CLASSIFIER",
        "counts": {
          "CLASSIFIER": 4,
          "DETECTOR": 1
        }
      }
    ],
    "03_dense_kitchen_scene": [
      {
        "scene_id": "03_dense_kitchen",
        "primary": "DETECTOR",
        "counts": {
          "DETECTOR": 3,
          "CLASSIFIER": 1,
          "FOOD_STATE": 3
        }
      }
    ],
    "04_ingredient_plus_packaged_food": [
      {
        "scene_id": "04_ingredient_packaged",
        "primary": "DETECTOR",
        "counts": {
          "CLASSIFIER": 1,
          "DETECTOR": 2
        }
      }
    ],
    "05_ingredient_plus_prepared_meal": [
      {
        "scene_id": "05_ingredient_prepared",
        "primary": "CLASSIFIER",
        "counts": {
          "CLASSIFIER": 1,
          "FOOD_STATE": 1,
          "DETECTOR": 1
        }
      }
    ],
    "06_multiple_prepared_components": [
      {
        "scene_id": "06_multi_prepared",
        "primary": "DETECTOR",
        "counts": {
          "DETECTOR": 3,
          "FOOD_STATE": 1
        }
      }
    ],
    "07_packaged_food_scene": [
      {
        "scene_id": "07_packaged",
        "primary": "CLASSIFIER",
        "counts": {
          "CLASSIFIER": 1
        }
      }
    ],
    "08_fruit_collection": [
      {
        "scene_id": "08_fruit_proxy",
        "primary": "CLASSIFIER",
        "counts": {
          "CLASSIFIER": 2,
          "DETECTOR": 1
        }
      }
    ],
    "09_snack_ready_to_eat_scene": [
      {
        "scene_id": "09_snack_rte",
        "primary": "DETECTOR",
        "counts": {
          "CLASSIFIER": 1,
          "DETECTOR": 2
        }
      }
    ],
    "10_raw_vs_partially_prepared": [
      {
        "scene_id": "10_raw_vs_partial",
        "primary": "CLASSIFIER",
        "counts": {
          "CLASSIFIER": 2,
          "DETECTOR": 1
        }
      }
    ],
    "11_cooked_vs_plated_vs_leftover": [
      {
        "scene_id": "11_state_triad",
        "primary": "CLASSIFIER",
        "counts": {
          "CLASSIFIER": 3,
          "DETECTOR": 1
        }
      }
    ],
    "12_recipe_document": [
      {
        "scene_id": "12_recipe_document",
        "primary": "CLASSIFIER",
        "counts": {
          "CLASSIFIER": 1
        }
      },
      {
        "scene_id": "12_nutrition_label",
        "primary": "CLASSIFIER",
        "counts": {
          "CLASSIFIER": 1
        }
      }
    ],
    "13_non_food": [
      {
        "scene_id": "13_empty_plate",
        "primary": "CLASSIFIER",
        "counts": {
          "CLASSIFIER": 1
        }
      },
      {
        "scene_id": "13_human_hand",
        "primary": "CLASSIFIER",
        "counts": {
          "CLASSIFIER": 1,
          "NON_FOOD_REJECTION": 1
        }
      },
      {
        "scene_id": "13_pantry_empty",
        "primary": "CLASSIFIER",
        "counts": {
          "CLASSIFIER": 1,
          "NON_FOOD_REJECTION": 1
        }
      },
      {
        "scene_id": "13_kitchen_mood",
        "primary": "CLASSIFIER",
        "counts": {
          "CLASSIFIER": 1,
          "NON_FOOD_REJECTION": 1
        }
      }
    ],
    "14_occlusion": [
      {
        "scene_id": "14_occlusion",
        "primary": "FOOD_STATE",
        "counts": {
          "FOOD_STATE": 1
        }
      }
    ],
    "15_overlap": [
      {
        "scene_id": "15_overlap",
        "primary": "FOOD_STATE",
        "counts": {
          "FOOD_STATE": 1
        }
      }
    ],
    "16_small_objects": [
      {
        "
```

## 15. Calibration results

```json
{
  "method": "temperature_scaling_v1",
  "temperature": 1.165985107421875,
  "fit_on": "val_only",
  "holdout_used_for_fit": false,
  "val_ece_before_note": 0.1788,
  "reliability": [
    {
      "bin": 1,
      "n": 7,
      "acc": 0.1429,
      "mean_conf": 0.165
    },
    {
      "bin": 2,
      "n": 19,
      "acc": 0.2632,
      "mean_conf": 0.271
    },
    {
      "bin": 3,
      "n": 61,
      "acc": 0.377,
      "mean_conf": 0.3466
    },
    {
      "bin": 4,
      "n": 114,
      "acc": 0.3684,
      "mean_conf": 0.4568
    },
    {
      "bin": 5,
      "n": 147,
      "acc": 0.4014,
      "mean_conf": 0.5472
    },
    {
      "bin": 6,
      "n": 150,
      "acc": 0.4133,
      "mean_conf": 0.6499
    },
    {
      "bin": 7,
      "n": 171,
      "acc": 0.5789,
      "mean_conf": 0.7485
    },
    {
      "bin": 8,
      "n": 195,
      "acc": 0.5846,
      "mean_conf": 0.8522
    },
    {
      "bin": 9,
      "n": 738,
      "acc": 0.7873,
      "mean_conf": 0.9717
    }
  ]
}
```

## 16. Security/regression results

- No production registry promotion performed.
- Experiment-only Food101 data forbids production candidacy.
- Existing V2 unit/regression suites unchanged by this experiment (shadow eval only).

## 17. Model SHA-256

- heads/model.pt: `14aa744e00521554fe551828289442c48064c2e9ebd11ddb2cd8c635ae2a5ceb`
- full: `50becee0b14e1e2db2adbd9292b58b044a8d9e1ddfd781eef489f0aab2d92ca9`

## 18. Dataset manifest/hash

- `{'sha256': '04f6dc34d0a395e50ef99ec6876cffb4bd3e63bfa4037ff8a1834e27f7a76e79', 'bytes': 3226}`

## 19. Registry status

- Challenger **NOT** registered as PRODUCTION.
- Artifact staged under `fv-v2-challenger/artifacts/challenger/` for shadow eval only.

## 20. Holdout integrity confirmation

- holdout_untouched=YES
- holdout_sha256=`fed84f55d4a101a1280563a291229d8646eecc778562b2706ff692fb15b9134b`
- Not used for training, calibration, thresholding, or model selection.

## 21. Production promotion decision

```json
{
  "PRODUCTION_PROMOTION": false,
  "reasons": [
    "strict_complete_rate_delta=-0.0244",
    "food101_not_mounted_oi_only_challenger",
    "missing_critical_labels_garlic_ginger_rice_biryani_yogurt_cream",
    "auto_promote_forbidden_until_production_eligible_data_and_material_strict_gain"
  ],
  "gates": {
    "holdout_untouched": true,
    "material_strict_improvement": false,
    "production_eligible_data_only": true,
    "no_critical_regression_checked": true
  }
}
```

## 22. Remaining blockers

- Critical labels still missing from OI-62 + Food101 expand: garlic, ginger, rice, biryani, yogurt, cream (and others per audit)
- Strict complete-scene rate still insufficient for production
- Experiment-only Food101 provenance blocks production promotion
- High-confidence misclassification risk remains without threshold lowering

## 23. Exact recommended next engineering action

1. Acquire **production-eligible** images for missing critical labels (garlic, ginger, rice, biryani, yogurt/cream, true non-food plates/hands) under KitchenIQ licence policy.
2. Retrain a production-eligible challenger (no Food101) once those labels have adequate support.
3. Re-run the identical 41-scene acceptance suite; promote only if strict complete-scene rate improves materially with no critical regressions.
4. Install Tesseract binary so OCR channel becomes AVAILABLE (orthogonal to classifier).

---

VISION ENGINE V2 CHALLENGER TRAINING COMPLETE — PRODUCTION PROMOTION: FALSE
