# Identity Failure Coverage Matrix

Generated: 2026-09-30T17:12:52.999027+00:00
Failure rows: 67

## Groups

- **B. model vocabulary missing**: 22
- **A. taxonomy missing**: 21
- **C. training data insufficient**: 12
- **D. detector failure**: 12

## Per-failure (compact)

| scene | expected | canonical | stage | taxonomy | model_vocab | prod_n | root |
|---|---|---|---|---|---|---|---|
| 01_tomato_pilot | tomato | tomato | CLASSIFIER | True | True | 80 | classifier_identity |
| 02_multi_raw_composite | tomato | tomato | CLASSIFIER | True | True | 80 | classifier_identity |
| 02_multi_raw_composite | garlic | garlic | CLASSIFIER | True | False | 72 | classifier_identity |
| 02_multi_raw_composite | ginger | ginger | CLASSIFIER | True | False | 66 | classifier_identity |
| 02_multi_raw_composite | rice | rice | CLASSIFIER | True | False | 80 | classifier_identity |
| 03_dense_kitchen | pizza | pizza | CLASSIFIER | True | True | 0 | training_data_insufficient |
| 03_dense_kitchen | fried_rice | cooked_rice | CLASSIFIER | True | False | 25 | model_vocab_missing |
| 03_dense_kitchen | caesar_salad | caesar_salad | DETECTOR | False | False | 0 | detector_failure |
| 03_dense_kitchen | french_fries | french_fries | CLASSIFIER | False | True | 0 | taxonomy_missing |
| 03_dense_kitchen | club_sandwich | sandwich | DETECTOR | False | True | 0 | detector_failure |
| 03_dense_kitchen | omelette | fried_egg | CLASSIFIER | True | False | 0 | model_vocab_missing |
| 04_ingredient_packaged | tomato | tomato | CLASSIFIER | True | True | 80 | classifier_identity |
| 04_ingredient_packaged | packaged_cheese | packaged_cheese | CLASSIFIER | True | False | 18 | classifier_identity |
| 05_ingredient_prepared | tomato | tomato | CLASSIFIER | True | True | 80 | classifier_identity |
| 05_ingredient_prepared | pizza | pizza | CLASSIFIER | True | True | 0 | training_data_insufficient |
| 06_multi_prepared | pizza | pizza | CLASSIFIER | True | True | 0 | training_data_insufficient |
| 06_multi_prepared | fried_rice | cooked_rice | DETECTOR | True | False | 25 | detector_failure |
| 06_multi_prepared | club_sandwich | sandwich | DETECTOR | False | True | 0 | detector_failure |
| 07_packaged | packaged_cheese | packaged_cheese | CLASSIFIER | True | False | 18 | classifier_identity |
| 08_fruit_proxy | guacamole | guacamole | CLASSIFIER | False | False | 0 | taxonomy_missing |
| 08_fruit_proxy | hummus | hummus | CLASSIFIER | False | False | 0 | taxonomy_missing |
| 09_snack_rte | donuts | doughnut | CLASSIFIER | False | True | 0 | taxonomy_missing |
| 09_snack_rte | ice_cream | ice_cream | DETECTOR | False | True | 0 | detector_failure |
| 10_raw_vs_partial | tomato | tomato | CLASSIFIER | True | True | 80 | classifier_identity |
| 10_raw_vs_partial | tomato_paste | tomato_paste | CLASSIFIER | True | False | 46 | classifier_identity |
| 11_state_triad | lasagna | lasagna | CLASSIFIER | False | False | 0 | taxonomy_missing |
| 11_state_triad | omelette | fried_egg | CLASSIFIER | True | False | 0 | model_vocab_missing |
| 11_state_triad | pad_thai | pad_thai | CLASSIFIER | False | False | 0 | taxonomy_missing |
| 12_recipe_document | recipe_document | recipe_document | CLASSIFIER | False | False | 0 | taxonomy_missing |
| 12_nutrition_label | recipe_document | recipe_document | CLASSIFIER | False | False | 0 | taxonomy_missing |
| 13_pantry_empty | non_food | non_food_object | CLASSIFIER | True | False | 0 | model_vocab_missing |
| 14_occlusion | salad | salad | CLASSIFIER | False | True | 0 | taxonomy_missing |
| 15_overlap | salad | salad | CLASSIFIER | False | True | 0 | taxonomy_missing |
| 16_small_objects | salad | salad | CLASSIFIER | False | True | 0 | taxonomy_missing |
| 17_partial_visibility | salad | salad | CLASSIFIER | False | True | 0 | taxonomy_missing |
| 18_low_light | salad | salad | CLASSIFIER | False | True | 0 | taxonomy_missing |
| 19_motion_blur | salad | salad | CLASSIFIER | False | True | 0 | taxonomy_missing |
| 20_perspective_distortion | salad | salad | CLASSIFIER | False | True | 0 | taxonomy_missing |
| 21_duplicate_tomatoes | tomato | tomato | CLASSIFIER | True | True | 80 | classifier_identity |
| 21_duplicate_tomatoes | tomato | tomato | CLASSIFIER | True | True | 80 | classifier_identity |
| 21_duplicate_tomatoes | tomato | tomato | CLASSIFIER | True | True | 80 | classifier_identity |
| 21_duplicate_tomatoes | tomato | tomato | CLASSIFIER | True | True | 80 | classifier_identity |
| 22_qty_one | tomato | tomato | CLASSIFIER | True | True | 80 | classifier_identity |
| 22_qty_four | tomato | tomato | CLASSIFIER | True | True | 80 | classifier_identity |
| 22_qty_four | tomato | tomato | CLASSIFIER | True | True | 80 | classifier_identity |
| 22_qty_four | tomato | tomato | CLASSIFIER | True | True | 80 | classifier_identity |
| 22_qty_four | tomato | tomato | CLASSIFIER | True | True | 80 | classifier_identity |
| 23_tomato_vs_tomato_paste | tomato | tomato | DETECTOR | True | True | 80 | detector_failure |
| 23_tomato_vs_tomato_paste | tomato_paste | tomato_paste | DETECTOR | True | False | 46 | detector_failure |
| 23_yogurt_vs_cream | yogurt | yogurt | DETECTOR | True | False | 70 | detector_failure |
| 23_yogurt_vs_cream | cream | cream | DETECTOR | True | False | 48 | detector_failure |
| 23_cheese_vs_packaged_cheese | cheese | cheese | DETECTOR | True | True | 50 | detector_failure |
| 23_cheese_vs_packaged_cheese | packaged_cheese | packaged_cheese | DETECTOR | True | False | 18 | detector_failure |
| 23_rice_vs_biryani | rice | rice | CLASSIFIER | True | False | 80 | classifier_identity |
| 23_rice_vs_biryani | biryani | biryani | DETECTOR | True | False | 40 | detector_failure |
| 23_fried_rice_not_raw | fried_rice | cooked_rice | CLASSIFIER | True | False | 25 | model_vocab_missing |
| 24_food_nonfood | pizza | pizza | CLASSIFIER | True | True | 0 | training_data_insufficient |
| 25_salad_bowl | salad | salad | CLASSIFIER | False | True | 0 | taxonomy_missing |
| 25_tomato_pasta | pasta | pasta | CLASSIFIER | True | True | 0 | training_data_insufficient |
| 25_complex_mixed | salad | salad | CLASSIFIER | False | True | 0 | taxonomy_missing |
| 25_complex_mixed | pasta | pasta | CLASSIFIER | True | True | 0 | training_data_insufficient |
| 25_complex_mixed | spring_rolls | spring_rolls | CLASSIFIER | False | False | 0 | taxonomy_missing |
| prepared_pizza | pizza | pizza | CLASSIFIER | True | True | 0 | training_data_insufficient |
| prepared_fried_rice | fried_rice | cooked_rice | CLASSIFIER | True | False | 25 | model_vocab_missing |
| prepared_bibimbap | bibimbap | bibimbap | CLASSIFIER | False | False | 0 | taxonomy_missing |
| prepared_lasagna | lasagna | lasagna | CLASSIFIER | False | False | 0 | taxonomy_missing |
| prepared_club_sandwich | club_sandwich | sandwich | CLASSIFIER | False | True | 0 | taxonomy_missing |
