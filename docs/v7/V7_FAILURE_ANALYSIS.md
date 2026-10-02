# V7 failure analysis (VAL; TEST not opened)

All specialists take the Adapter E embedding as input, train on TRAIN only (8 views per image: the original
plus 7 realistic augmentations), and use AdamW (lr 3e-4, wd 1e-4) for at most 20 epochs with patience 4. The
best epoch is chosen on VAL balanced accuracy. The MLP is kept only if it beats the linear baseline. The
selected margin is 0.00 for every pair.

| Pair | Train img | Val img | E errors | Recoverable | Model | Bal. acc | Acc. | Corrections | Regressions | Net |
|---|---|---|---|---|---|---|---|---|---|---|
| cream ↔ ice_cream | 30 / 30 | 4 / 10 | 4 | 4 | MLP | 0.775 | 0.786 | 1 | 0 | +1 |
| biryani ↔ fried_rice | 30 / 30 | 10 / 10 | 3 | 3 | MLP | 0.80 | 0.80 | 2 | 1 | +1 |
| cheese ↔ packaged_cheese | 30 / 18 | 5 / 4 | 3 | 3 | linear | 0.75 | 0.778 | 1 | 0 | +1 |
| caesar_salad ↔ salad | 30 / 30 | 10 / 10 | 2 | 2 | MLP | 1.00 | 1.00 | 2 | 0 | +2 |
| guacamole ↔ hummus | 30 / 30 | 10 / 10 | 2 | 2 | MLP | 0.85 | 0.85 | 1 | 0 | +1 |
| **Total** | | | 14 | 14 | | | | **7** | **1** | **+6** |

The control check covered only queries where both A and B were in E's top-5 and the ground truth was A or B:

| Pair | Active VAL queries | E top-1 there | Specialist acc. there |
|---|---|---|---|
| cream ↔ ice_cream | 33 (14 in pair) | 0.643 | 0.786 |
| biryani ↔ fried_rice | 43 (17) | 0.647 | 0.765 |
| cheese ↔ packaged_cheese | 15 (6) | 0.500 | 0.667 |
| caesar_salad ↔ salad | 40 (20) | 0.850 | 1.000 |
| guacamole ↔ hummus | 34 (17) | 0.824 | 0.882 |

## Ablation (margin 0; corrections / regressions / net)
| Pair | E only | E + linear | E + MLP |
|---|---|---|---|
| cream ↔ ice_cream | 0 | 1/0/+1 | 1/0/+1 |
| biryani ↔ fried_rice | 0 | 2/3/**−1** | 2/1/+1 |
| cheese ↔ packaged_cheese | 0 | 1/0/+1 | 0/0/0 |
| caesar_salad ↔ salad | 0 | 1/0/+1 | 2/0/+2 |
| guacamole ↔ hummus | 0 | 1/1/0 | 1/0/+1 |

Overall, the specialists recover 7 of the 14 recoverable errors in the selected pairs, and 7 of the 30
recoverable errors overall.

## Separability
- **Visually separable:** caesar_salad ↔ salad has balanced accuracy 1.00 and a learning curve flat at
  0.95–1.00. guacamole ↔ hummus reaches 0.85.
- **Fundamentally ambiguous at this data scale:**
  - biryani ↔ fried_rice: 0.75–0.80 with a noisy learning curve. Extrapolating to 0.90 balanced accuracy
    needs about 800 more images per class, which is not realistic.
  - cream ↔ ice_cream: the learning curve is flat at 0.775 from 24 to 60 images, so more of the same kind of
    image will not help.
- **Insufficient data:**
  - cheese ↔ packaged_cheese: only 18 packaged_cheese TRAIN images and 4 VAL images. The learning curve is
    flat at 0.75.
  - cream has only 4 VAL images, so its VAL estimate is unreliable.

## DATA_BOTTLENECK (no images downloaded; no money spent)
| Pair | Bottleneck | Additional images needed |
|---|---|---|
| biryani ↔ fried_rice | Visual ambiguity plus low diversity | About 800 per class by log-linear extrapolation to 0.90 bal. acc. Treat it as ambiguous rather than buy-able. |
| cream ↔ ice_cream | Flat curve: images lack distinguishing context (bowl vs cone, liquid vs scooped) | Extrapolation is not possible. Needs *different* images (containers, texture close-ups), about 30 per class, plus at least 10 more cream VAL images to measure the effect. |
| cheese ↔ packaged_cheese | Class imbalance (18 vs 30) and flat curve | At least 12 packaged_cheese TRAIN images to balance, plus wrapper/label views; at least 6 more VAL images per class. |
| guacamole ↔ hummus | Close to target | About 34 per class by extrapolation. |
| caesar_salad ↔ salad | None | 0 |

## Where the remaining limit is
- 12 of 42 E errors have the correct identity outside the top-5. No specialist can fix these, so the
  candidate-generation and reference-data limit stands.
- 16 recoverable errors fall on pairs that occur only once, so no specialist can be justified for them.
- Within the selected pairs, half of the recoverable errors (7 of 14) are still wrong.
