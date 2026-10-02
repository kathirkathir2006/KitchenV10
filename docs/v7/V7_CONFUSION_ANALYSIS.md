# V7 confusion analysis (VAL only; TEST not opened)

Source: frozen Adapter E on VAL, using the unchanged V5 scorer. The counts match the V6 failure analysis
(12 errors absent from the top-5). Pairs are unordered (A ↔ B) and sorted by recoverable errors.

| | Value |
|---|---|
| Known VAL queries | 232 |
| E correct (top-1) | 190 (0.819) |
| E errors | 42 |
| Recoverable (correct identity in E top-5) | 30 |
| Non-recoverable | 12 |
| **Theoretical V7 ceiling** | 220 / 232 = **0.948** |
| Required for success (+0.02) | ≥ 5 net queries |

| Pair | Errors | Recoverable | Non-recoverable | Mean E score gap |
|---|---|---|---|---|
| cream ↔ ice_cream | 4 | 4 | 0 | 0.021 |
| biryani ↔ fried_rice | 3 | 3 | 0 | 0.006 |
| cheese ↔ packaged_cheese | 3 | 3 | 0 | 0.130 |
| caesar_salad ↔ salad | 2 | 2 | 0 | 0.018 |
| guacamole ↔ hummus | 2 | 2 | 0 | 0.013 |
| biryani ↔ tomato_paste | 2 | 1 | 1 | 0.051 |
| 15 other pairs | 1 each | 1 each | 0 | — |
| 10 remaining pairs | 11 | 0 | 11 | — |

There are 31 confusion pairs in total. 16 recoverable errors fall on pairs with only one recoverable
error each (including biryani ↔ tomato_paste), so no specialist can be justified for them. Full per-error and per-pair data: `confusion_analysis.json`.
