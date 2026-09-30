# V3 Failure Matrix (41-scene @ 0.2927)

Generated: 2026-09-30T19:04:33.310200+00:00

## Substantial-improvement gate (FROZEN before V3 eval)

- min strict completeness: **0.439** (18/41)
- min delta vs 0.2927: **0.1463**
- max classifier failures: **20** (from 40)
- marginal band rejected: [0.3, 0.34]

Rationale: From 29 incomplete scenes and 40 CLASSIFIER failures: reject the brief's marginal band (≤0.34≈14/41). Require ≥18/41 (+6 scenes) and ≤20 classifier failures (≥50% reduction). Threshold frozen before any V3 candidate eval.

## Earliest causal stage counts

```json
{
  "food/non-food": 25,
  "open-set decision": 14,
  "detector": 12,
  "identity": 1
}
```

## Classifier subtype counts (A–J)

```json
{
  "C": 24,
  "B": 1,
  "A": 4,
  "D": 11
}
```

## Reported stage totals

```json
{
  "CLASSIFIER": 40,
  "DETECTOR": 12,
  "FOOD_NONFOOD": 2
}
```

Rows: 52. Full machine-readable: `docs/v3/v3_failure_matrix.json`.

## Implication for candidate spend

- Dominant earliest stages are **identity** / **food/non-food**, not detector-only.
- Therefore Candidate E (retrieval + specialists + evidence graph + open-set) is primary.
- Grounding DINO fallback is **optional** unless region-discovery rows dominate after E.
- Do NOT run another flat softmax classifier cycle.