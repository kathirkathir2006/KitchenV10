# 41-Scene Root-Cause Matrix (0.2927 replay analysis)

Generated: 2026-09-30T21:26:33.355721+00:00

Strict completeness: **0.2927** (12/41)
Substantial target (frozen): **0.439**

## Bottlenecks (by scenes blocked)

- PRIMARY: **OPEN_SET**
- SECONDARY: **FOOD_NONFOOD**
- TERTIARY: **DETECTOR**

```json
[
  {
    "stage": "OPEN_SET",
    "scenes_blocked": 10
  },
  {
    "stage": "FOOD_NONFOOD",
    "scenes_blocked": 9
  },
  {
    "stage": "DETECTOR",
    "scenes_blocked": 7
  },
  {
    "stage": "IDENTITY",
    "scenes_blocked": 3
  }
]
```

## Instance first-stage counts

```json
{
  "FOOD_NONFOOD": 17,
  "OPEN_SET": 20,
  "DETECTOR": 12,
  "IDENTITY": 3
}
```

## Acceptance-reported stages (for contrast)

```json
{
  "CLASSIFIER": 40,
  "DETECTOR": 12,
  "FOOD_NONFOOD": 2
}
```

Method: First-stage root cause prefers observation signals over matcher-reported CLASSIFIER. Tomato→non_food is FOOD_NONFOOD, not IDENTITY. DETECTOR kept when reported or no spatial support.

## Top missed identities (identity/open-set/prepared)

```json
[
  [
    "tomato",
    4
  ],
  [
    "fried_rice",
    3
  ],
  [
    "omelette",
    2
  ],
  [
    "lasagna",
    2
  ],
  [
    "non_food",
    2
  ],
  [
    "garlic",
    1
  ],
  [
    "ginger",
    1
  ],
  [
    "rice",
    1
  ],
  [
    "packaged_cheese",
    1
  ],
  [
    "guacamole",
    1
  ],
  [
    "hummus",
    1
  ],
  [
    "donuts",
    1
  ],
  [
    "pad_thai",
    1
  ],
  [
    "spring_rolls",
    1
  ],
  [
    "bibimbap",
    1
  ]
]
```