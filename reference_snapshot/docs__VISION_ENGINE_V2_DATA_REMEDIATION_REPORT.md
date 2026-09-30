# Vision Engine V2 — Production-Eligible Data Remediation Report

**Generated:** 2026-09-29T12:43:08Z
**Scope:** Data acquisition + audit only. NO training. NO production/holdout/taxonomy changes.

## Answers

1. **Valid production-eligible before (staging reuse estimate):** 227
2. **Valid production-eligible images added (new fetches):** food=225, hardneg=0 (plus reused staging food=85, hardneg=142)
3. **Rejected:** 108 — top reasons: {'wrong_semantics_reject_map': 64, 'zone_EXPERIMENT_ONLY:EVALUATION_ONLY': 36, 'no_results': 3, 'download_failed:URLError:<urlopen error [SSL: CERTIFICATE_VERIFY_FAILED] certifi': 3, 'license_rejected:cc_by_sa': 2}
4. **Final production-eligible per food target:** {'yogurt': 35, 'tomato': 40, 'rice': 40, 'garlic': 36, 'cream': 24, 'ginger': 33, 'packaged_cheese': 9, 'tomato_paste': 23, 'biryani': 20, 'fried_rice': 25, 'cheese': 25}
5. **Final hard-negative count:** 142 ({'foodlike_nonfood': 43, 'kitchen_surface': 46, 'empty_plate': 15, 'utensil': 15, 'human_hand': 15, 'empty_packaging': 8})
6. **Critical classes still zero-support:** []
7. **Below useful support (<20):** ['packaged_cheese']
8. **Sufficient (>= 40):** ['rice', 'tomato']
9. **Licence/provenance:** {'cc_by': 426, 'cc0': 26} / sources {'openverse': 323, 'open_images': 128, 'wikimedia_commons': 1}
10. **Duplicate rate:** 0.0044
11. **Real vs synthetic:** real=452 synthetic=0
12. **Experiment-only isolated:** 36 (separate manifest; not mixed into production-eligible)
13. **One controlled retraining cycle justified?** **YES** — Production-eligible package covers key absent identities at useful support (garlic/ginger/rice/yogurt >= 20), hard-negatives n=142 (>=40), and partial prepared/packaged coverage (biryani/fried_rice + tomato_paste/packaged_cheese). Controlled retrain justified.

## Policy notes

- Openverse CC0/CC-BY commercial filter with per-image provenance.
- Wikimedia used only for residual gaps; SA/NC/ND rejected.
- No bulk Open Images / Food101 download to Windows.
- Met Museum CC0 kept EXPERIMENT_ONLY (domain unsuitable).
- Hard negatives labelled **NONFOOD** only.

## Class support detail

```json
{
  "garlic": {
    "production_eligible_count": 36,
    "experiment_only_count": 0,
    "challenger_train_count_context": 0,
    "before_staging_audit": {
      "staging_candidates": 11,
      "production_eligible": 11
    },
    "sufficient": false,
    "useful": true,
    "zero_support": false,
    "sources": [
      "openverse"
    ],
    "real_world": true,
    "synthetic": 0
  },
  "ginger": {
    "production_eligible_count": 33,
    "experiment_only_count": 0,
    "challenger_train_count_context": 0,
    "before_staging_audit": {
      "staging_candidates": 8,
      "production_eligible": 8
    },
    "sufficient": false,
    "useful": true,
    "zero_support": false,
    "sources": [
      "openverse"
    ],
    "real_world": true,
    "synthetic": 0
  },
  "rice": {
    "production_eligible_count": 40,
    "experiment_only_count": 0,
    "challenger_train_count_context": 0,
    "before_staging_audit": {
      "staging_candidates": 24,
      "production_eligible": 24
    },
    "sufficient": true,
    "useful": true,
    "zero_support": false,
    "sources": [
      "openverse"
    ],
    "real_world": true,
    "synthetic": 0
  },
  "biryani": {
    "production_eligible_count": 20,
    "experiment_only_count": 0,
    "challenger_train_count_context": 0,
    "before_staging_audit": {},
    "sufficient": false,
    "useful": true,
    "zero_support": false,
    "sources": [
      "openverse"
    ],
    "real_world": true,
    "synthetic": 0
  },
  "yogurt": {
    "production_eligible_count": 35,
    "experiment_only_count": 0,
    "challenger_train_count_context": 0,
    "before_staging_audit": {
      "staging_candidates": 10,
      "production_eligible": 10
    },
    "sufficient": false,
    "useful": true,
    "zero_support": false,
    "sources": [
      "openverse"
    ],
    "real_world": true,
    "synthetic": 0
  },
  "cream": {
    "production_eligible_count": 24,
    "experiment_only_count": 0,
    "challenger_train_count_context": 0,
    "before_staging_audit": {
      "staging_candidates": 4,
      "production_eligible": 4
    },
    "sufficient": false,
    "useful": true,
    "zero_support": false,
    "sources": [
      "openverse"
    ],
    "real_world": true,
    "synthetic": 0
  },
  "tomato": {
    "production_eligible_count": 40,
    "experiment_only_count": 0,
    "challenger_train_count_context": 350,
    "before_staging_audit": {
      "staging_candidates": 22,
      "production_eligible": 22
    },
    "sufficient": true,
    "useful": true,
    "zero_support": false,
    "sources": [
      "open_images",
      "openverse"
    ],
    "real_world": true,
    "synthetic": 0
  },
  "tomato_paste": {
    "production_eligible_count": 23,
    "experiment_only_count": 0,
    "challenger_train_count_context": 0,
    "before_staging_audit": {
      "staging_candidates": 3,
      "production_eligible": 3
    },
    "sufficient": false,
    "useful": true,
    "zero_support": false,
    "sources": [
      "openverse"
    ],
    "real_world": true,
    "synthetic": 0
  },
  "fried_rice": {
    "production_eligible_count": 25,
    "experiment_only_count": 0,
    "challenger_train_count_context": 0,
    "before_staging_audit": {},
    "sufficient": false,
    "useful": true,
    "zero_support": false,
    "sources": [
      "openverse"
    ],
    "real_world": true,
    "synthetic": 0
  },
  "cheese": {
    "production_eligible_count": 25,
    "experiment_only_count": 0,
    "challenger_train_count_context": 250,
    "before_staging_audit": {},
    "sufficient": false,
    "useful": true,
    "zero_support": false,
    "sources": [
      "openverse"
    ],
    "real_world": true,
    "synthetic": 0
  },
  "packaged_cheese": {
    "production_eligible_count": 9,
    "experiment_only_count": 0,
    "challenger_train_count_context": 0,
    "before_staging_audit": {
      "staging_candidates": 3,
      "production_eligible": 3
    },
    "sufficient": false,
    "useful": false,
    "zero_support": false,
    "sources": [
      "openverse",
      "wikimedia_commons"
    ],
    "real_world": true,
    "synthetic": 0
  },
  "NONFOOD::empty_plate": {
    "production_eligible_count": 15,
    "label": "NONFOOD",
    "hardneg_category": "empty_plate",
    "sufficient_category": true,
    "sources": [
      "open_images"
    ]
  },
  "NONFOOD::cookware": {
    "production_eligible_count": 0,
    "label": "NONFOOD",
    "hardneg_category": "cookware",
    "sufficient_category": false,
    "sources": []
  },
  "NONFOOD::utensil": {
    "production_eligible_count": 15,
    "label": "NONFOOD",
    "hardneg_category": "utensil",
    "sufficient_category": true,
    "sources": [
      "open_images"
    ]
  },
  "NONFOOD::kitchen_surface": {
    "production_eligible_count": 46,
    "label": "NONFOOD",
    "hardneg_category": "kitchen_surface",
    "sufficient_category": true,
    "sources": [
      "open_images",
      "openverse"
    ]
  },
  "NONFOOD::human_hand": {
    "production_eligible_count": 15,
    "label": "NONFOOD",
    "hardneg_category": "human_hand",
    "sufficient_category": true,
    "sources": [
      "open_images"
    ]
  },
  "NONFOOD::empty_packaging": {
    "production_eligible_count": 8,
    "label": "NONFOOD",
    "hardneg_category": "empty_packaging",
    "sufficient_category": true,
    "sources": [
      "openverse"
    ]
  },
  "NONFOOD::foodlike_nonfood": {
    "production_eligible_count": 43,
    "label": "NONFOOD",
    "hardneg_category": "foodlike_nonfood",
    "sufficient_category": true,
    "sources": [
      "open_images",
      "openverse"
    ]
  },
  "NONFOOD::kitchen_mood_nonfood": {
    "production_eligible_count": 0,
    "label": "NONFOOD",
    "hardneg_category": "kitchen_mood_nonfood",
    "sufficient_category": false,
    "sources": []
  }
}
```


## Supplemental hard-negative pass

- cookware=12
- kitchen_mood_nonfood=12
- Final hard-negative total: 166
- Retrain justified: True
- Reason: Controlled retrain justified for absent-class + hard-negative remediation: critical foods at useful support, hardneg n=166 (kitchen_mood_nonfood=12, cookware=12), prepared/packaged coverage present. packaged_cheese still below useful (9<20) but non-zero.
