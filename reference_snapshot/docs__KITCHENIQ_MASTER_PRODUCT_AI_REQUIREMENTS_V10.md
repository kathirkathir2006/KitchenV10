# KitchenIQ Master Product, AI, Vision & Dataset Requirements

| Field | Value |
|---|---|
| **VERSION** | **10.0** |
| **STATUS** | **AUTHORITATIVE** |
| **PURPOSE** | **PRODUCT + AI + VISION + DATASET + EVALUATION MASTER REQUIREMENTS** |
| **Document type** | Self-contained product / AI / vision / dataset / evaluation master |
| **Precedence** | Where an older document conflicts with V10, **V10 takes precedence** |
| **Preserved foundation** | Valid V9.4 and V9.5 requirements remain in force unless explicitly superseded below |
| **Scope of this update** | Document only — no code, models, datasets, taxonomy, APIs, thresholds, or holdout changes |

---

## Document authority

This document is the authoritative KitchenIQ product/AI requirements specification for Version 10.0.

It establishes one clear product objective, AI/vision objective, dataset strategy, evaluation strategy, and development order so that future engineering work is driven by **product value** rather than isolated classifier accuracy or arbitrary image-count targets.

V10 preserves (and does not silently remove):

- Security and safety requirements (V9.4)
- Uncertainty, provenance, registry, and holdout protections
- KitchenState singularity
- Cook Engine requirements
- Food Trajectory, Entity Lineage, Entity Reconciliation
- Kitchen Twin (simulation-only)
- Use First Autopilot
- Learning-loop and controlled promotion
- API contracts already established in V9.4 / V9.5

V10 **supersedes** any prior statement that defines dataset success as reaching an arbitrary total such as 8,000 / 10,000 / 20,000 images.

---

## 1. KitchenIQ product objective

> **KitchenIQ is a predictive household food intelligence system that converts uncertain observations of food, products, meals and household behaviour into a persistent, uncertainty-aware food state, predicts likely future food trajectories, and recommends safe, realistic actions that reduce food waste, unnecessary shopping and decision effort.**

This is the **single authoritative V10 product objective**.

### Clarification

KitchenIQ is **NOT** primarily an AI food scanner.

The camera / vision system is an **observation and evidence-generation layer**.

Product value comes from:

```text
OBSERVE
→ UNDERSTAND
→ MAINTAIN FOOD STATE
→ PREDICT
→ DECIDE
→ ACT
→ LEARN
```

---

## 2. The product problem

Households do not merely need to know “what food is in this image”.

They need to know:

- What food do I actually have?
- Is it raw, prepared, opened, cooked, leftover, or otherwise changed?
- How much do I probably have?
- Is this the same food I saw earlier?
- What is likely to happen to it next?
- What should I use first?
- What should I avoid buying again?
- What can I cook with what already exists?
- What needs confirmation?
- What should change when household circumstances change?

Therefore the system must turn **imperfect observations** into a trustworthy household food state, predict likely future states, and recommend the **smallest useful action**.

---

## 3. Product architecture

Authoritative product architecture:

```text
USER / HOUSEHOLD CONTEXT
        ↓
SECURE OBSERVATION INPUT
        ↓
VISION / PERCEPTION / EVIDENCE
        ↓
FOOD UNDERSTANDING
        ↓
PERSISTENT FOOD ENTITIES
        ↓
KITCHEN STATE
        ↓
FOOD TRAJECTORY ENGINE
        ↓
KITCHEN TWIN / FUTURE STATE SIMULATION
        ↓
USE-FIRST / DECISION ENGINE
        ↓
ACTION / RECOMMENDATION
        ↓
OUTCOME
        ↓
LEARNING / EVALUATION
```

### Authority rules

| Component | Authority |
|---|---|
| **VISION** | Observation authority only |
| **FOOD ENTITY** | Persistent identity authority |
| **KITCHEN STATE** | Current operational household-food-state authority |
| **TRAJECTORY** | Future-state prediction authority |
| **KITCHEN TWIN** | Simulation authority only — **NOT** a second KitchenState |
| **DECISION ENGINE** | Action recommendation authority |
| **POLICY / SAFETY** | Deterministic safety and policy authority |
| **LEARNING** | Improvement authority only after validation, quarantine, evaluation, and promotion controls |

Vision observations do **not** directly own KitchenState.

---

## 4. Vision Engine objective

> **The Vision Engine must produce reliable, uncertainty-aware observations and evidence that allow the downstream KitchenIQ system to maintain household food state.**

It must **NOT** be described as: “recognise every food perfectly.”

Where evidence permits, vision must answer:

1. Is there food?
2. What is the food / non-food category?
3. What food family is it?
4. What specific identity is supported?
5. Is it raw, partially prepared, cooked, plated, leftover, packaged / ready-to-eat, etc.?
6. What packaging state is visible?
7. What quantity / instance evidence exists?
8. Is OCR available?
9. Is a barcode available?
10. What evidence supports the result?
11. How confident is each attribute?
12. What remains unknown?
13. Does the system need user confirmation?

If evidence does not support an answer, the system must **abstain** or return uncertainty.

---

## 5. Vision architecture

Preserve the established V9.4 / V9.5 principle:

> Vision should be implemented as **specialist perception capabilities** rather than one giant closed-set classifier.

### Authoritative conceptual pipeline

```text
SECURE INPUT
→ IMAGE QUALITY
→ SCENE / REGION DISCOVERY
→ FOOD / NON-FOOD GATE
→ OBJECT / INSTANCE DETECTION
→ FOOD FAMILY
→ HIERARCHICAL IDENTITY
→ PREPARED MEAL RECOGNITION
→ FOOD STATE
→ PACKAGING STATE
→ OCR
→ BARCODE
→ QUANTITY / INSTANCE ESTIMATION
→ EVIDENCE FUSION
→ CONFIDENCE / ABSTENTION
→ CANONICALISATION HINTS
→ ENTITY-MATCH HINTS
→ VISUAL OBSERVATION
```

Do **not** make the classifier the sole source of truth.

### Preserved vision decisions and controls

- **DINOv2 ViT-B/14** architecture decision unless future measured evidence justifies change
- **Faster R-CNN** detector currently used
- Evidence fusion
- Confidence bands
- Hierarchical stopping
- Prepared-meal protection
- Open-world outcomes
- Fail-closed behaviour
- Secure input boundary
- Registry and promotion controls
- Holdout protection

---

## 6. Prepared meals

A prepared meal is a **CanonicalMeal / prepared-meal observation**.

Do **NOT** decompose a prepared meal into raw ingredients unless those raw ingredients are independently visible.

### Correct

```text
Visible plated biryani
→ Prepared Meal
→ Biryani
→ regional candidates if evidence supports them
→ confidence
→ Canonical Meal
```

### Incorrect

```text
Biryani
→ rice + onion + chicken
→ treat these as independent raw inventory items
```

For mixed scenes, independently visible components may each become separate observations.

---

## 7. Household Food Entity

**Food Entity** is persistent identity authority.

A food entity may progress through:

```text
PURCHASED
→ STORED
→ OPENED
→ PORTIONED
→ PREPARED
→ COOKED
→ LEFTOVER
→ CONSUMED
or
→ DISCARDED
```

The system must prevent:

- duplicate entities
- accidental quantity inflation
- raw / prepared duplication
- silent replacement
- silent deletion
- incorrect merges
- incorrect splits

Vision observations do not directly own KitchenState. Entity reconciliation, lineage, and KitchenState update rules remain as established in V9.5 unless superseded by a later authoritative change.

---

## 8. Food Trajectory Engine

Preserve V9.5 Food Trajectory.

KitchenIQ must eventually predict likely future states including:

- consumption
- waste
- leftovers
- freezing
- duplicate shopping
- missing ingredients
- use-first urgency
- likely over-purchase
- household absence
- meal opportunity

Predictions must be evaluated against actual outcomes.

**Prediction is NOT safety authority.**

**Unknown does not mean safe.**

---

## 9. Kitchen Twin

Kitchen Twin is a **simulation representation** of household food state.

It must support future scenarios such as:

- `USE_FIRST`
- `COOK_NOW`
- `COOK_LATER`
- `FREEZE`
- `DO_NOT_BUY`
- `SUBSTITUTE`
- `SKIP_MEAL`
- `HOUSEHOLD_AWAY`

Kitchen Twin must **never** become a second authoritative KitchenState.

---

## 10. Use First Autopilot

**Use First Autopilot** is a flagship product capability.

It should continuously determine which food is most important to use next based on evidence-supported factors such as:

- expected deterioration
- current food state
- quantity
- household consumption behaviour
- expiry information
- previous missed opportunities
- meal feasibility
- household preferences
- dietary constraints
- waste risk
- available alternatives

It should explain the reason for the recommendation.

Example:

> “Use the opened spinach first because it is already opened, likely to deteriorate sooner, and there is a suitable dinner option tonight.”

Do **not** make unsupported claims about exact spoilage.

---

## 11. Decision Engine

The decision engine must not be generic recipe spam.

### Possible actions

- `USE_FIRST`
- `COOK`
- `COMBINE`
- `FREEZE`
- `CHECK`
- `CONFIRM`
- `ADD_TO_PLAN`
- `DO_NOT_BUY`
- `SUBSTITUTE`
- `NO_ACTION`

The system should prefer the **smallest useful action**.

### Suppress recommendations when

- identity is unsupported
- safety-critical information is unknown
- allergen status is unknown
- confidence is insufficient
- the recommendation would create unnecessary work
- required ingredients are unavailable
- the underlying household state is unreliable

Cook Engine feasibility, safety gates, and deterministic Policy Engine authority remain preserved from V9.4 / V9.5.

---

## 12. Dataset strategy — V10 rule (superseding)

### THIS SECTION SUPERSEDES ANY PREVIOUS ARBITRARY 8,000 / 10,000 IMAGE TARGET

KitchenIQ must **NOT** define success as reaching a total number such as:

- 8,000 images
- 10,000 images
- 20,000 images

There is **no universal total-image requirement**.

The dataset must instead be driven by:

- capability coverage
- label coverage
- diversity
- failure modes
- downstream product value

---

## 13. Initial dataset target: 250 per core label

### FINAL LOCK — V10 DATASET PRINCIPLE

> **250 diverse, validated examples per core label is the initial planning target. It is not a mandatory total-image quota, and it must never be satisfied with redundant, low-quality or legally unsuitable data.**

> **Coverage quality, diversity, licence validity, state variation, scene complexity and downstream usefulness matter more than total image count.**

### Rule statement

**INITIAL CORE DATA TARGET = 250 DIVERSE, VALIDATED EXAMPLES PER CORE LABEL.**

- 250 is the **initial planning target for core labels**.
- It is **NOT** a requirement to manufacture data to reach exactly 250.
- If a label has 230 high-quality, legally usable, diverse examples, do **NOT** add 20 poor-quality or redundant images simply to reach 250.
- If a class has substantially more high-quality data from an appropriate free dataset, it **may exceed** 250.
- The target applies to **meaningful examples**, not files alone.
- 250 is **NOT** a universal requirement for every capability.

---

## 14. What “250” means

The 250 examples must represent useful variation.

Where applicable, variation should include:

- viewpoint
- scale
- lighting
- background
- clutter
- occlusion
- partial visibility
- perspective
- presentation
- packaging variation
- preparation variation
- camera variation
- real-world context

Do **not** count near-identical duplicates as meaningful coverage.

A dataset with 250 near-identical images is **NOT** considered equivalent to 250 diverse examples.

---

## 15. Dataset targets by capability

Do **NOT** force every capability into the same flat 250-image rule.

| Capability | Target approach |
|---|---|
| **CORE FOOD LABELS** | 250 diverse examples per core label (initial planning target) |
| **FOOD STATES** | Meaningful examples for each required state and important state transition — do not simply duplicate the same food photograph |
| **PACKAGED PRODUCTS** | Leverage legally usable product datasets and product metadata rather than manually collecting 250 images for every individual SKU |
| **PREPARED MEALS** | Prioritise important meal/dish families and regional variants where evidence supports the hierarchy |
| **NON-FOOD / HARD NEGATIVES** | Broad, diverse coverage — false positives are a critical KitchenIQ failure mode |
| **OCR** | Evaluate on text-bearing food / package scenes |
| **BARCODE** | Evaluate detection and decoding separately |
| **QUANTITY / INSTANCE COUNT** | Require multi-instance and occlusion examples |
| **REAL-WORLD SCENES** | Prioritise multi-object, cluttered, mixed food / non-food scenes |
| **HOUSEHOLD STATE / TRAJECTORY** | Not solved by more isolated images — requires temporal / event evidence and repeated observations |

---

## 16. Data source policy

- No paid datasets
- No paid APIs for the core training strategy
- No manual image-capture programme as a prerequisite
- No bulk Windows downloads
- Bulk acquisition / training / evaluation must happen through **Kaggle** or appropriate cloud execution when required

Potential free sources may include, **subject to individual licence verification**:

- Open Food Facts
- Open Images
- Openverse
- Wikimedia Commons
- USDA / government sources where rights are verified
- FoodSeg103
- UECFoodPix
- Food-101
- Food2K
- VireoFood-172
- Cooking State Recognition
- suitable public barcode datasets

Do **NOT** assume “free download” means commercial-training permission.

Every image / dataset must carry provenance and licensing information.

---

## 17. Four data zones

Preserve:

1. **UNTRUSTED**
2. **OPERATIONAL**
3. **LEARNING QUARANTINE**
4. **CURATED TRAINING**

Only **CURATED TRAINING** may train production models.

Research / non-commercial datasets may be used for:

- experiments
- benchmarking
- architecture comparison
- research evaluation

They must remain clearly isolated from production training unless their licensing permits production use.

---

## 18. Data provenance

Every production-training image must have, where available:

- dataset / source
- source URL
- image ID
- image hash
- licence
- commercial-use status
- modification status
- redistribution status
- attribution requirement
- share-alike requirement
- acquisition date
- class
- annotation source
- data zone
- validation status

Rejected or legally uncertain images must **not** enter CURATED TRAINING.

---

## 19. Data quality

Reject or quarantine:

- corrupt files
- invalid images
- wrong semantics
- wrong labels
- duplicates
- near-duplicates where leakage is possible
- unusably tiny images
- synthetic images where production evidence is required
- unclear licensing
- non-commercial-only data intended for production
- misleading annotations

Do **not** optimise for raw image count.

---

## 20. Experiment-only data

Research datasets may be used to benchmark approaches.

Examples include datasets whose terms are research / non-commercial.

They must be explicitly marked:

```text
EXPERIMENT_ONLY
```

They must **NOT** silently become production-training data.

Reports must always distinguish:

- **EXPERIMENT DATA**
- from
- **PRODUCTION-ELIGIBLE DATA**

---

## 21. Dataset coverage matrix

V10 requires a dataset coverage matrix containing, label-by-label:

- frozen KitchenIQ label
- food / non-food type
- capability
- target = 250 where applicable (core labels)
- current valid production images
- current experiment-only images
- additional valid images available from free sources
- remaining gap
- state coverage
- scene / context coverage
- licence status
- duplicate risk
- annotation quality
- production eligibility

Do **not** claim that the dataset is complete simply because the total image count is large.

---

## 22. Frozen 207-label taxonomy

Preserve the frozen taxonomy:

**207 labels:**

- 108 raw
- 72 packaged
- 17 prepared
- 10 non-food

Do **NOT** expand, collapse, or rename the taxonomy as part of V10.

Hierarchy may be used to provide broader recognition when exact label evidence is insufficient.

The system must stop at the deepest evidence-supported level.

---

## 23. Evaluation strategy

The primary metric must **NOT** be isolated classifier Top-1 accuracy.

KitchenIQ must evaluate:

### LEVEL 1 — MODEL

- classification
- detection
- segmentation
- calibration
- abstention

### LEVEL 2 — VISION

- food / non-food
- identity
- state
- OCR
- barcode
- quantity
- mixed scenes

### LEVEL 3 — FOOD ENTITY

- duplicate prevention
- entity continuity
- merge / split correctness

### LEVEL 4 — KITCHEN STATE

- state accuracy
- quantity changes
- raw / prepared transitions
- destructive update protection

### LEVEL 5 — TRAJECTORY

- prediction accuracy
- waste-risk prediction
- consumption prediction

### LEVEL 6 — DECISION

- Use First usefulness
- feasibility
- recommendation acceptance
- duplicate-shopping prevention

### LEVEL 7 — OUTCOME

- reduced waste
- fewer unnecessary purchases
- reduced manual inventory work
- useful household decision support

---

## 24. Real-world acceptance

Preserve the **41-scene real-world acceptance suite**.

The current strict scene-completeness result of approximately **24.39%** remains a **major release blocker**.

A model must **not** be promoted merely because:

- Top-1 accuracy increased
- validation accuracy increased
- benchmark accuracy increased

If real-world scene completeness, state understanding, or downstream product utility gets worse, the model must **not** be promoted.

**Strict scene completeness remains an authoritative release metric.**

**Top-1 accuracy is not the primary product release criterion.**

---

## 25. Current Vision status

Vision Engine V2 is **implemented** but production status remains **BLOCKED**.

Known major issues include:

- classifier identity failures
- multi-food scene failures
- food-state failures
- hard-negative failures
- non-food false positives
- duplicate instances
- quantity variation
- prepared-meal recognition
- recipe-document handling
- small-object recognition
- perspective variation
- OCR runtime dependency availability
- insufficient label / domain coverage

Do **not** describe V2 as production-ready merely because the implementation is complete.

---

## 26. Established baseline policy

Before another broad retraining cycle, KitchenIQ must benchmark established specialist approaches against the same acceptance criteria.

The objective is to answer:

> “Are we training the wrong architecture / data strategy?”

rather than automatically asking:

> “How do we improve the current classifier?”

Candidate approaches should be evaluated by capability:

- object detection
- segmentation
- food recognition
- grocery / product recognition
- prepared-meal recognition
- food-state recognition
- OCR
- barcode
- quantity / instance detection
- evidence fusion

Use established open / free approaches where feasible.

Do **not** copy proprietary competitor implementations.

The benchmark must determine whether KitchenIQ should:

- **A.** retain the current architecture
- **B.** add specialist models
- **C.** replace a weak specialist
- **D.** use an ensemble
- **E.** change the data strategy
- **F.** defer retraining until another dependency is fixed

---

## 27. Training decision rule

Do **NOT** start another training cycle automatically.

A new training cycle requires evidence that:

1. the relevant data gap is understood
2. the data is production-eligible where required
3. the failure mode is addressable through training
4. the architecture is appropriate
5. the acceptance metric likely to improve is defined
6. regression risk is measurable
7. the holdout remains untouched
8. the resulting model can be evaluated on the same real-world acceptance suite

If these conditions are not met, **do not train**.

Training is **evidence-gated**, not automatic.

---

## 28. Security and safety

Preserve all V9.4 security requirements:

- Secure Input Boundary
- fail-closed behaviour
- authentication / authorisation
- resource limits
- malformed input handling
- SSRF protection
- sandbox parsing
- model output validation
- safety validation
- deterministic Policy Engine
- quarantine
- poisoning detection
- registry
- provenance
- integrity / hash verification
- rollback
- audit trail
- adversarial evaluation

### ML must NEVER

- decide allergy safety
- override safety gates
- mark an unknown allergen as safe
- invent nutrition
- directly mutate KitchenState
- bypass deterministic policy
- silently create destructive state changes

---

## 29. Learning loop

Preserve:

```text
USER CORRECTION
→ VALIDATION
→ ABUSE / POISONING DETECTION
→ QUARANTINE
→ DATASET QUALITY ASSESSMENT
→ TRAINING
→ SECURITY EVALUATION
→ CHALLENGER MODEL
→ SHADOW TEST
→ CONTROLLED PROMOTION
```

Corrections must never directly modify production weights.

---

## 30. Development order

Authoritative order of work:

1. Product objective
2. Domain / state model
3. Vision output contract
4. Acceptance set
5. Failure analysis
6. Dataset coverage audit
7. Established baseline benchmark
8. Architecture decision
9. Highest-impact failure remediation
10. Vision → Food Entity
11. Food Entity → Kitchen State
12. Kitchen State → Trajectory
13. Trajectory → Decision
14. Decision → Outcome
15. Prediction-error learning
16. Broad retraining only when justified by evidence

---

## 31. What we are not doing

KitchenIQ is **NOT**:

- a generic food classifier
- a recipe generator with a camera
- an image-recognition demo
- a giant flat classification problem
- a product that depends on perfect recognition
- a product that requires an arbitrary large KitchenIQ-specific image total before development can proceed
- a system that blindly trains on every free dataset
- a system that treats research-only datasets as production data
- a system that promotes a model because Top-1 accuracy improved

---

## 32. What success looks like

The long-term success condition is:

> A household can show KitchenIQ imperfect real-world observations and the system can maintain a trustworthy representation of what food probably exists, what state it is in, what is likely to happen next, and what useful action should happen now.

Example:

```text
SCAN KITCHEN
→ identify visible food
→ identify uncertainty
→ match existing entities
→ update food state
→ predict likely waste / use
→ recommend what to use first
→ generate feasible meal / action
→ user acts
→ KitchenIQ observes outcome
→ future prediction improves
```

---

## 33. Product North Star

> **KitchenIQ should not merely tell a household what food it can see. It should understand enough of the household's changing food state to predict what matters next and help the household act before waste, unnecessary shopping or decision effort occurs.**

---

## 34. Preserved V9.4 / V9.5 systems (summary)

The following remain in force under V10 unless a later authoritative document explicitly supersedes them:

| Area | Status under V10 |
|---|---|
| Secure Input Boundary / fail-closed | Preserved |
| Deterministic Policy Engine / safety gates | Preserved |
| Model registry, provenance, integrity, rollback | Preserved |
| Protected holdout | Preserved |
| KitchenState singularity | Preserved |
| Cook Engine feasibility / meal recommendation foundation | Preserved |
| Food Trajectory | Preserved and clarified |
| Entity Lineage / Reconciliation | Preserved |
| Kitchen Twin (simulation only) | Preserved and clarified |
| Use First Autopilot | Preserved and elevated as flagship |
| Learning loop / challenger / shadow / promotion | Preserved |
| Existing API contracts | Preserved (no API changes in this document update) |
| Frozen 207-label taxonomy | Preserved unchanged |
| DINOv2 + Faster R-CNN current vision stack | Preserved unless evidence justifies change |
| 41-scene real-world acceptance suite | Preserved as authoritative |

Detailed contracts for these systems remain documented in the V9.5 Architecture Specification and related Cook Engine documents. Where those documents conflict with V10 on product objective, dataset strategy, evaluation primacy, training gating, or vision’s role as observation/evidence, **V10 wins**.

---

## 35. V10 change log

- Product objective clarified.
- Vision repositioned as observation / evidence layer.
- Household food state made the central product representation.
- Food Trajectory / Kitchen Twin / Use First Autopilot preserved and clarified.
- Dataset strategy changed from arbitrary large image totals to capability-driven coverage.
- **250 diverse examples per core label** established as the **initial planning target**.
- Research / experiment data separated from production-eligible data.
- Real-world scene completeness elevated above isolated Top-1 accuracy.
- Established specialist baseline benchmarking made mandatory before broad retraining.
- Training is now evidence-gated rather than automatic.
- V9.4 / V9.5 security, safety, registry, holdout, and state-authority rules preserved.

---

## 36. Document quality locks

This V10 document intentionally:

- contains **no** required dataset target of 8,000 / 10,000 / 20,000 images
- treats **250** only as the **INITIAL CORE LABEL PLANNING TARGET**
- does **not** apply 250 as a universal requirement for every capability
- forbids research-only datasets from silently entering production training
- leaves the frozen **207-label** taxonomy unchanged
- records Vision V2 production status as **BLOCKED**
- keeps the **41-scene** acceptance suite authoritative
- keeps **strict scene completeness** as a release metric
- rejects Top-1 accuracy as the primary product release criterion
- is self-contained and understandable without reading prior documents
- defines exactly **one** authoritative product objective
- resolves V9.4 / V9.5 conflicts in favour of V10 where stated

---

*End of KitchenIQ Master Product, AI, Vision & Dataset Requirements — VERSION 10.0*
