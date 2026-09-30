# KitchenIQ Architecture Specification V9.5

**Product / Engine Version: 9.5** · **Document Revision: 31** · 25 September 2026  
**Supersedes:** Version 9.4 (9 September 2026) as the locked product architecture  
**Document title (authoritative):** KitchenIQ Architecture Specification V9.5  
**Former title:** KitchenIQ — Cook Engine Architecture (v9 / v9.4)  
**Status:** Authoritative implementation contract — product north star, V9.5 predictive household food system, **preserved** v9.4 security + recognition + cook-engine foundation, API contracts, visual understanding, predictive ML security  
**Sources:** ChatGPT + Perplexity research (Parts I–II); V9.5 product & engine strategic update (Part 0 + AUTHORITATIVE M–U)  
**Confidential** — Vallinate Technologies

Parts I and II are the source research. They must not be implemented where they conflict.

**The section `AUTHORITATIVE SPECIFICATION` at the end of this document supersedes any conflicting architecture, pseudocode, pipeline order, data model, scoring or API described earlier.** V9.5 sections **M–U** extend and, where stated, supersede **A–L** without discarding the v9.4 security, recognition, or cook-engine foundation. Do not treat the spec as implementable without those API contracts; they are defined there.

**Strategic rule:** Do not throw away v9.4. Do not merely rename the existing engine. Do not define success as expanding a closed-set classifier. V9.4 remains the security + recognition + cooking-engine foundation. V9.5 adds Food Trajectory, Entity Lineage, Entity Reconciliation, Household Behaviour, Kitchen Twin, Future State Simulation, Use First Autopilot, prediction-error learning, temporal evaluation, and product-level outcome metrics.

**Implementation status vocabulary (mandatory in this document):** `SPECIFIED` | `IMPLEMENTED` | `PARTIALLY IMPLEMENTED` | `NOT IMPLEMENTED` | `BLOCKED` | `EXPERIMENTAL`. Architecture requirements in Part 0 / M–U are **SPECIFIED** unless the Implementation Status Matrix marks otherwise. Do not claim SPECIFIED features are IMPLEMENTED.

---

## Changelog — Document Revision 31 (V9.5)

| Item | Status in this revision |
|---|---|
| Product North Star (predictive household food system) | SPECIFIED |
| Top-level V9.5 stack (Vision V2 → Twin → Use First → Learning) | SPECIFIED |
| Food Trajectory Engine | SPECIFIED (NOT IMPLEMENTED in code) |
| Food Entity Lineage | SPECIFIED (NOT IMPLEMENTED in code) |
| Entity Reconciliation | SPECIFIED (NOT IMPLEMENTED as first-class module) |
| Household Behaviour Model | SPECIFIED (NOT IMPLEMENTED in code) |
| Kitchen Twin | SPECIFIED (NOT IMPLEMENTED in code) |
| Future-State Simulation (scenario layer beyond cook simulator) | SPECIFIED; cook `State Simulation` remains IMPLEMENTED (v9.4) |
| Use First Autopilot | SPECIFIED (NOT IMPLEMENTED in code) |
| Prediction-error learning | SPECIFIED (NOT IMPLEMENTED in code) |
| Temporal / multi-observation evaluation | SPECIFIED (NOT IMPLEMENTED as product gate) |
| Product Intelligence Acceptance (Levels 1–7) | SPECIFIED |
| Vision Engine V2 open-world + hierarchy | SPECIFIED; current production vision PARTIALLY IMPLEMENTED / real-world dense acceptance BLOCKED |
| v9.4 Secure Input, Safety/Policy, Cook pipeline, Registry, Quarantine, APIs | PRESERVED — do not weaken |

**Rev 30 → 31:** Title renamed to Architecture Specification V9.5; authority role table; KitchenState singularity clarified; scenario IDs normalised; Implementation Status Matrix + integrity checklist (U); grocery-list excluded from primary product framing; change report requirements locked.

---

## Authority roles (non-negotiable)

| Role | Authority | Must not |
|---|---|---|
| **VISION ENGINE V2** | Observation capability (hypotheses) | Own or mutate KitchenState; invent facts from LOW confidence |
| **KITCHENSTATE** | **Sole** authoritative operational physical food state | Be duplicated by Twin/Trajectory stores as a second truth |
| **FOOD TRAJECTORY** | PREDICTED future evolution | Become OBSERVED fact; determine safety |
| **KITCHEN TWIN** | Simulation / what-if representation | Replace KitchenState; invent safety PASS |
| **USE FIRST AUTOPILOT** | Action selection / recommendation / plan updates | Override Safety/Policy; invent expiry; silent overwrite of confirmed user data |
| **POLICY / SAFETY ENGINE** | Deterministic safety authority | Be bypassed by predictions or autopilot |
| **FLUTTER** | Presentation / client only | Contain scoring, allergy, ranking, simulation, keys |

**Household Food State** (product phrase) = the understood household food picture derived from **KitchenState** + validated observations + context. It is **not** a second writable KitchenState authority.

---

# Part 0 — KitchenIQ V9.5 Product & Engine Strategic Update

## 0.1 Product North Star

KitchenIQ is a **predictive household food system** that understands the current state of household food, predicts what is likely to happen next, and determines the safest and most useful next action.

KitchenIQ is **NOT** primarily:

- a pantry scanner
- a food-recognition demo
- a recipe generator
- a meal-planning application
- a grocery list application

Those are **supporting capabilities**.

**Primary product description:** Food-specific household intelligence system.

**Product outcome loop:**

```
Observe
   ↓
Understand
   ↓
Establish Household Food State
   ↓
Predict Food Trajectory
   ↓
Evaluate Future Outcomes
   ↓
Determine Next Best Action
   ↓
Act / Recommend
   ↓
Observe Result
   ↓
Learn
```

The product must remain useful even when household inventory is incomplete or individual visual observations are uncertain.

### Product positioning

KitchenIQ is a predictive household food system rather than a conventional pantry tracker.

It combines food-specific visual intelligence, persistent food-state tracking, household behaviour modelling, future-state simulation and safe decision-making to determine what the household should do with its food next.

Supporting capabilities include: food recognition, pantry automation, meal planning, recipe generation, shopping intelligence, expiry awareness, waste reduction, household personalisation. These are components of the larger system.

### Competitive positioning (careful wording)

KitchenIQ's intended differentiation is the integration of food-specific recognition, persistent food entity/state tracking, household behavioural learning, trajectory prediction and future-state decision-making into one closed-loop household food system.

**Do not claim** unsupported market-superiority language such as “first in the world,” “the first household food system,” or that KitchenIQ will “beat Google/Gemini at image recognition.”

**Defensibility (moat):**

KitchenIQ's defensibility comes from converting repeated multimodal observations into a persistent, food-specific household state and learning the trajectory of that state over time.

```
Food Vision
  + Food Ontology
  + Entity Lineage
  + Household State
  + Historical Observations
  + Consumption Behaviour
  + Trajectory Models
  + Kitchen Twin
  + Decision Engine
  + Validated Outcome Feedback
```

That is substantially harder to reproduce than a single image-recognition feature.

### Language that must not appear as product truth

| Remove / avoid | Replace with |
|---|---|
| “KitchenIQ recognises every food.” | “KitchenIQ recognises supported food entities when evidence is sufficient and explicitly handles unknown or uncertain observations when evidence is insufficient.” |
| “AI scanner” as primary product description | “Food-specific household intelligence system.” |
| “Beat Google/Gemini at image recognition.” | “Use general visual capability as an input layer while differentiating through persistent food-state understanding, trajectory prediction and household decision intelligence.” |
| “The first household food system…” | “KitchenIQ's intended product position is a predictive household food system…” |
| MVP = “world's best food-recognition system” | MVP = demonstrate observe → reliable-enough food state → high-priority trajectories → safe Use First decision that improves the next household action |

### Current production honesty

Prior component/integration gates (including vision PRODUCTION_GATE where recorded) prove infrastructure. They do **not** automatically prove robust real-world dense-scene understanding or product intelligence. Vision remains subject to real-world acceptance. Product readiness requires the Product Intelligence Acceptance Layer (section 0.14 / AUTHORITATIVE T).

## 0.2 Top-level V9.5 architecture (locked)

Keep the v9.4 recognition → canonicalisation → safety → recommendation → decision pipeline. Add the household-intelligence stack:

```
KITCHENIQ
|
├── Vision Engine V2          (observation capability)
├── Household Context
├── Household Food State      (derived view; KitchenState is writable authority)
├── Food Trajectory Engine    (PREDICTED)
├── Kitchen Twin              (simulation)
├── Future-State Simulation
├── Use First Autopilot       (recommend / prioritise / plan — not safety)
├── Action / Recommendation Engine  (Cook Engine + ranking; Safety Gate inside)
└── Validated Learning Loop
```

The existing Cook Engine and Safety/Policy Engine remain authoritative for recipe decisions and safety.

### Locked end-to-end stack

```
┌──────────────────────────────────────────────┐
│                KITCHENIQ V9.5                │
└──────────────────────────────────────────────┘
                     │
                     ▼
             SECURE INPUT BOUNDARY
                     │
                     ▼
             VISION ENGINE V2
                     │
        ┌────────────┼─────────────┐
        ▼            ▼             ▼
     VISUAL        OCR          BARCODE
        │            │             │
        └────────────┼─────────────┘
                     ▼
             EVIDENCE FUSION
                     │
                     ▼
             CANONICALISATION
                     │
                     ▼
             ENTITY RECONCILIATION
                     │
                     ▼
              KITCHEN STATE
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
 FOOD TRAJECTORY ENGINE   HOUSEHOLD BEHAVIOUR
          │                     │
          └──────────┬──────────┘
                     ▼
                KITCHEN TWIN
                     │
                     ▼
            FUTURE SIMULATION
                     │
                     ▼
             CANDIDATE ACTIONS
                     │
                     ▼
               SAFETY GATE
                     │
                     ▼
               FEASIBILITY
                     │
                     ▼
          CULINARY COMPATIBILITY
                     │
                     ▼
              STATE SIMULATION
                     │
                     ▼
                FUTURE VALUE
                     │
                     ▼
            HOUSEHOLD / PERSONAL
                     │
                     ▼
                  RANKING
                     │
                     ▼
             USE FIRST AUTOPILOT
                     │
                     ▼
              EXPLANATION
                     │
                     ▼
          OUTPUT SECURITY VALIDATION
                     │
                     ▼
              USER / HOUSEHOLD
                     │
                     ▼
               ACTUAL OUTCOME
                     │
                     ▼
          VALIDATED LEARNING LOOP
                     │
                     └──────────────► future models
```

## 0.3 Food Trajectory Engine

The Food Trajectory Engine predicts the likely **future** state of household food rather than treating inventory as static records.

For each canonical food entity it reasons over: identity, location, quantity, food state, packaging state, observed condition, acquisition time, expected use window, household consumption history, preferences, meal plans, schedule/context, previous observations, corrections, consumption, waste, substitution behaviour.

It produces: **Current State + Trajectory + Risk + Potential Outcomes + Recommended Action**.

Example (illustrative — predictions are never OBSERVED facts):

```json
{
  "food_entity": "tomato",
  "current_state": "fresh",
  "quantity_estimate": 4,
  "trajectory": {
    "likely_use_window": "short",
    "waste_risk": "medium"
  },
  "recommended_action": "use_first",
  "reason_codes": [
    "high_near_term_use_priority",
    "available_meal_compatibility"
  ],
  "confidence": 0.87,
  "status": "predicted"
}
```

Evidence statuses (extends v9.4): **OBSERVED | INFERRED | PREDICTED | UNCERTAIN**.

A food trajectory is first-class: Food Entity → Current State → Historical States → Consumption History → Predicted Trajectory → Future Risk → Potential Actions.

Example paths: Purchased → Stored → Opened → Partially consumed → Remaining → Use First candidate → Consumed; or Purchased → Stored → Unused → Waste risk increasing → Use First intervention → Consumed.

## 0.4 Kitchen Twin

Kitchen Twin is a digital representation and simulation of the household food environment — **not** merely another inventory screen.

It represents: current household food state + food trajectories + consumption behaviour + planned meals + shopping state + predicted future state.

It must support:

- **Current-state:** What do I have? What needs attention? What should I use first? What can I cook now?
- **Future-state:** What will likely be wasted? What happens if I buy this / don't shop? What should I freeze? What remains while away?
- **Constraints:** Can I feed four without shopping? Can I delay shopping? What to use before buying more? What to substitute?

## 0.5 Future State Simulation

```
Current KitchenState → Scenario → Simulation → Future KitchenState → Compare Outcomes
```

Scenarios include (canonical IDs): `BUY`, `DO_NOT_BUY`, `COOK_NOW`, `COOK_LATER`, `FREEZE`, `USE_FIRST`, `SUBSTITUTE`, `SKIP_MEAL`, `CHANGE_MEAL`, `HOUSEHOLD_AWAY`.

Estimates (predictions, not guarantees): remaining food, utilisation, waste risk, shopping requirement, meal availability, household fit, cost implications.

“What happens if…?” answers **must** be based on simulated KitchenState, not generic chatbot reasoning.

## 0.6 Use First Autopilot (flagship workflow)

KitchenIQ continuously determines which household food should receive attention next, explains why, and updates meal and shopping recommendations when food state changes.

```
SCAN → UNDERSTAND → UPDATE FOOD STATE → CALCULATE TRAJECTORIES
→ IDENTIFY USE-FIRST CANDIDATES → CHECK SAFETY → CHECK MEAL COMPATIBILITY
→ SIMULATE OPTIONS → SELECT ACTION → EXPLAIN → UPDATE PLAN
```

### Autopilot must not override safety

Autopilot may: RECOMMEND, PRIORITISE, SIMULATE, UPDATE PLANS.

Autopilot must **not** independently: declare unsafe food safe; override allergy gates or household restrictions; invent expiry; convert uncertain observations into facts; silently modify confirmed user data.

The deterministic Policy Engine / Safety Gate remains authoritative:

```
Food Trajectory Prediction → Candidate Action → Safety Gate → Feasibility
→ Simulation → Ranking → Decision
```

**Prediction ≠ Safety determination.** “Likely consumed tomorrow” does **not** mean “safe to consume tomorrow.”

## 0.7 Food Entity Lineage + Entity Reconciliation

Persistent food identity / lineage:

```
Purchase → Package → Opened Package → Portion → Prepared Food → Leftover → Consumed / Discarded
```

Entity reconciliation for every new observation:

```
New Observation → Canonicalisation → Entity Matching
→ Existing Entity? YES → Update → Lineage → State transition
                 NO  → Create → Lineage → State transition
```

Must prevent: duplicate entities; accidental quantity inflation/replacement; repeated scans creating new inventory; cooked food counted as new raw ingredients; opened packages becoming unrelated products.

## 0.8 Vision Engine V2 (observation generator, not KitchenState authority)

```
Image → Scene Understanding → Region / Instance Discovery → Food / Non-Food
→ Broad Food Category → Fine Identity → Food State → OCR / Barcode / Package Evidence
→ Quantity / Instance Analysis → Entity Matching → Confidence / Alternatives → Canonicalisation
```

**Vision is an observation generator, not the authority for KitchenState.**

Open-world statuses: KNOWN | UNKNOWN_FOOD | UNKNOWN_NONFOOD | UNCERTAIN | CONFLICTING_EVIDENCE.

Never force “choose one of my known classes.” Prefer abstention / confirmation:

```json
{
  "food_status": "food",
  "identity": null,
  "alternatives": ["ginger", "turmeric"],
  "status": "requires_confirmation"
}
```

Identity and state remain separate. State may be multi-label where appropriate: raw, cut, mixed, cooked, fried, baked, plated, opened, sealed, frozen, leftover, unknown.

Evidence fusion: Visual + OCR + Barcode + Previous observation + Location + Quantity change + User confirmation. No evidence source may silently masquerade as another (e.g. missing barcode → `barcode_evidence = unavailable`, never `barcode_verified = true`).

## 0.9 Household Behaviour Model + prediction-error learning

Learn (with provenance): purchase frequency, consumption rate, meal frequency, preferences, avoidance, substitution, portion/leftover/waste behaviour, shopping intervals; who consumes what; which meals are cooked; which recommendations are accepted/rejected.

**Observed Behaviour ≠ Predicted Behaviour.**

Learning loop (keeps v9.4 zones — Untrusted / Operational / Learning Quarantine / Curated Training):

```
Observation → Prediction → Recommendation → User/Household Action → Actual Outcome
→ Prediction Error → Validated Learning Record → Quarantine → Training / Model Evaluation
```

Do **not** directly train production weights from user behaviour.

## 0.10 Recommendation objective upgrade

v9.4 cook ranking remains. Candidate generation may now use Food Trajectory Engine + Kitchen Twin + Household Behaviour Model:

```
Candidate Generation → Safety → Feasibility → Culinary Compatibility
→ State Simulation → Future Value → Household Fit → Ranking
```

Ranking may consider near-term utilisation, predicted waste avoidance, future meal preservation, shopping avoidance, household behaviour — **never** overriding hard safety gates.

### Core product interactions

**“What should we do now?”** returns ACTION + WHY + EXPECTED OUTCOME + ALTERNATIVE + CONFIDENCE.

**“What happens if…?”** returns Kitchen Twin simulation outcomes.

## 0.11 Canonical domain objects (V9.5)

```
FoodEntity → FoodObservation[] → FoodStateHistory → FoodLineage → FoodTrajectory
→ FutureScenario[] → ActionCandidate[] → Decision → Outcome
```

Also: FoodState, HouseholdBehaviour, FutureScenario, ActionCandidate, Decision, Outcome (alongside existing KitchenState, HouseholdContext, CanonicalIngredient/Meal, VisualObservation).

Confidence attaches not only to vision but to identity, state, quantity, entity match, consumption prediction, waste prediction, trajectory, future scenario, and recommended action — with `status: predicted` where applicable.

## 0.12 Data strategy (retain Vision V2 + add temporal)

Retain: public bootstrap + production-eligible + hard negatives + user-confirmation data.

**Add trajectory / temporal data:** observation t0 → t1 → t2 → actual outcome. The Food Trajectory Engine cannot be properly validated from isolated images alone.

## 0.13 Evaluation layers

| Layer | Question |
|---|---|
| Single-image | Can KitchenIQ understand this image? |
| Multi-observation | Can it maintain identity across observations? |
| Temporal trajectory | Can it predict what happens next? |
| Decision | Did it recommend the appropriate next action? |

Yields: Vision Accuracy + State Accuracy + Entity Continuity + Trajectory Accuracy + Decision Accuracy.

## 0.14 Acceptance philosophy (Product Intelligence Layer)

Keep real-world vision acceptance. Add Product Intelligence Acceptance above it:

```
LEVEL 1 — MODEL
LEVEL 2 — VISION
LEVEL 3 — FOOD ENTITY
LEVEL 4 — KITCHEN STATE
LEVEL 5 — TRAJECTORY
LEVEL 6 — DECISION
LEVEL 7 — OUTCOME
```

Top-1 / mAP / even perfect recognition do **not** alone mean KitchenIQ passes.

Ultimate evaluation: Did KitchenIQ correctly understand the household food state and make an appropriate, safe, explainable next-action decision?

### Mandatory flagship scenario: USE-FIRST END-TO-END TEST

Initial scan → multiple entities → reconciliation → food state → inventory reconcile → trajectory → waste risk → meal candidates → safety → Kitchen Twin simulates → Use First decision → user cooks → new observation → state transition → trajectory recalculated → shopping/meal plan updated.

Must prove survival across **multiple observations over time**, not one photograph.

## 0.15 Success metrics

**Vision:** scene completeness, food recall, food/non-food precision, identity accuracy, unknown handling, state accuracy, duplicate detection, OCR/barcode success, calibration, latency.

**Product intelligence:** food-state accuracy, entity reconciliation accuracy, trajectory / waste-risk / consumption prediction accuracy, Use First acceptance, shopping avoidance, meal utilisation, inventory automation rate.

**Business/product:** weekly active households, repeat Use First, inventory without manual entry, meals from existing food, avoidable purchase reduction, avoidable waste reduction, four-week prediction improvement.

## 0.16 MVP objective (V9.5)

Demonstrate that KitchenIQ can observe a household kitchen, establish a sufficiently reliable food state, identify high-priority food trajectories, and produce a **safe Use First** decision that improves the next household action.

---

# Part I — ChatGPT architecture

The key conclusion is: do not build the intelligence in Flutter. Flutter should be the presentation/client layer. The cooking intelligence should live behind an API, with the database holding the household state and the backend owning recognition, canonicalisation, safety, feasibility, localization, simulation, ranking and learning.

The research supports the building blocks—ingredient/recipe/user graphs, ingredient substitution, VLM + retrieval, health-aware recommendation, and staged retrieval/ranking already exist. The opportunity for KitchenIQ is how these are combined into a state-transition decision engine, not claiming that graphs or AI recommendations themselves are novel. Your source already points in this direction.

## 1. The architecture I recommend

```
                    FLUTTER APP
                         │
                         │ HTTPS / JSON
                         ▼
                ┌──────────────────┐
                │   API LAYER      │
                │ Auth / Sessions   │
                └────────┬─────────┘
                         │
                         ▼
              ┌───────────────────────┐
              │ COOK ORCHESTRATOR     │
              │ authoritative pipeline│
              └───────────┬───────────┘
                          │
                          ▼
 Capture → Canonicalise → Household Context
                          │
                          ▼
                 Candidate Generation
                          │
                          ▼
                    Safety Gate
                          │
                          ▼
                    Feasibility
                          │
                          ▼
               Culinary Compatibility
                          │
                          ▼
                  State Simulation
                          │
                          ▼
                    Future Value
                          │
                          ▼
            Personal / Household Fit
                          │
                          ▼
                      Ranking
                          │
                          ▼
               Diversity / Reranking
                          │
                          ▼
                    Explanation
                          │
                          ▼
                    RESULTS API
                          │
                          ▼
                     FLUTTER UI
```

This is the only pipeline. Candidate generation → safety → feasibility → later ranking → diversity reranking. Do not implement a different order from Part II or from older diagrams.

## 2. Flutter's job

Flutter should handle:

- Camera
- Scan UI
- Ingredient cards
- Manual ingredient entry
- Cuisine selection
- Localization selection
- Loading states
- Results UI
- Recipe detail
- User actions

Flutter should NOT contain:

- recipe scoring
- allergy logic
- ingredient normalization
- cuisine reasoning
- opportunity-cost calculations
- future-state simulation
- recommendation ranking
- LLM prompts
- API keys
- database credentials

That is the most important architectural decision.

## 3. The actual Cook algorithm

### Stage 1 — Capture

User chooses:

Let's Cook

Then:

```
Camera
   ↓
Scan 1..N ingredients
   ↓
Detection results
   ↓
User confirms / removes / adds
```

The VLM should return structured detections:

```json
{
  "name": "tomato",
  "confidence": 0.96,
  "quantity_estimate": null,
  "unit": null,
  "source": "camera"
}
```

Don't immediately treat the VLM output as truth.

Recognition → canonicalisation → confidence → user confirmation.

Current research already shows VLMs can be useful for ingredient recognition, but also demonstrates that recognition and recommendation are separate problems.

## 4. Canonical Ingredient Engine

This is extremely important.

User scans:

coriander

The system should resolve:

```
raw:
"coriander"

canonical:
coriander_leaf

aliases:
cilantro
fresh coriander
coriander leaves
```

Another example:

```
raw:
"chicken"

canonical:
chicken_meat

possible forms:
breast
thigh
whole chicken
minced chicken
```

Every ingredient should have:

```
Ingredient
├── canonical_id
├── display_name
├── aliases
├── category
├── culinary_roles
├── allergens
├── nutrition references
├── substitutions
├── cuisine relationships
├── regional relationships
└── confidence

versatility, scarcity-in-this-kitchen, perishability-now, future optionality and predicted waste are **DerivedKitchenFeatures**, not permanent fields on CanonicalIngredient. See AUTHORITATIVE SPECIFICATION.
```

Food knowledge graphs already demonstrate the value of linking recipe, nutrient and food ontology data.

## 5. Household constraint resolution

This is where KitchenIQ becomes different from a generic recipe search.

The request must contain:

```
target_scope:
    SELF
    HOUSEHOLD
    SELECTED_MEMBERS
```

For example:

**Self**

Use:

- User profile
- user's allergies
- user's dietary constraints
- user's preferences
- user's health goals

**Household**

Combine the relevant household members.

But don't simply combine every preference into a hard restriction.

Separate:

**HARD**

- allergy
- explicit exclusion
- strict dietary restriction
- safety restriction

**SOFT**

- likes
- dislikes
- health goals
- cuisine preference
- spice preference
- cooking time

Allergy handling needs to be treated seriously. For example, FDA identifies nine major allergens in the US and emphasizes avoiding allergenic foods for people with food allergies.

And importantly:

unknown allergen information ≠ safe.

If recipe provenance is insufficient to confidently evaluate an allergy, the engine should flag it for verification or exclude it from a “safe” recommendation.

## 6. Candidate generation

Do not ask an LLM:

“Here are 10 ingredients. Give me recipes.”

Instead generate candidates from several sources.

### Source A — Recipe database

Search based on:

- ingredient coverage
- ingredient roles
- cuisine
- localization
- semantic similarity

### Source B — Local recipes

Your own curated recipe database.

### Source C — Hybrid recipes

Existing recipe + validated substitutions.

### Source D — Generated culinary route

Only when retrieval doesn't provide enough viable candidates.

This is important because the research already shows retrieval + generation approaches, so KitchenIQ shouldn't make generation the default.

## 7. Feasibility engine

This replaces simplistic:

matched ingredients = 7

Instead every recipe gets:

**Required roles**

- protein
- base
- aromatic
- cooking medium

**Preferred roles**

- acid
- herb
- heat
- garnish

**Core ingredients**

Ingredients that define the dish.

**Optional ingredients**

Ingredients whose absence doesn't break it.

**Substitutable ingredients**

Ingredients that can be replaced safely/culinarily.

So:

**Biryani**

```
Rice        CORE
Chicken     CORE
Onion       IMPORTANT
Saffron     OPTIONAL
Mint        PREFERRED
Salt        STAPLE
```

This is much better than counting ingredients.

## 8. The part I would make KitchenIQ's core

### State Transition Engine

Every candidate gets simulated.

```
CURRENT KITCHEN

Chicken 1kg
Rice 1kg
Yoghurt 500g
Onion 5
Tomato 4
Spinach 300g

Candidate:

Chicken Biryani

Simulate:

AFTER COOKING

Chicken 600g
Rice 400g
Yoghurt 200g
Onion 3
Tomato 4
Spinach 300g
```

Then ask:

What did this meal do to the kitchen?

Your source correctly identifies this as the important conceptual shift: a meal is a transition between kitchen states.

## 9. Future Kitchen Value

Now we go one step further.

Suppose:

**Meal A**

Uses:

- chicken
- rice
- yoghurt
- onion

**Meal B**

Uses:

- chicken
- onion

Meal A might match more ingredients.

But if consuming the rice + yoghurt destroys tomorrow's strongest meal opportunity, Meal B may actually be the better decision.

So:

```
MEAL VALUE

Immediate suitability
+
Ingredient value used
+
Waste avoided
+
Future meal potential
-
Missing ingredient burden
-
Shopping burden
-
Opportunity cost
```

This is the core algorithm I would build around.

Not:

"Which recipe matches most?"

But:

"Which cooking action creates the best next kitchen state?"

That is the strongest part of the architecture proposed in your source.

## 10. Ingredient value needs to be contextual

Do not assign:

1 ingredient = 1 point

Instead:

```
ingredient_value =
    quantity
    × derived.culinary_role_value
    × derived.perishability
    × derived.freshness
    × derived.scarcity
    × derived.future_optionality
    × confidence

`derived.*` comes from DerivedKitchenFeatures. Do not store versatility as a hand-assigned CanonicalIngredient field.
```

Therefore:

one unused onion ≠ one unused saffron packet.

The engine learns that some ingredients have much greater opportunity value.

## 11. Cuisine engine

Cuisine should not just be a filter.

For:

Let's Cook

Cuisine is optional.

For:

International in Your Local Way

Cuisine becomes a candidate-generation constraint.

Example:

```
Ingredients
↓
Choose:
Thai
↓
Localize Me
↓
United Kingdom
↓
England
↓
London
```

The engine should then consider:

- Thai culinary identity
- local ingredient availability
- local substitutions
- user ingredients
- household constraints

It should not simply rename an existing recipe.

Your source specifically identifies this weakness in the old architecture: international adaptation should influence candidate generation, not merely happen after ranking.

## 12. “Localize Me” should be its own transformation layer

I would model:

```
ORIGINAL CULINARY TARGET
        ↓
Cuisine
        ↓
Country
        ↓
Region / State
        ↓
City / Local context
        ↓
LOCALIZATION PROFILE
        ↓
Candidate generation
```

For example:

```
Italian
↓
India
↓
Kerala
```

is not:

“Make Italian food with Indian ingredients.”

It means:

Preserve the culinary identity of the source cuisine while adapting ingredient availability, regional preferences and practical cooking context.

That distinction matters.

## 13. Ranking

After all that, rank.

I would use a structured score:

```
FINAL SCORE

Safety
Feasibility
Ingredient Value
Culinary Coherence
Future Kitchen Value
Waste Avoidance
Cuisine Fit
Localization Fit
Household Fit
Personal Preference
Cooking Burden
Missing Ingredient Burden
Confidence
```

But:

Safety isn't a score.

It is a gate.

```
UNSAFE
   ↓
REJECT
```

Then everything else gets ranked.

## 14. Don't return six nearly identical dishes

After ranking:

```
Chicken Biryani
Chicken Pulao
Chicken Rice
Chicken Masala Rice
Chicken Biryani variation
```

is a bad result.

Use a diversity reranker based on:

- ingredient overlap
- cooking technique
- flavour profile
- cuisine
- regional identity
- dish family

Graph-based culinary relationships are already well studied; FlavorGraph, for example, uses recipe co-occurrence and food/chemical relationships to represent culinary compatibility.

KitchenIQ's value is in using those relationships inside the state-transition decision process, not merely building another ingredient graph.

## 15. LLM's job

This is another thing I would lock down.

LLM CAN:

- interpret ambiguous user language
- normalize unusual ingredient names
- explain recommendations
- propose controlled substitutions
- generate a candidate when retrieval fails
- translate/localize culinary instructions
- create natural recipe descriptions

LLM CANNOT:

- decide allergy safety
- override hard constraints
- be the primary ranking engine
- invent nutrition facts
- decide whether an ingredient is safe
- directly modify Kitchen State

The source's proposal to keep the LLM bounded is correct.

## 16. Database architecture

I would structure the backend around these domains:

```
users
households
household_members
profiles

ingredients
ingredient_aliases
ingredient_roles
ingredient_substitutions
ingredient_allergens

recipes
recipe_ingredients
recipe_steps
recipe_cuisines
recipe_regions

culinary_relationships
cuisine_profiles
localization_profiles

kitchen_items
kitchen_item_events
kitchen_states

cook_sessions
cook_session_ingredients
cook_candidates
cook_decisions

user_recipe_events
user_substitution_events
user_cooking_events

ranking_versions
decision_explanations
```

And importantly:

Don't store only the current pantry.

Store events.

```
BOUGHT
OPENED
SCANNED
COOKED
CONSUMED
LEFTOVER
FROZEN
DISCARDED
```

That eventually gives you the Food Trajectory architecture.

## 17. Vector search

I would not immediately introduce a separate vector database.

If you're already using PostgreSQL, use Postgres + pgvector initially.

It supports exact and approximate nearest-neighbour search and can keep vectors alongside your relational data.

Use it for:

- recipe embeddings
- ingredient embeddings
- cuisine representations
- semantic retrieval

But keep the hard logic relational/deterministic.

## 18. Nutrition/food knowledge

For structured nutrition data, FoodData Central provides an API and food details/search endpoints.

But don't make KitchenIQ dependent on one external API for every recommendation.

Use:

```
Canonical internal ingredient
        ↓
external mappings
        ↓
cached structured data
```

External providers become data sources, not your application's brain.

## 19. Production architecture

I would aim for:

```
Flutter
   │
   ▼
API Gateway / FastAPI
   │
   ├── Auth
   ├── Cook Session API
   ├── Profile API
   ├── Kitchen API
   └── Results API
            │
            ▼
       Cook Engine
            │
      ┌─────┼─────────┐
      ▼     ▼         ▼
  Retrieval Safety  State
      │     │       Simulator
      └─────┼─────────┘
            ▼
         Ranker
            │
            ▼
       Explanation
            │
            ▼
         Response
```

- PostgreSQL
- Redis
- pgvector
- Object Storage
- External recipe/nutrition APIs
- LLM/VLM providers

Crucially:

Flutter never talks directly to Spoonacular/Gemini/FoodData Central.

All credentials stay server-side.

## 20. How Cursor should implement this

Do NOT give Cursor:

“Rewrite the whole KitchenIQ cook engine.”

That will produce a giant mess.

Give Cursor phases.

### Phase 0 — Architecture audit

Cursor must first inspect:

- Flutter project
- backend
- ingredient_engine.py
- database
- API routes
- auth
- recipe provider
- Gemini/VLM integration
- current ranking
- current international logic

No code changes.

Output:

```
CURRENT ARCHITECTURE
FILES
DATA FLOW
CURRENT ALGORITHM
CURRENT DATABASE
CURRENT API
WHAT SHOULD STAY
WHAT SHOULD MOVE
WHAT SHOULD BE DELETED
```

### Phase 1 — Establish backend boundary

Create/clean:

```
/api
/domain
/services
/engine
/repositories
/models
/providers
/tests
```

Flutter becomes:

```
UI
↓
API client
↓
DTO/state
```

No recommendation logic in widgets.

### Phase 2 — Cook domain model

Implement:

- CookSession
- IngredientObservation
- CanonicalIngredient
- KitchenState
- HouseholdContext
- CuisineIntent
- LocalizationIntent
- Candidate
- FeasibilityResult
- SafetyResult
- StateTransition
- RecipeDecision

These become typed backend domain objects.

### Phase 3 — Ingredient pipeline

Implement:

```
image
 ↓
recognition
 ↓
normalization
 ↓
canonical ingredient
 ↓
confidence
 ↓
confirmation
 ↓
CookSession
```

Support:

- 1 ingredient
- multiple ingredients
- duplicate detection
- manual addition
- removal
- correction

### Phase 4 — Safety engine

Implement independently.

```
CookSession
+
TargetProfile
+
Recipe
↓
SafetyEngine
↓
PASS / REJECT / VERIFY
```

Never bury safety inside the ranker.

### Phase 5 — Feasibility engine

Implement:

- required roles
- core ingredients
- optional ingredients
- substitution rules
- quantity coverage
- missing burden

Return structured explanations.

### Phase 6 — Candidate engine

Implement adapters:

```
RecipeProvider
 ├── LocalRecipeProvider
 ├── ExternalRecipeProvider
 └── GeneratedCandidateProvider
```

This means you can replace Spoonacular later without rewriting KitchenIQ.

### Phase 7 — State simulator

This is the big one.

Implement:

```
simulate(candidate, kitchen_state)
```

Output:

- remaining quantities
- consumed quantities
- new leftovers
- waste risk
- stranded ingredients
- future meal opportunities

Never mutate the real kitchen state during simulation.

Only:

```
simulation → recommendation
```

When user actually cooks:

```
Cook confirmation
↓
real KitchenState mutation
```

### Phase 8 — Future value engine

Implement:

```
future_value(before, after)
```

Initially it can use deterministic heuristics.

Later it can learn.

Don't start with ML.

You need correct state modelling before machine learning.

### Phase 9 — Ranking engine

Implement:

```
retrieve
→ safety
→ feasibility
→ state simulation
→ future value
→ personal fit
→ cuisine/localization
→ diversity
→ final result
```

Every result should contain score components internally.

This makes debugging possible.

### Phase 10 — Localization engine

Implement:

- CuisineIntent
- LocalizationProfile
- RegionalIngredientRules
- RegionalSubstitutions
- CulinaryIdentity

Then:

```
International
↓
Cuisine
↓
Localize Me
↓
Country
↓
Region
↓
City
↓
localized candidate generation
```

Do not implement localization as a prompt slapped onto the final recipe.

### Phase 11 — Learning

Only after the deterministic engine works.

Track:

- shown
- opened
- started
- cooked
- completed
- skipped
- substituted
- liked
- disliked
- rejected

Then learn:

- preferred cuisines
- accepted substitutions
- actual cooking behaviour
- actual ingredient usage
- actual meal acceptance

This is where KitchenIQ's long-term behaviour model starts becoming valuable.

### Phase 12 — Production hardening

Cursor must add:

- request IDs
- cook session IDs
- idempotency
- timeouts
- retries
- caching
- provider fallbacks
- structured logs
- ranking version
- engine version
- feature flags
- metrics
- tracing
- validation
- authorization
- concurrency handling
- stale-session handling
- deterministic fallback

The source already correctly identifies these production requirements.

## 21. The master rule for Cursor

Give Cursor this rule at the very top:

DO NOT IMPLEMENT THE COOK ENGINE IN FLUTTER.

Flutter is the client/presentation layer.

All ingredient intelligence, canonicalisation, household constraint resolution, safety, feasibility, candidate generation, cuisine/localization reasoning, state simulation, future-value calculation, ranking and learning must live in the backend/domain layer.

Inspect the existing backend and preserve/reuse its architecture where sound. Do not duplicate business logic in Flutter.

If a backend already exists, extend it. If ingredient_engine.py is the current engine, refactor it into proper domain services rather than moving its logic into Dart.

Do not rewrite the entire application.

Implement one bounded phase at a time.

After every phase:

- run tests,
- verify API contracts,
- verify existing functionality,
- commit/checkpoint,
- only then proceed.

## The architecture I would actually lock

```
                USER
                 │
                 ▼
             Flutter
                 │
                 ▼
             Cook API
                 │
                 ▼
          ┌──────────────┐
          │ Cook Engine  │
          └──────┬───────┘
                 │
     ┌───────────┼────────────┐
     ▼           ▼            ▼
 Ingredient   Household    Cuisine /
 Intelligence  Context     Localization
     │           │            │
     └───────────┼────────────┘
                 ▼
          Candidate Retrieval
                 │
                 ▼
            Safety Gate
                 │
                 ▼
          Feasibility Engine
                 │
                 ▼
       Culinary Compatibility
                 │
                 ▼
          State Simulator
                 │
                 ▼
        Future Kitchen Value
                 │
                 ▼
     Personal / Household Fit
                 │
                 ▼
              Ranker
                 │
                 ▼
           Diversity
                 │
                 ▼
            Explanation
                 │
                 ▼
              Flutter
                 │
                 ▼
           USER DECISION
                 │
                 ▼
            Cook Event
                 │
                 ▼
          Kitchen State
                 │
                 ▼
           Learning Loop
```

That is the architecture I would build.

And importantly, I'm not recommending that we immediately build a GNN, train a custom model, or throw an agent at it. Ingredient graphs, health-aware recommenders, substitution models, VLM recognition and recipe retrieval already exist in research.

The first moat should be the quality of the structured Kitchen State + transition simulator + contextual ingredient value + constrained culinary decisioning. Then we can add learned models where the data proves they improve decisions.

# Part II — Perplexity production redesign

Yes. I understand completely.

You're not asking for cosmetic improvements or marketing language. You want KitchenIQ's cook engine to be architecturally superior at the code, algorithm, data model, and production-implementation level—something that creates a genuine technical moat, not just a feature list.

I've just completed a prior-art scan. The important finding: ingredient graphs, multi-meal optimisation, VLM → recipe retrieval, and knowledge-graph-based recommenders all exist independently in research. That means our novelty cannot be "we use graphs" or "we optimise multiple meals." Those are building blocks, not the moat.

The moat must come from how KitchenIQ combines them into something structurally different: a state-transition decision engine that reasons about future kitchen value, ingredient roles, culinary coherence, and opportunity cost—even for a single "what should I cook now?" query.

Below is the production-grade architecture and logic redesign. I'm going to be brutally honest about what should be deleted, what should survive, and what needs to be fundamentally rewritten.

## Core philosophical shift

**Current mental model:**

"Which recipe matches the most scanned ingredients?"

**New mental model:**

"What is the highest-value culinary state transition available from my current kitchen state?"

Formally:

```
Recommendation
  = arg max_{a ∈ Actions} (
        V_immediate(a)
      + V_future(S_after(a))
      − C_opportunity(a)
      − C_risk(a)
    )
```

Where:

- `S_before` = current kitchen state
- `a` = cooking action (recipe, hybrid, or generated dish)
- `S_after(a)` = simulated remaining inventory
- `V_future` = expected value of future meals from that remainder
- `C_opportunity` = cost of consuming ingredients that enable better future options
- `C_risk` = waste, safety, missing-core, shopping burden

This is fundamentally different from ranking by `unused_scan_count` → `missed_count` → `native_score`.

## Production architecture: authoritative pipeline

The older “eight-layer” diagram that started at Hard Gates is **not** the pipeline. Capture, canonicalisation, household context and candidate generation happen first. Then:

```
Capture
  → Canonicalise
  → Household Context
  → Candidate Generation
  → Safety Gate
  → Feasibility
  → Culinary Compatibility
  → State Simulation
  → Future Value
  → Personal / Household Fit
  → Ranking
  → Diversity / Reranking
  → Explanation
```

Every layer produces structured, testable outputs. Nothing is a black box.

## Data model redesign

Do not put household constraints, culinary identity or derived scores into KitchenState. Three objects:

### KitchenState — physical food state

```python
class KitchenItem:
    canonical_id: str
    quantity: float          # normalized units
    unit: str
    storage: str             # fridge | freezer | pantry | unknown
    acquired_at: datetime | None
    expires_on: date | None
    opened: bool
    recognition_confidence: float  # 0–1, observation quality not a derived feature

class KitchenState:
    items: List[KitchenItem]
    as_of: datetime
```

### HouseholdContext — people, safety, preferences

```python
class HouseholdContext:
    target_scope: str  # SELF | HOUSEHOLD | SELECTED_MEMBERS
    member_ids: List[str]
    hard_allergens: List[str]
    hard_exclusions: List[str]
    hard_dietary: List[str]  # e.g. strict vegan, halal
    soft_likes: List[str]
    soft_dislikes: List[str]
    health_goals: List[str]
    cuisine_preferences: List[str]
    home_country: str
    home_region: str
    spice_preference: str | None
    time_budget_minutes: int | None
```

### DerivedKitchenFeatures — computed, not stored as identity

```python
class DerivedKitchenFeatures:
    culinary_role_value: Dict[str, float]   # 0–1
    perishability: Dict[str, float]         # 0–1, from category + storage + expiry
    freshness: Dict[str, float]             # 0–1, from elapsed time + storage, not cook duration
    scarcity: Dict[str, float]              # 0–1, this kitchen, now
    future_optionality: Dict[str, float]    # 0–1
    predicted_waste_risk: Dict[str, float]  # 0–1
    versatility: Dict[str, float]           # 0–1, derived from role count / graph degree
```

`versatility` is **derived**: e.g. min(1.0, distinct_roles / 4) or min(1.0, graph_degree / degree_norm). It is not a permanent manually assigned CanonicalIngredient field.

### CanonicalIngredient

```python
class CanonicalIngredient:
    id: str
    name: str
    display_name: str
    aliases: List[str]
    category: str  # protein, vegetable, spice, staple, ...
    roles: List[str]  # primary_protein, aromatic, seasoning, base, ...
    allergens: List[str]
    substitutions: List[SubstitutionRule]
    cuisine_relationships: List[str]
    regional_relationships: List[str]
    nutrition_ref: str | None
    typical_quantity_per_meal: float | None
```

### RecipeDecision (output object)

```python
class RecipeDecision:
    recipe_id: str
    safety: SafetyResult
    feasibility: FeasibilityScore
    haul_utilisation: float
    culinary_coherence: float
    ingredient_value_used: float
    missing_burden: float
    cuisine_fit: float
    native_fit: float
    personal_fit: float
    future_inventory_value: float
    waste_avoidance: float
    shopping_burden: float
    confidence: float
    explanation: List[str]
    state_after: KitchenState
```

Every field is computable and testable. No magical "AI score."

## Layer 1: Hard gates (non-negotiable)

What changes: Safety moves from a cosmetic badge to a hard exclusion gate.

```python
def safety_gate(ingredients, recipe, user_profile) -> SafetyResult:
    # Hard exclude if:
    # - Recipe contains allergen that user is allergic to
    # - Recipe violates hard dietary constraint (e.g., halal, vegan if strict)
    # - Ingredient recognition confidence too low for safety-critical items

    if any(allergen in recipe.allergens for allergen in user_profile.hard_allergens):
        return SafetyResult(reject=True, reason="allergen_conflict")

    if not recipe.satisfies_hard_constraints(user_profile):
        return SafetyResult(reject=True, reason="dietary_violation")

    return SafetyResult(reject=False)
```

This runs before any ranking. If safety fails, the recipe is excluded entirely.

## Layer 2: Feasibility engine (role-based, not string-based)

What changes: Move from "ingredient present/absent" to "culinary roles satisfied."

A recipe defines required and preferred roles:

```python
class RecipeRequirement:
    required_roles: List[str]  # ["primary_protein", "aromatic", "cooking_medium"]
    preferred_roles: List[str]  # ["acid", "heat", "herb"]
    core_ingredients: List[str]  # ingredients whose absence breaks the dish
    optional_ingredients: List[str]
```

Feasibility scoring:

```python
def feasibility_score(recipe, kitchen_state) -> FeasibilityScore:
    available_roles = extract_roles(kitchen_state.ingredients)

    missing_required = set(recipe.required_roles) - available_roles
    missing_preferred = set(recipe.preferred_roles) - available_roles

    missing_core = [ing for ing in recipe.core_ingredients
                    if not is_substitutable(ing, kitchen_state)]

    # Score components
    n_required = len(recipe.required_roles)
    if n_required == 0:
        role_satisfaction = 1.0
    else:
        role_satisfaction = 1.0 - (len(missing_required) / n_required)
    core_integrity = 1.0 if not missing_core else 0.0
    n_preferred = len(recipe.preferred_roles)
    if n_preferred == 0:
        optional_gap = 0.0
    else:
        optional_gap = len(missing_preferred) / n_preferred

    return FeasibilityScore(
        role_satisfaction=role_satisfaction,
        core_integrity=core_integrity,
        optional_gap=optional_gap,
        explanation=build_feasibility_explanation(...)
    )
```

This means:

- Missing saffron in biryani → low impact (optional, not core role)
- Missing rice in biryani → catastrophic (core structural ingredient)
- Missing onion → may be substitutable or acceptable depending on dish

## Layer 3: Culinary compatibility field (ingredient graph + roles)

What changes: Ingredients are nodes in a weighted graph with typed edges representing culinary relationships.

Graph edges can represent:

- Co-occurrence frequency (NPMI-weighted)
- Flavor compound sharing
- Regional pairing patterns (e.g., chicken + curry leaves + coconut → South Indian signature)
- Role compatibility (protein + aromatic + acid + heat)

Culinary coherence score:

```python
def culinary_coherence(recipe, kitchen_state, region=None) -> float:
    # Build ingredient subgraph for this recipe
    subgraph = extract_recipe_subgraph(recipe.ingredients, ingredient_graph)

    # Measure edge density / average weight
    coherence = average_edge_weight(subgraph)

    # If region specified, check regional signature match
    if region:
        regional_signature = compute_regional_signature(recipe.ingredients, region)
        coherence *= regional_signature

    return coherence
```

This prevents nonsensical combinations from ranking highly even if ingredient overlap is high.

## Layer 4: State simulator (the key innovation)

What changes: Every candidate recipe produces a KitchenState representing remaining **quantities, consumed items, leftovers and stranded items**. Freshness/expiry is not advanced by recipe cooking time.

```python
def simulate_after_state(recipe, kitchen_state: KitchenState) -> KitchenState:
    remaining = deepcopy(kitchen_state)
    remaining.as_of = kitchen_state.as_of  # same clock; time decay is not cook duration

    for ingredient in recipe.ingredients:
        quantity_needed = recipe.quantity_map[ingredient]
        item = remaining.item(ingredient)
        if item is None:
            continue
        item.quantity = max(0, item.quantity - quantity_needed)

    remaining.stranded = identify_stranded_ingredients(remaining)
    return remaining
```

The simulator changes quantities, consumed items, leftovers, and state transitions. Freshness/expiry advances based on elapsed time and storage conditions, not recipe cooking duration. Compute freshness in DerivedKitchenFeatures from `as_of`, `acquired_at`, `expires_on` and `storage`.

This enables the next layer.

## Layer 5: Future value estimator (opportunity cost)

What changes: The system evaluates what cooking this meal destroys for future meals.

```python
def future_inventory_value(state_after, state_before, derived: DerivedKitchenFeatures) -> float:
    value_lost = 0.0

    for item in state_before.items:
        after = state_after.item(item.canonical_id)
        after_qty = after.quantity if after else 0.0
        consumed = item.quantity - after_qty
        if consumed <= 0:
            continue
        cid = item.canonical_id
        ingredient_value = (
            derived.versatility.get(cid, 0.5)
            * derived.scarcity.get(cid, 0.5)
            * derived.perishability.get(cid, 0.5)
            * derived.future_optionality.get(cid, 0.5)
        )
        value_lost += consumed * ingredient_value

    waste_risk_threshold = config.waste_risk_threshold  # cook-rank-v1: 0.60
    ingredients_used_near_spoilage = [
        item.canonical_id
        for item in state_before.items
        if (item.quantity - (state_after.item(item.canonical_id).quantity if state_after.item(item.canonical_id) else 0.0)) > 0
        and derived.predicted_waste_risk.get(item.canonical_id, 0.0) > waste_risk_threshold
    ]
    # ingredients_used_near_spoilage = ingredients consumed by the candidate
    # whose current waste risk exceeds the configured threshold.

    waste_avoided = sum(
        derived.perishability.get(i, 0) * derived.freshness.get(i, 0)
        for i in ingredients_used_near_spoilage
    )
    return -value_lost + waste_avoided
```

Weights and normalisation for this term are in Scoring Specification. Do not invent new factors in code.

This is where the system can prefer Recipe B over Recipe A even if A has higher immediate ingredient overlap—because B leaves the rice and yogurt available for another high-value meal tomorrow.

## Layer 6: Personal fit (cuisine, native, preferences)

What changes: Personalisation becomes a weighted soft layer, not the primary ranking signal. The numeric mix `0.4 * haul + 0.25 * cuisine + ...` below is **research illustration only**. Implementation must use Scoring Specification `cook-rank-v1`. Do not copy these weights into code.

```python
def personal_fit_score(recipe, user_profile, haul_coverage) -> float:
    # Cuisine fit
    cuisine_match = recipe.cuisine in user_profile.preferred_cuisines

    # Native fit (home cuisine preference)
    native_match = recipe.cuisine == user_profile.home_cuisine

    # Household fit (existing logic, but now one component)
    household_fit = compute_household_fit(recipe, user_profile)

    # Weight haul coverage heavily (as per your acceptance constraint)
    # But don't let it completely overpower personal fit
    score = (
        0.4 * haul_coverage +
        0.25 * cuisine_match +
        0.2 * native_match +
        0.15 * household_fit
    )

    return score
```

International adaptation should influence candidate generation, not just post-hoc retitling.

## Layer 7: Diversity + marginal information

What changes: Avoid returning six variations of "Chicken Biryani."

```python
def diversify_results(candidates, k=6) -> List[RecipeDecision]:
    selected = []
    remaining = candidates

    while len(selected) < k and remaining:
        # Pick highest-scoring candidate
        best = max(remaining, key=lambda c: c.total_score)
        selected.append(best)

        # Penalise near-duplicates in next iteration
        remaining = [
            c for c in remaining
            if culinary_distance(c, best) > config.diversity_threshold
        ]

    return selected
```

Culinary distance can be computed from:

- Ingredient overlap
- Cooking method
- Flavor profile
- Regional signature

## Layer 8: Explanation generator (LLM-bounded)

What changes: LLM generates natural-language explanations from structured data, not the ranking itself.

```python
def generate_explanation(decision: RecipeDecision) -> str:
    # LLM receives structured data, not raw ingredients
    prompt = f"""
    Recipe: {decision.recipe_name}

    Strengths:
    - Uses {decision.haul_utilisation*100:.0f}% of meaningful scanned ingredients
    - Missing burden: {decision.missing_burden:.2f} (low = good)
    - Future inventory value preserved: {decision.future_inventory_value:.2f}
    - Culinary coherence: {decision.culinary_coherence:.2f}

    Explain why this is the best meal to cook now in 2-3 sentences.
    Focus on ingredient utilisation, what cooking this does to remaining inventory,
    and why it fits the user's cuisine preference.
    """

    return call_gemini(prompt, temperature=0.3)
```

LLM is bounded. It cannot override safety gates or feasibility scores.

## Candidate generation: beyond 12 Spoonacular recipes

Current limitation: 12 external recipes + local fallback is too narrow.

New approach: Multi-source retrieval with deduplication:

- Spoonacular (primary)
- Local recipe database (pre-indexed with ingredient graphs)
- Generated candidates (only when retrieval fails to produce viable options)
- Hybrid candidates (known recipe + feasible substitutions)

Deduplication uses semantic recipe identity:

- Ingredient set similarity (Jaccard or cosine on role vectors)
- Cooking method
- Flavor profile embedding

```python
def generate_candidates(kitchen_state, k=50) -> List[Candidate]:
    candidates = []

    # Source 1: Spoonacular
    spoonacular_results = query_spoonacular(kitchen_state.ingredients, limit=20)
    candidates.extend(spoonacular_results)

    # Source 2: Local database (graph-based retrieval)
    local_results = graph_retrieve(kitchen_state, ingredient_graph, limit=20)
    candidates.extend(local_results)

    # Source 3: Generated (only if needed)
    if len(viable(candidates)) < 5:
        generated = generate_culinary_route(kitchen_state, validator=culinary_validator)
        candidates.extend(generated)

    # Deduplicate
    candidates = semantic_deduplicate(candidates)

    return candidates[:k]
```

## Production engineering requirements

### Idempotency and determinism

- Every request includes a `session_id` and `scan_id`
- Results are cached by `(scan_hash, user_profile_hash, cuisine, timestamp_bucket)`
- Retry logic with exponential backoff for Spoonacular/Gemini failures
- Deterministic fallback: if external APIs fail, use local database only

### Timeout and failure handling

```python
def cook_with_timeouts(kitchen_state, timeout_ms=3000):
    try:
        with timeout(timeout_ms):
            return cook_engine(kitchen_state)
    except TimeoutError:
        # Fallback to cached results or local-only retrieval
        return fallback_cook(kitchen_state)
    except GeminiError:
        # Continue without LLM explanations
        return cook_engine(kitchen_state, use_llm=False)
    except SpoonacularError:
        # Use local database only
        return cook_engine(kitchen_state, external_sources=False)
```

### Observability

- Log every decision with all score components
- Track which candidates were rejected at each gate
- Monitor latency per layer
- A/B test ranking weight adjustments via feature flags

### Testing (adversarial suite)

Test cases must include:

- 1 ingredient, 3 ingredients, 30 ingredients
- Duplicates, aliases, ambiguous recognition
- Culturally ambiguous ingredients (e.g., "coriander" vs "cilantro")
- Pantry staples only
- Missing core ingredient
- Allergen hidden in recipe
- Conflicting allergies
- International + region combinations
- External API unavailable
- Gemini unavailable
- Zero candidates
- Duplicate candidates
- Huge candidate sets (100+)
- Malicious/invalid ingredient input
- Contradictory profile data
- Stale session
- Concurrent requests

## What to delete from current implementation

- `unused_scan_count` as primary ranking signal → Replace with `ingredient_value_used` weighted by role/scarcity/perishability
- Simple `missed_count` → Replace with feasibility engine (core vs optional, role satisfaction)
- Post-hoc International adaptation only → Move cuisine influence earlier in candidate generation
- LLM as ranking brain → LLM generates explanations only, not scores
- Cosmetic safety badges → Move safety to hard gate before ranking

## What survives (but gets upgraded)

- Whole-haul coverage priority → Survives, but becomes `haul_utilisation` weighted by ingredient value, not count
- Native cuisine preference → Survives as one component of personal fit, not the dominant signal
- Household fit → Survives but integrated into personal fit layer
- Missing ingredient categorisation → Upgraded to feasibility engine with core/optional/substitutable distinctions

## The moat (why this is hard to copy)

Competitors can replicate individual pieces:

- Ingredient graphs exist
- Multi-meal optimisation exists in research
- Knowledge-graph recommenders exist

What they cannot easily copy:

- **State-transition reasoning in real-time** — evaluating opportunity cost of consuming specific ingredients for a single "cook now" query
- **Ingredient value as a function of role, scarcity, perishability, and future optionality** — not just presence/absence
- **Culinary compatibility field** — combining co-occurrence graphs, flavor chemistry, and regional signatures into a coherence score
- **Integrated feedback loop** — user actions (what they actually cook, what they substitute) update the ingredient value model and personal fit weights over time
- **Explainable, testable scoring** — every recommendation can be audited and reproduced

This creates a system where the combination of ingredient intelligence + culinary knowledge + personal context + feasibility + state simulation + future value forms a defensible technical architecture.

## Next step: code-level redesign

I'm ready to take your actual `ingredient_engine.py` and rewrite it at the function/class level:

- Define the exact objects, data structures, and scoring equations
- Specify API contracts and database implications
- Write pseudocode for each layer
- Identify which functions to delete, which to keep, which to rewrite
- Design the adversarial test suite

---

# Scoring Specification

Implementation must load weights from versioned configuration. **No arbitrary weights may be introduced during implementation.** Changing a weight requires a new `ranking_version`, not a code literal.

## Config identity

```
ranking_version: cook-rank-v1
engine_version: cook-engine-v1
```

Store both on every `cook_decision` row.

## Hard constraints (not scores)

These are gates. They never enter the weighted sum.

| Gate | PASS | REJECT | VERIFY |
|---|---|---|---|
| Safety | No hard allergen/dietary hit; recipe ingredient list provenance is sufficient | Hard allergen or hard dietary violation | Allergen/dietary data missing or recipe provenance insufficient |
| Core integrity | All CORE recipe ingredients present or safely substitutable | A CORE ingredient is missing and not substitutable | CORE match depends on LOW-confidence recognition |
| Required roles | `role_satisfaction >= 0.34` | `role_satisfaction < 0.34` | Required role filled only by MEDIUM/LOW confidence items |

- REJECT → drop from `suggested_meals` / `more_meals`. May appear only in `blocked_meals` with reason.
- VERIFY → must not be labelled safe. Do not use for safety-critical “safe to eat” claims. May rank in a verify list or be excluded from the safe list (product default: **exclude from safe list**).
- unknown allergen information ≠ safe. VERIFY, not PASS.

## Normalisation

Every score component is clipped to **[0, 1]** before weighting.

| Component | Definition | 0 | 1 |
|---|---|---|---|
| `feasibility` | `0.7 * role_satisfaction + 0.3 * (1 - optional_gap)` after core gate. If `required_roles` is empty, `role_satisfaction = 1.0`. If `preferred_roles` is empty, `optional_gap = 0.0`. | unusable | all required roles + no preferred gap |
| `haul_utilisation` | value-weighted covered scan / total scan value using DerivedKitchenFeatures | none of the haul used | entire haul value used |
| `ingredient_value_used` | consumed quantity × derived ingredient_value, min-max normalised across the candidate set in this request | least value used in set | most value used in set |
| `culinary_coherence` | mean recipe-subgraph edge weight, optionally × regional_signature | incoherent | strong pairing |
| `future_kitchen_value` | `minmax(future_inventory_value)` across this request’s PASS set | worst remainder | best remainder |
| `waste_avoidance` | derived waste risk of **consumed** near-spoilage items | none | all high-risk items used |
| `cuisine_fit` | 1 if recipe matches chosen cuisine/region intent; 0.5 if cuisine unset (Let’s Cook); 0 if contradicts chosen cuisine | mismatch | match |
| `localization_fit` | 1 if candidate generated under localization profile; 0.5 if no localization intent; 0 if contradicts local availability rules | mismatch | match |
| `household_fit` | 1 minus normalised count of soft-constraint misses (likes/dislikes/time) | all soft misses | no soft misses |
| `personal_preference` | native/home cuisine match 1, else preferred-cuisine 0.7, else 0.4 | poor | native |
| `missing_ingredient_burden` | missing non-core, non-staple items / recipe non-staple count | none missing | all optional extras missing |
| `shopping_burden` | 1 if any missing CORE (should already be REJECT); else 0.5 if any missing IMPORTANT; else 0 | no shop | would need shop |
| `cooking_burden` | `min(1, ready_in_minutes / time_budget)` if time_budget set; else `min(1, ready_in_minutes / 90)` | quick | long |
| `opportunity_cost` | 1 - future_kitchen_value (so consuming scarce enabling ingredients is expensive) | no future damage | high future damage |
| `confidence` | min confidence of observations used to claim CORE/safety facts | unusable | high |

`unused_scan_count` and raw `missed_count` are **not** score components.

## Weighted rank (`cook-rank-v1`)

Only PASS candidates.

```
rank_score =
    w_feasibility          * feasibility
  + w_haul                 * haul_utilisation
  + w_ingredient_value     * ingredient_value_used
  + w_coherence            * culinary_coherence
  + w_future               * future_kitchen_value
  + w_waste                * waste_avoidance
  + w_cuisine              * cuisine_fit
  + w_locale               * localization_fit
  + w_household            * household_fit
  + w_personal             * personal_preference
  - w_missing              * missing_ingredient_burden
  - w_shop                 * shopping_burden
  - w_cook                 * cooking_burden
  - w_opportunity          * opportunity_cost
```

`cook-rank-v1` weights (YAML/config, not source literals):

```yaml
ranking_version: cook-rank-v1
weights:
  w_feasibility: 0.16
  w_haul: 0.22
  w_ingredient_value: 0.10
  w_coherence: 0.08
  w_future: 0.10
  w_waste: 0.06
  w_cuisine: 0.06
  w_locale: 0.04
  w_household: 0.05
  w_personal: 0.05
  w_missing: 0.04
  w_shop: 0.02
  w_cook: 0.01
  w_opportunity: 0.09
diversity_threshold: 0.35
role_satisfaction_reject_below: 0.34
waste_risk_threshold: 0.60
culinary_distance:
  w_ingredient_set: 0.40
  w_cooking_method: 0.20
  w_flavour_profile: 0.25
  w_dish_family: 0.15
```

Let’s Cook: cuisine intent unset → `cuisine_fit` uses the 0.5 rule; do not zero `w_cuisine` in code. International: cuisine/region are candidate-generation constraints **and** `cuisine_fit` / `localization_fit` scores.

## Tie-breakers (stable, in this order)

1. Higher `haul_utilisation`
2. Higher `future_kitchen_value`
3. Lower `missing_ingredient_burden`
4. Higher `personal_preference` (native)
5. Lexicographic `recipe_id`

## Diversity / reranking

After rank_score sort, greedy select while `culinary_distance(candidate, already_picked) > diversity_threshold` (`0.35` in cook-rank-v1).

```
culinary_distance(a, b) =
    w_ingredient_set   * ingredient_set_distance(a, b)
  + w_cooking_method   * cooking_method_distance(a, b)
  + w_flavour_profile  * flavour_profile_distance(a, b)
  + w_dish_family      * dish_family_distance(a, b)
```

`culinary_distance` is a weighted combination of ingredient-set distance, cooking-method distance, flavour-profile distance and dish-family distance; **weights are versioned in ranking configuration** (`cook-rank-v1` values above). Each component is in **[0, 1]** (1 = different). Ingredient-set distance = `1 - Jaccard(canonical_ids)`. Cooking-method, flavour-profile and dish-family distances = 0 if the labels are equal, 1 if either is missing or they differ. Do not invent another set of weights.

If the pool is exhausted before `k`, fill from remaining by rank_score (do not invent a second threshold).

## Penalties already covered

Do not add extra ad-hoc penalties (e.g. `-0.2 if pasta`). Household unsafe is a gate. Diabetes caution is a badge, not a rank penalty, unless it is a **hard** dietary exclude in HouseholdContext.

---

# Uncertainty

**System-wide rule:** unknown ≠ safe. Insufficient provenance ≠ PASS.

## Bands

| Band | Confidence `c` | Behaviour |
|---|---|---|
| HIGH | `c >= 0.85` | Normal processing |
| MEDIUM | `0.55 <= c < 0.85` | Confirmation required for new observations; reduced `confidence` component; cannot alone justify PASS on safety-critical facts |
| LOW | `c < 0.55` | Do not use for safety-critical decisions; do not fill CORE or allergen claims |

## Domain rules

| Fact | HIGH | MEDIUM | LOW / missing |
|---|---|---|---|
| Recognition | Canonicalise and rank | Ask user to confirm; keep in session only after confirm | Ignore for CORE/safety; may show as “unrecognised” / UNKNOWN_* |
| Quantity | Use in simulator | Use with wider leftover bands; lower ingredient_value | Simulator treats quantity as unknown; do not claim leftover grams |
| Allergen information | Safety PASS/REJECT as normal | VERIFY if recipe-side; confirm if observation-side | VERIFY / REJECT from safe list. Never PASS |
| Recipe ingredients | Feasibility as normal | Prefer VERIFY if CORE depends on them | Cannot PASS safety or core integrity |
| Substitutions | Apply if substitution rule is validated | Mark adapted; lower coherence slightly via missing provenance (do not invent a new weight) | Do not auto-substitute |
| Localization | Generate under profile | Weaker localization_fit (already in [0,1] table) | Do not claim “local” |
| Nutrition | May attach cached facts | Show with caveat | Do not invent nutrition; omit |
| Prepared-meal identity | Canonical meal if visual+textual evidence HIGH | Keep dish-family hypotheses; user confirm | Do not decompose a plated meal into raw ingredients as facts |
| Regional/local variant | Only if visual, textual, ingredient, preparation or context evidence HIGH | Hypothesis only; do not write as KitchenState | Do not assign region merely because the dish name is associated with a cuisine |
| Non-food image | Reject; do not enter cook session haul | Ask user | Reject; do not invent ingredients |
| Trajectory / waste / consumption prediction (V9.5) | May inform Use First ranking as PREDICTED | Show with confidence; never OBSERVED | Do not drive safety or invent expiry |
| Entity match (V9.5) | Reconcile / update lineage | Confirm if ambiguous | Create uncertain entity or ask; do not inflate quantity |

Claim statuses: **OBSERVED | INFERRED | PREDICTED | UNCERTAIN**. LLM must not override these bands. **Prediction ≠ Safety determination.**

---

# Visual Understanding

KitchenIQ recognition must distinguish raw ingredients, packaged food, prepared/cooked meals, recipe documents, and non-food images before downstream processing.

KitchenIQ recognises **supported** food entities when evidence is sufficient and explicitly handles unknown or uncertain observations when evidence is insufficient. Vision is an **observation generator**, not the authority for KitchenState.

Prepared-meal recognition must not be treated as ordinary ingredient recognition.

## Required capabilities

The recognition pipeline must support:

- **Raw ingredient recognition** — e.g. tomato, onion, chicken.
- **Prepared-meal recognition** — e.g. biryani, pasta, curry, soup, sandwich.
- **Multi-component meal recognition** — identify multiple visible dishes/components when present.
- **Food-state recognition** — raw, partially prepared, cooked, plated, leftover, packaged/ready-to-eat where visually inferable.
- **Dish-level identification** — identify the most likely canonical dish rather than incorrectly decomposing a prepared meal into raw ingredients.
- **Regional/localization recognition** — determine likely cuisine, region, state/province, and relevant local variant where evidence supports it.
- **Uncertainty handling** — never convert visual uncertainty into fact. Maintain candidate hypotheses and confidence.
- **Non-food rejection** — recognise when the image is not a food/recipe/ingredient input.
- **Mixed-input handling** — an image may contain both ingredients and prepared food; each must be independently classified.

## Visual pipeline

This pipeline is **inside Capture**. It is not the cook Decision Engine pipeline. “Candidate generation” here means visual hypotheses (ingredient IDs or dish IDs), not recipe retrieval.

```
Image
→ Visual Classification
→ Food-State Detection
→ Ingredient / Meal Recognition
→ Visual Candidate Generation
→ Canonicalisation
→ Localization (hypothesis)
→ Confidence
→ Validation
→ KitchenIQ Decision Engine
```

Prepared-meal path:

```
Image → Prepared Meal → Biryani → Regional Candidate(s) → Localization Confidence → Canonical Meal
```

Do not send a plated biryani through the raw-ingredient recogniser and then treat “rice, onion, chicken” as the haul unless the image also independently shows those raw items.

## Hierarchical recognition

When sufficient evidence exists:

```
Food → Prepared/Raw → Dish Family → Dish → Regional Variant → Local Variant
```

Stop at the deepest level that the evidence and confidence bands support. Do not invent a local variant from a dish name prior.

## Evidence and confidence

The model must preserve the original visual evidence and recognition confidence so downstream systems can distinguish:

- **observed facts** (what is visible)
- **inferred attributes** (likely dish, likely state)
- **uncertain hypotheses** (regional variant, exact recipe)

Localization must operate on both ingredients and prepared dishes. A detected dish should not automatically be assigned a regional identity merely because its name is associated with a cuisine. Regional classification must use available visual, textual, ingredient, preparation and contextual evidence.

Do not use a generic image classifier as the final decision layer. Recognition must be recipe/food-domain-specific and integrate with KitchenIQ's existing canonicalisation, confidence, localization and household-state architecture.

## Offline improvement

This layer must support continuous offline improvement through:

- validated user corrections
- evaluation datasets
- difficult-case datasets
- controlled model versioning (`food-vision-vN`)

Individual user corrections must never directly modify production model weights. Corrections enter **Untrusted**, then **Learning Quarantine** until validated. Only **Curated Training** data, registered in the Model + Dataset Registry, may train. That path is defined in **Predictive & Adaptive ML Security Architecture**: User Correction → Validation → Abuse/Poisoning Detection → Quarantine → Dataset Quality Assessment → Training → Security Evaluation → Challenger Model → Shadow Testing → Controlled Promotion.

---

# Predictive & Adaptive ML Security Architecture

KitchenIQ Recipe Intelligence must treat every external input, model interaction, dataset contribution and user correction as potentially hostile. Security must protect not only against known and historical attacks, but also detect, model and prepare for emerging and previously unseen attack patterns.

The security architecture must operate as a continuous defensive loop:

```
Threat Intelligence
→ Threat Forecasting
→ Attack Simulation
→ Preventive Controls
→ Runtime Detection
→ Containment
→ Recovery
→ Security Learning
```

Protective Security ML continuously detects and prioritises risk across the system; deterministic security controls and a policy engine enforce containment, while governed deployment controls prevent unsafe autonomous changes.

## 1. Secure Input Boundary

Every uploaded image, screenshot, PDF, document, URL or other input must pass through a security boundary before reaching the Recipe Intelligence pipeline.

The system must:

- validate file type and actual file content
- enforce size, dimension and processing limits
- detect malicious or malformed files
- sandbox document/image parsing
- prevent parser exploitation
- isolate untrusted content from internal services
- prevent malicious URLs from reaching internal infrastructure
- apply authentication, authorization and rate limits
- prevent resource-exhaustion attacks

No untrusted input may directly access the model, database, internal network or training pipeline.

`POST /api/v1/cook/session/{id}/vision` and any future document/URL ingest must fail closed at this boundary. Flutter never talks to Gemini, Spoonacular or training stores.

## 2. Protective ML Layer

Protective Security ML continuously detects and prioritises risk across the system; deterministic security controls and a policy engine enforce containment, while governed deployment controls prevent unsafe autonomous changes.

KitchenIQ must use a dedicated protective ML/security layer to identify abnormal and potentially malicious behaviour that deterministic security rules may not recognise.

It should analyse:

- input characteristics
- request behaviour
- account behaviour
- API usage patterns
- model interaction patterns
- repeated recognition failures
- abnormal correction patterns
- dataset anomalies
- model-performance anomalies
- infrastructure behaviour

The protective model must detect both known attack signatures and behavioural deviations associated with previously unseen attacks.

This layer is **not** the Recipe Intelligence ranker and **not** the visual classifier. It does not decide dishes. It does not enforce Allow / Quarantine / Block / Ask. It emits **risk and evidence** for the Policy Engine.

## 3. Policy Engine

Enforcement sits after Protective ML, not inside it.

```
Protective ML → Risk/Evidence → Policy Engine → Allow / Quarantine / Block / Ask
```

The Policy Engine is deterministic. It consumes Protective ML risk scores, evidence, input-boundary results and standing security policy. It is the only component that may choose:

- **Allow** — continue processing
- **Quarantine** — isolate the input, correction or artefact; do not train on it; do not write KitchenState
- **Block** — reject and stop
- **Ask** — request user confirmation before continuing

Protective ML must not change policy, weights or infrastructure. Policy changes require governed deployment.

## 4. Output Security Validation

Every Recipe Intelligence ML output must pass a deterministic validation chain before it is a result:

```
ML output → schema → ingredient/unit validation → safety validation → result
```

- **schema** — required fields, types, enums, confidence bands
- **ingredient/unit validation** — canonical ids, units, quantities; no free-text ingredients as facts
- **safety validation** — allergens, hard constraints, unknown ≠ safe
- **result** — only a payload that passed all three stages may be returned or written

Failed validation is not a recipe. Outcomes are Block, Ask, Quarantine, or `unable_to_determine`. Unknown must never become invented information.

## 5. Data Zones

Data must not mix across zones.

| Zone | Holds | May train production models |
|---|---|---|
| **Untrusted** | uploads, URLs, screenshots, PDFs, raw API payloads, unvalidated corrections | no |
| **Operational** | KitchenState, sessions, confirmed observations, rankings served to the user | no |
| **Learning Quarantine** | suspicious or coordinated corrections, anomalous datasets, failed security eval | no |
| **Curated Training** | provenance-checked, quality-evaluated, security-evaluated labelled data | yes, only this zone |

Untrusted never writes Operational as fact. Operational never copies into Curated Training. Learning Quarantine promotes to Curated Training only through the training path in section 8.

## 6. Model + Dataset Registry

All production models and datasets are registered before use. The registry records:

- version
- provenance
- integrity (hash / signature)
- evaluation (quality + security)
- rollback target

A model or dataset must not enter production because it scores better on a limited test set. Promotion requires a registry entry that passed quality evaluation **and** security evaluation. Rollback uses the previous registry version, not an unregistered artefact.

## 7. Adversarial ML Defence

The Recipe Intelligence models must be tested and protected against attacks designed specifically to manipulate ML systems, including:

- adversarial image manipulation
- recognition evasion
- model confusion
- model extraction attempts
- inference attacks
- training-data poisoning
- malicious dataset contamination
- malicious feedback/correction attacks
- automated probing intended to discover model behaviour

Security evaluation must be performed continuously against an evolving adversarial test set.

A model must not enter production on quality metrics alone. It must pass **quality evaluation + security evaluation**.

## 8. Training & Feedback Protection

User corrections must never directly modify production model weights.

The learning path must be:

```
User Correction
→ Validation
→ Abuse/Poisoning Detection
→ Quarantine
→ Dataset Quality Assessment
→ Training
→ Security Evaluation
→ Challenger Model
→ Shadow Testing
→ Controlled Promotion
```

Suspicious or coordinated corrections must be isolated in **Learning Quarantine** rather than incorporated into training data.

Training datasets in **Curated Training** must maintain provenance, versioning, integrity checks and lineage in the Model + Dataset Registry so that contaminated data can be identified and removed.

## 9. Predictive Threat Intelligence

Security must not depend only on previously known vulnerabilities.

KitchenIQ must continuously analyse:

- emerging attack techniques
- changes in ML attack patterns
- vulnerabilities affecting dependencies and infrastructure
- newly discovered model attacks
- abnormal system behaviour
- historical attack trends
- weaknesses discovered through internal security testing

The system should generate forward-looking threat scenarios and estimate which attack paths are most relevant to the current architecture.

These predicted threats must become new security tests and defensive requirements.

## 10. Continuous Attack Simulation

The security system must regularly simulate realistic attacks in an isolated environment.

Examples include:

```
malicious input → parser → ML model → API → database → training pipeline
```

The objective is to identify attack paths before an attacker can exploit them.

Discovered weaknesses must produce:

```
Finding → Risk Assessment → Remediation → Retest → Security Baseline Update
```

Simulations must not run against production KitchenState, production weights or real user data.

## 11. Zero-Day / Unknown-Threat Defence

The system must not assume that every threat has a known signature.

Unknown threats should be detected through:

- behavioural anomaly detection
- statistical deviation
- request-pattern analysis
- model-output anomalies
- unexpected data distributions
- unusual user/account behaviour
- infrastructure anomalies
- cross-layer correlation

When confidence of malicious behaviour is high, the Policy Engine must Block or Quarantine rather than Allow processing to continue.

## 12. Safe Containment & Fallback

When security or model integrity becomes uncertain:

```
Normal Processing
→ Detection
→ Risk Classification
→ Isolation
→ Safe Fallback
→ Evidence Preservation
→ Recovery
```

The fallback must never silently produce an untrusted recipe result.

Possible Policy Engine outcomes are:

- **Allow** — use a validated deterministic path, or a result that passed Output Security Validation
- **Ask** — request user confirmation
- **Block** — temporarily reject the input
- **Quarantine** — isolate the input
- return a safe “unable to determine” result

Unknown must never become invented information. This is the same rule as Uncertainty: unknown ≠ safe, unknown ≠ a recipe.

## 13. Model Integrity & Supply-Chain Security

All production models, datasets, dependencies and build artifacts must be recorded in the **Model + Dataset Registry** with:

- version control
- provenance
- integrity verification
- evaluation (quality + security)
- controlled promotion
- dependency security scanning
- access control
- rollback capability
- audit history

A model or dataset must not enter production simply because it performs better on a limited test set.

It must pass both:

Quality evaluation + Security evaluation.

Unregistered artefacts must not serve production.

## 14. Autonomous Security Changes — Safety Rule

The protective ML system may detect, predict, simulate and recommend security changes.

It must not independently modify production model weights, security policies or critical infrastructure solely from its own prediction.

Production changes require controlled validation and deployment.

## 15. Security Learning Loop

The security system continuously learns from:

Attacks + Near Misses + False Positives + False Negatives + Red-Team Results + Threat Intelligence + Model Failures

but learning must occur through controlled datasets and evaluated model versions.

Production security must therefore evolve without allowing attackers to teach the system directly.

## Core Security Principle

KitchenIQ must be designed to defend against what is known, detect what is unknown, anticipate what is emerging, and improve its defences without allowing the learning system itself to become an attack surface.

---

# AUTHORITATIVE SPECIFICATION

**The section `AUTHORITATIVE SPECIFICATION` at the end of this document supersedes any conflicting architecture, pseudocode, ordering, scoring or API described earlier in this document.** Parts I and II remain research context. Part 0 (V9.5 Product & Engine) is product-authoritative. Cursor and implementers must follow this section when they disagree. The specification is not complete without section **J (API contracts)**, **K (Visual understanding)**, **L (ML security)**, and **M–U (V9.5 predictive household food system + integrity)**. Do not invent endpoints, payloads, a generic-classifier final layer, or a path that lets untrusted input reach models, databases or training.

**Document Revision 31** locks **KitchenIQ Architecture Specification V9.5**. V9.4 security + recognition + cook-engine rules in A–L remain in force unless M–U explicitly extends them. Predictions never become OBSERVED facts. Autopilot never overrides Safety / Policy Engine. There is exactly **one** authoritative `KitchenState` and exactly **one** authoritative Safety/Policy Engine.

## A. Boundary

Flutter is presentation/client only: camera, scan UI, ingredient cards, meal cards, manual add/remove, cuisine selection, region/localization selection, loading, results, recipe detail, user actions, confirmation of MEDIUM-band detections.

Flutter must not contain: scoring, allergy logic, canonicalisation, cuisine reasoning, opportunity cost, simulation, ranking, LLM prompts, API keys, database credentials, **visual classification as the final decision**, food-state or regional identity as fact, protective-ML scoring, Policy Engine decisions, or security-policy changes.

All of that lives in the backend cook engine. Flutter talks only to KitchenIQ APIs.

## B. Pipeline (only order)

v9.4 cook decision order (unchanged once candidates exist):

```
Capture (Visual Understanding)
→ Canonicalise
→ Entity Reconciliation (V9.5 — before KitchenState mutation)
→ Household Context
→ Candidate Generation
→ Safety Gate
→ Feasibility
→ Culinary Compatibility
→ State Simulation
→ Future Value
→ Personal / Household Fit
→ Ranking
→ Diversity / Reranking
→ Explanation
→ Output Security Validation
```

V9.5 household-intelligence stack (feeds candidates / Use First; does not bypass Safety):

```
Vision Engine V2 + Household Context
→ Household Food State
→ Food Trajectory Engine + Household Behaviour Model
→ Kitchen Twin
→ Future State Simulation
→ Candidate Actions / Use First Autopilot
→ Safety Gate → … → Ranking → Explanation → Output Security Validation
```

No Recipe Intelligence result is returned until Output Security Validation passes: schema → ingredient/unit validation → safety validation → result.

Security decisions on inputs use:

```
Protective ML → Risk/Evidence → Policy Engine → Allow / Quarantine / Block / Ask
```

Capture itself is:

```
Image
→ Visual Classification
→ Food-State Detection
→ Ingredient / Meal Recognition
→ Visual Candidate Generation
→ Canonicalisation
→ Localization hypotheses
→ Confidence
→ Validation
```

Then Entity Reconciliation against KitchenState. The Decision Engine starts at Household Context (and may consult Trajectory / Kitchen Twin for candidates). Do not collapse Visual Candidate Generation into recipe Candidate Generation.

Do not start the pipeline at Safety. Do not adapt International recipes only after ranking. Cuisine and localization constrain **recipe Candidate Generation**, then still pass every later stage. Visual localization is a **hypothesis** on the observation, not an automatic cuisine_intent.

**Vision is an observation generator, not the authority for KitchenState.**

## C. Objects

- `KitchenState` = physical food only (items, quantities, storage, timestamps).
- `HouseholdContext` = members, hard/soft constraints, home place, preferences.
- `DerivedKitchenFeatures` = perishability, freshness (time+storage), scarcity, future optionality, culinary role value, predicted waste, **derived versatility**.
- `CanonicalIngredient` = identity, aliases, category, roles, allergens, substitutions, cuisine/region links, nutrition ref. **No versatility field. No scarcity field.**
- `CanonicalMeal` = dish identity (dish family, dish, optional regional/local variant). Prepared meals canonicalise here, not as a bag of raw ingredients.
- `VisualObservation` = one classified region of an image (or the whole image): visual_class, food_state, hypotheses, evidence_ref, model_version, fact/inference/uncertain split.
- **V9.5:** `FoodEntity`, `FoodState` / `FoodStateHistory`, `FoodObservation`, `FoodLineage`, `FoodTrajectory`, `HouseholdBehaviour`, `FutureScenario`, `ActionCandidate`, `Decision`, `Outcome`.
- Predictions on trajectories, waste, consumption, scenarios and actions carry confidence and status **PREDICTED** (never OBSERVED).
- **Household Food State** (product term) is the understood household food picture; it is **derived from** KitchenState + validated observations. It is **not** a second KitchenState write authority.

## D. Simulator

Simulate remaining quantities, consumption, leftovers, stranded items. Do not multiply freshness by recipe cooking time. Freshness/expiry from elapsed time and storage only.

Never mutate real KitchenState during recommendation. Mutate only after an explicit cook confirmation event.

## E. Scoring

Use Scoring Specification `cook-rank-v1` only. Safety is a gate. unknown allergen ≠ safe. No extra weights. Diversity uses `culinary_distance` weights from the same config. `ingredients_used_near_spoilage` is consumed items whose `predicted_waste_risk` exceeds `waste_risk_threshold`.

## F. Uncertainty

Use the Uncertainty section. LOW confidence must not drive safety-critical or CORE decisions.

Evidence / claim statuses: **OBSERVED | INFERRED | PREDICTED | UNCERTAIN**. Predictions must never be written as OBSERVED facts into KitchenState.

Open-world visual outcomes: **KNOWN | UNKNOWN_FOOD | UNKNOWN_NONFOOD | UNCERTAIN | CONFLICTING_EVIDENCE**. The system must not force a closed-set class when evidence is insufficient.

## G. LLM

May: ambiguous language, unusual names, explanations from structured scores, controlled substitution proposals when retrieval fails, instruction localisation, descriptions.

Must not: decide allergy safety, override gates, rank, invent nutrition, mark an ingredient safe, write KitchenState.

## H. International

Let’s Cook: cuisine optional; native/home is personal_preference only.

International: user chooses cuisine then region (culinary region, not a required GPS country/city form). That intent is `CuisineIntent` + `LocalizationIntent` for **candidate generation**. Do not rename a ranked Western dish as the adaptation step.

## I. Implementation rule

One bounded phase at a time. After each phase: tests, API contracts, existing behaviour, checkpoint. Do not rewrite the whole app. Extend `ingredient_engine.py` into domain services; do not move the engine into Dart.

Existing `POST /api/v1/ingredients/*` may remain during migration. New cook intelligence must use the contracts in **J**. Flutter must not keep calling ad-hoc scan ranking once the cook session APIs exist.

## J. API contracts

Base path: `/api/v1`. JSON. Auth required. All timestamps ISO-8601 UTC. `ranking_version` and `engine_version` are returned, never invented by the client.

### POST `/api/v1/cook/session`

Start a cook session.

Request:

```json
{
  "path": "lets_cook",
  "target_scope": "HOUSEHOLD",
  "member_ids": []
}
```

| Field | Type | Required | Notes |
|---|---|---|---|
| path | `"lets_cook"` \| `"international"` | yes | |
| target_scope | `"SELF"` \| `"HOUSEHOLD"` \| `"SELECTED_MEMBERS"` | yes | |
| member_ids | string[] | if SELECTED_MEMBERS | |

Response `201`:

```json
{
  "ok": true,
  "session_id": "c0ffee",
  "path": "lets_cook",
  "target_scope": "HOUSEHOLD",
  "status": "capturing",
  "engine_version": "cook-engine-v1"
}
```

### POST `/api/v1/cook/session/{id}/ingredients`

Add, confirm, correct or remove observations. Body is a batch.

Request:

```json
{
  "observations": [
    {
      "client_observation_id": "obs-1",
      "raw_name": "coriander",
      "quantity_estimate": null,
      "unit": null,
      "confidence": 0.96,
      "source": "camera",
      "action": "upsert"
    }
  ]
}
```

| Field | Type | Required |
|---|---|---|
| observations[].client_observation_id | string | yes |
| observations[].raw_name | string | yes unless action is `remove` |
| observations[].quantity_estimate | number \| null | no |
| observations[].unit | string \| null | no |
| observations[].confidence | number 0–1 | yes for camera |
| observations[].source | `"camera"` \| `"manual"` \| `"correction"` | yes |
| observations[].action | `"upsert"` \| `"confirm"` \| `"remove"` | yes |

Response `200`:

```json
{
  "ok": true,
  "session_id": "c0ffee",
  "ingredients": [
    {
      "observation_id": "obs-1",
      "raw_name": "coriander",
      "canonical_id": "coriander_leaf",
      "display_name": "Coriander leaf",
      "confidence": 0.96,
      "confidence_band": "HIGH",
      "needs_confirmation": false,
      "quantity_estimate": null,
      "unit": null
    }
  ]
}
```

Recognition is not truth until `confirm` when band is MEDIUM. This endpoint is for **confirmed haul items** (ingredients or accepted meals). Images go through **POST `/api/v1/cook/session/{id}/vision`** first.

### POST `/api/v1/cook/session/{id}/vision`

Classify one image. Does not mutate KitchenState. Does not treat a prepared meal as a raw-ingredient list. The request **must** pass the Secure Input Boundary (type/content validation, size/dimension limits, malware/malformed detection, sandboxed parse, authz, rate limit) **before** any visual model. Failure is fail-closed: no model call, no DB write, no training ingest.

Request:

```json
{
  "image_base64": "<base64>",
  "client_image_id": "img-1"
}
```

| Field | Type | Required |
|---|---|---|
| image_base64 | string | yes |
| client_image_id | string | yes |

Response `200`:

```json
{
  "ok": true,
  "session_id": "c0ffee",
  "image_id": "img-1",
  "evidence_ref": "vision/sha256-…",
  "model_version": "food-vision-v1",
  "rejected": false,
  "observations": [
    {
      "observation_id": "vis-1",
      "visual_class": "prepared_meal",
      "food_state": "plated",
      "kind": "meal",
      "raw_label": "biryani",
      "hierarchy": {
        "food": true,
        "prepared_or_raw": "prepared",
        "dish_family": "biryani",
        "dish": "chicken_biryani",
        "regional_variant": null,
        "local_variant": null
      },
      "hypotheses": [
        {
          "canonical_id": "chicken_biryani",
          "type": "meal",
          "confidence": 0.81,
          "evidence": ["visual"]
        }
      ],
      "localization": {
        "cuisine": "indian",
        "region": null,
        "state": null,
        "local_variant": null,
        "confidence": 0.31,
        "status": "HYPOTHESIS"
      },
      "components": [],
      "confidence": 0.81,
      "confidence_band": "MEDIUM",
      "fact_vs_inference": {
        "observed": ["plated rice dish", "visible meat"],
        "inferred": ["likely chicken biryani"],
        "uncertain": ["regional variant"]
      }
    }
  ]
}
```

| `visual_class` | Meaning |
|---|---|
| `raw_ingredient` | Uncooked ingredient |
| `packaged_food` | Packaged / labelled food |
| `prepared_meal` | Cooked/plated/leftover dish |
| `recipe_document` | Recipe card, page, screenshot of a recipe |
| `non_food` | Reject |
| `mixed` | Image-level flag; children are independent observations |

| `food_state` | When allowed |
|---|---|
| `raw` \| `partially_prepared` \| `cooked` \| `plated` \| `leftover` \| `packaged_ready_to_eat` | Only if visually inferable; else `null` |

`localization.status` is `FACT` only on HIGH evidence from visual/textual/ingredient/preparation/context — **never** from dish-name-to-cuisine lookup alone.

If `visual_class` is `non_food`, `rejected` is true, `observations` may be empty, error is not required. Do not invent ingredients.

Mixed image: one parent `mixed` optional; each ingredient and each prepared component is its own observation with its own class, state, hypotheses and confidence.

Prepared meal: `kind` = `meal`, `canonical_id` is a `CanonicalMeal`. Do not auto-expand to raw ingredients in the haul. Optional `components[]` are **visible** extra items only, each independently classified.

`model_version` is required. User corrections of these observations are stored as events; they must not update production weights.

### POST `/api/v1/cook/recommendations`

Run the pipeline. Does not mutate kitchen inventory.

Request:

```json
{
  "session_id": "c0ffee",
  "target_scope": "HOUSEHOLD",
  "ingredients": [
    {
      "canonical_id": "chicken_meat",
      "raw_name": "chicken",
      "quantity_estimate": 1.0,
      "unit": "kg",
      "confidence": 0.96
    }
  ],
  "cuisine_intent": {
    "style": "indian",
    "region": "chettinad"
  },
  "localization_intent": {
    "profile_id": null,
    "country": null,
    "region": null,
    "city": null
  },
  "household_context": {
    "member_ids": [],
    "hard_allergens": ["peanut"],
    "hard_exclusions": [],
    "hard_dietary": [],
    "home_country": "India",
    "home_region": "Tamil Nadu"
  },
  "limit": 6
}
```

| Field | Type | Required | Notes |
|---|---|---|---|
| session_id | string | yes | Must exist |
| target_scope | enum | yes | SELF / HOUSEHOLD / SELECTED_MEMBERS |
| ingredients | object[] | yes | Canonicalised session haul; empty array is invalid |
| ingredients[].canonical_id | string | yes | |
| ingredients[].raw_name | string | yes | |
| ingredients[].quantity_estimate | number \| null | no | |
| ingredients[].unit | string \| null | no | |
| ingredients[].confidence | number | yes | |
| cuisine_intent | object \| null | no | Required for `international`; omit or null for Let’s Cook |
| cuisine_intent.style | string | if object | e.g. indian |
| cuisine_intent.region | string \| null | no | Culinary region, not GPS |
| localization_intent | object \| null | no | Optional; not a required country/city gate |
| household_context | object | yes | Server still reloads members; client may send cache |
| limit | int | no | Default 6, max 8 |

Response `200`:

```json
{
  "ok": true,
  "session_id": "c0ffee",
  "ranking_version": "cook-rank-v1",
  "engine_version": "cook-engine-v1",
  "suggested": [ { "$ref": "CookCandidate" } ],
  "more": [ { "$ref": "CookCandidate" } ],
  "blocked": [ { "$ref": "CookCandidate" } ]
}
```

`CookCandidate`:

```json
{
  "candidate_id": "cand-01",
  "recipe": {
    "recipe_id": "local-chicken-biryani",
    "title": "Chicken biryani",
    "ready_in_minutes": 50,
    "image": null,
    "source": "local"
  },
  "safety": {
    "status": "PASS",
    "reason": null
  },
  "feasibility": {
    "role_satisfaction": 1.0,
    "core_integrity": 1.0,
    "optional_gap": 0.1,
    "missing_core": [],
    "explanation": []
  },
  "score_components": {
    "feasibility": 0.97,
    "haul_utilisation": 1.0,
    "ingredient_value_used": 0.82,
    "culinary_coherence": 0.88,
    "future_kitchen_value": 0.61,
    "waste_avoidance": 0.40,
    "cuisine_fit": 0.5,
    "localization_fit": 0.5,
    "household_fit": 1.0,
    "personal_preference": 1.0,
    "missing_ingredient_burden": 0.0,
    "shopping_burden": 0.0,
    "cooking_burden": 0.55,
    "opportunity_cost": 0.39,
    "confidence": 0.96,
    "rank_score": 0.71
  },
  "state_after": {
    "as_of": "2026-09-09T12:00:00Z",
    "items": [
      {
        "canonical_id": "chicken_meat",
        "quantity": 0.6,
        "unit": "kg",
        "storage": "fridge"
      }
    ],
    "stranded": []
  },
  "explanation": [
    "Uses the full haul including rice and yoghurt.",
    "Leaves spinach unused for a later meal."
  ],
  "confidence": 0.96,
  "adaptations": {
    "cuisine_adapted": false,
    "substitutions": [],
    "title_original": null
  }
}
```

| Response field | Required | Notes |
|---|---|---|
| candidate_id | yes | Stable for this session+ranking_version |
| recipe | yes | Identity for UI |
| safety | yes | PASS / REJECT / VERIFY |
| feasibility | yes | Structured; not a single magic score |
| score_components | yes | Every cook-rank-v1 component plus `rank_score` |
| state_after | yes | Simulated KitchenState; not persisted |
| explanation | yes | Strings from structured data; LLM may polish, not invent scores |
| confidence | yes | Min band used for CORE/safety facts |
| adaptations | yes | Substitutions / cuisine rewrite flags |

Blocked candidates still return `safety` + `feasibility` + `candidate_id` + `recipe`; `score_components` may be omitted if REJECT at the safety gate.

### POST `/api/v1/cook/decision`

User action after recommendations. This is the only path that may mutate real KitchenState.

Request:

```json
{
  "session_id": "c0ffee",
  "candidate_id": "cand-01",
  "action": "cooked",
  "substitutions": []
}
```

| Field | Type | Required |
|---|---|---|
| session_id | string | yes |
| candidate_id | string | yes |
| action | `"shown"` \| `"opened"` \| `"started"` \| `"cooked"` \| `"completed"` \| `"skipped"` \| `"rejected"` | yes |
| substitutions | object[] | no |

Response `200`:

```json
{
  "ok": true,
  "session_id": "c0ffee",
  "candidate_id": "cand-01",
  "action": "cooked",
  "kitchen_state_mutated": true,
  "engine_version": "cook-engine-v1"
}
```

`kitchen_state_mutated` is true only for `cooked` / `completed`. Simulation from recommendations is never a mutation.

Error shapes (all cook APIs):

```json
{ "ok": false, "error": "session_not_found", "message": "..." }
```

| HTTP | error |
|---|---|
| 400 | `ingredients_required`, `invalid_target_scope`, `invalid_path`, `cuisine_intent_required`, `image_required`, `non_food_image`, `malformed_input` |
| 401 | `unauthorized` |
| 403 | `forbidden` |
| 404 | `session_not_found`, `candidate_not_found` |
| 409 | `stale_session` |
| 413 | `payload_too_large` |
| 415 | `unsupported_media_type` |
| 422 | `input_quarantined`, `unable_to_determine` |
| 429 | `rate_limited` |
| 503 | `provider_unavailable` (deterministic local fallback still attempted for recommendations **only if** the security boundary passed) |

Security reject and quarantine must not return a fabricated recipe. Use `unable_to_determine` or `input_quarantined`.

## K. Visual understanding (authoritative)

Supersedes Part I Stage 1 if it implied “every photo is raw ingredients.”

1. Classify **before** recognising names: raw ingredient, packaged food, prepared meal, recipe document, non-food, mixed.
2. Prepared meals take the path Image → Prepared Meal → dish hypotheses → regional candidates only with evidence → localization confidence → CanonicalMeal. They are not ordinary ingredient recognition.
3. Hierarchical stop: Food → Prepared/Raw → Dish Family → Dish → Regional Variant → Local Variant, only as deep as evidence+bands allow.
4. Preserve evidence_ref, model_version, observed vs inferred vs uncertain. Uncertainty never becomes KitchenState fact.
5. Localization on dishes and ingredients uses visual, textual, ingredient, preparation and contextual evidence. Dish-name cuisine association is not enough to write a region.
6. Mixed images: classify each region independently.
7. Non-food: reject; do not create haul items.
8. Generic ImageNet-style classifiers are not the final decision layer. Domain food/recipe models only, then KitchenIQ canonicalisation, confidence, localization and household architecture.
9. Offline improvement via corrections, eval sets, hard cases, versioned models. **User corrections never write production weights.**
10. Flutter displays hypotheses and asks for confirmation. The Decision Engine consumes only validated observations.

## L. Predictive & adaptive ML security (authoritative)

Supersedes any implication that vision or ranking APIs may call models with raw uploads, that corrections train production weights, or that Protective Security ML protects the entire system by itself.

1. Treat every external input, model interaction, dataset contribution and user correction as potentially hostile.
2. Defensive loop only: Threat Intelligence → Threat Forecasting → Attack Simulation → Preventive Controls → Runtime Detection → Containment → Recovery → Security Learning.
3. Secure Input Boundary before Recipe Intelligence. No untrusted input to model, database, internal network or training pipeline.
4. Protective Security ML continuously detects and prioritises risk across the system; deterministic security controls and a policy engine enforce containment, while governed deployment controls prevent unsafe autonomous changes. It does not rank recipes and does not enforce Allow / Quarantine / Block / Ask.
5. Enforcement path: Protective ML → Risk/Evidence → Policy Engine → Allow / Quarantine / Block / Ask. Policy Engine is deterministic.
6. Output Security Validation after Recipe Intelligence ML: ML output → schema → ingredient/unit validation → safety validation → result. Failed validation is not a recipe.
7. Four data zones only: Untrusted, Operational, Learning Quarantine, Curated Training. Only Curated Training may train production models.
8. Model + Dataset Registry required for version, provenance, integrity, evaluation and rollback. Unregistered artefacts must not serve production.
9. Adversarial evaluation is continuous (evasion, extraction, inference, poisoning, malicious corrections, probing).
10. Learning path: User Correction → Validation → Abuse/Poisoning Detection → Quarantine → Dataset Quality Assessment → Training → Security Evaluation → Challenger Model → Shadow Testing → Controlled Promotion. Coordinated/suspicious corrections stay in Learning Quarantine.
11. Predictive threat intelligence produces new tests and controls, not silent production edits.
12. Attack simulation only in isolation. Finding → Risk Assessment → Remediation → Retest → Security Baseline Update.
13. Unknown threats: anomaly/correlation → Policy Engine Block or Quarantine when malice confidence is high. Do not keep processing as normal.
14. Uncertain integrity: isolate, preserve evidence, safe fallback. Never silently emit an untrusted recipe. Policy outcomes: Allow, Ask, Block, Quarantine, or `unable_to_determine`. Unknown is never invented.
15. Quality evaluation **and** security evaluation required before registry promotion.
16. Protective ML may recommend changes. It must not autonomously change production weights, security policies or critical infrastructure.
17. Security learning uses controlled datasets and versioned models. Attackers must not teach production directly.
18. Core principle: defend known, detect unknown, anticipate emerging, improve defences without making the learner an attack surface.

## M. V9.5 Product North Star (authoritative)

KitchenIQ is a predictive household food system. It is not primarily a pantry scanner, food-recognition demo, recipe generator, meal planner, or grocery-list application — those are supporting components. Primary description: **food-specific household intelligence system**.

MVP: observe a kitchen → establish sufficiently reliable food state → identify high-priority trajectories → produce a **safe Use First** decision that improves the next household action.

## N. Food Trajectory Engine (authoritative)

1. Inventory is not static: each `FoodEntity` has current state, history, predicted trajectory, risk, potential outcomes, and recommended action.
2. Trajectory outputs are **PREDICTED** (with confidence + evidence). They never become OBSERVED facts.
3. **Prediction ≠ Safety.** Trajectory may say “likely consumed tomorrow”; Safety Gate alone decides consume-safety.
4. Use First Autopilot may recommend / prioritise / simulate / update plans. It must not override allergy gates, invent expiry, convert uncertainty into fact, or silently modify confirmed user data.

## O. Entity Lineage & Reconciliation (authoritative)

1. Maintain `FoodLineage` across purchase → package → opened → portion → prepared → leftover → consumed/discarded.
2. Every new observation: Canonicalise → Entity Match → Update or Create → Lineage → State transition.
3. Prevent duplicate entities, quantity inflation/replacement, scan-duplication, cooked-as-new-raw, and opened-package identity breaks.

## P. Kitchen Twin & Future Simulation (authoritative)

1. Kitchen Twin = current food state + trajectories + behaviour + planned meals + shopping + predicted future — not an inventory screen alone.
2. Future simulation: Current KitchenState → Scenario → Future KitchenState → Compare Outcomes. Scenario IDs: `BUY`, `DO_NOT_BUY`, `COOK_NOW`, `COOK_LATER`, `FREEZE`, `USE_FIRST`, `SUBSTITUTE`, `SKIP_MEAL`, `CHANGE_MEAL`, `HOUSEHOLD_AWAY`.
3. “What should we do now?” and “What happens if…?” are first-class product interactions. Answers must use Twin/simulation structured outputs, not unconstrained chatbot invention.
4. Kitchen Twin **reads** KitchenState; it does not replace KitchenState as the operational authority.

## Q. Vision Engine V2 constraints (authoritative extension of K)

1. Vision generates observations; KitchenState authority remains backend reconciliation + user confirmation bands.
2. Open-world abstention required (UNKNOWN_*/UNCERTAIN/CONFLICTING_EVIDENCE).
3. Identity ⊥ State (multi-label state attributes allowed).
4. Evidence fusion must not masquerade channels (unavailable ≠ verified).
5. Retain Vision V2 data strategy; **add temporal/trajectory datasets** (t0/t1/t2 + outcome).

## R. Household Behaviour & Learning (authoritative)

1. `HouseholdBehaviour` models observed patterns with provenance; predicted behaviour is separate and confidence-tagged.
2. Prediction-error learning records forecast vs actual (consumption, waste, Use First outcomes).
3. Learning path remains: Validated Learning Record → Quarantine → … → Controlled Promotion. **No direct user-behaviour → production weights.**
4. Zones unchanged: Untrusted, Operational, Learning Quarantine, Curated Training.

## S. Ranking objective (authoritative extension of E)

Candidate generation may use Trajectory + Twin + Behaviour. Ranking may weigh near-term utilisation, waste avoidance, future meal preservation, shopping avoidance, household fit — **after** Safety Gate. Safety remains a hard gate, not a score.

## T. Acceptance & metrics (authoritative)

1. Keep real-world vision acceptance. Add Product Intelligence Acceptance:

```
LEVEL 1 MODEL → 2 VISION → 3 FOOD ENTITY → 4 KITCHEN STATE
→ 5 TRAJECTORY → 6 DECISION → 7 OUTCOME
```

2. Mandatory **USE-FIRST END-TO-END TEST** across multiple observations over time.
3. Metrics include vision (scene completeness, unknown handling, …), product intelligence (reconciliation, trajectory, Use First acceptance, …), and business/product outcomes (waste/purchase reduction, meals from existing food, …).
4. Top-1 / mAP alone never declare KitchenIQ product-ready.

## U. Document integrity & implementation status (authoritative)

### U.1 Integrity checklist (must remain true)

1. Exactly **one** authoritative `KitchenState` (operational physical food). Twin/Trajectory are derived / predictive layers.
2. Exactly **one** authoritative Safety / Policy Engine (deterministic). Autopilot and Trajectory cannot override it.
3. Flutter remains presentation/client only.
4. ML / vision never directly mutate KitchenState; mutations require validated observations, reconciliation, confirmation bands, and explicit cook/confirm events as already specified in A–J.
5. Predictions (`PREDICTED`) cannot become facts without an explicit state-transition / evidence process.
6. User corrections remain quarantined before training (Learning Quarantine → Curated Training → … → Controlled Promotion).
7. Holdout data remains protected and must not be mutated for training or to force PASS.
8. Production model replacement only via registry challenger/shadow/promotion — never by silent overwrite.
9. Existing API contracts in **J** remain; new V9.5 product APIs (Use First / What-if) are **SPECIFIED** for a future phase and must not break J.
10. Prepared-meal protection, confidence bands HIGH/MEDIUM/LOW, Secure Input Boundary, and Output Security Validation remain in force.

### U.2 Implementation Status Matrix (as of Document Revision 31)

| Capability | Status |
|---|---|
| Secure Input Boundary | IMPLEMENTED |
| Fail-closed security / Policy Engine | IMPLEMENTED |
| Cook Engine pipeline (safety→…→ranking→explanation→output validation) | IMPLEMENTED (core) |
| KitchenState (operational) | IMPLEMENTED |
| Confidence bands + VisualObservation evidence model | IMPLEMENTED |
| Canonicalisation / prepared-meal contracts | IMPLEMENTED |
| Learning Quarantine + Curated Training zones | IMPLEMENTED |
| Model + Dataset Registry / select_servable / promotion | IMPLEMENTED |
| Vision detector + DINOv2 classifier production artefacts | PARTIALLY IMPLEMENTED |
| Real-world dense vision acceptance | BLOCKED |
| OCR / barcode production providers | NOT IMPLEMENTED (stub / channel only) |
| Food Trajectory Engine | SPECIFIED / NOT IMPLEMENTED |
| Food Entity Lineage | SPECIFIED / NOT IMPLEMENTED |
| Entity Reconciliation (first-class) | SPECIFIED / NOT IMPLEMENTED |
| Kitchen Twin | SPECIFIED / NOT IMPLEMENTED |
| Future-State Simulation (product scenarios) | SPECIFIED / NOT IMPLEMENTED |
| Use First Autopilot | SPECIFIED / NOT IMPLEMENTED |
| Household Behaviour Model | SPECIFIED / NOT IMPLEMENTED |
| Prediction-error learning | SPECIFIED / NOT IMPLEMENTED |
| Product Intelligence Acceptance Levels 1–7 | SPECIFIED / NOT IMPLEMENTED |
| USE-FIRST END-TO-END automated acceptance | SPECIFIED / NOT IMPLEMENTED |
| Temporal trajectory evaluation harness | SPECIFIED / NOT IMPLEMENTED |
| Expanding flat closed-set classifier as final architecture | REJECTED (do not pursue as V9.5 direction) |

### U.3 Conflicts resolved in this revision

| Topic | Resolution |
|---|---|
| “Household Food State” vs KitchenState | Product view derived from KitchenState; KitchenState remains sole write authority |
| Cook State Simulation vs Kitchen Twin | Cook simulator (v9.4) remains for recipe leftover/stranded simulation; Twin scenarios are the broader product what-if layer — both allowed; Twin must not invent a second KitchenState |
| COOK_TOMORROW vs COOK_LATER | Canonical ID = `COOK_LATER` |
| Vision “owns” inventory | Rejected — vision is observation-only |
| Predictions as safety | Rejected — Prediction ≠ Safety |

### U.4 No V9.6 / no parallel redesign

This document is the locked V9.5 implementation contract. Do not spawn a competing architecture document or a speculative V9.6 redesign in place of implementing against this specification.

