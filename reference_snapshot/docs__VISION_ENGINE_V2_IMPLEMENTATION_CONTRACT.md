# KitchenIQ — Vision Engine V2 Implementation Contract

**Document type:** Implementation contract + codebase gap analysis  
**Product architecture baseline:** KitchenIQ Architecture Specification V9.5 (Document Revision 31)  
**Contract version:** 1.0  
**Date:** 25 September 2026  
**Scope:** Vision Engine V2 only  
**Status vocabulary:** `SPECIFIED` | `IMPLEMENTED` | `PARTIALLY IMPLEMENTED` | `NOT IMPLEMENTED` | `BLOCKED` | `EXPERIMENTAL`

**Absolute constraints for this document’s producing task:** no code changes, no training, no dataset download, no weight/holdout changes, no V9.5 architecture-doc edits, no Trajectory / Twin / Use First / Behaviour work.

**Non-negotiable product rule:** Vision is an **observation generator**. Vision **MUST NOT** directly mutate authoritative `KitchenState`.

---

## 0. Executive summary

### Current production path (code)

```
POST /api/v1/cook/session/{id}/vision
→ Secure Input Boundary (evaluate_image_upload)
→ vision.orchestrator
→ KitchenIqRegistryVisionProvider
→ food_vision.production_algorithm.pipeline.run_production_algorithm
     → select_object_detector (Faster R-CNN food_region / non_food_region)
     → crop_region (+ IoU de-dup / max 12)
     → DINOv2 ViT-B/14 + multi-heads (FoodVisionV2) via inference._forward (cached)
     → fuse_observation_channels (visual-primary; OCR stub empty; barcode unused)
     → optional mixed parent when ≥2 isolated food regions
     → prepare CanonicalMeal / prepared_meal non-decomposition
→ validate_provider_result
→ persist VisualObservation (session-scoped)
→ optional canonicalise_observation / localization enrich
```

**Does not mutate KitchenState** on the vision route (confirmed in `routes/cook.py` session_vision docstring and decision route separation).

### Adequacy verdict

The current path is a **closed-world-leaning** detect → crop → fixed fine-label classifier stack with:

- Detector: 3-class Faster R-CNN (`background` / `food_region` / `non_food_region`) — **IMPLEMENTED**
- Classifier: DINOv2 ViT-B/14 heads-only on OI slug label space — **PARTIALLY IMPLEMENTED**
- Open-world abstention: LOW → `unable_to_determine` — **PARTIAL**; no first-class `UNKNOWN_FOOD` / `UNKNOWN_NONFOOD` / `CONFLICTING_EVIDENCE` outcomes
- OCR: `KitchenIqOcrStub` returns `[]` — **NOT IMPLEMENTED**
- Barcode decoder: fusion channel exists; no vision decoder — **NOT IMPLEMENTED**
- Real-world dense acceptance: **BLOCKED** (measured; classifier-dominant failures)

Vision Engine V2 **extends and restructures** this path; it does **not** discard Secure Input, registry, orchestrator persistence, or DINOv2 by default.

### DINOv2 decision (contract)

**KEEP DINOv2 ViT-B/14 as the visual representation backbone** unless a later measured requirement proves it cannot support hierarchical/open-world heads. Justification: already integrated (`model_arch.FoodVisionV2`), production heads artifact exists, V9.5 forbids replacing it merely for novelty. What must change is **orchestration + head/taxonomy + open-world policy + evidence channels**, not automatic backbone replacement.

---

## 1. Current-state map (from code)

| COMPONENT | CURRENT IMPLEMENTATION | OWNER | INPUT | OUTPUT | REUSABLE? | REPLACE? | EXTEND? | BLOCKER? |
|---|---|---|---|---|---|---|---|---|
| Secure Input Boundary | `security/input_boundary.evaluate_image_upload` | security | base64/url + auth | BoundaryResult ALLOW/BLOCK/QUARANTINE | YES | NO | YES (as needed) | No |
| Vision route | `routes/cook.py` `session_vision` | API | JSON image | observations JSON | YES | NO | YES (schema fields) | No |
| Orchestrator | `vision/orchestrator.py` | vision | bytes + boundary | VisualObservation rows | YES | NO | YES | No |
| Provider factory | `vision/factory.py` → registry provider | vision | config | VisionProvider | YES | NO | YES | No |
| Production pipeline | `food_vision/production_algorithm/pipeline.py` | food_vision | image bytes | ProductionAlgorithmResult | YES (core) | NO | YES → V2 stages | Partial (dense) |
| Detector | Faster R-CNN `detection/torch_detector.py` + `arch.py` | food_vision.detection | RGB tensor | DetectionResult[] food/non_food | YES | NO | YES (classes/NMS) | Density/small-object |
| Null detector | `detection/null_detector.py` | detection | — | empty dets | YES | NO | NO | Fail-closed OK |
| Detector registry | `detection/registry.select_object_detector` | detection + ml_registry | model_id | ObjectDetector | YES | NO | YES | DB must have PRODUCTION |
| Crop | `detection/registry.crop_region` | detection | bbox | JPEG crop bytes | YES | NO | YES (padding policy) | No |
| Classifier backbone | DINOv2 ViT-B/14 via torch.hub | food_vision.model_arch | image tensor | 768-d embed | YES | NO (default) | YES | Hub/runtime dep |
| Classifier heads | FoodVisionV2 multi-heads (visual_class, food_state, kind, fine_label) | model_arch + inference | embed | logits | YES | NO | YES (hierarchy/open-world) | Label space mismatch |
| Inference serve | `inference.run_inference` / `run_region_inference` + process cache | food_vision | crop/whole | VisionProviderResult | YES | NO | YES | Closed-set force risk |
| Calibration | `food_vision/calibration.py` + registry cal JSON | food_vision | raw conf | calibrated conf | YES | NO | YES | Detector ECE high |
| Confidence bands | `cook_domain/confidence.py` HIGH≥0.85 MED≥0.55 | cook_domain | float | band | YES | NO | YES (predicted status) | Thresholds locked |
| Evidence fusion | `cook_domain/evidence_fusion.py` | cook_domain | channels | FusedEvidence | YES | NO | YES (status enums) | OCR/barcode empty |
| Multi-object contracts | `cook_domain/multi_object.py` | cook_domain | obs fields | MultiObjectObservation | YES | NO | YES | — |
| Canonicalisation | `cook_domain/canonicalisation/*` | cook_domain | observation | canonical hyp | YES | NO | YES | — |
| Localization | `cook_domain/localization/*` | cook_domain | obs + session | hyp only | YES | NO | YES | — |
| Validator | `vision/validator.py` | vision | ProviderObservation | validated list | YES | NO | YES (open-world) | — |
| VisualObservation ORM | `models/visual_observation.py` | models | fields | DB row | YES | NO | YES (open-world/status) | — |
| ProviderObservation DTO | `vision/contracts.py` | vision | — | DTO | YES | NO | YES | — |
| OCR | `vision/ocr.KitchenIqOcrStub` | vision | image | `[]` | Interface YES | Stub REPLACE w/ provider | YES | Missing OCR |
| Barcode | fusion channel only; no decoder | evidence_fusion | optional string | boost if present | Channel YES | NEW decoder | YES | Missing barcode |
| Gemini path | `legacy_gemini_adapter.py` | vision | — | opt-in legacy | KEEP opt-in only | DEPRECATE as prod default | NO | Must never be prod authority |
| Stub provider | `kitcheniq_stub.py` | vision | — | fail-closed tests | YES | NO | NO | — |
| ML registry | `ml_registry/engine.select_servable_model` | ml_registry | model_id | artifact path | YES | NO | YES (V2 versions) | — |
| Real-world acceptance | `food_vision/real_world_acceptance/*` | food_vision | fixtures | metrics JSON | YES | NO | YES (V2 categories) | Gate BLOCKED |
| Holdout | `experiments/holdout.py` + policy JSON | experiments | IDs | eval-only | YES | NO | NO | Must not mutate |
| Model food_state labels | `FOOD_STATE_LABELS` = raw/prepared/packaged/processed/unknown | model_arch | — | mapped via `map_model_food_state` | PARTIAL | EXTEND head | YES | V9.5 multi-label gap |
| Cook-domain FOOD_STATES | raw, partially_prepared, cooked, plated, leftover, packaged_ready_to_eat | confidence.py | — | validator set | YES | EXTEND for cut/fried/… | YES | Attribute gap |

---

## 2. Current → V2 mapping

| CURRENT COMPONENT | → V2 COMPONENT | ACTION | Reason (if REPLACE/DEPRECATE) |
|---|---|---|---|
| Secure Input Boundary | Secure Input Boundary | KEEP | — |
| cook vision API | VisionRequest / VisionResult API | EXTEND | Add open-world + evidence status fields; keep route |
| orchestrator persist | Observation Assembly + persist | EXTEND | — |
| production_algorithm.pipeline | Vision Engine V2 orchestrator stages | EXTEND | Restructure stages; keep fail-closed |
| Faster R-CNN detector | Region / Instance Discovery (+ food/non-food hint) | EXTEND | Not sufficient alone for dense kitchen; keep as discovery backbone until challenger proves better |
| Detector food/non_food labels | Food / Non-Food Classification | EXTEND | May later split dedicated classifier; keep hints |
| DINOv2 backbone | Visual representation backbone | KEEP | Justified by existing integration |
| Flat fine_label head (closed set) | Hierarchical Fine Identity + abstention | EXTEND → eventual REPLACE of **head/taxonomy**, not backbone | Closed-set OI slugs force wrong known class; V2 requires null identity + hierarchy stop |
| food_state head (5 labels) | Food-State Recognition (multi-attribute) | EXTEND | V9.5 requires cut/fried/baked/opened/sealed/frozen etc. |
| KitchenIqOcrStub | OCR Evidence provider | REPLACE (implementation) | Stub returns empty — cannot satisfy package/OCR eval |
| (none) | Barcode Evidence provider | NEW | No decoder exists |
| evidence_fusion | Evidence Fusion V2 | EXTEND | Add available/unavailable/failed/conflicting |
| multi_object / mixed parent | Scene Understanding + Observation Assembly | EXTEND | — |
| canonicalisation | Canonicalisation | KEEP / EXTEND | — |
| VisualObservation | VisionObservation persist model | EXTEND | Open-world + predicted status fields |
| quantity_uncertain default True | Quantity / Instance Analysis | EXTEND | Today mostly UNKNOWN — need explicit instance count vs quantity |
| (none) | Entity Matching interface (output only) | NEW | Prep for reconciliation; vision must not own KitchenState IDs |
| legacy_gemini as prod | — | DEPRECATE as production authority | Commercial VLM not KitchenIQ vision authority |
| Expanding 207-class flat classifier as final arch | — | DEPRECATE as target | Rejected by V9.5 |
| real_world_acceptance_v1 | V2 Acceptance Contract harness | EXTEND | Add open-world / OCR / barcode / entity continuity |
| holdout policy | HOLDOUT zone | KEEP | Untouched |

---

## 3. V2 module boundaries (responsibilities + interfaces only)

Proposed package root (SPECIFIED, not created here):

`backend/app/services/food_vision/v2/`

Existing modules remain until migration; V2 stages may wrap them.

### A. Scene Understanding — `v2/scene.py`

**Responsibility:** Whole-image context: scene type (single/dense/mixed/document/non_food-dominant), estimated object density, lighting/quality flags (heuristic or model), whether multi-region processing is required.  
**Input:** decoded image + mime + evidence_ref.  
**Output:** `SceneContext` (scene_kind, density_band, quality_flags, notes).  
**Must not:** invent food identities; mutate KitchenState.

### B. Region / Instance Discovery — `v2/regions.py` (wraps detector)

**Responsibility:** Propose regions/instances with bboxes, scores, detector label hints; apply NMS/de-dup/max-k policy (versioned).  
**Input:** image bytes + SceneContext.  
**Output:** `VisionRegion[]`.  
**Must not:** assign fine food identity.

### C. Food / Non-Food Classification — `v2/food_gate.py`

**Responsibility:** Per-region food vs non-food vs document vs uncertain; may use detector hint + dedicated head.  
**Output:** `food_status` ∈ {food, non_food, document, uncertain}.  
**Must not:** map non-food to a pantry ingredient.

### D. Hierarchical Food Recognition — `v2/hierarchy.py`

**Responsibility:** Descend hierarchy only as deep as evidence+band allow; allow null fine identity.  
**Output:** hierarchy dict + `identity_status` (KNOWN|UNKNOWN_FOOD|UNCERTAIN|…).  
**Must not:** force deepest label; decompose prepared meals without independent regions.

### E. Food-State Recognition — `v2/food_state.py`

**Responsibility:** Independent multi-label state attributes; separate from identity.  
**Output:** `FoodStatePrediction` (attributes[], confidence, status).  
**Must not:** set state from dish name alone as FACT.

### F. OCR Evidence — `v2/ocr.py` (provider plug-in)

**Responsibility:** Extract text regions + confidences as **untrusted evidence**.  
**Output:** `EvidenceChannel(ocr)` with status available|unavailable|failed.  
**Must not:** treat OCR as KitchenState truth or safety fact.

### G. Barcode Evidence — `v2/barcode.py` (NEW)

**Responsibility:** Detect/decode barcodes/QR on packages when present.  
**Output:** `EvidenceChannel(barcode)` with status available|unavailable|failed + payload.  
**Must not:** claim barcode_verified when unavailable.

### H. Quantity / Instance Analysis — `v2/quantity.py`

**Responsibility:** Instance count vs quantity estimate; default UNKNOWN when insufficient.  
**Output:** `QuantityEvidence`.  
**Must not:** fabricate grams.

### I. Evidence Fusion — extend `cook_domain/evidence_fusion.py`

**Responsibility:** Fuse channels with provenance; mark conflicting; never masquerade channels.  
**Output:** `FusedEvidence` + per-channel status.

### J. Confidence / Uncertainty — extend `cook_domain/confidence.py` + calibration

**Responsibility:** Band assignment; abstention; calibration application; claim statuses OBSERVED|INFERRED|PREDICTED|UNCERTAIN.  
**Must not:** lower thresholds to force PASS.

### K. Canonicalisation — keep `cook_domain/canonicalisation`

**Responsibility:** Map validated observations → CanonicalIngredient / CanonicalMeal hypotheses.  
**Must not:** invent meal→ingredient bags.

### L. Observation Assembly — extend pipeline + orchestrator

**Responsibility:** Build `VisionResult` / ProviderObservation list; mixed parent optional; prepared_meal components rules; rejection flags.  
**Must not:** write KitchenState.

### M. Entity-Matching interface — `v2/entity_match_hints.py` (NEW, interface only)

**Responsibility:** Emit **hints** for future Entity Reconciliation (candidate refs, match evidence, new-entity recommendation).  
**Must not:** assign authoritative inventory entity IDs or mutate KitchenState.

---

## 4. Data contracts (schemas)

Preserve all existing `ProviderObservation` / `VisualObservation` fields unless obsolete. Additions are **additive**.

### 4.1 VisionRequest (API body — extends current)

```json
{
  "image_base64": "<base64>",
  "client_image_id": "img-1",
  "mime": "image/jpeg",
  "session_context": {}
}
```

Existing required fields remain. Optional future: `prior_observation_ids` (hints only).

### 4.2 VisionRegion

```json
{
  "region_id": "reg-…",
  "bbox": [x1, y1, x2, y2],
  "normalized": false,
  "detector_score": 0.0,
  "detector_label_hint": "food_region|non_food_region|null",
  "detector_model_version": "detector-v1.0.0",
  "isolation_status": "isolated|uncertain|whole_image"
}
```

### 4.3 VisionHypothesis

```json
{
  "canonical_id": "ing:tomato|meal:pizza|null",
  "type": "ingredient|meal|packaged|document|unknown",
  "label": "tomato",
  "confidence": 0.0,
  "evidence": ["visual", "ocr"],
  "hierarchy_level": "dish|category|…"
}
```

### 4.4 FoodState (prediction object)

```json
{
  "attributes": ["raw", "cut"],
  "primary": "raw",
  "confidence": 0.0,
  "confidence_band": "HIGH|MEDIUM|LOW",
  "status": "inferred|uncertain|observed",
  "model_version": "food-state-…"
}
```

Allowed attributes (V9.5): `raw`, `cut`, `mixed`, `cooked`, `fried`, `baked`, `plated`, `opened`, `sealed`, `frozen`, `leftover`, `unknown`  
(Map from existing cook-domain states: `partially_prepared`, `packaged_ready_to_eat` remain valid aliases during migration.)

### 4.5 Evidence / channel status

```json
{
  "channel": "VISION|OCR_TEXT|BARCODE_PACKAGE|CONTEXT|LOCALIZATION|PRIOR_OBSERVATION|USER_CONFIRMATION|PACKAGE",
  "status": "available|unavailable|failed|conflicting",
  "confidence": 0.0,
  "payload": {},
  "model_version": null
}
```

### 4.6 LocalizationEvidence

Existing `localization` object: `{ "status": "HYPOTHESIS", … }` — regional FACT forbidden from dish name alone. Keep.

### 4.7 QuantityEvidence

```json
{
  "instance_count": null,
  "quantity_estimate": null,
  "unit": null,
  "quantity_uncertain": true,
  "status": "uncertain|inferred|observed",
  "confidence": 0.0
}
```

### 4.8 Uncertainty

```json
{
  "identity_status": "KNOWN|UNKNOWN_FOOD|UNKNOWN_NONFOOD|UNCERTAIN|CONFLICTING_EVIDENCE|REJECTED",
  "confidence": 0.0,
  "confidence_band": "HIGH|MEDIUM|LOW",
  "fact_vs_inference": {
    "observed": [],
    "inferred": [],
    "uncertain": [],
    "predicted": []
  },
  "requires_confirmation": true
}
```

### 4.9 VisionObservation (assembled; extends ProviderObservation)

Must include: `client_observation_id`, `visual_class`, `kind`, `recognition`, `raw_label` (nullable when unknown), `food_state` / `food_state_prediction`, `hierarchy`, `hypotheses` / alternatives, `localization`, `components`, `fact_vs_inference` (+ predicted), `quantity_*`, `ocr_*`, `evidence_channels`, `evidence_ref`, `model_id`, `model_version`, `parent_client_observation_id`, `region`, `identity_status`, `entity_match_hint`, `rejected`.

**Null identity allowed:** `raw_label = null` or `""` with `identity_status = UNKNOWN_FOOD|UNCERTAIN` — **forbidden** to substitute a random known class.

### 4.10 VisionResult

```json
{
  "ok": true,
  "rejected": false,
  "reject_reason": null,
  "error": null,
  "evidence_ref": "vision/…",
  "model_id": "food-vision-classifier-v1",
  "model_version": "…",
  "detector_model_version": "detector-v1.0.0",
  "detector_available": true,
  "algorithm_version": "vision-engine-v2",
  "scene": {},
  "observations": [],
  "multi_object_evidence": {},
  "evidence_fusion": {},
  "engine_version": "cook-engine-v1"
}
```

Maps to existing `VisionProviderResult` + pipeline `to_dict()` with additive fields.

---

## 5. Open-world requirement (contract)

| Outcome | Meaning | Identity | KitchenState effect via vision alone |
|---|---|---|---|
| KNOWN | Supported identity with sufficient evidence | non-null canonical candidate allowed | none (observation only) |
| UNKNOWN_FOOD | Food-like; identity unsupported/abstained | **null** | none |
| UNKNOWN_NONFOOD | Non-food / clutter | null / non_food | reject haul |
| UNCERTAIN | Insufficient evidence to choose | null or alternatives only | confirmation required |
| CONFLICTING_EVIDENCE | Channels disagree | null / VERIFY | confirmation |
| REJECTED | Security or hard non-food / malformed | n/a | no observations / quarantined |

**Hard rule:** Classifier argmax over a closed label set **must not** be emitted as KNOWN when calibrated confidence &lt; MEDIUM or when open-world head says unknown. Prefer `unable_to_determine` / `UNKNOWN_*` / alternatives list.

Example (required shape):

```json
{
  "food_status": "food",
  "identity": null,
  "alternatives": ["ginger", "turmeric"],
  "status": "requires_confirmation",
  "identity_status": "UNCERTAIN"
}
```

---

## 6. Hierarchical recognition interface

```
Food
 → Food Type
 → Raw Ingredient | Prepared Meal | Packaged Food | Drink | Document | Non-Food
 → Category
 → Dish Family
 → Dish
 → Regional Variant
 → Local Variant
```

**Stop rule:** Fill only levels supported by evidence + confidence band (existing pipeline `_hierarchy_for_band` principle). Regional/local never FACT from dish name alone.

**Prepared meal rule:** `visual_class=prepared_meal` → `CanonicalMeal` path; `components=[]` unless independently detected regions exist for those ingredients. Independently visible items remain separate observations (salad tomatoes OK).

---

## 7. Food state (independent)

Identity ⊥ State. Multi-label attributes allowed (e.g. `opened` + `leftover`).

Migration: map model head labels through `map_model_food_state`; extend head/vocab to V9.5 attributes without collapsing identity into state.

---

## 8. Evidence fusion contract

Channels: VISION, OCR_TEXT, BARCODE_PACKAGE, PRIOR_OBSERVATION, PACKAGE, CONTEXT, LOCALIZATION, USER_CONFIRMATION, HOUSEHOLD (session).

Per channel: `available | unavailable | failed | conflicting`.

Rules:

- Missing OCR → `status: unavailable` (not verified).
- Conflicting OCR vs visual → CONFLICTING_EVIDENCE / VERIFY; do not silently average into false certainty.
- User confirmation is a separate channel after MEDIUM band rules.
- Fusion must not copy barcode payload into “visual observed” facts.

---

## 9. Entity Matching interface (vision → future reconciliation)

Vision emits **hints only**:

```json
{
  "observation_id": "…",
  "candidate_entity_refs": [],
  "match_evidence": ["bbox_overlap_prior", "label_similarity", "barcode"],
  "match_confidence": 0.0,
  "match_uncertainty": "UNCERTAIN",
  "recommend_new_entity": true
}
```

**Out of scope for Vision V2 implementation phase:** writing KitchenState, quantity mutation, lineage commits. Those belong to Entity Reconciliation (separate V9.5 workstream).

---

## 10. Security mapping (V9.4 → V2)

| Control | Current | V2 requirement | Gap? |
|---|---|---|---|
| Secure Input Boundary | IMPLEMENTED before provider | KEEP mandatory | No |
| Malformed / decode | pipeline `_decode_validate` | KEEP | No |
| Size/dimensions | boundary + decode limits | KEEP | Review max dims vs decompression bombs periodically |
| Sandboxing / parse isolation | PIL decode path | KEEP; no new parsers without isolation | Residual: ensure future OCR/barcode parsers sandboxed |
| Resource exhaustion | rate limit + size | KEEP | Monitor CPU time for dense multi-crop |
| SSRF | URL checks in boundary | KEEP for URL paths | No |
| AuthZ | boundary | KEEP | No |
| Rate limits | upload_rate_limiter | KEEP | No |
| Untrusted isolation | zones + no train from vision | KEEP | No |
| Output validation | validator + Output Security on recipes | Vision: validate_provider_result KEEP | Extend for open-world |
| Registry / integrity | select_servable + sha256 | KEEP for all V2 artefacts | No |
| Fail-closed | unable_to_determine | KEEP | No |
| Commercial VLM as authority | factory can select legacy_gemini | Must remain non-default / non-prod authority | **Gap:** misconfig risk if `VISION_PROVIDER=gemini` |

**Do not fix gaps in this contract task** — record only.

---

## 11. Model architecture contract (no training)

| Capability | Interface | Current | V2 |
|---|---|---|---|
| Scene / region discovery | Detector I/O DetectionResult | Faster R-CNN | KEEP; version NMS/max-k; challenger allowed later |
| Visual backbone | 768-d embed | DINOv2 ViT-B/14 | **KEEP** |
| Food / non-food | class or head | detector hints + visual_class head | EXTEND |
| Hierarchical recognition | multi-level logits / cascading | flat fine_label | EXTEND heads / taxonomy; stop rule |
| Food-state | multi-label | single 5-way head + map | EXTEND attributes |
| OCR | OcrProvider.extract | Stub | REPLACE stub with registered provider |
| Barcode | BarcodeProvider.decode | Missing | NEW |
| Quantity / instances | count + uncertain | uncertain default | EXTEND |
| Calibration | temperature / table | exists | KEEP; fit only on val; never holdout |
| Unknown / abstention | open-world / LOW band | PARTIAL | REQUIRED first-class |

**When may DINOv2 be replaced?** Only if measured evidence shows backbone limitation (e.g. cannot support required hierarchy/open-world after head/data fixes) and a challenger passes registry security+quality gates. Not in this contract phase.

---

## 12. Data contract (zones — acquire nothing now)

| Zone | Use |
|---|---|
| PRODUCTION-ELIGIBLE / CURATED_TRAINING | Only approved datasets for production weights |
| EXPERIMENT-ONLY | Food101 etc. — eval/experiment; never silent prod train |
| EVALUATION | real_world_vision_acceptance_v1 + V2 suites |
| HOLDOUT | Frozen; eval-only; no mutate; no calibration fit |

**Required category coverage (for future acquisition/eval — not now):** isolated food; dense kitchen; multi-item; overlap; food+non-food; packaged; raw; prepared; multi-prepared; duplicates; hard negatives; unfamiliar/unknown food; food-state; OCR/package; barcode; temporal pairs (for later entity continuity eval).

---

## 13. Acceptance contract (Vision V2)

**Harness base:** extend `real_world_vision_acceptance_v1` (KEEP).

**Categories (minimum):** single-item; multiple-item; dense kitchen; packaged; prepared meals; raw vs prepared; food/non-food; hard negatives; duplicates; quantity variation; occlusion; small objects; partial visibility; low light; motion blur; perspective; unknown food; state recognition; OCR; barcode; entity continuity (multi-observation).

**Metrics (no invented pass thresholds unless already locked elsewhere):** scene completeness; food recall; food/non-food performance; identity accuracy; unknown/abstention quality; state accuracy; duplicate handling; quantity accuracy; OCR performance; barcode performance; calibration (ECE/reliability); latency; regression (existing pytest suites).

**Existing locked band thresholds remain:** HIGH ≥ 0.85, MEDIUM ≥ 0.55, LOW &lt; 0.55 — do not lower to obtain PASS.

**Product gate reminder:** Vision PASS ≠ KitchenIQ product PASS (V9.5 Levels 1–7). This contract covers Vision levels 1–2 primarily.

---

## 14. Migration contract (no cutover now)

| Item | Spec |
|---|---|
| Old path | `KitchenIqRegistryVisionProvider` → `run_production_algorithm` (v9.4 algo version string) |
| New path | Same provider entrypoint → `run_vision_engine_v2` (or pipeline flag) |
| Compatibility | ProviderObservation / VisualObservation additive fields; API remains `/vision` |
| Registry | New model versions for heads/OCR/barcode; detector may keep detector-v1.0.0 until challenger |
| Serving manifest | Extend production_vision_serving_manifest with V2 stage versions |
| Shadow | Challenger V2 models in shadow; compare acceptance metrics |
| Promotion | Existing promote_to_production / select_servable only |
| Rollback | Previous PRODUCTION versions remain servable; feature flag `VISION_ENGINE=v1|v2` |
| Holdout | Untouched throughout |

---

## 15. Existing failure → V2 response matrix

Source evidence: `fv-real-world-acceptance-v1` taxonomy (CLASSIFIER-dominant, DETECTOR, FOOD_STATE, NON_FOOD) + pipeline/OCR/barcode code.

| Failure | CURRENT ROOT CAUSE | V2 DESIGN RESPONSE | DATA? | MODEL CHANGE? | ORCH.? | EVAL? |
|---|---|---|---|---|---|---|
| tomato recognition | Closed-set / domain mismatch; synthetic pilots; LOW/unable | Open-world + hierarchy; curated raw data later | YES | Heads/taxonomy EXTEND | YES abstention | YES |
| garlic / ginger | Same classifier slug confusion | Open-world alternatives; hard-neg pairs | YES | YES heads | YES | YES |
| rice | Confused with prepared rice dishes | Hierarchy raw vs prepared; prepared-meal protection | YES | YES | YES | YES |
| packaged cheese | Packaged vs fresh confusion; tiny pilots | Packaged path + OCR/barcode | YES | YES | YES fusion | YES |
| hard-negative pairs | Flat classifier | Hard-neg training + eval; abstention | YES | YES | YES | YES |
| multiple raw ingredients | Detector OK often; classifier fails; completeness | Scene completeness metrics; per-instance identity | YES | YES | YES | YES |
| multiple prepared meals | Partial detect/classify | Dense discovery + hierarchy dish level | YES | YES | YES | YES |
| cooked/plated/leftover | Weak food_state head labels | Independent multi-label state | YES | YES state head | YES | YES |
| recipe-document | Misclassified prepared_meal / unable | Document class + OCR; reject pantry invent | YES | YES | YES | YES |
| non-food rejection | False food regions + classifier | Stronger food/non-food + non_food_region hint (already partial) | YES | YES | YES | YES |
| duplicate instances | One-class≠one-instance | Instance analysis separate from identity | YES | MAYBE | YES | YES |
| quantity variation | quantity_uncertain always | Explicit QuantityEvidence UNKNOWN default | NO immediate | NO | YES | YES |
| dense scenes | Detector mAP ~0.36–0.40; completeness | Better discovery + eval; challenger detector if needed | YES | MAYBE detector | YES NMS | YES |
| OCR missing | Stub | OCR provider REPLACE | YES | YES OCR model/lib | YES | YES |
| barcode missing | No decoder | NEW barcode module | YES | YES | YES | YES |

---

## 16. Final gap report

### A. Current architecture
Detect (Faster R-CNN) → crop → DINOv2 multi-head closed-set labels → fusion (vision-only in practice) → validate → VisualObservation. KitchenState not mutated by vision.

### B. V2 target architecture
Secure Input → Scene → Regions → Food/Non-Food → Hierarchy → Food State → OCR/Barcode → Quantity → Entity-match hints → Fusion → Confidence → Canonicalisation → Observation output.

### C. Current → V2 mapping
See §2 (KEEP / EXTEND / REPLACE / DEPRECATE / NEW).

### D. Reusable components
Boundary, orchestrator, registry, detector stack, DINOv2 backbone, inference cache, fusion framework, multi_object, canonicalisation, localization, validator, VisualObservation, acceptance harness, confidence bands, holdout policy.

### E. Components requiring replacement
OCR stub implementation; closed-set “always emit a known fine label” behaviour; treating flat 207-class expansion as final architecture.

### F. New components required
Barcode provider; open-world identity status; hierarchical stop interface; quantity/instance module; entity-match hint interface; scene context module; V2 stage orchestration wrapper.

### G. API/schema changes
Additive fields on vision response / ProviderObservation / VisualObservation; no breaking removal of v9.4 fields.

### H. Model interface requirements
See §11. DINOv2 KEEP by default.

### I. Data requirements
See §12. No acquisition in this phase.

### J. Evaluation requirements
See §13. Extend real_world_vision_acceptance_v1.

### K. Security gaps
Misconfig risk for `VISION_PROVIDER=gemini`; future OCR/barcode parser isolation must be enforced when added; dense multi-crop CPU resource monitoring.

### L. Migration requirements
See §14. Shadow → challenger → promote; feature flag; rollback via registry.

### M. Failure → V2 matrix
See §15.

### N. Explicit blockers (before claiming Vision V2 production-ready)

1. Real-world acceptance overall **BLOCKED** / low scene completeness (measured).  
2. Classifier taxonomy / open-world abstention insufficient for proprietary kitchen labels.  
3. OCR **NOT IMPLEMENTED**.  
4. Barcode **NOT IMPLEMENTED**.  
5. Food-state attribute set incomplete vs V9.5 multi-label.  
6. No first-class entity-match hint contract in code yet.  
7. Holdout must remain frozen; no silent threshold lowering.

---

## 17. Dependency-ordered implementation sequence (DO NOT START)

1. **Contracts & schemas** — additive DTOs / identity_status / evidence channel status (API compatible).  
2. **Open-world policy in inference + validator** — null identity; forbid forced known class.  
3. **Hierarchy stop rules** — wire hierarchy levels to bands (extend existing).  
4. **Food-state attribute model** — extend mapping + head contract (train later).  
5. **Scene + region orchestration cleanup** — formalize SceneContext; keep detector.  
6. **Quantity / instance module** — explicit UNKNOWN.  
7. **Entity-match hint interface** — output only.  
8. **OCR provider** — replace stub behind registry (data+model later).  
9. **Barcode provider** — new (data+model later).  
10. **Evidence fusion status enums** — available/unavailable/failed/conflicting.  
11. **Acceptance harness extension** — unknown/OCR/barcode/entity continuity.  
12. **Challenger training** (Kaggle/CURATED only) — heads/taxonomy/state; DINOv2 keep unless measured.  
13. **Shadow + promote** via registry.  
14. **Only then** (separate workstreams): Entity Reconciliation consuming hints → Trajectory / Twin / Use First.

**Do not** jump to Food Trajectory, Kitchen Twin, or Use First Autopilot in the Vision V2 implementation phase.

---

## 18. Files created

| File | Role |
|---|---|
| `docs/VISION_ENGINE_V2_IMPLEMENTATION_CONTRACT.md` | Authoritative Vision V2 implementation contract + gap analysis (this document) |

**Not modified:** V9.5 architecture specification; production code; models; holdout; datasets.

---

## VISION V2 IMPLEMENTATION CONTRACT COMPLETE
