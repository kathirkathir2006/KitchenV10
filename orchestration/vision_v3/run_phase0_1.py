"""Phase 0–1: audit current 0.2927 system + 41-scene failure decomposition + substantial gate."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

NEW_ROOT = Path(r"C:\Projects\KitchenIQ-V10-AI")
OLD_ROOT = Path(r"C:\Projects\KitchenIQ-OS")
DOCS = NEW_ROOT / "docs" / "v3"
EXP = NEW_ROOT / "backend/instance/dev_experiments/v10-vision-v3"
ACC = (
    NEW_ROOT
    / "backend/instance/dev_experiments/v10-vision-recovery/acceptance/identity_coverage_v1_acceptance.json"
)
REQ = NEW_ROOT / "docs/vision/required_identity_vocabulary_v1.json"
COV = NEW_ROOT / "docs/vision/production_identity_coverage_v1.json"
MATRIX_PREV = NEW_ROOT / "docs/vision/identity_failure_coverage_matrix.json"

BASELINE = 0.2927
N_SCENES = 41
N_COMPLETE_BASE = 12  # 0.2927 * 41
N_INCOMPLETE = N_SCENES - N_COMPLETE_BASE
CLASSIFIER_FAILS_BASE = 40

# Substantial gate FIXED BEFORE any V3 evaluation (do not move after seeing results).
# Marginal band explicitly rejected by brief: 0.30–0.34.
# Derive from failure population:
#   - clear marginal ceiling (0.34 ≈ 14/41) by requiring ≥18/41 (+6 absolute scenes)
#   - major cut of classifier failures: ≥50% reduction (40 → ≤20)
SUBSTANTIAL_MIN_STRICT = 18 / 41  # 0.439024...
SUBSTANTIAL_MAX_CLASSIFIER_FAILURES = 20
SUBSTANTIAL_MIN_DELTA = round(SUBSTANTIAL_MIN_STRICT - BASELINE, 4)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


FAMILY_MAP = {
    "tomato": "produce",
    "onion": "produce",
    "potato": "produce",
    "carrot": "produce",
    "garlic": "produce",
    "ginger": "produce",
    "pepper": "produce",
    "cucumber": "produce",
    "lettuce": "produce",
    "apple": "produce",
    "banana": "produce",
    "orange": "produce",
    "lemon": "produce",
    "cheese": "dairy",
    "packaged_cheese": "packaged_dairy",
    "egg": "protein",
    "fried_egg": "prepared_egg",
    "omelette": "prepared_egg",
    "cooked_rice": "rice_grain",
    "fried_rice": "rice_grain",
    "biryani": "rice_grain",
    "pasta": "grain_starch",
    "bread": "grain_starch",
    "pizza": "prepared_meal",
    "salad": "prepared_meal",
    "sandwich": "prepared_meal",
    "soup": "prepared_meal",
    "curry": "prepared_meal",
    "doughnut": "prepared_sweet",
    "ice_cream": "prepared_sweet",
    "french_fries": "prepared_side",
    "human_hand": "non_food",
    "non_food": "non_food",
    "empty_plate": "non_food",
}


def classify_clf_subtype(gt: str, pred: str | None, obs: dict | None, iou: float) -> str:
    gt_n = (gt or "").lower().replace(" ", "_")
    pred_n = (pred or "").lower().replace(" ", "_") if pred else None
    obs = obs or {}
    status = (obs.get("status") or "").lower()
    visual = (obs.get("visual_class") or "").lower()
    is_food = obs.get("is_food")

    if iou < 0.1 and pred_n is None and status in {"non_food", ""} and is_food is False:
        return "C"  # unknown/nonfood forced — tomato→non_food pattern
    if status == "non_food" or visual == "non_food" or pred_n == "non_food":
        return "C"
    if pred_n is None:
        if gt_n in FAMILY_MAP and FAMILY_MAP[gt_n] == "prepared_meal":
            return "D"
        return "D"  # missing identity / abstain without match
    if FAMILY_MAP.get(gt_n) and FAMILY_MAP.get(pred_n) and FAMILY_MAP[gt_n] == FAMILY_MAP[pred_n] and gt_n != pred_n:
        return "B"
    if gt_n in {"cooked_rice", "fried_rice", "biryani"} and pred_n in {"rice", "cooked_rice", "fried_rice"}:
        return "B"
    if "raw" in (obs.get("food_state") or "") and gt_n.startswith("cooked"):
        return "F"
    if "cooked" in (obs.get("food_state") or "") and gt_n in {"tomato", "onion", "potato", "carrot"}:
        return "F"
    if FAMILY_MAP.get(gt_n) == "prepared_meal" and FAMILY_MAP.get(pred_n) not in {None, "prepared_meal", "prepared_egg", "prepared_sweet", "prepared_side", "rice_grain"}:
        return "G"
    if "packaged" in gt_n or "packaged" in (pred_n or ""):
        return "H"
    if iou > 0 and iou < 0.3:
        return "I"
    if pred_n and pred_n != gt_n:
        return "A"
    return "J"


def earliest_stage(fail: dict, obs_list: list[dict], scene: dict) -> str:
    stage = (fail.get("stage") or "").upper()
    gt = fail.get("gt")
    # find best overlapping obs
    matched_obs = None
    for o in obs_list:
        if (o.get("raw_label") or o.get("fine_label") or o.get("identity")) == fail.get("pred"):
            matched_obs = o
            break
    if not obs_list and stage == "DETECTOR":
        return "detector"
    if stage == "DETECTOR":
        return "detector"
    if stage == "FOOD_NONFOOD":
        return "food/non-food"
    # classifier with non_food obs → food/non-food earlier
    for o in obs_list:
        if o.get("is_food") is False or o.get("visual_class") == "non_food" or o.get("status") == "non_food":
            # if GT is food and only non_food obs → food/non-food
            if gt and gt not in {"human_hand", "non_food", "empty_plate"}:
                return "food/non-food"
    if stage == "CLASSIFIER":
        # if no identity and abstain → open-set / identity
        for o in obs_list:
            if o.get("status") in {"abstain", "unknown_food", "unable_to_determine"}:
                return "open-set decision"
        return "identity"
    return stage.lower() or "orchestration"


def build_failure_matrix(acc: dict) -> dict[str, Any]:
    rows = []
    subtype_c = Counter()
    stage_c = Counter()
    for scene in acc.get("scenes") or []:
        sid = scene.get("scene_id")
        obs = scene.get("observations") or []
        for fail in scene.get("failures") or []:
            gt = fail.get("gt")
            pred = fail.get("pred")
            iou = float(fail.get("iou") or 0)
            # attach an observation heuristically
            obs0 = obs[0] if obs else {}
            subtype = classify_clf_subtype(str(gt or ""), pred, obs0, iou) if (fail.get("stage") or "").upper() == "CLASSIFIER" else None
            early = earliest_stage(fail, obs, scene)
            if subtype:
                subtype_c[subtype] += 1
            stage_c[early] += 1
            rows.append(
                {
                    "scene_id": sid,
                    "category": scene.get("category"),
                    "gt": gt,
                    "pred": pred,
                    "iou": iou,
                    "reported_stage": fail.get("stage"),
                    "earliest_causal_stage": early,
                    "classifier_subtype": subtype,
                    "obs_status": obs0.get("status"),
                    "obs_identity": obs0.get("identity") or obs0.get("raw_label"),
                    "obs_visual_class": obs0.get("visual_class"),
                    "obs_food_state": obs0.get("food_state"),
                    "obs_is_food": obs0.get("is_food"),
                    "n_obs": scene.get("n_obs"),
                    "n_gt": scene.get("n_gt"),
                    "strict_scene_complete": scene.get("strict_scene_complete"),
                }
            )
    return {
        "version": "v3_failure_matrix_v1",
        "generated_at": now(),
        "baseline_strict": BASELINE,
        "source_acceptance": str(ACC),
        "n_rows": len(rows),
        "earliest_stage_counts": dict(stage_c),
        "classifier_subtype_counts": dict(subtype_c),
        "reported_stage_totals": acc.get("primary_stage_totals"),
        "substantial_improvement_gate": {
            "frozen_before_eval": True,
            "baseline_strict": BASELINE,
            "n_scenes": N_SCENES,
            "n_complete_baseline": N_COMPLETE_BASE,
            "n_incomplete_baseline": N_INCOMPLETE,
            "classifier_failures_baseline": CLASSIFIER_FAILS_BASE,
            "marginal_band_rejected": [0.30, 0.34],
            "min_strict_scene_completeness": round(SUBSTANTIAL_MIN_STRICT, 4),
            "min_absolute_delta": SUBSTANTIAL_MIN_DELTA,
            "min_complete_scenes": 18,
            "max_classifier_failures": SUBSTANTIAL_MAX_CLASSIFIER_FAILURES,
            "rationale": (
                "From 29 incomplete scenes and 40 CLASSIFIER failures: reject the brief's "
                "marginal band (≤0.34≈14/41). Require ≥18/41 (+6 scenes) and ≤20 classifier "
                "failures (≥50% reduction). Threshold frozen before any V3 candidate eval."
            ),
        },
        "rows": rows,
    }


def write_audit() -> None:
    cov_weights = (
        NEW_ROOT
        / "backend/instance/dev_experiments/v10-vision-recovery/models/identity_coverage_best.pt"
    )
    prod_clf = (
        OLD_ROOT
        / "backend/instance/dev_experiments/fv-production-final/registry_artifacts"
        / "food-vision-classifier-v1/classifier-prod-v1.0.0/model.pt"
    )
    md = f"""# KitchenIQ Vision V3 — Current System Audit

**Generated:** {now()}  
**Project:** `C:\\Projects\\KitchenIQ-V10-AI` (GitHub KitchenV10 only)  
**Old project:** READ-ONLY (`C:\\Projects\\KitchenIQ-OS`)

## Current detector
- Production KitchenIQ Faster R-CNN via `select_object_detector()` (old backend, read-only registry DB copy).
- Wired in `src/kitcheniq_v10/specialist/pipeline.py` and recovery ensemble.
- COCO torchvision FRCNN is fallback only — domain-wrong for 41-scene GT.
- **Must keep as baseline** for V3 Phase 3; RT-DETR is challenger only.

## Current classifier
- Baseline production: FoodVisionV2 + **DINOv2 ViT-B/14 frozen** multi-head (`visual_class`, `food_state`, `kind`, `label`), OI-62 vocab.
- Identity-coverage recovery candidate: DINOv2 ViT-B/14 frozen + **73-label** head (OI-62 ∪ trainable required).
  - Artifact: `{cov_weights}`
  - SHA256: `{sha256_file(cov_weights)}`
  - 41-scene strict: **0.2927** (12/41)
- Inference path: detect → crop → softmax top-1 label → specialist fusion → official observation schema.

## Current data flow
IMAGE → Faster R-CNN regions → FoodVisionV2 heads → `fuse_specialists` (fail-closed / abstain) → `obs_to_official` → `match_instances` / `scene_completeness`.

## Current APIs / harness
- Acceptance: `kitcheniq_v10.specialist.acceptance.run_acceptance`
- Official metrics imported from old project (read-only): `app.services.food_vision.real_world_acceptance.metrics`
- Fixtures: `backend/instance/dev_experiments/v10-specialist-ensemble/acceptance/fixtures/` (IMMUTABLE)
- Catalog SHA256: `{sha256_file(NEW_ROOT / 'backend/instance/dev_experiments/v10-specialist-ensemble/acceptance/fixtures/catalog/fixture_catalog.json')}`

## Current failure attribution (0.2927 run)
- CLASSIFIER = 40
- DETECTOR = 12
- FOOD_NONFOOD = 2
- Dominant blocker for incremental softmax heads remains **identity / open-set / food-gate**, not detector alone.

## Reusable components
- KIQ Faster R-CNN detector + registry load pattern
- Acceptance harness + official matchers (do not modify fixtures/scoring)
- Fusion fail-closed / prepared-meal protection patterns
- Production-eligible manifests + required identity vocabulary v1
- Canonical taxonomy CSV / entity interfaces (read-only from old where needed)

## Components that must be replaced for V3
- Single softmax “pick one label” as final identity
- Flat label space without hierarchical retrieval
- Missing evidence graph / competing hypotheses
- Missing calibrated open-set decisions (KNOWN / UNKNOWN_* / CONFLICTING / INSUFFICIENT)
- Missing cross-object relationship reasoning
- DINOv2-only representation → **DINOv3 primary frozen backbone** (controlled compare vs DINOv2 baseline)
- Optional RT-DETR region discovery (only if 41-scene region metrics improve)

## Security boundary (preserve)
- Fail-closed abstention, uncertainty bands, evidence provenance, holdout untouched, no KitchenState mutation, no allergen-safe claims, experiment-only isolation.
"""
    (DOCS / "V3_CURRENT_SYSTEM_AUDIT.md").write_text(md, encoding="utf-8")


def write_matrix_md(matrix: dict) -> None:
    gate = matrix["substantial_improvement_gate"]
    lines = [
        "# V3 Failure Matrix (41-scene @ 0.2927)",
        "",
        f"Generated: {matrix['generated_at']}",
        "",
        "## Substantial-improvement gate (FROZEN before V3 eval)",
        "",
        f"- min strict completeness: **{gate['min_strict_scene_completeness']}** ({gate['min_complete_scenes']}/41)",
        f"- min delta vs 0.2927: **{gate['min_absolute_delta']}**",
        f"- max classifier failures: **{gate['max_classifier_failures']}** (from {CLASSIFIER_FAILS_BASE})",
        f"- marginal band rejected: {gate['marginal_band_rejected']}",
        "",
        f"Rationale: {gate['rationale']}",
        "",
        "## Earliest causal stage counts",
        "",
        "```json",
        json.dumps(matrix["earliest_stage_counts"], indent=2),
        "```",
        "",
        "## Classifier subtype counts (A–J)",
        "",
        "```json",
        json.dumps(matrix["classifier_subtype_counts"], indent=2),
        "```",
        "",
        "## Reported stage totals",
        "",
        "```json",
        json.dumps(matrix["reported_stage_totals"], indent=2),
        "```",
        "",
        f"Rows: {matrix['n_rows']}. Full machine-readable: `docs/v3/v3_failure_matrix.json`.",
        "",
        "## Implication for candidate spend",
        "",
        "- Dominant earliest stages are **identity** / **food/non-food**, not detector-only.",
        "- Therefore Candidate E (retrieval + specialists + evidence graph + open-set) is primary.",
        "- Grounding DINO fallback is **optional** unless region-discovery rows dominate after E.",
        "- Do NOT run another flat softmax classifier cycle.",
    ]
    (DOCS / "v3_failure_matrix.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    EXP.mkdir(parents=True, exist_ok=True)
    write_audit()
    acc = json.loads(ACC.read_text(encoding="utf-8"))
    matrix = build_failure_matrix(acc)
    write_json(DOCS / "v3_failure_matrix.json", matrix)
    write_matrix_md(matrix)
    write_json(
        DOCS / "v3_substantial_gate.json",
        matrix["substantial_improvement_gate"],
    )
    write_json(
        EXP / "reports" / "phase0_1_summary.json",
        {
            "baseline": BASELINE,
            "n_rows": matrix["n_rows"],
            "earliest_stage_counts": matrix["earliest_stage_counts"],
            "classifier_subtype_counts": matrix["classifier_subtype_counts"],
            "substantial_gate": matrix["substantial_improvement_gate"],
            "generated_at": now(),
        },
    )
    print(json.dumps(matrix["substantial_improvement_gate"], indent=2))
    print("earliest", matrix["earliest_stage_counts"])
    print("subtypes", matrix["classifier_subtype_counts"])


if __name__ == "__main__":
    main()
