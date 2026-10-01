"""Vision V4 Phases 1–3: pipeline forensics + 41-scene first-stage root-cause (no training)."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

NEW_ROOT = Path(r"C:\Projects\KitchenIQ-V10-AI")
OLD_ROOT = Path(r"C:\Projects\KitchenIQ-OS")
DOCS = NEW_ROOT / "docs" / "v4"
ACC = (
    NEW_ROOT
    / "backend/instance/dev_experiments/v10-vision-recovery/acceptance/identity_coverage_v1_acceptance.json"
)
CATALOG = (
    NEW_ROOT
    / "backend/instance/dev_experiments/v10-specialist-ensemble/acceptance/fixtures/catalog/fixture_catalog.json"
)
WEIGHTS = (
    NEW_ROOT
    / "backend/instance/dev_experiments/v10-vision-recovery/models/identity_coverage_best.pt"
)

BASELINE = 0.2927
SUBSTANTIAL = 0.4390
IOU_MATCH = 0.3

NONFOOD_LABELS = {
    "non_food",
    "human_hand",
    "hand",
    "empty_plate",
    "utensil",
    "cookware",
}


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


def norm(s: str | None) -> str:
    return (s or "").lower().replace(" ", "_").strip()


def is_food_gt(label: str) -> bool:
    return norm(label) not in NONFOOD_LABELS and not norm(label).startswith("empty")


def build_forensics() -> dict[str, Any]:
    """Static end-to-end forensics of the current 0.2927 system from code + artifacts."""
    forensics = {
        "generated_at": now(),
        "baseline_strict": BASELINE,
        "weights": {
            "path": str(WEIGHTS),
            "sha256": sha256_file(WEIGHTS),
            "bytes": WEIGHTS.stat().st_size if WEIGHTS.is_file() else 0,
            "role": "identity_coverage_v1 FoodVisionV2 multi-head (DINOv2 ViT-B/14 frozen)",
        },
        "stages": [
            {
                "stage": "input",
                "model": None,
                "weights": None,
                "behaviour": "PIL RGB image from fixture path; JPEG bytes for detector",
                "threshold": None,
                "fail_closed": False,
                "notes": "No secure upload boundary in local acceptance path",
            },
            {
                "stage": "secure_boundary",
                "model": None,
                "weights": None,
                "behaviour": "Acceptance harness bypasses Flask secure input; production path uses OS food_vision routes",
                "threshold": None,
                "fail_closed": True,
                "notes": "DEAD in local 41-scene eval — not exercised",
            },
            {
                "stage": "image_preprocessing",
                "model": "resize 224 + ImageNet normalize",
                "weights": None,
                "behaviour": "Crop → RGB → 224x224 bilinear → mean/std",
                "threshold": None,
                "fail_closed": False,
                "notes": "Detector uses full-res JPEG bytes; classifier uses 224 crop",
            },
            {
                "stage": "detector",
                "model": "KitchenIQ production Faster R-CNN via select_object_detector()",
                "weights": "OS registry_final.db → food-vision detector artifact (read-only copy)",
                "behaviour": "Top-12 boxes by confidence; empty → whole-image fallback box",
                "threshold": "implicit detector score ranking; COCO FRCNN fallback score>=0.4 if KIQ load fails",
                "fail_closed": False,
                "fallback": "torchvision fasterrcnn_resnet50_fpn COCO (DOMAIN WRONG)",
                "notes": "Detector NOT retrained; 12 DETECTOR-stage failures at 0.2927",
            },
            {
                "stage": "crop_region_generation",
                "model": None,
                "behaviour": "Axis-aligned crop from bbox; no segmentation; no NMS beyond detector",
                "fail_closed": False,
                "notes": "Overlapping/small objects poorly served — no mask/proposal refinement",
            },
            {
                "stage": "food_nonfood",
                "model": "FoodVisionV2 visual_class head + identity top1 heuristics in SpecialistEnsemble",
                "weights": str(WEIGHTS),
                "behaviour": "visual_class==non_food OR top1 in {non_food,empty_plate,hand} → non_food",
                "threshold": "hardneg_strict mode raises non_food; coverage ensemble uses fusion rules",
                "fail_closed": "Aggressive non_food can veto true food (tomato→non_food observed)",
                "notes": "Acceptance often MISLABELS these as CLASSIFIER; first divergence is FOOD_NONFOOD",
            },
            {
                "stage": "classifier_identity",
                "model": "FoodVisionV2 fc_label over 73-class vocab (OI-62 ∪ trainable)",
                "weights": str(WEIGHTS),
                "behaviour": "softmax top-1; OOV → unknown in coverage path",
                "threshold": "temperature from blob calibration or 1.0; fusion MEDIUM/HIGH bands",
                "calibration": "temperature scalar if present in checkpoint",
                "fail_closed": "unsupported label → unknown (coverage)",
                "notes": "Closed-set softmax remains primary identity authority in 0.2927 system",
            },
            {
                "stage": "food_state",
                "model": "FoodVisionV2 fc_state + prepared specialist remaps",
                "weights": str(WEIGHTS),
                "behaviour": "raw/prepared/packaged/processed/unknown; prepared meals mapped toward plated",
                "fail_closed": False,
                "notes": "At 0.2927 food_state_agreement=1.0 on measured subset; V3 broke this to 0.0",
            },
            {
                "stage": "evidence_fusion",
                "model": "fuse_specialists deterministic rules",
                "weights": None,
                "behaviour": "food vs nonfood conflict; prepared labels; abstain bands",
                "ocr": "accepted as optional dict but unused in coverage acceptance path",
                "barcode": "accepted as optional dict but unused in coverage acceptance path",
                "fail_closed": True,
                "notes": "OCR/barcode are DEAD fields in current 41-scene path",
            },
            {
                "stage": "canonicalisation",
                "model": None,
                "behaviour": "obs_to_official maps FusedObservation → acceptance schema",
                "fail_closed": False,
                "notes": "No Food Entity canonicalisation engine in V10 src",
            },
            {
                "stage": "entity_matching",
                "model": None,
                "behaviour": "ABSENT in V10 — no runtime entity continuity",
                "fail_closed": False,
                "notes": "DEAD / not implemented",
            },
            {
                "stage": "final_observation",
                "model": "match_instances + scene_completeness (OS metrics, read-only)",
                "behaviour": "IoU match GT vs obs; failure_stage from matcher",
                "fail_closed": False,
                "notes": "Matcher may attribute FOOD_NONFOOD failures as CLASSIFIER when pred=null",
            },
        ],
        "duplicated_authorities": [
            "visual_class head AND identity label both gate food/non-food",
            "prepared specialist AND identity PREPARED_MEAL_LABELS set",
            "OS production classifier weights vs V10 coverage weights — only coverage used for 0.2927",
        ],
        "dead_or_unused": [
            "OCR evidence in fusion (None in acceptance)",
            "Barcode evidence in fusion (None in acceptance)",
            "Food Entity matching",
            "Secure input boundary in local eval",
            "Retrieval / evidence graph (V3 built but library n=0; not in 0.2927 path)",
        ],
        "stale_or_fallback_risks": [
            "COCO Faster R-CNN fallback if KIQ detector load fails",
            "Whole-image box when detector returns empty — creates false 'regions'",
        ],
    }
    return forensics


def first_stage_for_failure(
    *,
    gt_label: str,
    gt_bbox: list[float] | None,
    obs_list: list[dict[str, Any]],
    reported_stage: str | None,
    iou: float,
) -> str:
    """Earliest causal stage where reality diverges from GT."""
    gt_n = norm(gt_label)
    food_gt = is_food_gt(gt_n)

    # spatial: any obs with iou>0 recorded, or heuristic from reported stage
    if (reported_stage or "").upper() == "DETECTOR" or (iou < 1e-6 and not obs_list):
        return "DETECTOR"

    # If matcher says DETECTOR
    if (reported_stage or "").upper() == "DETECTOR":
        return "DETECTOR"

    # Prefer observation signals over reported CLASSIFIER
    # Case: observations exist but all non_food while GT is food
    if food_gt and obs_list:
        foodish = [
            o
            for o in obs_list
            if o.get("is_food") is True
            or norm(o.get("visual_class"))
            in {"raw_ingredient", "prepared_meal", "packaged_food", "packaged", "mixed"}
            or (
                norm(o.get("raw_label") or o.get("identity")) not in NONFOOD_LABELS
                and norm(o.get("status")) != "non_food"
            )
        ]
        if not foodish:
            # all obs non-food / abstain-as-nonfood
            nonfoodish = [
                o
                for o in obs_list
                if o.get("is_food") is False
                or norm(o.get("status")) == "non_food"
                or norm(o.get("visual_class")) == "non_food"
                or norm(o.get("raw_label") or o.get("identity")) in NONFOOD_LABELS
            ]
            if nonfoodish:
                return "FOOD_NONFOOD"

    # Unknown / abstain open-set
    if obs_list:
        for o in obs_list:
            st = norm(o.get("status"))
            if st in {"unknown_food", "abstain", "unable_to_determine", "conflicting", "confirm_required"}:
                # if GT food and no correct identity
                if food_gt and norm(o.get("identity") or o.get("raw_label")) != gt_n:
                    if st in {"unknown_food", "abstain", "unable_to_determine"}:
                        return "OPEN_SET"
                    if st == "conflicting":
                        return "FUSION"

    # Identity mismatch
    if food_gt:
        preds = [norm(o.get("identity") or o.get("raw_label")) for o in obs_list]
        if gt_n not in preds:
            # prepared meal mistaken?
            if any(p in {"tomato", "onion", "potato", "rice", "cheese"} for p in preds) and gt_n in {
                "biryani",
                "fried_rice",
                "pizza",
                "salad",
                "sandwich",
                "omelette",
            }:
                return "PREPARED_MEAL"
            return "IDENTITY"

    # State (only if identity matched somehow but scene still failed — rare in per-failure rows)
    if (reported_stage or "").upper() in {"FOOD_STATE", "STATE"}:
        return "FOOD_STATE"

    if (reported_stage or "").upper() == "FOOD_NONFOOD":
        return "FOOD_NONFOOD"

    if (reported_stage or "").upper() == "CLASSIFIER":
        # already handled non_food above; residual classifier
        return "IDENTITY"

    return (reported_stage or "ORCHESTRATION").upper()


def scene_blocking_stage(scene: dict[str, Any], failure_stages: list[str]) -> str | None:
    if scene.get("strict_scene_complete"):
        return None
    if not failure_stages:
        return "ORCHESTRATION"
    # priority order: detector first, then food_nonfood, then identity, etc.
    priority = [
        "DETECTOR",
        "REGION_DISCOVERY",
        "FOOD_NONFOOD",
        "IDENTITY",
        "PREPARED_MEAL",
        "FOOD_STATE",
        "OPEN_SET",
        "FUSION",
        "OCR",
        "BARCODE",
        "ENTITY",
        "RETRIEVAL",
        "ORCHESTRATION",
    ]
    counts = Counter(failure_stages)
    for p in priority:
        if counts.get(p):
            return p
    return failure_stages[0]


def build_root_cause_matrix(acc: dict, catalog: dict) -> dict[str, Any]:
    gt_by_scene: dict[str, list[dict]] = {}
    for sc in catalog.get("scenes") or []:
        sid = sc.get("scene_id") or sc.get("id")
        gt_by_scene[sid] = list(sc.get("instances") or [])

    rows = []
    scene_rows = []
    stage_fail_counts = Counter()
    scene_block_counts = Counter()
    identity_miss_counter = Counter()

    for scene in acc.get("scenes") or []:
        sid = scene.get("scene_id")
        obs = scene.get("observations") or []
        fails = scene.get("failures") or []
        gt_inst = gt_by_scene.get(sid) or []
        expected = [norm(g.get("label") or g.get("fine_label") or g.get("raw_label")) for g in gt_inst]
        detected_labels = [norm(o.get("identity") or o.get("raw_label")) for o in obs]
        fail_stages = []
        for f in fails:
            gt = f.get("gt")
            # find gt bbox if any
            gt_bbox = None
            for g in gt_inst:
                if norm(g.get("label") or g.get("fine_label")) == norm(gt):
                    gt_bbox = g.get("bbox")
                    break
            stage = first_stage_for_failure(
                gt_label=str(gt or ""),
                gt_bbox=gt_bbox,
                obs_list=obs,
                reported_stage=f.get("stage"),
                iou=float(f.get("iou") or 0),
            )
            fail_stages.append(stage)
            stage_fail_counts[stage] += 1
            if stage in {"IDENTITY", "PREPARED_MEAL", "OPEN_SET"}:
                identity_miss_counter[norm(gt)] += 1
            rows.append(
                {
                    "scene_id": sid,
                    "category": scene.get("category"),
                    "gt": gt,
                    "pred": f.get("pred"),
                    "iou": f.get("iou"),
                    "reported_stage": f.get("stage"),
                    "first_stage_root_cause": stage,
                    "n_obs": scene.get("n_obs"),
                    "n_gt": scene.get("n_gt"),
                    "obs_statuses": [o.get("status") for o in obs],
                    "obs_labels": detected_labels,
                }
            )

        block = scene_blocking_stage(scene, fail_stages)
        if block:
            scene_block_counts[block] += 1
        scene_rows.append(
            {
                "scene_id": sid,
                "category": scene.get("category"),
                "strict_scene_complete": scene.get("strict_scene_complete"),
                "expected_objects": expected,
                "detected_objects": detected_labels,
                "n_gt": scene.get("n_gt"),
                "n_obs": scene.get("n_obs"),
                "matched": scene.get("matched"),
                "object_prf": scene.get("object_prf"),
                "failure_first_stages": fail_stages,
                "scene_blocking_stage": block,
                "observations": [
                    {
                        "status": o.get("status"),
                        "identity": o.get("identity") or o.get("raw_label"),
                        "visual_class": o.get("visual_class"),
                        "food_state": o.get("food_state"),
                        "confidence": o.get("confidence"),
                        "is_food": o.get("is_food"),
                    }
                    for o in obs
                ],
            }
        )

    ranked = sorted(scene_block_counts.items(), key=lambda kv: -kv[1])
    primary = ranked[0][0] if ranked else "UNKNOWN"
    secondary = ranked[1][0] if len(ranked) > 1 else "NONE"
    tertiary = ranked[2][0] if len(ranked) > 2 else "NONE"

    return {
        "version": "v4_41_scene_root_cause_matrix",
        "generated_at": now(),
        "baseline_strict": BASELINE,
        "substantial_target": SUBSTANTIAL,
        "source_acceptance": str(ACC),
        "n_scenes": acc.get("n_scenes"),
        "n_strict_complete": acc.get("n_strict_complete"),
        "reported_stage_totals_from_acceptance": acc.get("primary_stage_totals"),
        "first_stage_failure_instance_counts": dict(stage_fail_counts),
        "scene_blocking_stage_counts": dict(scene_block_counts),
        "bottlenecks": {
            "PRIMARY": primary,
            "SECONDARY": secondary,
            "TERTIARY": tertiary,
            "ranking": [{"stage": k, "scenes_blocked": v} for k, v in ranked],
        },
        "top_missed_identities": identity_miss_counter.most_common(20),
        "food_state_agreement": acc.get("food_state_agreement"),
        "hardneg_nonfood": acc.get("hardneg_nonfood"),
        "instances": rows,
        "scenes": scene_rows,
        "method_note": (
            "First-stage root cause prefers observation signals over matcher-reported CLASSIFIER. "
            "Tomato→non_food is FOOD_NONFOOD, not IDENTITY. DETECTOR kept when reported or no spatial support."
        ),
    }


def write_forensics_md(f: dict) -> None:
    lines = [
        "# Current Pipeline Forensics (0.2927 system)",
        "",
        f"Generated: {f['generated_at']}",
        "",
        f"Weights: `{f['weights']['path']}`",
        f"SHA256: `{f['weights']['sha256']}`",
        "",
        "## Stage trace",
        "",
    ]
    for s in f["stages"]:
        lines.append(f"### {s['stage']}")
        for k, v in s.items():
            if k == "stage":
                continue
            lines.append(f"- **{k}**: {v}")
        lines.append("")
    lines.append("## Duplicated authorities")
    for x in f["duplicated_authorities"]:
        lines.append(f"- {x}")
    lines.append("")
    lines.append("## Dead / unused")
    for x in f["dead_or_unused"]:
        lines.append(f"- {x}")
    lines.append("")
    lines.append("## Stale / fallback risks")
    for x in f["stale_or_fallback_risks"]:
        lines.append(f"- {x}")
    (DOCS / "current_pipeline_forensics.md").write_text("\n".join(lines), encoding="utf-8")


def write_matrix_md(m: dict) -> None:
    b = m["bottlenecks"]
    lines = [
        "# 41-Scene Root-Cause Matrix (0.2927 replay analysis)",
        "",
        f"Generated: {m['generated_at']}",
        "",
        f"Strict completeness: **{BASELINE}** ({m['n_strict_complete']}/{m['n_scenes']})",
        f"Substantial target (frozen): **{SUBSTANTIAL}**",
        "",
        "## Bottlenecks (by scenes blocked)",
        "",
        f"- PRIMARY: **{b['PRIMARY']}**",
        f"- SECONDARY: **{b['SECONDARY']}**",
        f"- TERTIARY: **{b['TERTIARY']}**",
        "",
        "```json",
        json.dumps(b["ranking"], indent=2),
        "```",
        "",
        "## Instance first-stage counts",
        "",
        "```json",
        json.dumps(m["first_stage_failure_instance_counts"], indent=2),
        "```",
        "",
        "## Acceptance-reported stages (for contrast)",
        "",
        "```json",
        json.dumps(m["reported_stage_totals_from_acceptance"], indent=2),
        "```",
        "",
        f"Method: {m['method_note']}",
        "",
        "## Top missed identities (identity/open-set/prepared)",
        "",
        "```json",
        json.dumps(m["top_missed_identities"], indent=2),
        "```",
    ]
    (DOCS / "41_scene_root_cause_matrix.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    forensics = build_forensics()
    write_json(DOCS / "current_pipeline_forensics.json", forensics)
    write_forensics_md(forensics)

    acc = json.loads(ACC.read_text(encoding="utf-8"))
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    matrix = build_root_cause_matrix(acc, catalog)
    write_json(DOCS / "41_scene_root_cause_matrix.json", matrix)
    write_matrix_md(matrix)

    design = {
        "generated_at": now(),
        "primary": matrix["bottlenecks"]["PRIMARY"],
        "secondary": matrix["bottlenecks"]["SECONDARY"],
        "tertiary": matrix["bottlenecks"]["TERTIARY"],
        "substantial_target": SUBSTANTIAL,
        "selected_components": [],
        "rejected_components": [],
        "architecture_name": "V4_OPEN_WORLD_SCENE_ENGINE",
        "rationale": "",
    }
    # Evidence-based selection
    pri = design["primary"]
    sec = design["secondary"]
    # Always need open-world engine (phase 5 mandate) + scene reasoning
    selected = [
        "F_specialist_classifiers_as_evidence_only",
        "L_evidence_fusion",
        "M_open_set_rejection",
        "K_cross_object_graph",
        "J_food_state_specialist",
        "D_dinov2_retained_backbone",
    ]
    rejected = [
        "blind_dinov3_swap",
        "blind_rtdetr_swap",
        "small_softmax_retrain",
        "retrieval_without_library",
    ]
    if pri == "FOOD_NONFOOD" or sec == "FOOD_NONFOOD":
        selected.insert(0, "food_nonfood_gate_repair_with_topk_food_mass")
    if pri == "DETECTOR" or sec == "DETECTOR":
        selected.append("A_stronger_detector_or_proposal_refinement")
        # only if detector is primary — consider RT-DETR later on Kaggle
    if pri in {"IDENTITY", "OPEN_SET", "PREPARED_MEAL"} or sec in {"IDENTITY", "OPEN_SET", "PREPARED_MEAL"}:
        selected.append("E_hierarchical_retrieval_REQUIRES_n_items_gt_0")
        selected.append("candidate_generation_topk_not_argmax")
    if "FOOD_STATE" in {pri, sec, design["tertiary"]}:
        selected.append("state_head_decoupled_from_identity")

    # Do not spend GPU on RT-DETR unless detector is primary
    if pri != "DETECTOR":
        rejected.append("RT-DETR_primary_this_cycle")
    # DINOv3 only if we can get weights; else DINOv2
    rejected.append("DINOv3_until_weights_accessible")

    design["selected_components"] = selected
    design["rejected_components"] = rejected
    design["rationale"] = (
        f"Primary bottleneck={pri}, secondary={sec}. "
        "V4 keeps DINOv2 frozen representation; replaces flat argmax with top-k candidate generation + "
        "evidence fusion + open-set; repairs food/non-food false negatives that were mislabeled CLASSIFIER; "
        "adds scene-level cross-object confidence modifiers; builds retrieval library ONLY with assert n>0. "
        "Detector upgrade deferred unless detector ranks primary."
    )
    write_json(DOCS / "v4_architecture_selection.json", design)
    print(json.dumps(matrix["bottlenecks"], indent=2))
    print(json.dumps(design, indent=2)[:2000])


if __name__ == "__main__":
    main()
