"""41-scene acceptance for specialist ensemble — aligned to official metrics."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image

from kitcheniq_v10.specialist.pipeline import SpecialistEnsemble
from kitcheniq_v10.specialist.types import FusedObservation

OLD_ROOT = Path(r"C:\Projects\KitchenIQ-OS")


def _ensure_old_backend() -> None:
    backend = str(OLD_ROOT / "backend")
    if backend not in sys.path:
        sys.path.insert(0, backend)


def obs_to_official(o: FusedObservation) -> dict[str, Any]:
    """Map specialist fused observation → official acceptance observation schema."""
    status = o.status
    raw = o.identity or o.raw_label
    visual = o.visual_class
    if status == "non_food":
        raw = "non_food"
        visual = "non_food"
    elif status in {"abstain", "unknown_food", "conflicting", "confirm_required", "unable_to_determine"}:
        # Official matcher needs raw_label for identity hits; keep None when abstaining.
        if status == "unknown_food":
            visual = visual or "raw_ingredient"
        elif status == "conflicting":
            raw = None
    return {
        "raw_label": raw,
        "fine_label": o.identity,
        "bbox": o.bbox,
        "confidence": o.confidence,
        "visual_class": visual,
        "food_state": o.food_state,
        "is_food": o.is_food,
        "is_prepared_meal": o.is_prepared_meal,
        "status": status,
        "quantity_uncertain": True,
        "quantity_estimate": None,
        "components": None,
        "localization": {"detection_bbox": o.bbox} if o.bbox else {},
        "specialist_provenance": o.provenance,
        "abstain_reason": o.abstain_reason,
    }


def run_acceptance(
    *,
    catalog_path: Path,
    fixtures_root: Path,
    ensemble: SpecialistEnsemble,
    out_dir: Path,
    candidate_id: str,
) -> dict[str, Any]:
    _ensure_old_backend()
    from app.services.food_vision.real_world_acceptance.metrics import (
        match_instances,
        object_prf,
        scene_completeness,
    )

    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    scenes = catalog.get("scenes") or []
    results: list[dict[str, Any]] = []
    stage_fail: Counter[str] = Counter()
    n_strict = 0
    n_status_pass = 0
    food_state_rows: list[dict[str, Any]] = []
    hardneg_rows: list[dict[str, Any]] = []

    for scene in scenes:
        sid = scene.get("scene_id") or scene.get("id")
        img_info = scene.get("image") or {}
        raw_path = img_info.get("path") or ""
        img_path = Path(raw_path)
        if not img_path.is_file():
            name = img_path.name if img_path.name else f"{sid}.jpg"
            cand = fixtures_root / "generated" / name
            if cand.is_file():
                img_path = cand
            else:
                found = list((fixtures_root / "generated").glob(f"{sid}*"))
                img_path = found[0] if found else cand
        if not img_path.is_file():
            results.append(
                {
                    "scene_id": sid,
                    "error": "missing_image",
                    "strict_scene_complete": False,
                    "status": "FAIL",
                }
            )
            stage_fail["INFRASTRUCTURE"] += 1
            continue

        image = Image.open(img_path).convert("RGB")
        fused = ensemble.infer_image(image)
        observations = [obs_to_official(o) for o in fused]
        gt_instances = list(scene.get("instances") or [])

        matches = match_instances(gt_instances, observations)
        sc = scene_completeness(matches)
        prf = object_prf(matches, len(observations))

        failures: list[dict[str, Any]] = []
        for m in matches:
            if m.matched:
                continue
            stage = m.failure_stage or "CLASSIFIER"
            stage_fail[stage] += 1
            failures.append(
                {
                    "gt": (m.detail or {}).get("gt_label"),
                    "stage": stage,
                    "pred": (m.detail or {}).get("pred_label"),
                    "iou": m.iou,
                }
            )

        # Secondary: food-state agreement (does NOT gate strict completeness)
        for m in matches:
            if not m.matched or m.pred_idx is None:
                continue
            gt_inst = next(
                (
                    i
                    for i in gt_instances
                    if str(i.get("instance_id") or i.get("id") or "") == m.gt_id
                ),
                {},
            )
            want = gt_inst.get("food_state")
            if not want:
                continue
            got = observations[m.pred_idx].get("food_state")
            ok = str(got or "").lower() == str(want).lower() or (
                str(want).lower() == "plated" and str(got or "").lower() in {"cooked", "plated", "leftover"}
            )
            food_state_rows.append(
                {"scene_id": sid, "expected": want, "got": got, "ok": ok}
            )
            if not ok:
                stage_fail["FOOD_STATE"] += 1

        # Non-food / false-food secondary
        cat = scene.get("category") or ""
        false_food = [
            o
            for o in observations
            if o.get("visual_class") not in {"non_food", "recipe_document", None}
            and str(o.get("raw_label") or "")
            not in {"", "unable_to_determine", "non_food"}
            and (o.get("confidence") or 0) >= 0.55
            and o.get("is_food") is True
        ]
        nonfood_pass = None
        if cat == "13_non_food":
            nonfood_pass = len(false_food) == 0
            hardneg_rows.append(
                {
                    "scene_id": sid,
                    "false_food": len(false_food),
                    "pass": nonfood_pass,
                    "preds": [o.get("raw_label") for o in observations],
                }
            )
            if not nonfood_pass:
                stage_fail["FOOD_NONFOOD"] += 1

        status = "PASS" if sc["strict_scene_complete"] else (
            "PARTIAL" if sc["scene_completeness"] > 0 else "FAIL"
        )
        if cat == "13_non_food" and nonfood_pass:
            status = "PASS"
        if sc["strict_scene_complete"]:
            n_strict += 1
        if status == "PASS":
            n_status_pass += 1

        results.append(
            {
                "scene_id": sid,
                "category": cat,
                "n_gt": len([i for i in gt_instances if i.get("required", True)]),
                "n_obs": len(observations),
                "matched": sc["n_matched"],
                "completeness": sc["scene_completeness"],
                "strict_scene_complete": sc["strict_scene_complete"],
                "status": status,
                "object_prf": prf,
                "failures": failures,
                "observations": [
                    {
                        "status": o.get("status"),
                        "identity": o.get("fine_label"),
                        "raw_label": o.get("raw_label"),
                        "food_state": o.get("food_state"),
                        "confidence": o.get("confidence"),
                        "visual_class": o.get("visual_class"),
                        "is_food": o.get("is_food"),
                        "is_prepared_meal": o.get("is_prepared_meal"),
                        "abstain_reason": o.get("abstain_reason"),
                    }
                    for o in observations
                ],
            }
        )

    n = len(results) or 1
    fs_ok = sum(1 for r in food_state_rows if r.get("ok"))
    hn_ok = sum(1 for r in hardneg_rows if r.get("pass"))
    summary = {
        "candidate_id": candidate_id,
        "n_scenes": len(results),
        "n_strict_complete": n_strict,
        "strict_scene_completeness": round(n_strict / n, 4),
        "n_status_pass": n_status_pass,
        "status_pass_rate": round(n_status_pass / n, 4),
        "baseline_strict": 0.2439,
        "delta_vs_baseline": round(n_strict / n - 0.2439, 4),
        "primary_stage_totals": dict(stage_fail),
        "food_state_agreement": {
            "n": len(food_state_rows),
            "ok": fs_ok,
            "rate": round(fs_ok / max(1, len(food_state_rows)), 4),
        },
        "hardneg_nonfood": {
            "n": len(hardneg_rows),
            "ok": hn_ok,
            "rate": round(hn_ok / max(1, len(hardneg_rows)), 4),
        },
        "scenes": results,
        "acceptance_alignment": "official_match_instances_v1",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{candidate_id}_acceptance.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary
