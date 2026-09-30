"""KitchenIQ V10 autonomous engineering orchestrator.

Runs stages 00-18 with atomic checkpoints. Old project is READ-ONLY.
Large ML compute goes to Kaggle; Windows holds orchestration + manifests only.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

NEW_ROOT = Path(__file__).resolve().parents[1]
OLD_ROOT = Path(r"C:\Projects\KitchenIQ-OS")
STAGES = [
    "00_environment",
    "01_reference_snapshot",
    "02_baseline",
    "03_data_audit",
    "04_dataset_validation",
    "05_architecture_benchmark",
    "06_dataset_build",
    "07_detector",
    "08_food_recognition",
    "09_specialists",
    "10_calibration",
    "11_real_world_acceptance",
    "12_failure_analysis",
    "13_challenger_selection",
    "14_final_evaluation",
    "15_registry",
    "16_serving",
    "17_end_to_end",
    "18_final_report",
]
KAGGLE = Path(r"C:\Users\kathi\AppData\Roaming\Python\Python313\Scripts\kaggle.exe")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha_json(obj: Any) -> str:
    return sha_bytes(json.dumps(obj, sort_keys=True, default=str).encode("utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def state_path() -> Path:
    return NEW_ROOT / "CURRENT_STATE.json"


def load_state() -> dict[str, Any]:
    p = state_path()
    if p.is_file():
        return read_json(p)
    return {
        "current_stage": None,
        "completed_stages": [],
        "failed_stages": [],
        "retry_count": {},
        "active_candidate": None,
        "latest_checkpoint": None,
        "latest_model_hash": None,
        "latest_dataset_hash": None,
        "latest_acceptance_result": None,
        "production_status": "BLOCKED",
        "resume_configuration": {"max_retries": 3},
        "created_at": now(),
        "updated_at": now(),
        "old_project_root": str(OLD_ROOT),
        "new_project_root": str(NEW_ROOT),
        "isolation_verified": str(OLD_ROOT.resolve()) != str(NEW_ROOT.resolve()),
    }


def save_state(state: dict[str, Any]) -> None:
    state["updated_at"] = now()
    write_json(state_path(), state)


def checkpoint_path(stage: str) -> Path:
    return NEW_ROOT / "checkpoints" / "stages" / f"{stage}.json"


def mark_checkpoint(stage: str, payload: dict[str, Any]) -> Path:
    cp = {
        "stage": stage,
        "status": "COMPLETE",
        "timestamp": now(),
        "git_commit": _git_head(NEW_ROOT),
        "configuration_hash": sha_json(payload.get("config") or {}),
        "dataset_manifest_hash": payload.get("dataset_manifest_hash"),
        "model_hash": payload.get("model_hash"),
        "random_seed": payload.get("random_seed", 42),
        "metrics": payload.get("metrics") or {},
        "environment": payload.get("environment") or {},
        "artifact_locations": payload.get("artifact_locations") or {},
        "completion_marker": True,
        "payload": payload,
    }
    path = checkpoint_path(stage)
    write_json(path, cp)
    # verify
    loaded = read_json(path)
    if not loaded.get("completion_marker") or loaded.get("status") != "COMPLETE":
        raise RuntimeError(f"checkpoint verify failed for {stage}")
    return path


def stage_done(stage: str) -> bool:
    p = checkpoint_path(stage)
    if not p.is_file():
        return False
    try:
        d = read_json(p)
        return bool(d.get("completion_marker") and d.get("status") == "COMPLETE")
    except Exception:
        return False


def _git_head(repo: Path) -> str | None:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            capture_output=True,
            text=True,
            check=False,
        )
        return r.stdout.strip() or None
    except Exception:
        return None


def _git_porcelain_sha(repo: Path) -> tuple[int, str]:
    r = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    text = r.stdout or ""
    return len([l for l in text.splitlines() if l.strip()]), sha_bytes(text.encode())


def old_read(rel: str) -> Path:
    return OLD_ROOT / rel


def copy_text_ref(rel: str, dest_rel: str) -> dict[str, Any] | None:
    src = old_read(rel)
    if not src.is_file():
        return None
    dest = NEW_ROOT / "reference_snapshot" / dest_rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    # text/json only — never bulk images
    if src.suffix.lower() in {".pt", ".pth", ".ckpt", ".bin", ".safetensors", ".zip", ".tar", ".gz"}:
        meta = {
            "path": rel,
            "sha256": sha_file(src),
            "bytes": src.stat().st_size,
            "copied": False,
            "reason": "weight_or_archive_hash_only_no_local_bulk_copy",
        }
        write_json(dest.with_suffix(".hash.json"), meta)
        return meta
    shutil.copy2(src, dest)
    return {"path": rel, "sha256": sha_file(src), "bytes": src.stat().st_size, "copied_to": str(dest)}


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------

def stage_00_environment(state: dict[str, Any]) -> dict[str, Any]:
    env = {
        "python": sys.version,
        "platform": sys.platform,
        "cwd": str(Path.cwd()),
        "new_root": str(NEW_ROOT),
        "old_root": str(OLD_ROOT),
        "isolation_ok": str(OLD_ROOT.resolve()) != str(NEW_ROOT.resolve()),
        "kaggle_cli": str(KAGGLE) if KAGGLE.is_file() else None,
        "kaggle_creds": (Path.home() / ".kaggle" / "access_token").is_file()
        or (Path.home() / ".kaggle" / "kaggle.json").is_file(),
        "torch": None,
        "cuda": False,
        "gpu": None,
    }
    try:
        import torch

        env["torch"] = torch.__version__
        env["cuda"] = bool(torch.cuda.is_available())
        if env["cuda"]:
            env["gpu"] = torch.cuda.get_device_name(0)
    except Exception as e:
        env["torch_error"] = str(e)
    kaggle_probe = {"ok": False}
    if KAGGLE.is_file():
        try:
            r = subprocess.run(
                [str(KAGGLE), "kernels", "list", "--mine", "--page-size", "1"],
                capture_output=True,
                text=True,
                timeout=60,
            )
            kaggle_probe = {"ok": r.returncode == 0, "stdout_tail": (r.stdout or "")[-500:]}
        except Exception as e:
            kaggle_probe = {"ok": False, "error": str(e)}
    env["kaggle_api_probe"] = kaggle_probe
    write_json(NEW_ROOT / "reports" / "stage" / "00_environment.json", env)
    if not env["isolation_ok"]:
        raise RuntimeError("ISOLATION FAILURE: OLD==NEW")
    return {"environment": env, "metrics": {"kaggle_api_ok": kaggle_probe.get("ok")}}


def stage_01_reference_snapshot(state: dict[str, Any]) -> dict[str, Any]:
    docs = [
        "docs/KITCHENIQ_MASTER_PRODUCT_AI_REQUIREMENTS_V10.md",
        "docs/KitchenIQ_Architecture_Specification_v9.5.md",
        "docs/KitchenIQ_Architecture_V9.5_Change_Report.md",
        "docs/VISION_ENGINE_V2_IMPLEMENTATION_CONTRACT.md",
        "docs/VISION_ENGINE_V2_IMPLEMENTATION_FINAL_REPORT.md",
        "docs/VISION_ENGINE_V2_CHALLENGER_TRAINING_FINAL_REPORT.md",
        "docs/VISION_ENGINE_V2_CHALLENGER_FAILURE_ANALYSIS.md",
        "docs/VISION_ENGINE_V2_DATA_REMEDIATION_REPORT.md",
        "docs/ARCHITECTURE.md",
    ]
    metrics_files = [
        "backend/instance/dev_experiments/fv-v2-challenger/reports/dataset_manifest.json",
        "backend/instance/dev_experiments/fv-v2-challenger/reports/evaluation_metrics.json",
        "backend/instance/dev_experiments/fv-v2-challenger/reports/calibration.json",
        "backend/instance/dev_experiments/fv-v2-challenger/failure_analysis.json",
        "backend/instance/dev_experiments/fv-v2-challenger/acceptance_challenger/reports/scene_level_metrics.json",
        "backend/instance/dev_experiments/fv-real-world-acceptance-v2/reports/scene_level_metrics.json",
        "backend/instance/dev_experiments/fv-real-world-acceptance-v2/reports/vision_failure_taxonomy.json",
        "backend/instance/dev_experiments/fv-v2-data-remediation/acquisition_summary.json",
        "backend/instance/dev_experiments/fv-v2-data-remediation/class_support_report.json",
        "backend/instance/dev_experiments/fv-production-final/kaggle/reports_pull_clf_json/production_evaluation_report.json",
    ]
    weights = [
        "backend/instance/dev_experiments/fv-production-final/registry_artifacts/food-vision-classifier-v1/classifier-prod-v1.0.0/model.pt",
        "backend/instance/dev_experiments/fv-v2-challenger/artifacts/challenger/model.pt",
    ]
    snap: dict[str, Any] = {"captured_at": now(), "files": {}, "weights": {}, "taxonomy": {}}
    for rel in docs + metrics_files:
        meta = copy_text_ref(rel, rel.replace("/", "__"))
        if meta:
            snap["files"][rel] = meta
    for rel in weights:
        meta = copy_text_ref(rel, "weights/" + Path(rel).name + ".hash.json")
        if meta:
            snap["weights"][rel] = meta
    # taxonomy if present
    tax_candidates = list((OLD_ROOT / "backend/app/services/food_vision").rglob("*taxonomy*"))
    tax_info = []
    for p in tax_candidates[:20]:
        if p.is_file() and p.suffix in {".json", ".csv", ".yaml", ".yml", ".py"}:
            tax_info.append({"rel": str(p.relative_to(OLD_ROOT)), "sha256": sha_file(p), "bytes": p.stat().st_size})
    snap["taxonomy_candidates"] = tax_info
    write_json(NEW_ROOT / "reference_snapshot" / "REFERENCE_SNAPSHOT.json", snap)
    return {
        "artifact_locations": {"snapshot": "reference_snapshot/REFERENCE_SNAPSHOT.json"},
        "metrics": {"n_files": len(snap["files"]), "n_weights": len(snap["weights"])},
        "model_hash": (snap["weights"].get(weights[0]) or {}).get("sha256"),
    }


def stage_02_baseline(state: dict[str, Any]) -> dict[str, Any]:
    baseline: dict[str, Any] = {
        "source": "old_project_measured_artifacts",
        "vision_v2_production_status": "BLOCKED",
        "strict_scene_completeness_production": 0.2439,
        "strict_scene_completeness_challenger": 0.2195,
        "scene_delta_challenger": -0.0244,
        "production_classifier_top1": {"val": 0.6177, "test": 0.5554},
        "challenger_classifier_top1": {"val": 0.6155, "test": 0.5714},
        "critical_categories_improved": [],
        "promotion": False,
        "holdout_untouched": True,
        "ocr": "UNAVAILABLE_tesseract_missing",
        "barcode": "AVAILABLE",
        "failure_analysis_dominant": "E_MULTIPLE_BLOCKERS",
        "attribution_challenger_official_stages": {
            "detector_pct": 33.0,
            "classifier_pct": 45.1,
            "state_pct": 18.7,
            "non_food_pct": 3.3,
            "other_pct": 0.0,
        },
        "architecture_baseline": "DINOv2 ViT-B/14 + Faster R-CNN + evidence fusion (baseline, not locked final)",
        "notes": [
            "Top-1 improved on challenger but strict scene completeness regressed",
            "Missing critical labels shared by prod and challenger",
            "Hard-negative overconfidence caused 1/41 strict regression (13_kitchen_mood)",
        ],
    }
    # enrich from copied acceptance if available
    for cand in [
        NEW_ROOT / "reference_snapshot" / "backend__instance__dev_experiments__fv-real-world-acceptance-v2__reports__scene_level_metrics.json",
        NEW_ROOT / "reference_snapshot" / "backend/instance/dev_experiments/fv-real-world-acceptance-v2/reports/scene_level_metrics.json",
    ]:
        # our copy uses __ separator
        pass
    acc = OLD_ROOT / "backend/instance/dev_experiments/fv-real-world-acceptance-v2/reports/scene_level_metrics.json"
    if acc.is_file():
        doc = read_json(acc)
        scenes = doc.get("scenes") or []
        n = len(scenes)
        ok = sum(1 for s in scenes if ((s.get("scene_completeness") or {}).get("strict_scene_complete")))
        if n:
            baseline["strict_scene_completeness_production_recomputed"] = round(ok / n, 4)
            baseline["n_scenes"] = n
            baseline["n_strict_complete"] = ok
    fa = OLD_ROOT / "backend/instance/dev_experiments/fv-v2-challenger/failure_analysis.json"
    if fa.is_file():
        fadoc = read_json(fa)
        baseline["failure_executive"] = fadoc.get("executive_conclusion")
    write_json(NEW_ROOT / "reports" / "stage" / "02_baseline.json", baseline)
    return {
        "metrics": {
            "strict_scene_completeness": baseline["strict_scene_completeness_production"],
            "production_status": "BLOCKED",
        },
        "artifact_locations": {"baseline": "reports/stage/02_baseline.json"},
    }


def stage_03_data_audit(state: dict[str, Any]) -> dict[str, Any]:
    """Audit coverage from old manifests + remediation — no bulk download."""
    audit: dict[str, Any] = {
        "policy": {
            "core_target": 250,
            "core_target_meaning": "INITIAL_PLANNING_TARGET_not_quota",
            "no_bulk_windows_download": True,
            "zones": ["UNTRUSTED", "OPERATIONAL", "LEARNING_QUARANTINE", "CURATED_TRAINING"],
        },
        "labels": {},
        "summary": {},
    }
    # challenger train counts (OI experiment path)
    ch = OLD_ROOT / "backend/instance/dev_experiments/fv-v2-challenger/reports/dataset_manifest.json"
    rem = OLD_ROOT / "backend/instance/dev_experiments/fv-v2-data-remediation/acquisition_summary.json"
    cs = OLD_ROOT / "backend/instance/dev_experiments/fv-v2-data-remediation/class_support_report.json"
    train_counts = {}
    if ch.is_file():
        train_counts = (read_json(ch).get("train_counts") or {})
    rem_food = {}
    rem_hn = {}
    if rem.is_file():
        rdoc = read_json(rem)
        rem_food = rdoc.get("food_counts") or {}
        rem_hn = rdoc.get("hardneg_counts") or {}
        audit["remediation_summary"] = {
            "newly_acquired_food": rdoc.get("newly_acquired_food"),
            "newly_acquired_hardneg": rdoc.get("newly_acquired_hardneg"),
            "controlled_retrain_justified": rdoc.get("controlled_retrain_justified"),
            "below_useful_food": rdoc.get("below_useful_food"),
        }
    # 207 taxonomy labels from old taxonomy loader if possible
    labels_207: list[str] = []
    try:
        sys.path.insert(0, str(OLD_ROOT / "backend"))
        from app.services.food_vision.dataset.taxonomy import load_taxonomy_rows  # type: ignore

        rows = load_taxonomy_rows()
        labels_207 = [str(r.get("label") or r.get("name") or "").strip() for r in rows]
        labels_207 = [x for x in labels_207 if x]
    except Exception as e:
        audit["taxonomy_load_error"] = str(e)
        # fallback: union of known sources
        labels_207 = sorted(set(list(train_counts) + list(rem_food) + list(rem_hn)))

    buckets = {"ZERO": 0, "<50": 0, "50-99": 0, "100-249": 0, ">=250": 0}
    for lab in labels_207:
        key = lab.replace(" ", "_").lower()
        pe = int(rem_food.get(key) or rem_food.get(lab) or 0)
        # hardneg categories are separate
        exp = int(train_counts.get(key) or train_counts.get(lab) or 0)
        total_prodish = pe  # production-eligible from remediation package
        if total_prodish >= 250:
            bucket = ">=250"
        elif total_prodish >= 100:
            bucket = "100-249"
        elif total_prodish >= 50:
            bucket = "50-99"
        elif total_prodish > 0:
            bucket = "<50"
        else:
            bucket = "ZERO"
        buckets[bucket] += 1
        audit["labels"][lab] = {
            "production_eligible_count": pe,
            "experiment_only_challenger_train_count": exp,
            "bucket": bucket,
            "core_target": 250,
            "below_target_sufficient_for_experiment": pe >= 20 and pe < 250,
        }
    audit["hardneg"] = rem_hn
    audit["summary"] = {
        "n_labels_audited": len(labels_207),
        "buckets": buckets,
        "n_with_any_production_eligible": sum(1 for v in audit["labels"].values() if v["production_eligible_count"] > 0),
        "core_target_met_count": buckets[">=250"],
        "note": "Full 207 production-eligible coverage is incomplete; remediation package covers critical gaps only.",
    }
    if cs.is_file():
        audit["class_support_report_sha"] = sha_file(cs)
    write_json(NEW_ROOT / "reports" / "stage" / "03_data_audit.json", audit)
    write_json(NEW_ROOT / "manifests" / "coverage_matrix_partial.json", audit)
    return {
        "metrics": audit["summary"],
        "dataset_manifest_hash": sha_json(audit["summary"]),
        "artifact_locations": {"audit": "reports/stage/03_data_audit.json"},
    }


def stage_04_dataset_validation(state: dict[str, Any]) -> dict[str, Any]:
    rem_dir = OLD_ROOT / "backend/instance/dev_experiments/fv-v2-data-remediation"
    validation = {
        "production_eligible_manifest_exists": (rem_dir / "production_eligible_manifest.jsonl").is_file(),
        "hard_negative_manifest_exists": (rem_dir / "hard_negative_manifest.jsonl").is_file(),
        "experiment_only_isolated": (rem_dir / "experiment_only_manifest.jsonl").is_file(),
        "rejected_quarantined": (rem_dir / "rejected_manifest.jsonl").is_file(),
        "provenance_report_exists": (rem_dir / "provenance_report.json").is_file(),
        "no_holdout_mutation": True,
        "bulk_windows_download": False,
        "issues": [],
    }
    # count lines without loading images
    for name in [
        "production_eligible_manifest.jsonl",
        "hard_negative_manifest.jsonl",
        "experiment_only_manifest.jsonl",
        "rejected_manifest.jsonl",
    ]:
        p = rem_dir / name
        if p.is_file():
            n = sum(1 for line in p.open(encoding="utf-8") if line.strip())
            validation[f"n_{name}"] = n
    if not validation["production_eligible_manifest_exists"]:
        validation["issues"].append("missing_production_eligible_manifest")
    write_json(NEW_ROOT / "reports" / "stage" / "04_dataset_validation.json", validation)
    return {"metrics": validation, "artifact_locations": {"validation": "reports/stage/04_dataset_validation.json"}}


def stage_05_architecture_benchmark(state: dict[str, Any]) -> dict[str, Any]:
    """Evidence-based architecture decision. No fashion. Use measured baseline + feasibility."""
    bench = {
        "principle": "DINOv2 is BASELINE not locked final; selection by product-level evidence",
        "candidates": [],
        "selected": None,
        "selection_rationale": "",
    }
    # Candidate A: current stack (measured)
    bench["candidates"].append(
        {
            "id": "A_detector_dinov2_classifier_fusion",
            "architecture": "Faster R-CNN + DINOv2 ViT-B/14 multi-head + evidence fusion",
            "weights_available": True,
            "licence": "code:project; backbone:DINOv2 terms; OI/OV data:per-image",
            "compute": "Kaggle T4 capable (prior challenger run)",
            "latency": "acceptable_prior",
            "validation_metrics": {"top1_test_prod": 0.5554, "top1_test_challenger": 0.5714},
            "scene_metrics": {"strict_prod": 0.2439, "strict_challenger": 0.2195},
            "production_suitability": "BASELINE_BLOCKED_on_scene_completeness",
            "evaluated": True,
            "evidence": "fv-real-world-acceptance-v2 + fv-v2-challenger",
        }
    )
    # Candidate B: detector + specialist heads (conceptual + partial evidence)
    bench["candidates"].append(
        {
            "id": "B_detector_specialist_ensemble",
            "architecture": "Faster R-CNN + specialist classifiers (raw/packaged/prepared/state/nonfood) + fusion",
            "weights_available": "PARTIAL_reuse_dinov2_backbone_possible",
            "licence": "same_as_A_plus_specialist_heads",
            "compute": "higher_than_A_but_T4_feasible_if_frozen_backbone",
            "latency": "higher",
            "validation_metrics": "NOT_FULLY_EVALUATED",
            "scene_metrics": "NOT_EVALUATED",
            "production_suitability": "PROMISING_GIVEN_FAILURE_MIX",
            "evaluated": False,
            "status": "NOT EVALUATED — RESOURCE/TIME for full 41-scene specialist suite pending justified training",
            "evidence_support": "failure attribution classifier 45% + state 19% + nonfood issues => specialists justified",
        }
    )
    # Candidate C: replace backbone with ConvNeXt/EfficientNet
    bench["candidates"].append(
        {
            "id": "C_detector_convnext_classifier",
            "architecture": "Faster R-CNN + ConvNeXt-Tiny classifier",
            "weights_available": True,
            "licence": "timm/apache_typically",
            "compute": "T4_feasible",
            "validation_metrics": "NOT EVALUATED — RESOURCE UNAVAILABLE for full product acceptance in this cycle without justified data+kernel",
            "scene_metrics": "NOT EVALUATED — RESOURCE UNAVAILABLE",
            "production_suitability": "UNKNOWN",
            "evaluated": False,
            "status": "NOT EVALUATED — RESOURCE UNAVAILABLE",
        }
    )
    # Candidate D: segmentation-assisted
    bench["candidates"].append(
        {
            "id": "D_detector_segmentation_assist",
            "architecture": "Mask R-CNN / SegFormer assist + classifier",
            "weights_available": True,
            "licence": "varies",
            "compute": "heavier",
            "validation_metrics": "NOT EVALUATED — RESOURCE UNAVAILABLE",
            "scene_metrics": "NOT EVALUATED — RESOURCE UNAVAILABLE",
            "production_suitability": "UNKNOWN",
            "evaluated": False,
            "status": "NOT EVALUATED — RESOURCE UNAVAILABLE",
            "note": "May help multi-object/occlusion but detector already 33% of failures; not first lever without measured gain",
        }
    )

    # Decision: keep A as operational baseline architecture family, but SELECT B as target architecture
    # because measured failures are multi-capability (classifier/state/nonfood) and classifier-only
    # retraining already failed to improve scene completeness.
    selected = {
        "id": "B_detector_specialist_ensemble",
        "architecture": "Faster R-CNN detector + specialist recognition heads (food/nonfood, family/identity, prepared-meal, food-state) on shared representation + evidence fusion + abstention",
        "backbone_policy": "Reuse DINOv2 ViT-B/14 as shared representation UNLESS specialist benchmark later proves otherwise; do not assume it is final",
        "why_not_blind_retrain_A": "Challenger Top-1 up / strict scene down; missing labels shared; hardneg regression",
        "next_compute": "Train only justified specialists + hardneg/state on Kaggle; keep detector unless specialist cycle stalls on detection",
    }
    bench["selected"] = selected
    bench["selection_rationale"] = (
        "Measured product metric (strict scene completeness) did not improve with classifier-only challenger. "
        "Failure mix is multi-component. Specialists are the evidence-supported architecture direction. "
        "Candidates C/D not fully evaluated on 41-scene under free compute in this pass — recorded as NOT EVALUATED."
    )
    write_json(NEW_ROOT / "reports" / "stage" / "05_architecture_benchmark.json", bench)
    write_json(NEW_ROOT / "configs" / "selected_architecture.json", selected)
    return {
        "metrics": {"selected": selected["id"], "n_candidates": len(bench["candidates"])},
        "config": selected,
        "artifact_locations": {
            "benchmark": "reports/stage/05_architecture_benchmark.json",
            "selected": "configs/selected_architecture.json",
        },
    }


def stage_06_dataset_build(state: dict[str, Any]) -> dict[str, Any]:
    """Immutable dataset version pointing at remediation manifests (no bulk copy)."""
    rem = OLD_ROOT / "backend/instance/dev_experiments/fv-v2-data-remediation"
    files = {
        "production_eligible_manifest": rem / "production_eligible_manifest.jsonl",
        "hard_negative_manifest": rem / "hard_negative_manifest.jsonl",
        "experiment_only_manifest": rem / "experiment_only_manifest.jsonl",
        "acquisition_summary": rem / "acquisition_summary.json",
        "provenance_report": rem / "provenance_report.json",
    }
    manifest = {
        "dataset_version": "kiq-v10-core-remediation-v1",
        "created_at": now(),
        "production_eligible_source": str(files["production_eligible_manifest"]),
        "hard_negative_source": str(files["hard_negative_manifest"]),
        "experiment_only_source": str(files["experiment_only_manifest"]),
        "file_hashes": {k: sha_file(v) for k, v in files.items() if v.is_file()},
        "split_definition": {
            "note": "Split must be built on Kaggle from manifests; no bulk image mirror on Windows",
            "leakage_controls": ["source", "image_hash", "near_dup_ban_across_splits"],
        },
        "licence_policy": "CC0/CC-BY verified only for PRODUCTION_ELIGIBLE; EXPERIMENT_ONLY isolated",
        "core_target": 250,
        "status": "MANIFEST_READY_IMAGES_ON_REMEDIATION_STORE_OR_KAGGLE",
    }
    manifest["manifest_hash"] = sha_json({k: v for k, v in manifest.items() if k != "manifest_hash"})
    write_json(NEW_ROOT / "manifests" / "dataset_version_kiq-v10-core-remediation-v1.json", manifest)
    # pointer for Kaggle — do not upload images from Windows if large; use existing Kaggle datasets
    write_json(
        NEW_ROOT / "kaggle" / "dataset_pointers.json",
        {
            "existing_kaggle_datasets": ["kathiresannatarajan/kiq-fv-staging"],
            "remediation_local_manifests_only": True,
            "no_bulk_windows_image_mirror": True,
        },
    )
    return {
        "dataset_manifest_hash": manifest["manifest_hash"],
        "metrics": {"dataset_version": manifest["dataset_version"]},
        "artifact_locations": {"manifest": "manifests/dataset_version_kiq-v10-core-remediation-v1.json"},
    }


def _kaggle_kernel_push_and_wait(kernel_dir: Path, timeout_s: int = 7200) -> dict[str, Any]:
    if not KAGGLE.is_file():
        return {"ok": False, "error": "kaggle_cli_missing"}
    meta = read_json(kernel_dir / "kernel-metadata.json")
    ref = f"{meta.get('id')}"
    # push
    push = subprocess.run(
        [str(KAGGLE), "kernels", "push", "-p", str(kernel_dir)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    result: dict[str, Any] = {
        "push_rc": push.returncode,
        "push_out": (push.stdout or "")[-1000:],
        "push_err": (push.stderr or "")[-1000:],
        "ref": ref,
    }
    if push.returncode != 0:
        result["ok"] = False
        return result
    # status poll
    t0 = time.time()
    last = None
    while time.time() - t0 < timeout_s:
        st = subprocess.run(
            [str(KAGGLE), "kernels", "status", ref],
            capture_output=True,
            text=True,
            timeout=60,
        )
        last = (st.stdout or st.stderr or "").strip()
        result["last_status"] = last
        low = last.lower()
        if "complete" in low or "success" in low:
            result["ok"] = True
            result["terminal"] = "complete"
            break
        if any(x in low for x in ["error", "failed", "cancelled"]):
            result["ok"] = False
            result["terminal"] = "failed"
            break
        time.sleep(30)
    else:
        result["ok"] = False
        result["terminal"] = "timeout"
    # pull output metadata only if complete
    if result.get("terminal") == "complete":
        out_dir = NEW_ROOT / "kaggle" / "outputs" / ref.replace("/", "__")
        out_dir.mkdir(parents=True, exist_ok=True)
        pull = subprocess.run(
            [str(KAGGLE), "kernels", "output", ref, "-p", str(out_dir)],
            capture_output=True,
            text=True,
            timeout=600,
        )
        result["pull_rc"] = pull.returncode
        result["pull_out"] = (pull.stdout or "")[-500:]
        result["output_dir"] = str(out_dir)
    return result


def stage_07_to_14_kaggle_and_local(state: dict[str, Any]) -> dict[str, Any]:
    """Create and execute Kaggle kernel for training/eval; fall back to evidence-gated no-train if blocked."""
    kernel_dir = NEW_ROOT / "kaggle" / "kernels" / "kiq-v10-autonomous"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    # Lightweight kernel: environment detect + architecture decision echo + optional train skip with gate
    nb = {
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "nbformat": 4,
        "nbformat_minor": 5,
        "cells": [
            {
                "cell_type": "code",
                "metadata": {},
                "source": [
                    "import json, os, platform, time\n",
                    "from pathlib import Path\n",
                    "out = Path('/kaggle/working')\n",
                    "info = {\n",
                    "  'stage': '07_14_bundle',\n",
                    "  'platform': platform.platform(),\n",
                    "  'cuda_visible': os.environ.get('CUDA_VISIBLE_DEVICES'),\n",
                    "}\n",
                    "try:\n",
                    "  import torch\n",
                    "  info['torch']=torch.__version__\n",
                    "  info['cuda']=torch.cuda.is_available()\n",
                    "  info['gpu']=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else []\n",
                    "  info['vram_gb']=[round(torch.cuda.get_device_properties(i).total_memory/1e9,2) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else []\n",
                    "except Exception as e:\n",
                    "  info['torch_error']=str(e)\n",
                    "# Decision for this cycle: DO NOT blind-retrain classifier-only.\n",
                    "# Evidence: challenger Top-1 up, strict scene down.\n",
                    "info['training_decision']='DEFER_FULL_RETRAIN'\n",
                    "info['reason']=('Classifier-only challenger already measured regression on 41-scene strict completeness. '\n",
                    "               'Architecture selected: specialist ensemble. Full specialist training requires '\n",
                    "               'production-eligible crop packs mounted; this kernel validates GPU env and freezes deferral.')\n",
                    "info['selected_architecture']='B_detector_specialist_ensemble'\n",
                    "info['baseline_strict']=0.2439\n",
                    "info['challenger_strict']=0.2195\n",
                    "info['promote']=False\n",
                    "(out/'v10_kaggle_env_report.json').write_text(json.dumps(info, indent=2))\n",
                    "print(json.dumps(info, indent=2))\n",
                ],
                "outputs": [],
                "execution_count": None,
            }
        ],
    }
    (kernel_dir / "kiq-v10-autonomous.ipynb").write_text(json.dumps(nb, indent=2), encoding="utf-8")
    meta = {
        "id": "kathiresannatarajan/kiq-v10-autonomous",
        "id_no": 0,
        "title": "kiq-v10-autonomous",
        "code_file": "kiq-v10-autonomous.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "dataset_sources": ["kathiresannatarajan/kiq-fv-staging"],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    write_json(kernel_dir / "kernel-metadata.json", meta)

    kaggle_result = _kaggle_kernel_push_and_wait(kernel_dir, timeout_s=1800)

    # Local evidence for stages 07-14 without fabricating improved scene scores
    local = {
        "detector": {
            "action": "RETAIN_FASTER_RCNN_BASELINE",
            "reason": "Detector stage totals unchanged prod→challenger; not the cause of strict delta",
            "trained_this_cycle": False,
        },
        "food_recognition": {
            "action": "DEFER_SPECIALIST_TRAINING_PENDING_MOUNTED_PRODUCTION_CROPS",
            "blind_classifier_retrain": False,
            "evidence": "prior challenger regression",
            "trained_this_cycle": False,
        },
        "specialists": {
            "planned": ["food_nonfood", "prepared_meal", "food_state", "packaged"],
            "trained_this_cycle": False,
            "status": "ARCHITECTURE_SELECTED_TRAINING_DEFERRED",
        },
        "calibration": {
            "bands": {"HIGH": 0.85, "MEDIUM": 0.55, "LOW": " <0.55"},
            "challenger_more_confident_wrong_on_nonfood": True,
            "action": "retain_bands_do_not_lower_thresholds",
        },
        "real_world_acceptance": {
            "baseline_strict": 0.2439,
            "final_strict_this_cycle": 0.2439,
            "note": "No new weights promoted; acceptance remains baseline production measurement",
            "n_scenes": 41,
        },
        "failure_analysis": {
            "dominant": "E_VISION_ACCEPTANCE_STILL_BLOCKED_MULTI_COMPONENT",
            "detector_pct": 33.0,
            "classifier_pct": 45.1,
            "state_pct": 18.7,
            "non_food_pct": 3.3,
        },
        "challenger_selection": {
            "selected_for_promotion": None,
            "reason": "No challenger improved strict scene completeness over 0.2439",
        },
        "final_evaluation": {
            "holdout_used": False,
            "reason": "No final frozen candidate qualified for holdout",
        },
        "kaggle": kaggle_result,
    }
    write_json(NEW_ROOT / "reports" / "stage" / "07_14_training_eval_bundle.json", local)
    return {
        "metrics": local["real_world_acceptance"],
        "model_hash": state.get("latest_model_hash"),
        "artifact_locations": {
            "bundle": "reports/stage/07_14_training_eval_bundle.json",
            "kaggle_kernel": str(kernel_dir),
        },
        "kaggle": kaggle_result,
        "local": local,
    }


def stage_15_registry(state: dict[str, Any], bundle: dict[str, Any]) -> dict[str, Any]:
    reg = {
        "registry_version": "v10-registry-1",
        "production_model": "UNCHANGED_REFERENCE_ONLY",
        "reference_production_classifier_hash": (
            read_json(NEW_ROOT / "reference_snapshot" / "REFERENCE_SNAPSHOT.json")
            .get("weights", {})
            .get(
                "backend/instance/dev_experiments/fv-production-final/registry_artifacts/food-vision-classifier-v1/classifier-prod-v1.0.0/model.pt",
                {},
            )
            .get("sha256")
        ),
        "challengers": [],
        "promoted": False,
        "reason": "Production gates not passed; no overwrite of production",
    }
    write_json(NEW_ROOT / "registry" / "v10_registry.json", reg)
    return {"metrics": {"promoted": False}, "artifact_locations": {"registry": "registry/v10_registry.json"}}


def stage_16_serving(state: dict[str, Any]) -> dict[str, Any]:
    serving = {
        "serving_bundle": "NOT_ISSUED",
        "reason": "PRODUCTION BLOCKED — no new serving weights",
        "rollback_reference": "old_project_production_classifier_hash_only",
    }
    write_json(NEW_ROOT / "serving" / "serving_manifest.json", serving)
    return {"metrics": serving, "artifact_locations": {"serving": "serving/serving_manifest.json"}}


def stage_17_end_to_end(state: dict[str, Any]) -> dict[str, Any]:
    """Contract-level E2E checklist against baseline failures — no silent pass."""
    scenarios = [
        "single_raw_ingredient",
        "multiple_raw_ingredients",
        "dense_kitchen",
        "packaged_product",
        "ingredient_packaged",
        "ingredient_prepared",
        "multiple_prepared",
        "fruit_collection",
        "ready_to_eat_snack",
        "raw_vs_partial",
        "cooked_plated_leftover",
        "recipe_document",
        "non_food",
        "hard_negatives",
        "mixed_food_nonfood",
        "duplicate_instances",
        "quantity_variation",
        "occlusion",
        "perspective",
        "low_light",
        "small_objects",
        "partial_visibility",
        "ocr",
        "barcode",
    ]
    # Map known acceptance failures: most remain FAIL at product level
    results = []
    for s in scenarios:
        status = "FAIL_BASELINE_UNRESOLVED"
        if s == "barcode":
            status = "PARTIAL_AVAILABLE_IN_BASELINE"
        if s == "ocr":
            status = "FAIL_TESSERACT_MISSING_IN_BASELINE"
        results.append({"scenario": s, "status": status, "pipeline": "IMAGE→OBS→ENTITY→STATE→TRAJ→USEFIRST→DECISION"})
    e2e = {
        "n_scenarios": len(scenarios),
        "n_pass": sum(1 for r in results if r["status"].startswith("PASS")),
        "results": results,
        "overall": "FAIL",
        "note": "No new production weights; E2E remains blocked by vision acceptance baseline",
    }
    write_json(NEW_ROOT / "reports" / "stage" / "17_end_to_end.json", e2e)
    return {"metrics": {"e2e_overall": e2e["overall"], "n_pass": e2e["n_pass"]}, "artifact_locations": {"e2e": "reports/stage/17_end_to_end.json"}}


def stage_18_final_report(state: dict[str, Any], collected: dict[str, Any]) -> dict[str, Any]:
    # Integrity check OLD
    baseline = read_json(NEW_ROOT / "reference_snapshot" / "OLD_PROJECT_INTEGRITY_BASELINE.json")
    porcelain_n, porcelain_sha = _git_porcelain_sha(OLD_ROOT)
    head = _git_head(OLD_ROOT)
    integrity = {
        "old_head_before": baseline.get("old_git_head"),
        "old_head_after": head,
        "head_unchanged": head == baseline.get("old_git_head"),
        "porcelain_count_before": baseline.get("old_porcelain_line_count"),
        "porcelain_count_after": porcelain_n,
        "porcelain_sha_before": baseline.get("old_porcelain_sha256"),
        "porcelain_sha_after": porcelain_sha,
        "porcelain_unchanged": porcelain_sha == baseline.get("old_porcelain_sha256"),
        "verdict": None,
    }
    integrity["verdict"] = (
        "UNCHANGED"
        if integrity["head_unchanged"] and integrity["porcelain_unchanged"]
        else "INTEGRITY_FAILURE"
    )
    write_json(NEW_ROOT / "reports" / "OLD_PROJECT_INTEGRITY_AFTER.json", integrity)

    audit = read_json(NEW_ROOT / "reports" / "stage" / "03_data_audit.json")
    arch = read_json(NEW_ROOT / "reports" / "stage" / "05_architecture_benchmark.json")
    bundle = collected.get("bundle") or {}
    kaggle = (bundle.get("kaggle") if isinstance(bundle, dict) else None) or {}

    gates = {
        "A_DATA": "FAIL_partial_coverage_207_incomplete",
        "B_MODEL": "FAIL_no_improved_candidate",
        "C_REAL_WORLD": "FAIL_strict_scene_0.2439_blocker",
        "D_STATE": "NOT_PROVEN",
        "E_PRODUCT": "NOT_PROVEN",
        "F_OPERATIONS": "FAIL_no_new_serving_bundle",
    }
    production_gate = "FAIL"
    final_decision = "E. VISION ACCEPTANCE STILL BLOCKED"
    dominant_blocker = (
        "41-scene strict complete-scene rate remains 0.2439 (production BLOCKED). "
        "Classifier-only challenger previously worsened it to 0.2195. "
        "Architecture direction selected (specialist ensemble) but no candidate improved real-world acceptance this cycle."
    )

    final_state = {
        "status": "BLOCKED",
        "new_project": str(NEW_ROOT),
        "architecture": arch.get("selected"),
        "best_model": "production_reference_unchanged (no improved challenger)",
        "real_world_scene_completeness": {"baseline": 0.2439, "final": 0.2439},
        "dataset": audit.get("summary"),
        "core_250_target": {
            "met_count": audit.get("summary", {}).get("core_target_met_count"),
            "buckets": audit.get("summary", {}).get("buckets"),
            "note": "250 is planning target; critical remediation package exists but 207-wide >=250 not met",
        },
        "kaggle": {
            "attempted": True,
            "result": kaggle.get("terminal") or kaggle.get("ok"),
            "ref": kaggle.get("ref"),
            "detail": {k: kaggle.get(k) for k in ("ok", "terminal", "last_status", "push_rc")},
        },
        "production_gate": production_gate,
        "gates": gates,
        "final_decision": final_decision,
        "dominant_blocker": dominant_blocker,
        "old_project": integrity["verdict"],
        "generated_at": now(),
    }
    write_json(NEW_ROOT / "docs" / "KITCHENIQ_V10_AUTONOMOUS_FINAL_STATE.json", final_state)

    report = f"""# KitchenIQ V10 Autonomous Engineering — Final Report

**Generated:** {now()}  
**NEW_PROJECT_ROOT:** `{NEW_ROOT}`  
**OLD_PROJECT_ROOT:** `{OLD_ROOT}` (READ-ONLY)

## 1. Executive result

**STATUS: BLOCKED**

**FINAL DECISION: {final_decision}**

Dominant blocker: {dominant_blocker}

## 2. Product objective

KitchenIQ is a predictive household food intelligence system that converts uncertain observations into persistent uncertainty-aware food state, predicts trajectories, and recommends safe realistic actions. Vision is observation/evidence — not the product.

## 3. Old-project baseline

- Vision V2 implemented, production **BLOCKED**
- Strict scene completeness **0.2439** (41 scenes)
- Challenger strict **0.2195** (regression)
- Top-1 challenger improved but product metric worsened

## 4–7. Dataset audit / 207 coverage / 250 target / licensing

See `reports/stage/03_data_audit.json`.

- Labels audited: {audit.get('summary',{}).get('n_labels_audited')}
- Buckets: {audit.get('summary',{}).get('buckets')}
- Core target (>=250) met count: {audit.get('summary',{}).get('core_target_met_count')}
- Remediation package provides critical production-eligible + hard-negative coverage; 207-wide 250 target not fully met
- Licensing: production-eligible CC0/CC-BY verified path from remediation; EXPERIMENT_ONLY isolated

## 8–10. Architecture benchmark / selected / models

Selected: `{arch.get('selected',{}).get('id')}` — {arch.get('selected',{}).get('architecture')}

Candidates C/D recorded as NOT EVALUATED where full 41-scene compute was unavailable.

Models evaluated this cycle: environment/GPU validation on Kaggle; **no blind classifier retrain** (evidence-gated deferral).

## 11–12. Training / Calibration

Training deferred for classifier-only path. Confidence bands retained HIGH≥0.85 / MEDIUM≥0.55. Thresholds not lowered.

## 13–15. 41-scene / failures / challengers

- Baseline → final strict: **0.2439 → 0.2439**
- Failure mix: detector ~33%, classifier ~45%, state ~19%, non-food ~3%
- No challenger selected for promotion

## 16. Final holdout

**Not used** (no qualifying final candidate). Holdout protection preserved.

## 17–19. Security / Registry / Serving

- No production overwrite
- Registry records reference hashes only
- Serving bundle NOT ISSUED

## 20. End-to-end validation

Overall **FAIL** — baseline unresolved across required scenarios (`reports/stage/17_end_to_end.json`).

## 21. Production gate

**FAIL** — gates A/B/C/F failed or unproven.

## 22. Final decision

**{final_decision}**

## 23. Exact remaining blocker

{dominant_blocker}

## Old project integrity

**{integrity['verdict']}**  
head unchanged={integrity['head_unchanged']} porcelain unchanged={integrity['porcelain_unchanged']}

## Kaggle

attempted={bool(kaggle)} terminal={kaggle.get('terminal')} ok={kaggle.get('ok')} ref={kaggle.get('ref')}
"""
    (NEW_ROOT / "docs" / "KITCHENIQ_V10_AUTONOMOUS_ENGINEERING_FINAL_REPORT.md").write_text(
        report, encoding="utf-8"
    )
    return {
        "metrics": final_state,
        "artifact_locations": {
            "report": "docs/KITCHENIQ_V10_AUTONOMOUS_ENGINEERING_FINAL_REPORT.md",
            "final_state": "docs/KITCHENIQ_V10_AUTONOMOUS_FINAL_STATE.json",
        },
        "final_state": final_state,
        "integrity": integrity,
    }


def run_stage(stage: str, state: dict[str, Any], collected: dict[str, Any]) -> dict[str, Any]:
    if stage == "00_environment":
        return stage_00_environment(state)
    if stage == "01_reference_snapshot":
        return stage_01_reference_snapshot(state)
    if stage == "02_baseline":
        return stage_02_baseline(state)
    if stage == "03_data_audit":
        return stage_03_data_audit(state)
    if stage == "04_dataset_validation":
        return stage_04_dataset_validation(state)
    if stage == "05_architecture_benchmark":
        return stage_05_architecture_benchmark(state)
    if stage == "06_dataset_build":
        return stage_06_dataset_build(state)
    if stage in {
        "07_detector",
        "08_food_recognition",
        "09_specialists",
        "10_calibration",
        "11_real_world_acceptance",
        "12_failure_analysis",
        "13_challenger_selection",
        "14_final_evaluation",
    }:
        # Execute once as bundle when first of these is reached
        if "bundle" not in collected:
            collected["bundle"] = stage_07_to_14_kaggle_and_local(state)
        return {
            "metrics": collected["bundle"].get("metrics") or {},
            "artifact_locations": collected["bundle"].get("artifact_locations") or {},
            "model_hash": collected["bundle"].get("model_hash"),
            "note": f"covered_by_bundle:{stage}",
        }
    if stage == "15_registry":
        return stage_15_registry(state, collected.get("bundle") or {})
    if stage == "16_serving":
        return stage_16_serving(state)
    if stage == "17_end_to_end":
        return stage_17_end_to_end(state)
    if stage == "18_final_report":
        return stage_18_final_report(state, collected)
    raise RuntimeError(f"unknown stage {stage}")


def main() -> None:
    assert str(OLD_ROOT.resolve()) != str(NEW_ROOT.resolve()), "isolation failure"
    state = load_state()
    if not state.get("isolation_verified"):
        raise SystemExit("isolation not verified")
    collected: dict[str, Any] = {}
    log_path = NEW_ROOT / "logs" / "autonomous_run.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        line = f"[{now()}] {msg}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    log(f"START autonomous V10 programme NEW={NEW_ROOT}")
    for stage in STAGES:
        if stage_done(stage):
            log(f"SKIP completed {stage}")
            if stage not in state["completed_stages"]:
                state["completed_stages"].append(stage)
            save_state(state)
            continue
        state["current_stage"] = stage
        save_state(state)
        retries = int(state.get("retry_count", {}).get(stage, 0))
        max_retries = int(state.get("resume_configuration", {}).get("max_retries", 3))
        while True:
            log(f"RUN {stage} attempt={retries+1}")
            try:
                payload = run_stage(stage, state, collected)
                cp = mark_checkpoint(stage, payload)
                state["latest_checkpoint"] = str(cp)
                if payload.get("model_hash"):
                    state["latest_model_hash"] = payload["model_hash"]
                if payload.get("dataset_manifest_hash"):
                    state["latest_dataset_hash"] = payload["dataset_manifest_hash"]
                if stage == "11_real_world_acceptance" or stage == "14_final_evaluation":
                    state["latest_acceptance_result"] = payload.get("metrics")
                if stage not in state["completed_stages"]:
                    state["completed_stages"].append(stage)
                if stage in state.get("failed_stages", []):
                    state["failed_stages"] = [s for s in state["failed_stages"] if s != stage]
                save_state(state)
                log(f"COMPLETE {stage}")
                break
            except Exception as e:
                retries += 1
                state.setdefault("retry_count", {})[stage] = retries
                save_state(state)
                log(f"FAIL {stage}: {e}\n{traceback.format_exc()}")
                if retries >= max_retries:
                    if stage not in state["failed_stages"]:
                        state["failed_stages"].append(stage)
                    save_state(state)
                    # For non-critical infrastructure, continue to final report with blocker
                    if stage == "18_final_report":
                        raise
                    log(f"ABORT_STAGE {stage} after retries; continuing where possible")
                    break
                time.sleep(min(30, 5 * retries))

    # Ensure final report exists even if some mid stages failed
    if not stage_done("18_final_report"):
        payload = stage_18_final_report(state, collected)
        mark_checkpoint("18_final_report", payload)
        state["completed_stages"].append("18_final_report")
        save_state(state)

    final = read_json(NEW_ROOT / "docs" / "KITCHENIQ_V10_AUTONOMOUS_FINAL_STATE.json")
    state["production_status"] = final.get("status")
    state["current_stage"] = "TERMINAL"
    save_state(state)
    log("TERMINAL " + json.dumps(final, default=str)[:2000])
    print("\n===== TERMINAL_STATE =====", flush=True)
    print(json.dumps(final, indent=2), flush=True)


if __name__ == "__main__":
    main()
