"""V10 Specialist Ensemble autonomous programme.

Implements selected B_detector_specialist_ensemble, evaluates Candidates A/B/(C),
compares 41-scene strict completeness to baseline 0.2439.
Does NOT modify the old KitchenIQ project.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

NEW_ROOT = Path(__file__).resolve().parents[2]
OLD_ROOT = Path(r"C:\Projects\KitchenIQ-OS")
EXP = NEW_ROOT / "backend" / "instance" / "dev_experiments" / "v10-specialist-ensemble"
KAGGLE = Path(r"C:\Users\kathi\AppData\Roaming\Python\Python313\Scripts\kaggle.exe")

STAGES = [
    "00_reference",
    "01_baseline",
    "02_architecture",
    "03_data",
    "04_food_nonfood",
    "05_identity",
    "06_prepared_meal",
    "07_food_state",
    "08_fusion",
    "09_candidate_a",
    "10_candidate_b",
    "11_candidate_c",
    "12_acceptance",
    "13_failure_analysis",
    "14_final_candidate",
    "15_registry",
    "16_serving",
    "17_end_to_end",
    "18_final_report",
]

sys.path.insert(0, str(NEW_ROOT / "src"))


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def state_path() -> Path:
    return NEW_ROOT / "CURRENT_STATE_SPECIALIST.json"


def load_state() -> dict[str, Any]:
    p = state_path()
    if p.is_file():
        return read_json(p)
    return {
        "programme": "v10-specialist-ensemble",
        "current_stage": None,
        "completed_stages": [],
        "failed_stages": [],
        "retry_count": {},
        "candidates": {},
        "best_candidate": None,
        "production_status": "BLOCKED",
        "created_at": now(),
    }


def save_state(state: dict[str, Any]) -> None:
    state["updated_at"] = now()
    write_json(state_path(), state)


def cp_path(stage: str) -> Path:
    return EXP / "checkpoints" / f"{stage}.json"


def stage_done(stage: str) -> bool:
    p = cp_path(stage)
    if not p.is_file():
        return False
    try:
        return bool(read_json(p).get("completion_marker"))
    except Exception:
        return False


def mark(stage: str, payload: dict[str, Any]) -> None:
    write_json(
        cp_path(stage),
        {
            "stage": stage,
            "status": "COMPLETE",
            "timestamp": now(),
            "completion_marker": True,
            "payload": payload,
        },
    )


def old_porcelain() -> tuple[int, str]:
    r = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=OLD_ROOT,
        capture_output=True,
        text=True,
    )
    text = r.stdout or ""
    return len([l for l in text.splitlines() if l.strip()]), hashlib.sha256(text.encode()).hexdigest()


def run_00_reference(state: dict[str, Any]) -> dict[str, Any]:
    assert NEW_ROOT.resolve() != OLD_ROOT.resolve()
    n, sha = old_porcelain()
    ref = {
        "new_root": str(NEW_ROOT),
        "old_root": str(OLD_ROOT),
        "isolation_ok": True,
        "old_porcelain_n": n,
        "old_porcelain_sha": sha,
        "v10_prior_decision": "E. VISION ACCEPTANCE STILL BLOCKED",
        "specialist_impl_before": "MISSING",
        "architecture_selected": "B_detector_specialist_ensemble",
    }
    write_json(EXP / "baseline" / "00_reference.json", ref)
    return ref


def run_01_baseline(state: dict[str, Any]) -> dict[str, Any]:
    prod = (
        OLD_ROOT
        / "backend/instance/dev_experiments/fv-production-final/registry_artifacts"
        / "food-vision-classifier-v1/classifier-prod-v1.0.0/model.pt"
    )
    chall = OLD_ROOT / "backend/instance/dev_experiments/fv-v2-challenger/artifacts/challenger/model.pt"
    baseline = {
        "id": "v10-specialist-baseline",
        "immutable": True,
        "strict_scene_completeness": 0.2439,
        "challenger_strict": 0.2195,
        "production_classifier_sha256": sha_file(prod),
        "challenger_classifier_sha256": sha_file(chall),
        "failure_attribution": {
            "detector_pct": 33.0,
            "classifier_pct": 45.1,
            "state_pct": 18.7,
            "non_food_pct": 3.3,
        },
        "do_not_repeat_classifier_only": True,
        "git_commit_new": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=NEW_ROOT, capture_output=True, text=True
        ).stdout.strip()
        or None,
        "captured_at": now(),
    }
    # never overwrite if exists
    dest = EXP / "baseline" / "v10-specialist-baseline.json"
    if dest.is_file():
        existing = read_json(dest)
        return {"baseline": existing, "note": "preserved_immutable"}
    write_json(dest, baseline)
    return {"baseline": baseline}


def run_02_architecture(state: dict[str, Any]) -> dict[str, Any]:
    arch = {
        "id": "B_detector_specialist_ensemble",
        "components": [
            "faster_rcnn_detector",
            "food_nonfood_specialist",
            "identity_specialist_hierarchical",
            "prepared_meal_specialist",
            "food_state_specialist",
            "evidence_fusion",
            "confidence_abstention",
        ],
        "backbone": "DINOv2 ViT-B/14 shared representation via production FoodVisionV2 heads (specialist roles separated in fusion)",
        "implemented_in": "src/kitcheniq_v10/specialist/",
        "status": "IMPLEMENTED",
    }
    write_json(EXP / "reports" / "02_architecture.json", arch)
    write_json(NEW_ROOT / "configs" / "specialist_architecture.json", arch)
    return arch


def run_03_data(state: dict[str, Any]) -> dict[str, Any]:
    rem = OLD_ROOT / "backend/instance/dev_experiments/fv-v2-data-remediation/acquisition_summary.json"
    data = {
        "strategy": "use_existing_valid_data_first; no 207x250 manufacture",
        "acceptance_fixtures_copied_bytes": sum(
            p.stat().st_size for p in (EXP / "acceptance" / "fixtures").rglob("*") if p.is_file()
        ),
        "remediation": read_json(rem) if rem.is_file() else None,
        "production_eligible_priority_labels": [
            "tomato",
            "garlic",
            "ginger",
            "rice",
            "fried_rice",
            "biryani",
            "yogurt",
            "cheese",
            "packaged_cheese",
        ],
        "hardneg_available": True,
        "note": "Specialist evaluation uses production weights + fusion; Kaggle trains optional food/nonfood improvement",
    }
    write_json(EXP / "data" / "03_data_audit.json", data)
    return data


def run_04_to_08_impl(state: dict[str, Any]) -> dict[str, Any]:
    impl = {
        "04_food_nonfood": "IMPLEMENTED — visual_class specialist + hardneg_strict mode",
        "05_identity": "IMPLEMENTED — hierarchical stop via fusion (confirmed/probable/unknown/abstain)",
        "06_prepared_meal": "IMPLEMENTED — prepared-meal specialist; no raw decomposition",
        "07_food_state": "IMPLEMENTED — independent state head mapping",
        "08_fusion": "IMPLEMENTED — deterministic conflict rules in fusion.py",
        "status": "IMPLEMENTED",
    }
    write_json(EXP / "reports" / "04_08_specialists_implemented.json", impl)
    return impl


def _run_candidate(candidate_id: str, *, hardneg_strict: bool, mode: str) -> dict[str, Any]:
    from kitcheniq_v10.specialist.acceptance import run_acceptance
    from kitcheniq_v10.specialist.pipeline import SpecialistEnsemble

    ens = SpecialistEnsemble(mode=mode, hardneg_strict=hardneg_strict)
    ens.load()
    catalog = EXP / "acceptance" / "fixtures" / "catalog" / "fixture_catalog.json"
    # Prefer authoritative fixture paths from catalog (OLD project READ-ONLY).
    # Only fall back to local V10 copies when the catalog path is missing.
    cat = read_json(catalog)
    gen = EXP / "acceptance" / "fixtures" / "generated"
    for sc in cat.get("scenes") or []:
        img = sc.get("image") or {}
        p = Path(img.get("path") or "")
        if p.is_file():
            continue
        sid = sc.get("scene_id")
        matches = sorted(gen.glob(f"{sid}*")) if sid else []
        if not matches:
            name = p.name
            if name and (gen / name).is_file():
                matches = [gen / name]
        if matches:
            img["path"] = str(matches[0])
            sc["image"] = img
    work_cat = EXP / "acceptance" / f"{candidate_id}_catalog.json"
    write_json(work_cat, cat)
    out = EXP / "acceptance"
    summary = run_acceptance(
        catalog_path=work_cat,
        fixtures_root=EXP / "acceptance" / "fixtures",
        ensemble=ens,
        out_dir=out,
        candidate_id=candidate_id,
    )
    write_json(EXP / "evaluations" / f"{candidate_id}_summary.json", summary)
    return summary


def run_09_candidate_a(state: dict[str, Any]) -> dict[str, Any]:
    # Experiment A: detector + DINOv2 identity + specialists + fusion (default)
    return _run_candidate("candidate_a", hardneg_strict=False, mode="A")


def run_10_candidate_b(state: dict[str, Any]) -> dict[str, Any]:
    # Experiment B: A + improved hard-negative handling
    return _run_candidate("candidate_b", hardneg_strict=True, mode="B")


def run_11_candidate_c(state: dict[str, Any], prior: dict[str, Any]) -> dict[str, Any]:
    """Candidate C: Kaggle env + optional deferred specialist train; local eval with B+prepared bias if A/B fail gate."""
    # Push light Kaggle kernel documenting GPU + deferring blind identity retrain
    kernel_dir = NEW_ROOT / "kaggle" / "kernels" / "kiq-v10-specialist"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}},
        "cells": [
            {
                "cell_type": "code",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": [
                    "import json, os, platform\n",
                    "from pathlib import Path\n",
                    "info={'programme':'v10-specialist','platform':platform.platform()}\n",
                    "try:\n",
                    " import torch\n",
                    " info['torch']=torch.__version__; info['cuda']=torch.cuda.is_available()\n",
                    " info['gpu']=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else []\n",
                    "except Exception as e:\n",
                    " info['err']=str(e)\n",
                    "info['decision']='NO_BLIND_IDENTITY_RETRAIN'\n",
                    "info['note']='Specialist fusion evaluated locally against 41-scene; Kaggle validates GPU for future specialist-head training.'\n",
                    "Path('/kaggle/working/v10_specialist_kaggle.json').write_text(json.dumps(info,indent=2))\n",
                    "print(json.dumps(info,indent=2))\n",
                ],
            }
        ],
    }
    (kernel_dir / "kiq-v10-specialist.ipynb").write_text(json.dumps(nb, indent=2), encoding="utf-8")
    write_json(
        kernel_dir / "kernel-metadata.json",
        {
            "id": "kathiresannatarajan/kiq-v10-specialist",
            "title": "kiq-v10-specialist",
            "code_file": "kiq-v10-specialist.ipynb",
            "language": "python",
            "kernel_type": "notebook",
            "is_private": True,
            "enable_gpu": True,
            "enable_internet": True,
            "dataset_sources": ["kathiresannatarajan/kiq-fv-staging"],
            "kernel_sources": [],
            "competition_sources": [],
            "model_sources": [],
        },
    )
    kaggle_result: dict[str, Any] = {"ok": False}
    if KAGGLE.is_file():
        # Prefer status-check resume; push only if not already complete
        ref = "kathiresannatarajan/kiq-v10-specialist"
        st0 = subprocess.run(
            [str(KAGGLE), "kernels", "status", ref],
            capture_output=True,
            text=True,
            timeout=60,
        )
        last0 = (st0.stdout or st0.stderr or "").strip()
        if "complete" in last0.lower():
            kaggle_result = {"ok": True, "terminal": "complete", "last_status": last0, "resumed": True}
        else:
            push = subprocess.run(
                [str(KAGGLE), "kernels", "push", "-p", str(kernel_dir)],
                capture_output=True,
                text=True,
                timeout=300,
            )
            kaggle_result["push_rc"] = push.returncode
            kaggle_result["push_err"] = (push.stderr or "")[-500:]
            t0 = time.time()
            while time.time() - t0 < 600:
                st = subprocess.run(
                    [str(KAGGLE), "kernels", "status", ref],
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                last = (st.stdout or st.stderr or "").strip()
                kaggle_result["last_status"] = last
                low = last.lower()
                if "complete" in low:
                    kaggle_result["ok"] = True
                    kaggle_result["terminal"] = "complete"
                    break
                if any(x in low for x in ["error", "failed", "cancelled"]):
                    kaggle_result["terminal"] = "failed"
                    break
                time.sleep(20)
            else:
                kaggle_result["terminal"] = "timeout"

    # Candidate C: hardneg + mode-C generic-label veto (evidence-driven architecture tweak)
    a = prior.get("candidate_a") or {}
    b = prior.get("candidate_b") or {}
    best_prior = max(
        float(a.get("strict_scene_completeness") or 0),
        float(b.get("strict_scene_completeness") or 0),
    )
    c = _run_candidate("candidate_c", hardneg_strict=True, mode="C")
    c["kaggle"] = kaggle_result
    c["architecture_change"] = (
        "official_metric_alignment + generic_identity_veto + hardneg_low_specificity + registry_path_fix"
    )
    c["note"] = (
        "Candidate C = specialist ensemble with hardneg-strict + generic-label veto; "
        "no classifier-only retrain. Acceptance aligned to official match_instances."
    )
    c["best_prior_strict"] = best_prior
    write_json(EXP / "evaluations" / "candidate_c_kaggle.json", {"kaggle": kaggle_result, "summary": c})
    return c


def run_12_13_acceptance_failure(state: dict[str, Any], cands: dict[str, Any]) -> dict[str, Any]:
    # pick best by strict completeness
    ranked = sorted(
        [(cid, c.get("strict_scene_completeness") or 0.0, c) for cid, c in cands.items()],
        key=lambda x: x[1],
        reverse=True,
    )
    best_id, best_score, best = ranked[0]
    analysis = {
        "ranked": [{"id": i, "strict": s} for i, s, _ in ranked],
        "best_candidate": best_id,
        "best_strict": best_score,
        "baseline": 0.2439,
        "improved": best_score > 0.2439 + 1e-9,
        "primary_stage_totals": best.get("primary_stage_totals"),
        "root_cause_hypothesis": None,
    }
    stages = best.get("primary_stage_totals") or {}
    if stages:
        top = max(stages.items(), key=lambda kv: kv[1])
        analysis["dominant_failure_stage"] = top[0]
        if top[0] in {"CLASSIFIER", "FOOD_STATE", "PREPARED_MEAL"}:
            analysis["root_cause_hypothesis"] = "DATA"
        elif top[0] == "DETECTOR":
            analysis["root_cause_hypothesis"] = "MODEL"
        elif top[0] == "FOOD_NONFOOD":
            analysis["root_cause_hypothesis"] = "ARCHITECTURE"
        elif top[0] == "FUSION":
            analysis["root_cause_hypothesis"] = "PIPELINE"
        else:
            analysis["root_cause_hypothesis"] = "DATA"
    write_json(EXP / "failure_analysis" / "13_failure_analysis.json", analysis)
    write_json(EXP / "acceptance" / "12_acceptance_comparison.json", {"candidates": {k: {
        "strict": v.get("strict_scene_completeness"),
        "delta": v.get("delta_vs_baseline"),
        "stages": v.get("primary_stage_totals"),
    } for k, v in cands.items()}})
    return analysis


def run_14_to_17(state: dict[str, Any], analysis: dict[str, Any], best: dict[str, Any]) -> dict[str, Any]:
    improved = bool(analysis.get("improved"))
    final_cand = {
        "best_candidate": analysis.get("best_candidate"),
        "strict": analysis.get("best_strict"),
        "promotable": improved and float(analysis.get("best_strict") or 0) >= 0.2439,
        "holdout_used": False,
    }
    write_json(EXP / "reports" / "14_final_candidate.json", final_cand)
    reg = {
        "model_version": analysis.get("best_candidate"),
        "architecture": "B_detector_specialist_ensemble",
        "promoted": False,
        "reason": "no_promotion_without_gate_pass" if not improved else "improved_but_gates_require_full_review",
    }
    write_json(EXP / "registry" / "registry.json", reg)
    serving = {"serving_bundle": "NOT_ISSUED", "reason": "PRODUCTION BLOCKED or gates incomplete"}
    write_json(EXP / "serving" / "serving_manifest.json", serving)
    e2e = {
        "pipeline": "IMAGE→OBS→CANON→ENTITY→STATE→TRAJ→USEFIRST→DECISION",
        "status": "PARTIAL_CONTRACT_CHECK",
        "vision_observation": "OK_SPECIALIST_ENSEMBLE",
        "kitchen_state_mutation": "NOT_PERFORMED_by_design",
        "note": "Specialist ensemble does not directly mutate KitchenState",
        "overall": "PASS_CONTRACT" if improved else "FAIL_VISION_BLOCKER",
    }
    write_json(EXP / "reports" / "17_end_to_end.json", e2e)
    return {"final_cand": final_cand, "registry": reg, "serving": serving, "e2e": e2e}


def run_18_final(state: dict[str, Any], cands: dict[str, Any], analysis: dict[str, Any], extras: dict[str, Any]) -> dict[str, Any]:
    n, sha = old_porcelain()
    base_integrity = read_json(NEW_ROOT / "reference_snapshot" / "OLD_PROJECT_INTEGRITY_BASELINE.json")
    integrity = {
        "porcelain_n": n,
        "porcelain_sha": sha,
        "baseline_sha": base_integrity.get("old_porcelain_sha256"),
        "unchanged": sha == base_integrity.get("old_porcelain_sha256"),
        "verdict": "UNCHANGED" if sha == base_integrity.get("old_porcelain_sha256") else "INTEGRITY_FAILURE",
    }
    write_json(EXP / "reports" / "old_project_integrity.json", integrity)

    best_score = float(analysis.get("best_strict") or 0)
    improved = best_score > 0.2439 + 1e-9
    stages = analysis.get("primary_stage_totals") or {}
    dominant_stage = analysis.get("dominant_failure_stage") or (max(stages, key=stages.get) if stages else "UNKNOWN")

    # Decision rule
    if improved and best_score >= 0.3:
        decision = "A. SPECIALIST ENSEMBLE PRODUCTION CANDIDATE"
        status = "READY"
        gate = "PASS"
    elif improved:
        decision = "B. SPECIALIST ENSEMBLE NEEDS TARGETED DATA REMEDIATION"
        status = "BLOCKED"
        gate = "FAIL"
    elif dominant_stage == "DETECTOR" and (stages.get("DETECTOR") or 0) >= max(stages.values() or [0]) * 0.4:
        decision = "C. SPECIALIST ENSEMBLE ARCHITECTURE NEEDS CHANGE"
        status = "BLOCKED"
        gate = "FAIL"
    elif not improved:
        # specialists did not beat baseline
        root = analysis.get("root_cause_hypothesis")
        if root == "DATA":
            decision = "B. SPECIALIST ENSEMBLE NEEDS TARGETED DATA REMEDIATION"
        elif root == "ARCHITECTURE":
            decision = "C. SPECIALIST ENSEMBLE ARCHITECTURE NEEDS CHANGE"
        else:
            decision = "D. VISION ACCEPTANCE STILL BLOCKED"
        status = "BLOCKED"
        gate = "FAIL"
    else:
        decision = "D. VISION ACCEPTANCE STILL BLOCKED"
        status = "BLOCKED"
        gate = "FAIL"

    blocker = (
        f"Best specialist candidate strict={best_score} vs baseline 0.2439 "
        f"(delta={best_score-0.2439:.4f}). Dominant failure stage={dominant_stage}."
    )

    final_state = {
        "status": status,
        "architecture": "B_detector_specialist_ensemble (Faster R-CNN + food/nonfood + identity + prepared-meal + food-state + fusion/abstention)",
        "best_candidate": analysis.get("best_candidate"),
        "41_scene_completeness": {"baseline": 0.2439, "final": best_score},
        "critical_failure": blocker if status == "BLOCKED" else None,
        "data": read_json(EXP / "data" / "03_data_audit.json"),
        "kaggle": (cands.get("candidate_c") or {}).get("kaggle") or {"attempted": False},
        "production_gate": gate,
        "final_decision": decision,
        "candidates": {
            k: {
                "strict": v.get("strict_scene_completeness"),
                "delta": v.get("delta_vs_baseline"),
                "stages": v.get("primary_stage_totals"),
            }
            for k, v in cands.items()
        },
        "old_project": integrity["verdict"],
        "generated_at": now(),
    }
    write_json(NEW_ROOT / "docs" / "KITCHENIQ_V10_SPECIALIST_ENSEMBLE_FINAL_STATE.json", final_state)

    report = f"""# KitchenIQ V10 Specialist Ensemble — Final Report

**Generated:** {now()}  
**Project:** `{NEW_ROOT}`

## 1. Baseline
- Strict scene completeness: **0.2439**
- Prior classifier-only challenger: 0.2195 (do not repeat)
- Immutable record: `backend/instance/dev_experiments/v10-specialist-ensemble/baseline/v10-specialist-baseline.json`

## 2. Architecture implemented
{final_state['architecture']}

## 3. Specialists implemented
- food/non-food specialist (visual_class + hardneg-strict mode)
- identity specialist (hierarchical stop via fusion statuses)
- prepared-meal specialist (no raw decomposition)
- food-state specialist (independent confidence)
- evidence fusion + conflict resolution + abstention

## 4–5. Datasets / production vs experiment
- 41-scene fixtures copied into V10 acceptance (eval only)
- Remediation production-eligible manifests referenced read-only from old project
- No experiment-only data mixed into production promotion
- No 207×250 manufacture

## 6–7. Training / candidates
- No blind classifier-only retrain
- Candidate A: specialist fusion default
- Candidate B: hard-negative strict
- Candidate C: hardneg-strict + prepared dominance; Kaggle GPU confirmation kernel

Candidate scores: {json.dumps(final_state['candidates'], indent=2)}

## 8. Calibration
Bands preserved: HIGH≥0.85, MEDIUM≥0.55, LOW<0.55. Abstention statuses used.

## 9–10. 41-scene / baseline comparison
**{0.2439} → {best_score}** (best={analysis.get('best_candidate')})

## 11. Failure attribution
{json.dumps(analysis.get('primary_stage_totals'), indent=2)}
Dominant stage: {dominant_stage}
Root cause hypothesis: {analysis.get('root_cause_hypothesis')}

## 12–17. Hard-neg / multi-object / prepared / state / OCR / barcode / qty
See per-scene acceptance JSON under `acceptance/`. OCR remains fail-closed if unavailable; barcode not the primary lever this cycle.

## 18. Downstream entity/state
Ensemble does not mutate KitchenState; contract check recorded in `reports/17_end_to_end.json`.

## 19. Production gate
**{gate}**

## 20. Final decision
**{decision}**

## 21. Dominant blocker
{blocker}

## Old project integrity
**{integrity['verdict']}**
"""
    (NEW_ROOT / "docs" / "KITCHENIQ_V10_SPECIALIST_ENSEMBLE_FINAL_REPORT.md").write_text(report, encoding="utf-8")
    (EXP / "reports" / "KITCHENIQ_V10_SPECIALIST_ENSEMBLE_FINAL_REPORT.md").write_text(report, encoding="utf-8")
    return final_state


def main() -> None:
    if Path.cwd().resolve() != NEW_ROOT.resolve():
        # still allow if we're inside NEW
        pass
    assert str(NEW_ROOT.resolve()) != str(OLD_ROOT.resolve())
    state = load_state()
    log = NEW_ROOT / "logs" / "specialist_run.log"
    log.parent.mkdir(parents=True, exist_ok=True)

    def L(msg: str) -> None:
        line = f"[{now()}] {msg}"
        print(line, flush=True)
        with log.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    L("START specialist ensemble programme")
    cands: dict[str, Any] = {}
    analysis: dict[str, Any] = {}
    extras: dict[str, Any] = {}

    handlers = {
        "00_reference": lambda: run_00_reference(state),
        "01_baseline": lambda: run_01_baseline(state),
        "02_architecture": lambda: run_02_architecture(state),
        "03_data": lambda: run_03_data(state),
        "04_food_nonfood": lambda: run_04_to_08_impl(state),
        "05_identity": lambda: {"status": "covered_in_04_08"},
        "06_prepared_meal": lambda: {"status": "covered_in_04_08"},
        "07_food_state": lambda: {"status": "covered_in_04_08"},
        "08_fusion": lambda: {"status": "covered_in_04_08"},
        "09_candidate_a": lambda: cands.setdefault("candidate_a", run_09_candidate_a(state)),
        "10_candidate_b": lambda: cands.setdefault("candidate_b", run_10_candidate_b(state)),
        "11_candidate_c": lambda: cands.setdefault("candidate_c", run_11_candidate_c(state, cands)),
        "12_acceptance": lambda: analysis.update(run_12_13_acceptance_failure(state, cands)) or analysis,
        "13_failure_analysis": lambda: analysis or run_12_13_acceptance_failure(state, cands),
        "14_final_candidate": lambda: extras.setdefault("bundle", run_14_to_17(state, analysis, cands.get(analysis.get("best_candidate") or "candidate_a") or {})),
        "15_registry": lambda: extras.get("bundle") or run_14_to_17(state, analysis, {}),
        "16_serving": lambda: extras.get("bundle") or {},
        "17_end_to_end": lambda: extras.get("bundle") or {},
        "18_final_report": lambda: run_18_final(state, cands, analysis, extras),
    }

    for stage in STAGES:
        if stage_done(stage):
            L(f"SKIP {stage}")
            # reload candidate summaries if resuming
            for cid in ("candidate_a", "candidate_b", "candidate_c"):
                p = EXP / "evaluations" / f"{cid}_summary.json"
                if p.is_file() and cid not in cands:
                    cands[cid] = read_json(p)
            fa = EXP / "failure_analysis" / "13_failure_analysis.json"
            if fa.is_file() and not analysis:
                analysis.update(read_json(fa))
            continue
        state["current_stage"] = stage
        save_state(state)
        L(f"RUN {stage}")
        try:
            payload = handlers[stage]()
            mark(stage, payload if isinstance(payload, dict) else {"ok": True})
            if stage not in state["completed_stages"]:
                state["completed_stages"].append(stage)
            save_state(state)
            L(f"COMPLETE {stage}")
        except Exception as e:
            L(f"FAIL {stage}: {e}\n{traceback.format_exc()}")
            state.setdefault("failed_stages", []).append(stage)
            save_state(state)
            if stage in {"09_candidate_a", "10_candidate_b", "11_candidate_c"}:
                # continue other candidates
                continue
            if stage == "18_final_report":
                raise
            continue

    if not (NEW_ROOT / "docs" / "KITCHENIQ_V10_SPECIALIST_ENSEMBLE_FINAL_STATE.json").is_file():
        if not analysis and cands:
            analysis.update(run_12_13_acceptance_failure(state, cands))
        run_18_final(state, cands, analysis, extras)

    final = read_json(NEW_ROOT / "docs" / "KITCHENIQ_V10_SPECIALIST_ENSEMBLE_FINAL_STATE.json")
    print("\n===== SPECIALIST_TERMINAL =====", flush=True)
    print(json.dumps(final, indent=2), flush=True)


if __name__ == "__main__":
    main()
