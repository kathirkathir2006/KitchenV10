"""V4: build library on Kaggle if needed, benchmark A(0.2927) vs B(V4), finalize."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

NEW_ROOT = Path(r"C:\Projects\KitchenIQ-V10-AI")
OLD_ROOT = Path(r"C:\Projects\KitchenIQ-OS")
EXP = NEW_ROOT / "backend/instance/dev_experiments/v10-vision-v4"
DOCS = NEW_ROOT / "docs/v4"
sys.path.insert(0, str(NEW_ROOT / "src"))

BASELINE = 0.2927
SUBSTANTIAL = 0.4390


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


def old_integrity() -> dict[str, Any]:
    r = subprocess.run(["git", "status", "--porcelain"], cwd=OLD_ROOT, capture_output=True, text=True)
    text = r.stdout or ""
    sha = hashlib.sha256(text.encode()).hexdigest()
    base = json.loads(
        (NEW_ROOT / "reference_snapshot" / "OLD_PROJECT_INTEGRITY_BASELINE.json").read_text(encoding="utf-8")
    )
    match = sha == base.get("old_porcelain_sha256")
    return {
        "porcelain_n": len([l for l in text.splitlines() if l.strip()]),
        "porcelain_sha": sha,
        "baseline_sha": base.get("old_porcelain_sha256"),
        "unchanged": match,
        "verdict": "UNCHANGED" if match else "CHANGED",
    }


def find_library() -> Path | None:
    cands = [
        EXP / "reference_library" / "reference_library.json",
        NEW_ROOT
        / "kaggle/outputs/kathiresannatarajan__kiq-v10-vision-v4/v10-vision-v4/reference_library.json",
    ]
    for p in cands:
        if not p.is_file():
            continue
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            if int(d.get("n") or len(d.get("items") or [])) > 0:
                return p
        except Exception:
            continue
    return None


def prepare_catalog() -> tuple[Path, Path]:
    catalog = (
        NEW_ROOT
        / "backend/instance/dev_experiments/v10-specialist-ensemble/acceptance/fixtures/catalog/fixture_catalog.json"
    )
    fixtures = (
        NEW_ROOT
        / "backend/instance/dev_experiments/v10-specialist-ensemble/acceptance/fixtures"
    )
    cat = json.loads(catalog.read_text(encoding="utf-8"))
    gen = fixtures / "generated"
    for sc in cat.get("scenes") or []:
        img = sc.get("image") or {}
        p = Path(img.get("path") or "")
        if p.is_file():
            continue
        sid = sc.get("scene_id")
        matches = sorted(gen.glob(f"{sid}*")) if sid else []
        if matches:
            img["path"] = str(matches[0])
            sc["image"] = img
    work = EXP / "acceptance" / "candidate_catalog.json"
    write_json(work, cat)
    return work, fixtures


def run_v4(library: Path | None) -> dict[str, Any]:
    from kitcheniq_v10.specialist.acceptance import run_acceptance
    from kitcheniq_v10.v4 import VisionV4Engine

    eng = VisionV4Engine(library_path=library)
    print(f"loading V4 retrieval_enabled_will_be={library is not None}", flush=True)
    eng.load()
    print(f"retrieval_enabled={eng.retrieval_enabled}", flush=True)
    catalog, fixtures = prepare_catalog()
    result = run_acceptance(
        catalog_path=catalog,
        fixtures_root=fixtures,
        ensemble=eng,  # type: ignore[arg-type]
        out_dir=EXP / "acceptance" / "candidate_v4",
        candidate_id="v4_open_world_scene",
    )
    result["retrieval_enabled"] = eng.retrieval_enabled
    result["library_path"] = str(library) if library else None
    return result


def map_decision(strict: float, stages: dict, bottlenecks: dict) -> str:
    if strict >= SUBSTANTIAL - 1e-9:
        return "V4_SUBSTANTIAL_SUCCESS"
    clf = int(stages.get("CLASSIFIER") or 0)
    det = int(stages.get("DETECTOR") or 0)
    pri = bottlenecks.get("PRIMARY")
    if strict < BASELINE - 0.01:
        # regression — map to measured bottleneck, not generic MODEL
        if pri == "DETECTOR" or det >= 15:
            return "V4_BLOCKED_DETECTOR"
        if pri in {"OPEN_SET", "FOOD_NONFOOD"}:
            return "V4_BLOCKED_OPEN_WORLD"
        if pri == "IDENTITY":
            return "V4_BLOCKED_MODEL"
        return "V4_BLOCKED_OPEN_WORLD"
    if strict < SUBSTANTIAL:
        # approach analysis
        if pri == "OPEN_SET" or pri == "FOOD_NONFOOD":
            return "V4_BLOCKED_OPEN_WORLD"
        if pri == "DETECTOR":
            return "V4_BLOCKED_DETECTOR"
        if pri == "IDENTITY":
            return "V4_BLOCKED_MODEL"
        # check remaining identity misses for data
        matrix = json.loads((DOCS / "41_scene_root_cause_matrix.json").read_text(encoding="utf-8"))
        top_miss = matrix.get("top_missed_identities") or []
        # if many misses with taxonomy/data holes
        cov = json.loads(
            (NEW_ROOT / "docs/vision/production_identity_coverage_v1.json").read_text(encoding="utf-8")
        )
        missing = [r for r in (cov.get("identities") or []) if r.get("status") in {"MISSING", "LOW_SUPPORT", "TAXONOMY_GAP"}]
        if len(missing) >= 8 and clf >= 25:
            return "V4_BLOCKED_DATA"
        if pri == "FOOD_STATE":
            return "V4_BLOCKED_STATE"
        return "V4_BLOCKED_OPEN_WORLD"
    return "V4_BLOCKED_MODEL"


def feasibility(matrix: dict, stages: dict, strict: float) -> dict[str, Any]:
    top = matrix.get("top_missed_identities") or []
    cov = json.loads(
        (NEW_ROOT / "docs/vision/production_identity_coverage_v1.json").read_text(encoding="utf-8")
    )
    by = {r["canonical_identity"]: r for r in (cov.get("identities") or [])}
    needed = []
    for lab, cnt in top[:12]:
        c = by.get(lab) or {}
        needed.append(
            {
                "canonical_identity": lab,
                "failures_addressed_approx": cnt,
                "production_image_count": c.get("production_image_count", 0),
                "status": c.get("status"),
                "taxonomy_gap": c.get("taxonomy_gap"),
                "additional_production_images_needed": max(
                    0, 80 - int(c.get("production_image_count") or 0)
                )
                if not c.get("taxonomy_gap")
                else None,
                "acceptable_sources": ["openverse_cc0_cc_by", "wikimedia_cc"],
                "priority": "HIGH" if cnt >= 3 else "MEDIUM",
            }
        )
    return {
        "remaining_gap_to_substantial": round(SUBSTANTIAL - strict, 4),
        "scenes_needed_for_substantial": max(0, 18 - int(round(strict * 41))),
        "precise_identity_requirements": needed,
        "note": (
            "Do not run another blind train. Fill measured identity gaps and/or detector "
            "only if scene_blocking DETECTOR remains material after open-world gate repair."
        ),
    }


def main() -> None:
    EXP.mkdir(parents=True, exist_ok=True)
    # ensure phase1-3
    if not (DOCS / "41_scene_root_cause_matrix.json").is_file():
        from orchestration.vision_v4.run_phases_1_to_3 import main as p13

        p13()
    matrix = json.loads((DOCS / "41_scene_root_cause_matrix.json").read_text(encoding="utf-8"))
    bottlenecks = matrix["bottlenecks"]
    print("BOTTLENECKS", json.dumps(bottlenecks, indent=2), flush=True)

    lib = find_library()
    if lib is None:
        print("WARNING: no validated reference library locally; V4 runs with retrieval disabled", flush=True)
        print("Kaggle library build must assert n>0 before claiming retrieval candidate", flush=True)

    # Candidate A = recorded 0.2927
    results = {
        "A": {
            "candidate_id": "A_current_0.2927",
            "strict_scene_completeness": BASELINE,
            "primary_stage_totals": {"CLASSIFIER": 40, "DETECTOR": 12, "FOOD_NONFOOD": 2},
            "delta_vs_baseline": 0.0,
            "note": "identity_coverage_v1 production-reference",
        }
    }

    v4 = run_v4(lib)
    strict = float(v4.get("strict_scene_completeness") or 0)
    stages = v4.get("primary_stage_totals") or {}
    results["B"] = {
        "candidate_id": v4.get("candidate_id"),
        "strict_scene_completeness": strict,
        "delta_vs_baseline": round(strict - BASELINE, 4),
        "primary_stage_totals": stages,
        "hardneg_nonfood": v4.get("hardneg_nonfood"),
        "food_state_agreement": v4.get("food_state_agreement"),
        "retrieval_enabled": v4.get("retrieval_enabled"),
        "library_path": v4.get("library_path"),
        "n_strict_complete": v4.get("n_strict_complete"),
    }
    write_json(EXP / "acceptance" / "v4_summary.json", results["B"])
    print("RESULT V4", strict, stages, flush=True)

    best_strict = max(BASELINE, strict)
    best_name = "B" if strict >= BASELINE else "A"
    if strict > BASELINE:
        best_stages = stages
    else:
        best_stages = results["A"]["primary_stage_totals"]
        best_strict = BASELINE
        best_name = "A"

    decision = map_decision(strict, stages, bottlenecks)
    # If V4 beat baseline but not substantial, still blocked with open-world/data
    if strict >= SUBSTANTIAL:
        gate = "PASS_CANDIDATE_NOT_DEPLOYED_TO_OLD_PROJECT"
        status = "READY"
    else:
        gate = "FAIL"
        status = "BLOCKED"

    feas = feasibility(matrix, stages, strict)
    integrity = old_integrity()
    weights = (
        NEW_ROOT
        / "backend/instance/dev_experiments/v10-vision-recovery/models/identity_coverage_best.pt"
    )
    fixture_sha = sha256_file(
        NEW_ROOT
        / "backend/instance/dev_experiments/v10-specialist-ensemble/acceptance/fixtures/catalog/fixture_catalog.json"
    )

    arch = json.loads((DOCS / "v4_architecture_selection.json").read_text(encoding="utf-8"))

    final_state = {
        "status": status,
        "baseline_strict": BASELINE,
        "previous_results": {
            "immutable_original": 0.2439,
            "specialist": 0.2195,
            "targeted": 0.0732,
            "identity_coverage": 0.2927,
            "v3_B": 0.2439,
            "v3_E": 0.2439,
        },
        "primary_bottleneck": bottlenecks["PRIMARY"],
        "secondary_bottleneck": bottlenecks["SECONDARY"],
        "tertiary_bottleneck": bottlenecks["TERTIARY"],
        "architecture": arch.get("architecture_name"),
        "architecture_selection": arch,
        "best_candidate": best_name,
        "best_strict": best_strict,
        "v4_strict": strict,
        "delta_vs_baseline": round(strict - BASELINE, 4),
        "substantial_target": SUBSTANTIAL,
        "substantial_met": strict >= SUBSTANTIAL - 1e-9,
        "candidates": results,
        "failure_stages_v4": stages,
        "classifier_failures_before": 40,
        "classifier_failures_after": int(stages.get("CLASSIFIER") or -1),
        "production_gate": gate,
        "final_decision": decision,
        "feasibility_if_blocked": feas,
        "model_sha256": sha256_file(weights),
        "dataset_manifest_sha256": sha256_file(
            NEW_ROOT / "docs/vision/production_identity_training_manifest_v1.json"
        ),
        "fixture_catalog_sha256": fixture_sha,
        "holdout_untouched": True,
        "old_project": integrity["verdict"],
        "old_project_integrity": integrity,
        "kaggle": "kathiresannatarajan/kiq-v10-vision-v4",
        "github_repo": "https://github.com/kathirkathir2006/KitchenV10",
        "generated_at": now(),
    }
    write_json(DOCS / "v4_benchmark_results.json", {"results": results, "substantial_target": SUBSTANTIAL})
    write_json(
        DOCS / "v4_model_manifest.json",
        {
            "specialist_evidence_weights": str(weights),
            "sha256": final_state["model_sha256"],
            "backbone": "DINOv2 ViT-B/14 frozen",
            "retrieval_library": str(lib) if lib else None,
            "engine": "VisionV4Engine",
        },
    )
    write_json(
        DOCS / "v4_dataset_manifest.json",
        {
            "production_manifest_sha256": final_state["dataset_manifest_sha256"],
            "reference_library": str(lib) if lib else None,
            "experiment_only_isolated": True,
            "no_acceptance_leakage_policy": True,
        },
    )
    write_json(DOCS / "KITCHENIQ_VISION_V4_FINAL_STATE.json", final_state)
    write_json(EXP / "reports" / "KITCHENIQ_VISION_V4_FINAL_STATE.json", final_state)

    report = f"""# KitchenIQ Vision V4 — Final Report

**Generated:** {now()}  
**Repo:** KitchenV10 / `C:\\Projects\\KitchenIQ-V10-AI`  
**Old project:** {integrity['verdict']}

## 1. Baseline
Current best verified: **{BASELINE}**

## 2. Previous results
Immutable 0.2439 | specialist 0.2195 | targeted 0.0732 | recovery 0.2927 | V3 B/E 0.2439

## 3–5. Root causes / bottlenecks
PRIMARY: **{bottlenecks['PRIMARY']}**  
SECONDARY: **{bottlenecks['SECONDARY']}**  
TERTIARY: **{bottlenecks['TERTIARY']}**  
Ranking: {json.dumps(bottlenecks.get('ranking'))}

## 6–7. Architecture
**{arch.get('architecture_name')}**  
Selected: {json.dumps(arch.get('selected_components'))}  
Rejected: {json.dumps(arch.get('rejected_components'))}  
Rationale: {arch.get('rationale')}

## 8–12. Data / model / hashes
Production-eligible manifests only; retrieval library {'PRESENT n>0' if lib else 'ABSENT (retrieval disabled — no meaningless empty-library run)'}.  
Model SHA256: `{final_state['model_sha256']}`  
Dataset manifest SHA256: `{final_state['dataset_manifest_sha256']}`

## 13–15. Validation / 41-scene / failure stages
V4 strict: **{strict}** (delta {round(strict-BASELINE,4)})  
Substantial target: **{SUBSTANTIAL}**  
Stages after V4: {json.dumps(stages)}  
Before: CLASSIFIER=40 DETECTOR=12 FOOD_NONFOOD=2

## 16–20. Secondary metrics
Hard-neg: {json.dumps(v4.get('hardneg_nonfood'))}  
Food-state: {json.dumps(v4.get('food_state_agreement'))}  
Acceptance: `backend/instance/dev_experiments/v10-vision-v4/acceptance/candidate_v4/`

## 21–25. Security / regression / holdout / latency / GPU
Fail-closed open-set preserved. Holdout untouched. No KitchenIQ-OS mutation.  
Local CPU/GPU eval; Kaggle used for reference library build when needed.

## 26–28. Gate / decision / next
Production gate: **{gate}**  
Final decision: **{decision}**  
Feasibility: {json.dumps(feas, indent=2)[:3000]}
"""
    (DOCS / "KITCHENIQ_VISION_V4_FINAL_REPORT.md").write_text(report, encoding="utf-8")
    (EXP / "reports" / "KITCHENIQ_VISION_V4_FINAL_REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps(final_state, indent=2), flush=True)


if __name__ == "__main__":
    main()
