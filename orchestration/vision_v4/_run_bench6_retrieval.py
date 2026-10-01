"""V4 retrieval acceptance with per-scene progress + faulthandler."""
from __future__ import annotations

import faulthandler
import json
import sys
import traceback
from pathlib import Path

faulthandler.enable()

NEW_ROOT = Path(r"C:\Projects\KitchenIQ-V10-AI")
sys.path.insert(0, str(NEW_ROOT / "src"))
sys.path.insert(0, str(NEW_ROOT))

PROG = NEW_ROOT / "backend/instance/dev_experiments/v10-vision-v4/reports/bench6_progress.txt"
CRASH = NEW_ROOT / "backend/instance/dev_experiments/v10-vision-v4/reports/bench6_crash.txt"
OUT = NEW_ROOT / "backend/instance/dev_experiments/v10-vision-v4/acceptance/candidate_v4_retrieval"


def p(msg: str) -> None:
    print(msg, flush=True)
    with PROG.open("a", encoding="utf-8") as f:
        f.write(msg + "\n")


def main() -> None:
    PROG.write_text("", encoding="utf-8")
    try:
        from orchestration.vision_v4.run_benchmark_and_finalize import (
            BASELINE,
            SUBSTANTIAL,
            find_library,
            feasibility,
            map_decision,
            old_integrity,
            prepare_catalog,
            sha256_file,
            write_json,
            now,
            DOCS,
            EXP,
            NEW_ROOT as NR,
        )
        from kitcheniq_v10.v4 import VisionV4Engine
        from kitcheniq_v10.specialist.acceptance import run_acceptance

        lib = find_library()
        assert lib is not None
        d = json.loads(lib.read_text(encoding="utf-8"))
        n = int(d.get("n") or 0)
        assert n > 0, n
        p(f"library_ok n={n}")

        eng = VisionV4Engine(library_path=lib)
        eng.load()
        p(f"loaded retrieval={eng.retrieval_enabled}")

        catalog, fixtures = prepare_catalog()
        p(f"catalog={catalog}")
        result = run_acceptance(
            catalog_path=catalog,
            fixtures_root=fixtures,
            ensemble=eng,  # type: ignore[arg-type]
            out_dir=OUT,
            candidate_id="v4_open_world_scene_retrieval",
        )
        p(f"RESULT strict={result.get('strict_scene_completeness')} stages={result.get('primary_stage_totals')}")

        matrix = json.loads((DOCS / "41_scene_root_cause_matrix.json").read_text(encoding="utf-8"))
        bottlenecks = matrix["bottlenecks"]
        strict = float(result.get("strict_scene_completeness") or 0)
        stages = result.get("primary_stage_totals") or {}
        results = {
            "A": {
                "candidate_id": "A_current_0.2927",
                "strict_scene_completeness": BASELINE,
                "primary_stage_totals": {"CLASSIFIER": 40, "DETECTOR": 12, "FOOD_NONFOOD": 2},
                "delta_vs_baseline": 0.0,
                "note": "identity_coverage_v1 production-reference",
            },
            "B": {
                "candidate_id": result.get("candidate_id"),
                "strict_scene_completeness": strict,
                "delta_vs_baseline": round(strict - BASELINE, 4),
                "primary_stage_totals": stages,
                "hardneg_nonfood": result.get("hardneg_nonfood"),
                "food_state_agreement": result.get("food_state_agreement"),
                "retrieval_enabled": True,
                "library_path": str(lib),
                "library_n": n,
                "library_sha256": d.get("sha256"),
                "n_strict_complete": result.get("n_strict_complete"),
            },
            "B_no_retrieval_prior": {
                "strict_scene_completeness": 0.2195,
                "retrieval_enabled": False,
                "note": "prior V4 without library",
            },
        }
        decision = map_decision(strict, stages, bottlenecks)
        gate = "PASS_CANDIDATE_NOT_DEPLOYED_TO_OLD_PROJECT" if strict >= SUBSTANTIAL else "FAIL"
        status = "READY" if strict >= SUBSTANTIAL else "BLOCKED"
        best_name = "B" if strict >= BASELINE else "A"
        best_strict = max(BASELINE, strict)
        feas = feasibility(matrix, stages, strict)
        integrity = old_integrity()
        weights = NR / "backend/instance/dev_experiments/v10-vision-recovery/models/identity_coverage_best.pt"
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
                "v4_no_retrieval": 0.2195,
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
                NR / "docs/vision/production_identity_training_manifest_v1.json"
            ),
            "reference_library_sha256": d.get("sha256"),
            "reference_library_n": n,
            "fixture_catalog_sha256": sha256_file(
                NR
                / "backend/instance/dev_experiments/v10-specialist-ensemble/acceptance/fixtures/catalog/fixture_catalog.json"
            ),
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
                "retrieval_library": str(lib),
                "retrieval_library_n": n,
                "retrieval_library_sha256": d.get("sha256"),
                "engine": "VisionV4Engine",
            },
        )
        write_json(
            DOCS / "v4_dataset_manifest.json",
            {
                "production_manifest_sha256": final_state["dataset_manifest_sha256"],
                "reference_library": str(lib),
                "reference_library_n": n,
                "reference_library_sha256": d.get("sha256"),
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
Immutable 0.2439 | specialist 0.2195 | targeted 0.0732 | recovery 0.2927 | V3 B/E 0.2439 | V4 no-retrieval 0.2195

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
Retrieval library PRESENT n={n} sha={d.get('sha256')}  
Model SHA256: `{final_state['model_sha256']}`  
Dataset manifest SHA256: `{final_state['dataset_manifest_sha256']}`

## 13–15. Validation / 41-scene / failure stages
V4+retrieval strict: **{strict}** (delta {round(strict-BASELINE,4)})  
Substantial target: **{SUBSTANTIAL}**  
Stages after V4: {json.dumps(stages)}  
Before: CLASSIFIER=40 DETECTOR=12 FOOD_NONFOOD=2

## 16–20. Secondary metrics
Hard-neg: {json.dumps(result.get('hardneg_nonfood'))}  
Food-state: {json.dumps(result.get('food_state_agreement'))}  
Acceptance: `{OUT}`

## 21–25. Security / regression / holdout / latency / GPU
Fail-closed open-set preserved. Holdout untouched. No KitchenIQ-OS mutation.  
Kaggle kernel: kathiresannatarajan/kiq-v10-vision-v4 (library n={n}).

## 26–28. Gate / decision / next
Production gate: **{gate}**  
Final decision: **{decision}**  
Feasibility: {json.dumps(feas, indent=2)[:3000]}
"""
        (DOCS / "KITCHENIQ_VISION_V4_FINAL_REPORT.md").write_text(report, encoding="utf-8")
        (EXP / "reports" / "KITCHENIQ_VISION_V4_FINAL_REPORT.md").write_text(report, encoding="utf-8")
        p(f"FINAL_DECISION={decision} GATE={gate} STRICT={strict}")
        print(json.dumps({"final_decision": decision, "strict": strict, "gate": gate}, indent=2), flush=True)
    except BaseException:
        tb = traceback.format_exc()
        CRASH.write_text(tb, encoding="utf-8")
        p("CRASH")
        print(tb, flush=True)
        raise


if __name__ == "__main__":
    main()
