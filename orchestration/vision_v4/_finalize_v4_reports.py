"""Patch V4 final artifacts: OPEN_WORLD decision, library sha, feasibility."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

NEW = Path(r"C:\Projects\KitchenIQ-V10-AI")
OLD = Path(r"C:\Projects\KitchenIQ-OS")
DOCS = NEW / "docs/v4"
EXP = NEW / "backend/instance/dev_experiments/v10-vision-v4"
LIB = EXP / "reference_library/reference_library.json"
SUM = EXP / "reports/reference_library_summary.json"

BASELINE = 0.2927
SUBSTANTIAL = 0.4390
V4_STRICT = 0.2195


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha_file(p: Path) -> str | None:
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    summary = json.loads(SUM.read_text(encoding="utf-8")) if SUM.is_file() else {}
    lib = json.loads(LIB.read_text(encoding="utf-8"))
    lib_n = int(lib.get("n") or len(lib.get("items") or []))
    assert lib_n > 0
    lib_sha = summary.get("sha256") or lib.get("sha256") or sha_file(LIB)
    # persist sha into library file if missing
    if not lib.get("sha256") and lib_sha:
        lib["sha256"] = lib_sha
        LIB.write_text(json.dumps(lib), encoding="utf-8")

    state = json.loads((DOCS / "KITCHENIQ_VISION_V4_FINAL_STATE.json").read_text(encoding="utf-8"))
    state["final_decision"] = "V4_BLOCKED_OPEN_WORLD"
    state["production_gate"] = "FAIL"
    state["status"] = "BLOCKED"
    state["v4_strict"] = V4_STRICT
    state["best_strict"] = BASELINE
    state["best_candidate"] = "A"
    state["substantial_target"] = SUBSTANTIAL
    state["substantial_met"] = False
    state["reference_library_n"] = lib_n
    state["reference_library_sha256"] = lib_sha
    state["exact_blocker"] = (
        "V4 open-world scene engine (top-k + fusion + open-set + food-gate + retrieval n=108) "
        f"regressed to {V4_STRICT} vs baseline {BASELINE}; over-abstention / weak identity evidence "
        "remains the scene-blocking failure mode (PRIMARY=OPEN_SET). Retrieval library present but "
        "does not lift 41-scene strict completeness."
    )
    state["exact_next_technical_requirement"] = (
        "Do not launch another flat softmax retrain or empty-library retrieval. "
        "Required next: (1) production-eligible hard-negative + calibrated open-set thresholds "
        "tuned on identity_coverage failures without sacrificing the 12 currently-complete scenes; "
        "(2) for the measured misses — tomato(4), fried_rice(3), omelette(2), lasagna(2), "
        "non_food(2) — either raise specialist recall above the open-set gate with >=80 "
        "verified production images each for fried_rice/omelette (currently 0) and fill taxonomy "
        "gaps lasagna/guacamole/hummus, OR replace the crop identity path with an open-vocab "
        "detector+segmenter that emits regions before open-set (DETECTOR blocks 7 scenes); "
        "(3) keep A=0.2927 as production reference until a candidate >=0.4390."
    )
    feas = state.get("feasibility_if_blocked") or {}
    feas["remaining_gap_to_substantial_from_best"] = round(SUBSTANTIAL - BASELINE, 4)
    feas["remaining_gap_v4_candidate_to_substantial"] = round(SUBSTANTIAL - V4_STRICT, 4)
    feas["v4_vs_baseline"] = round(V4_STRICT - BASELINE, 4)
    feas["retrieval_library_n"] = lib_n
    feas["retrieval_library_sha256"] = lib_sha
    feas["retrieval_did_not_improve_strict"] = True
    state["feasibility_if_blocked"] = feas
    if "candidates" in state and "B" in state["candidates"]:
        state["candidates"]["B"]["library_sha256"] = lib_sha
        state["candidates"]["B"]["library_n"] = lib_n
        state["candidates"]["B"]["retrieval_enabled"] = True
    state["generated_at"] = now()

    # old integrity recheck
    import subprocess

    r = subprocess.run(["git", "status", "--porcelain"], cwd=OLD, capture_output=True, text=True)
    text = r.stdout or ""
    porcelain_sha = hashlib.sha256(text.encode()).hexdigest()
    base = json.loads((NEW / "reference_snapshot/OLD_PROJECT_INTEGRITY_BASELINE.json").read_text(encoding="utf-8"))
    unchanged = porcelain_sha == base.get("old_porcelain_sha256")
    state["old_project"] = "UNCHANGED" if unchanged else "CHANGED"
    state["old_project_integrity"] = {
        "porcelain_n": len([l for l in text.splitlines() if l.strip()]),
        "porcelain_sha": porcelain_sha,
        "baseline_sha": base.get("old_porcelain_sha256"),
        "unchanged": unchanged,
        "verdict": "UNCHANGED" if unchanged else "CHANGED",
    }

    bench = {
        "results": state["candidates"],
        "substantial_target": SUBSTANTIAL,
        "primary_bottleneck": state["primary_bottleneck"],
        "secondary_bottleneck": state["secondary_bottleneck"],
        "final_decision": state["final_decision"],
    }
    (DOCS / "v4_benchmark_results.json").write_text(json.dumps(bench, indent=2), encoding="utf-8")
    (DOCS / "v4_model_manifest.json").write_text(
        json.dumps(
            {
                "specialist_evidence_weights": str(
                    NEW
                    / "backend/instance/dev_experiments/v10-vision-recovery/models/identity_coverage_best.pt"
                ),
                "sha256": state.get("model_sha256"),
                "backbone": "DINOv2 ViT-B/14 frozen",
                "retrieval_library": str(LIB),
                "retrieval_library_n": lib_n,
                "retrieval_library_sha256": lib_sha,
                "engine": "VisionV4Engine",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (DOCS / "v4_dataset_manifest.json").write_text(
        json.dumps(
            {
                "production_manifest_sha256": state.get("dataset_manifest_sha256"),
                "reference_library": str(LIB),
                "reference_library_n": lib_n,
                "reference_library_sha256": lib_sha,
                "experiment_only_isolated": True,
                "no_acceptance_leakage_policy": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (DOCS / "KITCHENIQ_VISION_V4_FINAL_STATE.json").write_text(
        json.dumps(state, indent=2, default=str), encoding="utf-8"
    )
    (EXP / "reports/KITCHENIQ_VISION_V4_FINAL_STATE.json").write_text(
        json.dumps(state, indent=2, default=str), encoding="utf-8"
    )

    report = f"""# KitchenIQ Vision V4 — Final Report

**Generated:** {now()}  
**Repo:** KitchenV10 / `C:\\Projects\\KitchenIQ-V10-AI`  
**Old project:** {state['old_project']}

## 1. Baseline
Current best verified: **{BASELINE}**

## 2. Previous results
Immutable 0.2439 | specialist 0.2195 | targeted 0.0732 | recovery 0.2927 | V3 B/E 0.2439 | V4 no-retrieval 0.2195 | V4+retrieval(n={lib_n}) 0.2195

## 3–5. Root causes / bottlenecks
PRIMARY: **OPEN_SET** (10 scenes blocked)  
SECONDARY: **FOOD_NONFOOD** (9 scenes)  
TERTIARY: **DETECTOR** (7 scenes)  
IDENTITY: 3 scenes

## 6–7. Architecture
**V4_OPEN_WORLD_SCENE_ENGINE** — DINOv2 retained; top-k candidates; evidence fusion; open-set; food-gate repair; scene context; hierarchical retrieval with assert n>0.  
Rejected: blind DINOv3, blind RT-DETR, small softmax retrain, empty-library retrieval.

## 8–12. Data / model / hashes
Retrieval library PRESENT n={lib_n} sha=`{lib_sha}`  
Model SHA256: `{state.get('model_sha256')}`  
Dataset manifest SHA256: `{state.get('dataset_manifest_sha256')}`  
Fixture catalog SHA256: `{state.get('fixture_catalog_sha256')}`

## 13–15. Validation / 41-scene / failure stages
V4+retrieval strict: **{V4_STRICT}** (delta {round(V4_STRICT-BASELINE,4)})  
Best remains A: **{BASELINE}**  
Substantial target: **{SUBSTANTIAL}**  
Stages after V4: CLASSIFIER=43 DETECTOR=12 FOOD_NONFOOD=2  
Before (A): CLASSIFIER=40 DETECTOR=12 FOOD_NONFOOD=2

## 16–20. Secondary metrics
Hard-neg rate: 0.5 (2/4)  
Food-state agreement: 1.0 (16/16)  
Open-set over-abstention dominated (tomato→unknown_food even with retrieval top containing tomato at low score)

## 21–25. Security / regression / holdout / latency / GPU
Fail-closed preserved. Holdout untouched. KitchenIQ-OS {state['old_project']}.  
Kaggle: kathiresannatarajan/kiq-v10-vision-v4 (library build). Local CPU 41-scene eval.

## 26–28. Gate / decision / next
Production gate: **FAIL**  
Final decision: **V4_BLOCKED_OPEN_WORLD**  
Exact blocker: {state['exact_blocker']}  
Exact next: {state['exact_next_technical_requirement']}

## Feasibility (precise)
{json.dumps(feas, indent=2)[:4000]}
"""
    (DOCS / "KITCHENIQ_VISION_V4_FINAL_REPORT.md").write_text(report, encoding="utf-8")
    (EXP / "reports/KITCHENIQ_VISION_V4_FINAL_REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps({
        "final_decision": state["final_decision"],
        "v4_strict": V4_STRICT,
        "best": BASELINE,
        "lib_n": lib_n,
        "lib_sha": lib_sha,
        "old": state["old_project"],
    }, indent=2))


if __name__ == "__main__":
    main()
