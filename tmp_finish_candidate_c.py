"""Run candidate C eval only; write summary; then finalize report stages."""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

NEW_ROOT = Path(r"C:\Projects\KitchenIQ-V10-AI")
sys.path.insert(0, str(NEW_ROOT / "src"))
sys.path.insert(0, str(NEW_ROOT / "orchestration" / "specialist"))

from run_specialist_programme import (  # noqa: E402
    EXP,
    load_state,
    mark,
    read_json,
    run_12_13_acceptance_failure,
    run_14_to_17,
    run_18_final,
    save_state,
    write_json,
    _run_candidate,
)

print("START candidate_c isolated", flush=True)
try:
    c = _run_candidate("candidate_c", hardneg_strict=True, mode="C")
    c["kaggle"] = {"ok": True, "terminal": "complete", "resumed": True, "note": "prior kernel"}
    c["architecture_change"] = (
        "official_metric_alignment + generic_identity_veto + hardneg_low_specificity + registry_path_fix"
    )
    write_json(EXP / "evaluations" / "candidate_c_kaggle.json", {"kaggle": c["kaggle"], "summary": c})
    mark("11_candidate_c", c)
    state = load_state()
    if "11_candidate_c" not in state.get("completed_stages", []):
        state.setdefault("completed_stages", []).append("11_candidate_c")
    save_state(state)
    print("C strict", c.get("strict_scene_completeness"), "delta", c.get("delta_vs_baseline"), flush=True)

    cands = {
        "candidate_a": read_json(EXP / "evaluations" / "candidate_a_summary.json"),
        "candidate_b": read_json(EXP / "evaluations" / "candidate_b_summary.json"),
        "candidate_c": c,
    }
    analysis = run_12_13_acceptance_failure(state, cands)
    mark("12_acceptance", analysis)
    mark("13_failure_analysis", analysis)
    for s in ("12_acceptance", "13_failure_analysis"):
        if s not in state["completed_stages"]:
            state["completed_stages"].append(s)
    save_state(state)
    print("analysis best", analysis.get("best_candidate"), analysis.get("best_strict"), flush=True)

    best = cands.get(analysis.get("best_candidate") or "candidate_a") or {}
    extras = {"bundle": run_14_to_17(state, analysis, best)}
    for s in ("14_final_candidate", "15_registry", "16_serving", "17_end_to_end"):
        mark(s, extras["bundle"])
        if s not in state["completed_stages"]:
            state["completed_stages"].append(s)
    final = run_18_final(state, cands, analysis, extras)
    mark("18_final_report", final)
    if "18_final_report" not in state["completed_stages"]:
        state["completed_stages"].append("18_final_report")
    save_state(state)
    print("FINAL", json.dumps({k: final.get(k) for k in ("status", "best_candidate", "41_scene_completeness", "production_gate", "final_decision", "old_project")}, indent=2), flush=True)
except Exception:
    print("FATAL", traceback.format_exc(), flush=True)
    raise
