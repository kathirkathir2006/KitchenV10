import json
from pathlib import Path

D = Path(r"C:\Projects\KitchenIQ-V10-AI\docs\v7")
r = json.loads((D / "specialist_results.json").read_text(encoding="utf-8"))
ca = json.loads((D / "confusion_analysis.json").read_text(encoding="utf-8"))
sel = json.loads((D / "specialist_selection.json").read_text(encoding="utf-8"))
print("E", {k: r["E_val"][k] for k in ("known_identity_top1_raw", "known_identity_top5_raw", "known_family_top1_raw", "known_state_top1_raw", "known_accepted_correct", "known_coverage_identity", "unknown_rejection", "false_confirmation", "objective")})
for n, c in r["combinations"].items():
    print(n, c["pairs"], c["changes"], "sum_indiv", c["sum_of_individual_net"], "interaction", c["interaction_net_loss"], "multi", c["queries_with_multiple_specialists_firing"])
    print("  ", {k: c["metrics"][k] for k in ("known_identity_top1_raw", "known_identity_top5_raw", "known_family_top1_raw", "known_state_top1_raw", "known_accepted_correct", "known_coverage_identity", "unknown_rejection", "false_confirmation", "objective")})
for s in r["specialists"]:
    print(s["pair"], s["train_images"], s["val_images"], "err", s["E_errors_involving_pair"], "rec", s["recoverable_E_errors"], s["chosen_model"], "ep", s["best_epoch"], "m", s["selected_margin"], s["control"])
    for n, v in s["val_binary"].items():
        print("   ", n, v["accuracy"], v["balanced_accuracy"], "abl@0", s["ablation_by_margin"][n]["0.0"], "@0.15", s["ablation_by_margin"][n]["0.15"])
    lc = s["learning_curve"]
    print("    LC", [(p["n_images"], p["val_balanced_accuracy"]) for p in lc["points"]], lc.get("status"), lc.get("additional_per_class"))
print("pairs top", [(p["pair"], p["total_errors"], p["recoverable_errors"], p["average_score_gap"]) for p in ca["pairs"][:14]])
print("nonqual with rec>=2", [(c["pair"], c["reasons"]) for c in sel["candidates"] if not c["qualifies"] and c["recoverable_errors"] >= 2])
print("n pairs", len(ca["pairs"]), "rec1 pairs", sum(p["recoverable_errors"] == 1 for p in ca["pairs"]), "rec1 total", sum(p["recoverable_errors"] for p in ca["pairs"] if p["recoverable_errors"] == 1))
print("changed", [(x["gt"], x["E_top5"][0], x["V7_top5"][0], x["E_ok"], x["V7_ok"]) for x in r["final_changed_or_wrong_queries"] if x["E_ok"] != x["V7_ok"]])
