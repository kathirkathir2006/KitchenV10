"""Diagnose why frozen-DINOv2 retrieval collapses onto one label (library-side vs query-side)."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(r"C:\Projects\KitchenIQ-V10-AI")
LIB = ROOT / "backend/instance/dev_experiments/v10-vision-v4/reference_library/reference_library.json"
EXP = ROOT / "backend/instance/dev_experiments/v10-vision-v5/v5_exp_AB_retrieval_001"


def main() -> None:
    items = json.loads(LIB.read_text(encoding="utf-8"))["items"]
    E = np.stack([np.asarray(i["embedding"], np.float32) for i in items])
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    labels = np.array([i["canonical_identity"] for i in items])
    S = E @ E.T
    np.fill_diagonal(S, np.nan)
    out: dict = {}
    # hubness: how often each item is someone's nearest neighbour
    nn = np.nanargmax(S, axis=1)
    out["lib_nn_in_degree_by_label"] = dict(Counter(labels[nn]).most_common())
    # intra vs inter class similarity
    intra, mean_to_all = {}, {}
    for l in sorted(set(labels)):
        m = labels == l
        sub = S[np.ix_(m, m)]
        intra[l] = round(float(np.nanmean(sub)), 3)
        mean_to_all[l] = round(float(np.nanmean(S[m])), 3)
    out["intra_class_mean_sim"] = intra
    out["mean_sim_to_whole_library"] = mean_to_all
    # duplicates (near-identical references)
    dup = int(np.nansum(S > 0.97) // 2)
    out["near_duplicate_pairs_gt_0.97"] = dup
    out["provenance_samples"] = defaultdict(list)
    for i in items:
        if len(out["provenance_samples"][i["canonical_identity"]]) < 3:
            out["provenance_samples"][i["canonical_identity"]].append(
                {"prov": str(i.get("provenance"))[:160], "licence": i.get("licence"), "zone": i.get("data_zone")})
    rows = json.loads((EXP / "per_region_failures.json").read_text(encoding="utf-8"))
    out["query_A_sim_stats"] = {
        "mean": round(float(np.mean([r["A_sim"] for r in rows])), 3),
        "max": max(r["A_sim"] for r in rows),
        "min": min(r["A_sim"] for r in rows),
    }
    out["query_top1_label_counts"] = dict(Counter(r["A_top1"] for r in rows).most_common())
    out["tomato_rows"] = [{k: r[k] for k in ("scene_id", "A_top1", "A_sim", "B_top5")} for r in rows if r["gt"] == "tomato"][:6]
    print(json.dumps(out, indent=2, default=list))
    (EXP / "hubness_diagnosis.json").write_text(json.dumps(out, indent=2, default=list), encoding="utf-8")


if __name__ == "__main__":
    main()
