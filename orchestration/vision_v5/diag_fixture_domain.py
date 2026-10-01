"""Classify each frozen fixture as synthetic-drawing vs photographic (read-only) and split A/B results."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(r"C:\Projects\KitchenIQ-V10-AI")
CATALOG = ROOT / "backend/instance/dev_experiments/v10-specialist-ensemble/acceptance/fixtures/catalog/fixture_catalog.json"
EXP = ROOT / "backend/instance/dev_experiments/v10-vision-v5/v5_exp_AB_retrieval_001"
GEN = Path(r"C:\Projects\KitchenIQ-OS\backend\instance\dev_experiments\fv-real-world-acceptance-v2\fixtures\generated")


def domain(p: Path) -> dict:
    img = Image.open(p).convert("RGB")
    a = np.asarray(img.resize((128, 128)))
    q = (a // 8).reshape(-1, 3)
    uniq = len({tuple(x) for x in q})
    grad = float(np.abs(np.diff(a.astype(float), axis=1)).mean())
    synthetic = uniq < 600
    return {"size": img.size, "unique_colours_q": uniq, "mean_grad": round(grad, 2),
            "domain": "SYNTHETIC_DRAWING" if synthetic else "PHOTO"}


def main() -> None:
    cat = json.loads(CATALOG.read_text(encoding="utf-8"))
    dom = {}
    for sc in cat["scenes"]:
        p = Path((sc.get("image") or {}).get("path") or "")
        if not p.is_file():
            found = sorted(GEN.glob(f"{sc['scene_id']}*"))
            p = found[0] if found else p
        dom[sc["scene_id"]] = domain(p)
    rows = json.loads((EXP / "per_region_failures.json").read_text(encoding="utf-8"))
    split = defaultdict(lambda: {"regions": 0, "covered": 0, "A_top1_ok": 0, "B_top5_ok": 0, "scenes": set()})
    for r in rows:
        d = dom[r["scene_id"]]["domain"]
        s = split[d]
        s["regions"] += 1
        s["scenes"].add(r["scene_id"])
        if r["covered_by_library"]:
            s["covered"] += 1
            s["A_top1_ok"] += r["A_top1"] == r["gt"]
            s["B_top5_ok"] += r["gt"] in r["B_top5"]
    summary = {k: {**v, "scenes": len(v["scenes"])} for k, v in split.items()}
    out = {"per_scene": dom, "split": summary,
           "n_synthetic_scenes": sum(1 for v in dom.values() if v["domain"] == "SYNTHETIC_DRAWING")}
    (EXP / "fixture_domain.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({"split": summary, "n_synthetic_scenes": out["n_synthetic_scenes"],
                      "synthetic": sorted(k for k, v in dom.items() if v["domain"] == "SYNTHETIC_DRAWING")}, indent=2))


if __name__ == "__main__":
    main()
