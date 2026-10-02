"""41-scene GT-region comparison: frozen DINOv2 baseline vs Adapter C, using ONLY frozen artifacts.

Same reference library v1, same scorer/decision code as Kaggle (imported), val-calibrated thresholds
from the frozen config. Reported split by photo vs synthetic scenes. Nothing here is fitted.
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(r"C:\Projects\KitchenIQ-V10-AI")
sys.path.insert(0, str(ROOT / "kaggle/v5_src"))
import kiq_v5_stage as S  # noqa: E402

CATALOG = ROOT / "backend/instance/dev_experiments/v10-specialist-ensemble/acceptance/fixtures/catalog/fixture_catalog.json"
GEN = Path(r"C:\Projects\KitchenIQ-OS\backend\instance\dev_experiments\fv-real-world-acceptance-v2\fixtures\generated")
CATALOG_SHA = "38fc3eaee75c76fa5fd1dc3d4177af4abde2cfeb3e410d290fb524f9e30a1523"
LIBDIR = ROOT / "docs/v5/reference_library_v1"
CFGDIR = ROOT / "docs/v5/adapter_c_v1"
DOMAIN = ROOT / "backend/instance/dev_experiments/v10-vision-v5/v5_exp_AB_retrieval_001/fixture_domain.json"
OUT = ROOT / "backend/instance/dev_experiments/v10-vision-v5/v5_exp_C_41scene_001"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    assert sha(CATALOG) == CATALOG_SHA, "benchmark changed — STOP"
    frozen = json.loads((CFGDIR / "FROZEN.json").read_text(encoding="utf-8"))
    cfg_p = CFGDIR / "adapter_c_frozen_config.json"
    assert sha(cfg_p) == frozen["config_sha256"], "config not the frozen one"
    cfg = json.loads(cfg_p.read_bytes())
    ck = CFGDIR / cfg["adapter"]["checkpoint"]
    assert sha(ck) == cfg["adapter"]["checkpoint_sha256"], "checkpoint changed"
    lib = json.loads((LIBDIR / "reference_library_v1_manifest.json").read_bytes())
    assert sha(LIBDIR / "reference_library_v1_manifest.json") == cfg["library_manifest_sha256"]
    emb = json.loads((LIBDIR / "reference_library_v1_dinov2_embeddings.json").read_bytes())["embeddings"]
    Rf = np.stack([np.asarray(emb[r["ref_id"]], np.float32) for r in lib["items"]])
    rlab = np.array([r["label"] for r in lib["items"]])
    classes, fam, state = cfg["classes"], dict(cfg["families"]), dict(cfg["states"])
    known = set(classes)
    domain = json.loads(DOMAIN.read_text(encoding="utf-8"))["per_scene"]

    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", trust_repo=True).eval()
    S._DINO, S.DEV = model, torch.device("cpu")
    a = cfg["adapter"]["cfg"]
    adapter = S.AdapterC(len(classes), a["hidden"], a["dim"], a["dropout"])
    adapter.load_state_dict(torch.load(ck, map_location="cpu")["state_dict"])

    cat = json.loads(CATALOG.read_text(encoding="utf-8"))
    crops, gt, meta = [], [], []
    for sc in cat["scenes"]:
        p = Path((sc.get("image") or {}).get("path") or "")
        if not p.is_file():
            p = sorted(GEN.glob(f"{sc['scene_id']}*"))[0]
        img = Image.open(p).convert("RGB")
        W, H = img.size
        for inst in sc.get("instances") or []:
            if not inst.get("required", True):
                continue
            b = inst.get("bbox")
            crops.append(img if not b else img.crop((max(0, int(b[0])), max(0, int(b[1])), min(W, int(b[2])), min(H, int(b[3])))))
            lab = str(inst["label"])
            fam.setdefault(lab, "document" if "document" in lab else "unmapped")
            state.setdefault(lab, inst.get("food_state") or "none")
            gt.append({"label": lab, "family": fam[lab], "state": state[lab],
                       "role": "known" if lab in known else "open_set_unknown", "openverse_id": None})
            meta.append({"scene_id": sc["scene_id"], "domain": domain[sc["scene_id"]]["domain"]})
    X = np.concatenate([S.embed_batch(crops[i:i + 16]) for i in range(0, len(crops), 16)])

    systems = {}
    for name, Q, R, t in (("baseline", X, Rf, cfg["baseline"]["thresholds"]),
                          ("adapter_c", S.adapter_embed(adapter, X), S.adapter_embed(adapter, Rf), cfg["adapter"]["thresholds"])):
        sc = S.label_scores(Q, R, rlab, classes)
        systems[name] = (sc, S.decide(sc, classes, fam, t["t_id"], t["m_id"], t["t_fam"]))

    rows = []
    for k, g in enumerate(gt):
        row = {**meta[k], "gt": g["label"], "gt_family": g["family"], "role": g["role"]}
        for name, (sc, dec) in systems.items():
            top1 = classes[int(np.argmax(sc[k]))]
            d = dec[k]
            row[name] = {"top1": top1, "top5": [classes[j] for j in np.argsort(-sc[k])[:5]], "decision": d["level"],
                         "label": d["label"], "family": d["family"], "score": round(d["score"], 4),
                         "divergence": S.earliest_divergence(g, d, top1, fam, known)}
        b_ok, a_ok = row["baseline"]["divergence"] == "OK", row["adapter_c"]["divergence"] == "OK"
        row["change"] = "fixed" if a_ok and not b_ok else ("regressed" if b_ok and not a_ok else ("both_ok" if a_ok else "both_fail"))
        rows.append(row)

    def subset(dom: str | None) -> dict:
        idx = [i for i, m in enumerate(meta) if dom is None or m["domain"] == dom]
        out = {"n_regions": len(idx)}
        for name, (sc, dec) in systems.items():
            out[name] = S.detail_metrics(sc[idx], [dec[i] for i in idx], [gt[i] for i in idx], classes, fam, state, known)
            scenes = defaultdict(list)
            for i in idx:
                scenes[meta[i]["scene_id"]].append(rows[i][name]["divergence"] == "OK")
            out[name]["oracle_region_scene_complete"] = f"{sum(all(v) for v in scenes.values())}/{len(scenes)}"
            out[name]["divergence_counts"] = dict(Counter(rows[i][name]["divergence"] for i in idx))
        out["change_counts"] = dict(Counter(rows[i]["change"] for i in idx))
        return out

    result = {"experiment_id": "v5_exp_C_41scene_001", "benchmark_catalog_sha256": CATALOG_SHA,
              "frozen_config_sha256": frozen["config_sha256"], "library_manifest_sha256": cfg["library_manifest_sha256"],
              "region_source": "ground-truth boxes (diagnostic; detector excluded)",
              "all": subset(None), "photo": subset("PHOTO"), "synthetic": subset("SYNTHETIC_DRAWING")}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (OUT / "per_region.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
