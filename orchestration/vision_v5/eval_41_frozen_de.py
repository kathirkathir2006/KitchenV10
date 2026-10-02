"""41-scene GT-region comparison of frozen baseline, Adapter C, Adapter D, Adapter E (frozen artifacts only).

Run only after D and E are frozen. Nothing is fitted or selected here; photo and synthetic scenes are reported
separately. Uses the assembled D/E stage module, whose scorer/decision code is the unchanged shared code.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(r"C:\Projects\KitchenIQ-V10-AI")
MOD = ROOT / "kaggle/v5_selftest/kiq_v5_stage.py"
spec = importlib.util.spec_from_file_location("kiq_v5_de_stage", MOD)
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)

CATALOG = ROOT / "backend/instance/dev_experiments/v10-specialist-ensemble/acceptance/fixtures/catalog/fixture_catalog.json"
GEN = Path(r"C:\Projects\KitchenIQ-OS\backend\instance\dev_experiments\fv-real-world-acceptance-v2\fixtures\generated")
CATALOG_SHA = "38fc3eaee75c76fa5fd1dc3d4177af4abde2cfeb3e410d290fb524f9e30a1523"
LIBDIR = ROOT / "docs/v5/reference_library_v1"
DOMAIN = ROOT / "backend/instance/dev_experiments/v10-vision-v5/v5_exp_AB_retrieval_001/fixture_domain.json"
OUT = ROOT / "backend/instance/dev_experiments/v10-vision-v5/v5_exp_DE_41scene_001"
NAMES = ["baseline", "adapter_c", "adapter_d", "adapter_e"]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def frozen(x: str) -> tuple[dict, Path]:
    d = ROOT / f"docs/v5/adapter_{x}_v1"
    fz = json.loads((d / "FROZEN.json").read_text(encoding="utf-8"))
    cp = d / f"adapter_{x}_frozen_config.json"
    assert sha(cp) == fz["config_sha256"], f"{x} config not the frozen one"
    cfg = json.loads(cp.read_bytes())
    ck = d / cfg["adapter"]["checkpoint"]
    assert sha(ck) == cfg["adapter"]["checkpoint_sha256"], f"{x} checkpoint changed"
    return cfg, ck


def main() -> None:
    assert sha(CATALOG) == CATALOG_SHA, "benchmark changed — STOP"
    assert S.scorer_source_sha() == "ec2dd8a44f48246bd810d619b3495acb9a4a4ecc40994d92809ce20767e9a270"
    cfgs = {x: frozen(x) for x in ("c", "d", "e")}
    ccfg = cfgs["c"][0]
    lib = json.loads((LIBDIR / "reference_library_v1_manifest.json").read_bytes())
    assert sha(LIBDIR / "reference_library_v1_manifest.json") == ccfg["library_manifest_sha256"] == S.EXPECTED_LIBRARY_SHA
    emb = json.loads((LIBDIR / "reference_library_v1_dinov2_embeddings.json").read_bytes())["embeddings"]
    Rf = np.stack([np.asarray(emb[r["ref_id"]], np.float32) for r in lib["items"]])
    rlab = np.array([r["label"] for r in lib["items"]])
    classes, fam, state = ccfg["classes"], dict(ccfg["families"]), dict(ccfg["states"])
    assert classes == cfgs["d"][0]["classes"] == cfgs["e"][0]["classes"]
    known = set(classes)
    domain = json.loads(DOMAIN.read_text(encoding="utf-8"))["per_scene"]

    S._DINO = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", trust_repo=True).eval()
    S.DEV = torch.device("cpu")
    models, thr = {}, {"baseline": ccfg["baseline"]["thresholds"]}
    for x, kls in (("c", None), ("d", S.AdapterD), ("e", S.AdapterE)):
        cfg, ck = cfgs[x]
        a = cfg["adapter"]["cfg"]
        m = S.AdapterC(len(classes), a["hidden"], a["dim"], a["dropout"]) if x == "c" else kls(a["hidden"], a["dim"], a["dropout"])
        m.load_state_dict(torch.load(ck, map_location="cpu", weights_only=False)["state_dict"])
        models[f"adapter_{x}"], thr[f"adapter_{x}"] = m, cfg["adapter"]["thresholds"]

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
    for n in NAMES:
        Q, R = (X, Rf) if n == "baseline" else (S.adapter_embed(models[n], X), S.adapter_embed(models[n], Rf))
        sc = S.label_scores(Q, R, rlab, classes)
        t = thr[n]
        systems[n] = (sc, S.decide(sc, classes, fam, t["t_id"], t["m_id"], t["t_fam"]))

    pairs = [("baseline", "adapter_d"), ("baseline", "adapter_e"), ("adapter_c", "adapter_d"), ("adapter_c", "adapter_e")]
    rows = []
    for k, g in enumerate(gt):
        row = {**meta[k], "gt": g["label"], "gt_family": g["family"], "role": g["role"]}
        for n, (sc, dec) in systems.items():
            top1 = classes[int(np.argmax(sc[k]))]
            d = dec[k]
            row[n] = {"top1": top1, "decision": d["level"], "label": d["label"], "family": d["family"],
                      "score": round(d["score"], 4), "divergence": S.earliest_divergence(g, d, top1, fam, known)}
        row["change"] = {f"{a}->{b}": S.change(row[a]["divergence"] == "OK", row[b]["divergence"] == "OK") for a, b in pairs}
        rows.append(row)

    def subset(dom: str | None) -> dict:
        idx = [i for i, m in enumerate(meta) if dom is None or m["domain"] == dom]
        out = {"n_regions": len(idx), "n_scenes": len({meta[i]["scene_id"] for i in idx})}
        for n, (sc, dec) in systems.items():
            out[n] = S.detail_metrics(sc[idx], [dec[i] for i in idx], [gt[i] for i in idx], classes, fam, state, known)
            scenes = defaultdict(list)
            for i in idx:
                scenes[meta[i]["scene_id"]].append(rows[i][n]["divergence"] == "OK")
            out[n]["oracle_region_scene_complete"] = f"{sum(all(v) for v in scenes.values())}/{len(scenes)}"
            out[n]["divergence_counts"] = dict(Counter(rows[i][n]["divergence"] for i in idx))
        out["change_counts"] = {f"{a}->{b}": dict(Counter(rows[i]["change"][f"{a}->{b}"] for i in idx)) for a, b in pairs}
        return out

    result = {"experiment_id": "v5_exp_DE_41scene_001", "benchmark_catalog_sha256": CATALOG_SHA,
              "config_sha256": {x: json.loads((ROOT / f"docs/v5/adapter_{x}_v1/FROZEN.json").read_text())["config_sha256"] for x in "cde"},
              "library_manifest_sha256": S.EXPECTED_LIBRARY_SHA, "used_for_selection": False,
              "region_source": "ground-truth boxes (diagnostic; detector excluded)",
              "all": subset(None), "photo": subset("PHOTO"), "synthetic": subset("SYNTHETIC_DRAWING")}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (OUT / "per_region.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
