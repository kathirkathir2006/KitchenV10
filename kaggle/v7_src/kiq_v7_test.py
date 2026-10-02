"""V7 ONE-SHOT TEST: frozen Adapter E vs E + frozen V7 specialists on real_photo_set_v1 TEST + 41-scene benchmark.

Nothing is fitted, selected or re-thresholded. Refuses to run twice (docs/v7/test/V7_TEST_DONE.json).
Test embeddings come from Kaggle kernel kiq-v7-test-embed v1 (frozen DINOv2 view 0, hash-verified images).
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
spec = importlib.util.spec_from_file_location("kiq_v7", ROOT / "kaggle/v7_src/kiq_v7_specialists.py")
V7 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(V7)
V6, S = V7.V6, V7.S
CFG = ROOT / "docs/v7/v7_specialists_frozen_config.json"
CFG_SHA = "c29079096ed20ecd78d90bf0b29d86201241f705757d152f1dcef915c1b297b5"
CK_SHA = "eeae5fbeaf84969af199b263bde13ebb8a69895fcb20f7fdbd0dfbc0fc29eb25"
EMB = ROOT / "kaggle/outputs/kiq-v7-test-embed/v5_v7_test_embed"
HIST = ROOT / "docs/v5/adapter_de_v1_eval/de_test_results.json"
OUT = ROOT / "docs/v7/test"
CATALOG = ROOT / "backend/instance/dev_experiments/v10-specialist-ensemble/acceptance/fixtures/catalog/fixture_catalog.json"
CATALOG_SHA = "38fc3eaee75c76fa5fd1dc3d4177af4abde2cfeb3e410d290fb524f9e30a1523"
GEN = Path(r"C:\Projects\KitchenIQ-OS\backend\instance\dev_experiments\fv-real-world-acceptance-v2\fixtures\generated")
DOMAIN = ROOT / "backend/instance/dev_experiments/v10-vision-v5/v5_exp_AB_retrieval_001/fixture_domain.json"
KEYS = ("known_identity_top1_raw", "known_identity_top5_raw", "known_family_top1_raw", "known_state_top1_raw",
        "known_accepted_correct", "known_coverage_identity", "unknown_rejection", "false_confirmation", "objective")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_v7(dim: int) -> list[dict]:
    assert sha(CFG) == CFG_SHA, "V7 config changed"
    cfg = json.loads(CFG.read_bytes())
    assert cfg["frozen"] and not cfg["test_used"]
    ck = ROOT / "docs/v7" / cfg["checkpoint"]
    assert sha(ck) == CK_SHA == cfg["checkpoint_sha256"], "V7 checkpoint changed"
    blob = torch.load(ck, map_location="cpu", weights_only=False)
    specs = []
    for p in cfg["pairs"]:
        b = blob[p["pair"]]
        m = V7.head(b["kind"], dim)
        m.load_state_dict(b["state_dict"])
        m.eval()
        specs.append({**p, "model": {"model": m, "mu": b["mu"].numpy(), "sd": b["sd"].numpy()}})
    return specs, cfg


def evaluate(I: dict, gt: list[dict], XB: np.ndarray, specs: list[dict], tag: str) -> tuple[dict, list[dict]]:
    """Baseline/C/D/E/E+V7 on a set of frozen DINOv2 query embeddings with frozen thresholds."""
    J = {**I, "gval": gt}
    Rraw = I["Xtr"][I["ref_idx"], 0]
    E = I["models"]["adapter_e"]
    qE = S.adapter_embed(E, XB)
    sc = {}
    for n in ("baseline", "adapter_c", "adapter_d", "adapter_e"):
        X, R = (XB, Rraw) if n == "baseline" else (S.adapter_embed(I["models"][n], XB), S.adapter_embed(I["models"][n], Rraw))
        sc[n] = S.label_scores(X, R, I["rlab"], I["classes"])
    EV = {"scores": sc["adapter_e"], "top5": [[I["classes"][j] for j in np.argsort(-r)[:5]] for r in sc["adapter_e"]]}
    pA = {sp["pair"]: V7.specialist_prob(sp["model"], qE) for sp in specs}
    orders, fired = V7.apply(EV, specs, pA, {sp["pair"]: sp["margin"] for sp in specs})
    sc["adapter_e_v7"] = V7.permuted_scores(EV, orders, I["classes"])
    thr = {"baseline": I["cfgs"]["adapter_c"]["baseline"]["thresholds"], "adapter_c": I["cfgs"]["adapter_c"]["adapter"]["thresholds"],
           "adapter_d": I["cfgs"]["adapter_d"]["adapter"]["thresholds"], "adapter_e": I["cfgs"]["adapter_e"]["adapter"]["thresholds"],
           "adapter_e_v7": I["cfgs"]["adapter_e"]["adapter"]["thresholds"]}
    met, dec = {}, {}
    for n, s in sc.items():
        t = thr[n]
        dec[n] = S.decide(s, I["classes"], I["fam"], t["t_id"], t["m_id"], t["t_fam"])
        m = S.detail_metrics(s, dec[n], gt, I["classes"], I["fam"], I["state"], I["known"])
        met[n] = {k: round(float(v), 4) for k, v in m.items()}
    per = []
    for q, g in enumerate(gt):
        row = {"id": g["openverse_id"], "gt": g["label"], "role": g["role"], "E_top5": EV["top5"][q], "V7_top5": orders[q],
               "specialists_fired": fired[q]}
        for n in ("adapter_e", "adapter_e_v7"):
            top1 = I["classes"][int(np.argmax(sc[n][q]))]
            row[n] = {"top1": top1, "decision": dec[n][q]["level"], "label": dec[n][q]["label"],
                      "divergence": S.earliest_divergence(g, dec[n][q], top1, I["fam"], I["known"])}
        kn = g["label"] in I["known"]
        e_ok, v_ok = row["adapter_e"]["top1"] == g["label"], row["adapter_e_v7"]["top1"] == g["label"]
        row["top1_change"] = None if not kn else ("fixed" if v_ok and not e_ok else ("broken" if e_ok and not v_ok else ("both_ok" if v_ok else "both_wrong")))
        per.append(row)
    kn = [r for r in per if r["top1_change"]]
    by_pair = {sp["pair"]: {"fired": sum(sp["pair"] in r["specialists_fired"] for r in per),
                            "fixed": sum(sp["pair"] in r["specialists_fired"] and r["top1_change"] == "fixed" for r in kn),
                            "broken": sum(sp["pair"] in r["specialists_fired"] and r["top1_change"] == "broken" for r in kn),
                            "fired_on_unknown": sum(sp["pair"] in r["specialists_fired"] and r["role"] != "known" for r in per)}
               for sp in specs}
    unk = [r for r in per if r["gt"] not in I["known"]]
    summary = {"set": tag, "n": len(gt), "n_known": len(kn), "metrics": met,
               "top1_changes_E_to_V7": dict(Counter(r["top1_change"] for r in kn)), "per_specialist": by_pair,
               "divergence": {n: dict(Counter(r[n]["divergence"] for r in per)) for n in ("adapter_e", "adapter_e_v7")},
               "open_set": {n: {"n": len(unk), "rejected": sum(r[n]["decision"] != "identity" for r in unk),
                                "identity_false_accepts": sum(r[n]["decision"] == "identity" for r in unk),
                                "family_backoff": sum(r[n]["decision"] == "family" for r in unk)} for n in ("adapter_e", "adapter_e_v7")}}
    return summary, per


def scene_set(I: dict) -> tuple[list[dict], np.ndarray, list[dict]]:
    assert sha(CATALOG) == CATALOG_SHA, "benchmark changed — STOP"
    domain = json.loads(DOMAIN.read_text(encoding="utf-8"))["per_scene"]
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
            I["fam"].setdefault(lab, "document" if "document" in lab else "unmapped")
            I["state"].setdefault(lab, inst.get("food_state") or "none")
            gt.append({"label": lab, "family": I["fam"][lab], "state": I["state"][lab],
                       "role": "known" if lab in I["known"] else "open_set_unknown", "openverse_id": f"{sc['scene_id']}#{len(gt)}"})
            meta.append({"scene_id": sc["scene_id"], "domain": domain[sc["scene_id"]]["domain"]})
    S._DINO = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", trust_repo=True).eval()
    X = np.concatenate([S.embed_batch(crops[i:i + 16]) for i in range(0, len(crops), 16)])
    return gt, X, meta


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    done = OUT / "V7_TEST_DONE.json"
    if done.exists():
        raise SystemExit("V7 test already evaluated once — refusing to re-run")
    I = V6.load_inputs()
    specs, cfg = load_v7(I["cfgs"]["adapter_e"]["adapter"]["cfg"]["dim"])
    meta = json.loads((EMB / "test_items.json").read_bytes())
    Xte = np.load(EMB / "test_dinov2_view0.npy")
    assert sha(EMB / "test_dinov2_view0.npy") == meta["embedding_sha256"] and meta["v7_config_sha256"] == CFG_SHA
    test = [i for i in json.loads(V6.PHOTO.read_bytes())["items"] if i["split"] == "test"]
    assert [i["sha256"] for i in test] == meta["sha256"] and Xte.shape == (len(test), 768)
    for i in test:
        I["fam"].setdefault(i["label"], i["family"])
        I["state"].setdefault(i["label"], i["state"])
    gte = S.gt_rows(test)
    t_sum, t_per = evaluate(I, gte, Xte, specs, "real_photo_set_v1 test")
    hist = json.loads(HIST.read_text(encoding="utf-8"))["metrics"]
    repro = {n: all(abs(t_sum["metrics"][n][k] - hist[n][k]) < 1e-3 for k in KEYS) for n in ("baseline", "adapter_c", "adapter_d", "adapter_e")}
    t_sum["reproduces_historical_frozen_test"] = repro
    gs, Xs, ms = scene_set(I)
    s_sum, s_per = evaluate(I, gs, Xs, specs, "41-scene GT regions")
    splits = {}
    for dom in (None, "PHOTO", "SYNTHETIC_DRAWING"):
        idx = [k for k, m in enumerate(ms) if dom is None or m["domain"] == dom]
        sub = [s_per[k] for k in idx]
        scenes = defaultdict(lambda: defaultdict(list))
        for k in idx:
            for n in ("adapter_e", "adapter_e_v7"):
                scenes[n][ms[k]["scene_id"]].append(s_per[k][n]["divergence"] == "OK")
        splits[dom or "ALL"] = {"n_regions": len(idx),
                                "top1_known": {n: round(float(np.mean([r[n]["top1"] == r["gt"] for r in sub if r["top1_change"]] or [0])), 4)
                                               for n in ("adapter_e", "adapter_e_v7")},
                                "top1_changes": dict(Counter(r["top1_change"] for r in sub if r["top1_change"])),
                                "scene_complete": {n: f"{sum(all(v) for v in scenes[n].values())}/{len(scenes[n])}" for n in scenes}}
    s_sum["by_domain"] = splits
    E, V = t_sum["metrics"]["adapter_e"], t_sum["metrics"]["adapter_e_v7"]
    ch = t_sum["top1_changes_E_to_V7"]
    checks = {"top1_gain_ge_0.02": V["known_identity_top1_raw"] - E["known_identity_top1_raw"] >= 0.02 - 1e-9,
              "fixed_ge_5": ch.get("fixed", 0) >= 5, "broken_le_2": ch.get("broken", 0) <= 2,
              "unknown_rejection_ge_E": V["unknown_rejection"] >= E["unknown_rejection"],
              "false_confirmation_le_E_plus_0.005": V["false_confirmation"] <= E["false_confirmation"] + 0.005}
    verdict = "CONFIRMED" if all(checks.values()) else "NOT CONFIRMED"
    res = {"stage": "V7 one-shot test", "evaluated_once": True, "verdict": verdict, "checks": checks,
           "v7_config_sha256": CFG_SHA, "v7_checkpoint_sha256": CK_SHA, "thresholds_recalibrated": False,
           "weights_updated": False, "test_embedding_source": "kathiresannatarajan/kiq-v7-test-embed v1",
           "test_embedding_sha256": meta["embedding_sha256"], "benchmark_catalog_sha256": CATALOG_SHA,
           "test": t_sum, "scenes_41": s_sum, **I["provenance"]}
    (OUT / "v7_test_results.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    (OUT / "v7_test_per_item.json").write_text(json.dumps(t_per, indent=1, default=float), encoding="utf-8")
    (OUT / "v7_41scene_per_region.json").write_text(json.dumps([{**m, **r} for m, r in zip(ms, s_per)], indent=1, default=float), encoding="utf-8")
    done.write_text(json.dumps({"results_sha256": sha(OUT / "v7_test_results.json"), "verdict": verdict}, indent=1), encoding="utf-8")
    print(json.dumps({"verdict": verdict, "checks": checks, "repro": repro,
                      "test": {n: {k: t_sum["metrics"][n][k] for k in KEYS} for n in t_sum["metrics"]},
                      "changes": ch, "per_specialist": t_sum["per_specialist"], "open_set": t_sum["open_set"],
                      "divergence": t_sum["divergence"], "scenes": splits,
                      "scene_metrics": {n: {k: s_sum["metrics"][n][k] for k in KEYS} for n in ("adapter_e", "adapter_e_v7")}}, indent=1))


if __name__ == "__main__":
    main()
