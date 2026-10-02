"""KitchenIQ V6 candidate reranker over frozen Adapter E top-5 (TRAIN fit, VAL selection; TEST never opened).

Inputs are frozen artifacts only:
  - real_photo_set_v1 manifest (item order / labels / sha)              docs/v5/real_photo_set_v1
  - frozen DINOv2 view embeddings of TRAIN+VAL produced by kiq-v5-adapter-e v1 (emb_cache.npz)
  - reference_library_v1 (train only)                                    docs/v5/reference_library_v1
  - frozen Adapter C/D/E configs + checkpoints                           docs/v5/adapter_{c,d,e}_v1
V5 code is imported read-only (scorer, decide, calibrate, detail_metrics, adapters). Nothing in V5 is modified.

usage: python kaggle/v6_src/kiq_v6_reranker.py
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(r"C:\Projects\KitchenIQ-V10-AI")
V5MOD = ROOT / "kaggle/v5_selftest/kiq_v5_stage.py"
PHOTO = ROOT / "docs/v5/real_photo_set_v1/manifest.json"
LIBDIR = ROOT / "docs/v5/reference_library_v1"
CACHE = ROOT / "kaggle/outputs/kiq-v5-adapter-e/v5_adapter_e/emb_cache.npz"
OUT = ROOT / "backend/instance/dev_experiments/v10-vision-v6/v6_reranker_001"
FREEZE = ROOT / "docs/v6/reranker_v1"
SCORER_SRC_SHA = "ec2dd8a44f48246bd810d619b3495acb9a4a4ecc40994d92809ce20767e9a270"
TOPK = 5
SUPPORT_K = 10
WEIGHT_DECAYS = (1e-4, 1e-3, 1e-2)
SEED = 1234

spec = importlib.util.spec_from_file_location("kiq_v5_frozen", V5MOD)
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)
S.DEV = torch.device("cpu")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def sha_obj(o) -> str:
    return hashlib.sha256(json.dumps(o, sort_keys=True).encode()).hexdigest()


# ------------------------------------------------------------- frozen inputs
def load_inputs() -> dict:
    assert S.scorer_source_sha() == SCORER_SRC_SHA, "V5 scorer changed"
    assert sha(PHOTO) == S.EXPECTED_PHOTO_SHA, "photo manifest changed"
    lib_p = LIBDIR / "reference_library_v1_manifest.json"
    assert sha(lib_p) == S.EXPECTED_LIBRARY_SHA, "library changed"
    items = [i for i in json.loads(PHOTO.read_bytes())["items"] if i["split"] in {"train", "val"}]
    assert not any(i["split"] == "test" for i in items)
    train = [i for i in items if i["split"] == "train"]
    val = [i for i in items if i["split"] == "val"]
    z = np.load(CACHE)
    Xtr, Xva = z["Xtr"], z["Xva"]
    assert Xtr.shape == (len(train), 8, 768) and Xva.shape == (len(val), 768)
    lib = json.loads(lib_p.read_bytes())
    emb = json.loads((LIBDIR / "reference_library_v1_dinov2_embeddings.json").read_bytes())["embeddings"]
    tr_pos = {i["sha256"]: k for k, i in enumerate(train)}
    ref_idx = np.array([tr_pos[r["sha256"]] for r in lib["items"]])
    lib_emb = np.stack([np.asarray(emb[r["ref_id"]], np.float32) for r in lib["items"]])
    align = float(np.abs(Xtr[ref_idx, 0] - lib_emb).max())
    assert align < 1e-5, f"embedding cache does not align with frozen library ({align})"
    models, cfgs = {}, {}
    for x in "cde":
        d = ROOT / f"docs/v5/adapter_{x}_v1"
        fz = json.loads((d / "FROZEN.json").read_text())
        cp = d / f"adapter_{x}_frozen_config.json"
        assert sha(cp) == fz["config_sha256"]
        cfg = json.loads(cp.read_bytes())
        ck = d / cfg["adapter"]["checkpoint"]
        assert sha(ck) == cfg["adapter"]["checkpoint_sha256"]
        a = cfg["adapter"]["cfg"]
        m = (S.AdapterC(len(cfg["classes"]), a["hidden"], a["dim"], a["dropout"]) if x == "c"
             else (S.AdapterD if x == "d" else S.AdapterE)(a["hidden"], a["dim"], a["dropout"]))
        m.load_state_dict(torch.load(ck, map_location="cpu", weights_only=False)["state_dict"])
        models[f"adapter_{x}"], cfgs[f"adapter_{x}"] = m, cfg
    ecfg = cfgs["adapter_e"]
    classes, fam, state = ecfg["classes"], dict(ecfg["families"]), dict(ecfg["states"])
    return {"train": train, "val": val, "Xtr": Xtr, "Xva": Xva, "ref_idx": ref_idx,
            "ref_sha": np.array([r["sha256"] for r in lib["items"]]), "rlab": np.array([r["label"] for r in lib["items"]]),
            "models": models, "cfgs": cfgs, "classes": classes, "fam": fam, "state": state, "known": set(classes),
            "gval": S.gt_rows(val), "provenance": {
                "photo_manifest_sha256": sha(PHOTO), "library_manifest_sha256": sha(lib_p),
                "embedding_cache": "kaggle kernel kathiresannatarajan/kiq-v5-adapter-e v1 /v5_adapter_e/emb_cache.npz",
                "embedding_cache_sha256": sha(CACHE), "cache_library_alignment_maxabs": align,
                "adapter_e_config_sha256": sha(ROOT / "docs/v5/adapter_e_v1/adapter_e_frozen_config.json"),
                "adapter_e_checkpoint_sha256": ecfg["adapter"]["checkpoint_sha256"], "scorer_source_sha256": SCORER_SRC_SHA}}


# ------------------------------------------------------------- candidate evidence
FEATURES_E = ["rank", "is_rank1", "score", "gap_from_top", "separation", "proto_cos", "ref_top1", "ref_top3_mean",
              "ref_top3_std", "ref_max", "ref_mean", "ref_median", "ref_std", "n_refs", "support_count",
              "support_frac", "ref_consistency", "family_score", "family_separation", "family_agree_top5",
              "state_score", "state_separation", "state_agree_top5"]
FEATURES_DINO = ["dino_score", "dino_rank", "dino_gap_from_top", "dino_separation"]


def query_rows(qE: np.ndarray, qB: np.ndarray, RE: np.ndarray, RB: np.ndarray, rl: np.ndarray, I: dict) -> tuple[list[dict], np.ndarray]:
    """Top-5 candidates of one query under Adapter E (V5 scorer) + per-candidate evidence."""
    classes, fam, state = I["classes"], I["fam"], I["state"]
    sE = S.label_scores(qE[None], RE, rl, classes)[0]
    sB = S.label_scores(qB[None], RB, rl, classes)[0]
    order = np.argsort(-sE)
    bord = np.argsort(-sB)
    brank = {classes[j]: r + 1 for r, j in enumerate(bord)}
    top = [classes[j] for j in order[:TOPK]]
    cos_all = RE @ qE
    nn_lab = rl[np.argsort(-cos_all)[:SUPPORT_K]]
    fam_best, st_best = {}, {}
    for j, c in enumerate(classes):
        fam_best[fam[c]] = max(fam_best.get(fam[c], -9.0), float(sE[j]))
        st_best[state[c]] = max(st_best.get(state[c], -9.0), float(sE[j]))
    rows = []
    for r, c in enumerate(top, 1):
        j = classes.index(c)
        m = rl == c
        cos = np.sort(cos_all[m])[::-1]
        Rc = RE[m]
        proto = S.norm(Rc.mean(0))
        pair = Rc @ Rc.T
        n = len(Rc)
        cons = float((pair.sum() - n) / max(1, n * (n - 1)))
        others = np.delete(sE, j)
        bo = np.delete(sB, j)
        fs, ss = fam_best[fam[c]], st_best[state[c]]
        rows.append({
            "candidate": c, "family": fam[c], "state": state[c],
            "rank": r, "is_rank1": float(r == 1), "score": float(sE[j]), "gap_from_top": float(sE[j] - sE[order[0]]),
            "separation": float(sE[j] - others.max()), "proto_cos": float(qE @ proto), "ref_top1": float(cos[0]),
            "ref_top3_mean": float(cos[:3].mean()), "ref_top3_std": float(cos[:3].std()), "ref_max": float(cos.max()),
            "ref_mean": float(cos.mean()), "ref_median": float(np.median(cos)), "ref_std": float(cos.std()),
            "n_refs": float(n), "support_count": float((nn_lab == c).sum()), "support_frac": float((nn_lab == c).mean()),
            "ref_consistency": cons, "family_score": fs,
            "family_separation": fs - max([v for k, v in fam_best.items() if k != fam[c]] or [-9.0]),
            "family_agree_top5": float(sum(fam[o] == fam[c] for o in top if o != c)),
            "state_score": ss, "state_separation": ss - max([v for k, v in st_best.items() if k != state[c]] or [-9.0]),
            "state_agree_top5": float(sum(state[o] == state[c] for o in top if o != c)),
            "dino_score": float(sB[j]), "dino_rank": float(brank[c]), "dino_gap_from_top": float(sB[j] - sB[bord[0]]),
            "dino_separation": float(sB[j] - bo.max())})
    return rows, sE


def build_examples(I: dict) -> dict:
    E = I["models"]["adapter_e"]
    Rraw = I["Xtr"][I["ref_idx"], 0]
    RE = S.adapter_embed(E, Rraw)
    trE = S.adapter_embed(E, I["Xtr"][:, 0])
    vaE = S.adapter_embed(E, I["Xva"])
    out = {}
    for split, items, QE, QB in (("train", I["train"], trE, I["Xtr"][:, 0]), ("val", I["val"], vaE, I["Xva"])):
        rows, scores, self_excluded = [], [], 0
        for k, it in enumerate(items):
            keep = I["ref_sha"] != it["sha256"]          # leave-one-image-out: no self-evidence by sha256
            self_excluded += int((~keep).sum())
            assert it["sha256"] not in set(I["ref_sha"][keep])
            rr, sE = query_rows(QE[k], QB[k], RE[keep], Rraw[keep], I["rlab"][keep], I)
            for row in rr:
                row.update({"q": k, "query_id": it["openverse_id"], "query_sha256": it["sha256"], "gt": it["label"],
                            "target": float(row["candidate"] == it["label"])})
            rows += rr
            scores.append(sE)
        out[split] = {"rows": rows, "scores": np.stack(scores), "self_refs_excluded": self_excluded}
    return out


# ------------------------------------------------------------- logistic regression (PyTorch, L2)
def fit_lr(X: np.ndarray, y: np.ndarray, wd: float) -> dict:
    torch.manual_seed(SEED)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xt = torch.from_numpy((X - mu) / sd).float()
    yt = torch.from_numpy(y).float()
    w = torch.zeros(X.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], lr=1.0, max_iter=500, tolerance_grad=1e-9, tolerance_change=1e-12,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(Xt @ w + b, yt) + wd * (w ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return {"w": w.detach().numpy(), "b": float(b.detach()), "mu": mu, "sd": sd, "wd": wd,
            "train_loss": float(closure().detach())}


def predict(m: dict, X: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-(((X - m["mu"]) / m["sd"]) @ m["w"] + m["b"])))


# ------------------------------------------------------------- evaluation
def rerank_scores(scores: np.ndarray, rows: list[dict], prob: np.ndarray, classes: list[str]) -> tuple[np.ndarray, list[list[str]]]:
    """Permute E's own top-5 score values onto the reranked order. Non-candidates untouched, thresholds untouched."""
    new, orders = scores.copy(), []
    for q in range(len(scores)):
        idx = [i for i, r in enumerate(rows) if r["q"] == q]
        cand = [rows[i]["candidate"] for i in idx]
        vals = sorted([float(scores[q, classes.index(c)]) for c in cand], reverse=True)
        rer = [cand[i] for i in np.argsort(-prob[idx], kind="stable")]
        for v, c in zip(vals, rer):
            new[q, classes.index(c)] = v
        orders.append(rer)
    return new, orders


def metrics(sc: np.ndarray, I: dict, t: dict) -> dict:
    dec = S.decide(sc, I["classes"], I["fam"], t["t_id"], t["m_id"], t["t_fam"])
    m = S.detail_metrics(sc, dec, I["gval"], I["classes"], I["fam"], I["state"], I["known"])
    m.update(S.val_rates(dec, I["gval"], I["known"]))
    return {k: float(v) for k, v in m.items()}, dec


def oracle(scores: np.ndarray, I: dict) -> dict:
    kn = [i for i, g in enumerate(I["gval"]) if g["label"] in I["known"]]
    top1 = [I["classes"][int(np.argmax(scores[i]))] == I["gval"][i]["label"] for i in kn]
    in5 = [I["gval"][i]["label"] in [I["classes"][j] for j in np.argsort(-scores[i])[:TOPK]] for i in kn]
    return {"n_known_val": len(kn), "current_top1": round(float(np.mean(top1)), 4),
            "top5_oracle_ceiling": round(float(np.mean(in5)), 4),
            "max_possible_gain": round(float(np.mean(in5)) - float(np.mean(top1)), 4),
            "errors_with_correct_in_top5": int(sum(a and not b for a, b in zip(in5, top1))),
            "errors_correct_absent_from_top5": int(sum(not a for a in in5))}


def frozen_val(I: dict) -> dict:
    Rraw = I["Xtr"][I["ref_idx"], 0]
    out = {}
    for n in ("baseline", "adapter_c", "adapter_d", "adapter_e"):
        X, R = (I["Xva"], Rraw) if n == "baseline" else (S.adapter_embed(I["models"][n], I["Xva"]), S.adapter_embed(I["models"][n], Rraw))
        t = I["cfgs"]["adapter_c"]["baseline"]["thresholds"] if n == "baseline" else I["cfgs"][n]["adapter"]["thresholds"]
        out[n] = metrics(S.label_scores(X, R, I["rlab"], I["classes"]), I, t)[0]
    return out


def failure_analysis(I: dict, ex: dict, prob: np.ndarray, rr_orders: list[list[str]]) -> dict:
    rows, classes, fam, state = ex["rows"], I["classes"], I["fam"], I["state"]
    recs, agg, conf_e, conf_r = [], Counter(), Counter(), Counter()
    for q, g in enumerate(I["gval"]):
        if g["label"] not in I["known"]:
            continue
        idx = [i for i, r in enumerate(rows) if r["q"] == q]
        orig = [rows[i]["candidate"] for i in idx]
        e_ok, r_ok = orig[0] == g["label"], rr_orders[q][0] == g["label"]
        if not (orig.count(g["label"])):
            agg["correct_absent_from_top5"] += 1
        if e_ok and r_ok:
            continue
        kind = "wrong->correct" if r_ok else ("correct->wrong" if e_ok else ("absent_from_top5" if g["label"] not in orig else "wrong->wrong"))
        agg[kind] += 1
        pred = rr_orders[q][0]
        if not r_ok:
            agg["family_confusion" if fam[pred] != g["family"] else "same_family_identity_confusion"] += 1
            agg["state_confusion" if state[pred] != g["state"] else "same_state"] += 1
            conf_r[f"{g['label']} -> {pred}"] += 1
        if not e_ok:
            conf_e[f"{g['label']} -> {orig[0]}"] += 1
        recs.append({"query_id": g["openverse_id"], "ground_truth": g["label"], "gt_family": g["family"], "gt_state": g["state"],
                     "transition": kind, "original_e_top5": orig, "reranked_top5": rr_orders[q],
                     "correct_original_rank": orig.index(g["label"]) + 1 if g["label"] in orig else None,
                     "correct_new_rank": rr_orders[q].index(g["label"]) + 1 if g["label"] in orig else None,
                     "candidate_probabilities": {rows[i]["candidate"]: round(float(prob[i]), 4) for i in idx},
                     "reference_statistics": {rows[i]["candidate"]: {k: round(rows[i][k], 4) for k in (
                         "score", "proto_cos", "ref_top1", "ref_top3_mean", "ref_top3_std", "ref_mean", "ref_median",
                         "ref_std", "support_count", "family_score", "dino_rank")} for i in idx}})
    return {"aggregate": dict(agg), "top_confusions_adapter_e": conf_e.most_common(20),
            "top_confusions_reranked": conf_r.most_common(20), "records": recs}


def main() -> None:
    I = load_inputs()
    OUT.mkdir(parents=True, exist_ok=True)
    ex = build_examples(I)
    tr, va = ex["train"], ex["val"]
    e_val = frozen_val(I)
    E = e_val["adapter_e"]
    t_e = I["cfgs"]["adapter_e"]["adapter"]["thresholds"]
    e_sc_check = metrics(va["scores"], I, t_e)[0]
    assert abs(e_sc_check["known_identity_top1_raw"] - E["known_identity_top1_raw"]) < 1e-9, "candidate scores != frozen E"
    orc = oracle(va["scores"], I)
    print("ORACLE", orc, flush=True)
    tr_in5 = np.mean([any(r["target"] for r in tr["rows"] if r["q"] == q) for q in range(len(I["train"]))])

    feature_sets = {"E_evidence": FEATURES_E, "E_plus_frozen_dino_evidence": FEATURES_E + FEATURES_DINO}
    grid = []
    for fs_name, feats in feature_sets.items():
        Xtr = np.array([[r[f] for f in feats] for r in tr["rows"]], np.float64)
        ytr = np.array([r["target"] for r in tr["rows"]])
        Xva = np.array([[r[f] for f in feats] for r in va["rows"]], np.float64)
        for wd in WEIGHT_DECAYS:
            m = fit_lr(Xtr, ytr, wd)
            p = predict(m, Xva)
            new, orders = rerank_scores(va["scores"], va["rows"], p, I["classes"])
            vm = metrics(new, I, t_e)[0]
            safe = {"top1_gt_E": vm["known_identity_top1_raw"] > E["known_identity_top1_raw"],
                    "unknown_rejection_ge_E": vm["unknown_rejection"] >= E["unknown_rejection"],
                    "false_confirmation_le_E_plus_0.005": vm["false_confirmation"] <= E["false_confirmation"] + 0.005}
            grid.append({"feature_set": fs_name, "weight_decay": wd, "features": feats, "val": vm,
                         "train_top1_reranked": None, "passes": all(safe.values()), "constraints": safe,
                         "weights": dict(zip(feats, np.round(m["w"], 4).tolist())), "_m": m, "_p": p, "_orders": orders})
            print(fs_name, wd, {k: vm[k] for k in ("known_identity_top1_raw", "known_accepted_correct",
                                                   "unknown_rejection", "false_confirmation", "objective")}, safe, flush=True)

    passing = [g for g in grid if g["passes"]]
    pick = max(passing or grid, key=lambda g: (g["val"]["known_identity_top1_raw"], g["val"]["objective"], g["weight_decay"]))
    fa = failure_analysis(I, va, pick["_p"], pick["_orders"])
    fa_e = failure_analysis(I, va, -np.array([r["rank"] for r in va["rows"]], float), [
        [r["candidate"] for r in va["rows"] if r["q"] == q] for q in range(len(I["val"]))])
    verdict = "PASSED" if passing else "FAILED"
    gain = pick["val"]["known_identity_top1_raw"] - E["known_identity_top1_raw"]
    n_known = orc["n_known_val"]
    diag = {
        "e_errors_total": orc["errors_with_correct_in_top5"] + orc["errors_correct_absent_from_top5"],
        "e_errors_correct_in_top5": orc["errors_with_correct_in_top5"],
        "e_errors_correct_absent": orc["errors_correct_absent_from_top5"],
        "top_e_confusions": fa_e["top_confusions_adapter_e"][:10],
        "share_of_e_errors_in_top10_pairs": round(sum(v for _, v in fa_e["top_confusions_adapter_e"][:10]) /
                                                  max(1, orc["errors_with_correct_in_top5"] + orc["errors_correct_absent_from_top5"]), 4)}
    report = {"stage": "V6 candidate reranker", "test_opened": False, "test_used": False, "verdict": verdict,
              "provenance": I["provenance"], "self_refs_excluded": {"train": tr["self_refs_excluded"], "val": va["self_refs_excluded"]},
              "n_rows": {"train": len(tr["rows"]), "val": len(va["rows"])},
              "train_queries_with_correct_in_top5": round(float(tr_in5), 4),
              "train_vs_val_mean_top1_score": [round(float(tr["scores"].max(1).mean()), 4), round(float(va["scores"].max(1).mean()), 4)],
              "oracle_val": orc, "val_frozen_models": e_val,
              "grid": [{k: v for k, v in g.items() if not k.startswith("_")} for g in grid],
              "selected": {"feature_set": pick["feature_set"], "weight_decay": pick["weight_decay"], "val": pick["val"],
                           "raw_top1_gain_vs_E": round(gain, 4), "gain_in_queries": round(gain * n_known),
                           "passes_safety": pick["passes"]},
              "diagnosis_inputs": diag}
    (OUT / "v6_val_results.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    (OUT / "v6_val_failure_analysis.json").write_text(json.dumps({"selected": report["selected"], "reranked": fa,
                                                                  "adapter_e_only": {k: fa_e[k] for k in ("aggregate", "top_confusions_adapter_e")}},
                                                                 indent=1), encoding="utf-8")
    print("VERDICT", verdict, json.dumps(report["selected"]), flush=True)
    if passing:
        FREEZE.mkdir(parents=True, exist_ok=True)
        (FREEZE / ".gitattributes").write_text("* -text\n", encoding="utf-8")
        m = pick["_m"]
        ck = FREEZE / "v6_reranker_final.pt"
        torch.save({"w": torch.from_numpy(m["w"]), "b": m["b"], "mu": torch.from_numpy(m["mu"]),
                    "sd": torch.from_numpy(m["sd"]), "features": pick["features"], "weight_decay": m["wd"]}, ck)
        cfg = {"config_id": "kiq_v6_reranker_v1", "frozen": True, "test_used": False, "selection_split": "val",
               "fit_split": "train (leave-one-image-out by sha256)", "base_model": "frozen Adapter E",
               "candidates": TOPK, "features": pick["features"], "weight_decay": pick["weight_decay"],
               "decision": "E frozen thresholds; E top-5 score values permuted onto reranked order",
               "thresholds": t_e, "val_metrics": pick["val"], "checkpoint": ck.name, "checkpoint_sha256": sha(ck),
               **I["provenance"]}
        cp = FREEZE / "v6_reranker_frozen_config.json"
        cp.write_text(json.dumps(cfg, indent=1, sort_keys=True), encoding="utf-8")
        shutil.copyfile(OUT / "v6_val_failure_analysis.json", FREEZE / "v6_val_failure_analysis.json")
        shutil.copyfile(OUT / "v6_val_results.json", FREEZE / "v6_val_results.json")
        print("FROZEN config", sha(cp), "ckpt", sha(ck))


if __name__ == "__main__":
    main()
