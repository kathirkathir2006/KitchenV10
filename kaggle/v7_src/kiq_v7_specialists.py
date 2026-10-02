"""KitchenIQ V7 pairwise specialists over frozen Adapter E top-5 (TRAIN fit, VAL selection; TEST never opened).

Frozen inputs only (verified by sha in kiq_v6_reranker.load_inputs): real_photo_set_v1 manifest, frozen DINOv2
TRAIN(8 views)/VAL embeddings from kiq-v5-adapter-e v1, reference_library_v1, Adapter C/D/E.
TRAIN augmentation = the 7 realistic views already embedded by V5 (crop scale 0.6-1.0, h-flip,
brightness 0.75-1.25, colour 0.75-1.25); no new transforms.

usage: python kaggle/v7_src/kiq_v7_specialists.py
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(r"C:\Projects\KitchenIQ-V10-AI")
spec = importlib.util.spec_from_file_location("kiq_v6_frozen", ROOT / "kaggle/v6_src/kiq_v6_reranker.py")
V6 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(V6)
S = V6.S
DOCS = ROOT / "docs/v7"
V6_FA = ROOT / "docs/v6/reranker_v1/v6_val_failure_analysis.json"
V6_RES = ROOT / "docs/v6/reranker_v1/v6_val_results.json"
TOPK = 5
MIN_RECOVERABLE = 2
MIN_TRAIN_PER_CLASS = 10
MIN_VAL_PER_CLASS = 3
MAX_SPECIALISTS = 5
MARGINS = (0.0, 0.05, 0.10, 0.15)
LR, WD, MAX_EPOCHS, PATIENCE, BATCH = 3e-4, 1e-4, 20, 4, 64
SEED = 1234
TARGET_BAL_ACC = 0.90


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def wjson(name: str, obj) -> str:
    p = DOCS / name
    p.write_text(json.dumps(obj, indent=1, default=float), encoding="utf-8")
    return sha(p)


def pair_key(a: str, b: str) -> str:
    return " <-> ".join(sorted((a, b)))


# ------------------------------------------------------------- Phase 0/1: E errors + confusion table
def e_val(I: dict) -> dict:
    E = I["models"]["adapter_e"]
    RE = S.adapter_embed(E, I["Xtr"][I["ref_idx"], 0])
    sc = S.label_scores(S.adapter_embed(E, I["Xva"]), RE, I["rlab"], I["classes"])
    return {"scores": sc, "top5": [[I["classes"][j] for j in np.argsort(-sc[q])[:TOPK]] for q in range(len(sc))]}


def confusion_analysis(I: dict, EV: dict) -> dict:
    cls, gv = I["classes"], I["gval"]
    kn = [q for q, g in enumerate(gv) if g["label"] in I["known"]]
    errors = []
    for q in kn:
        g, t5 = gv[q]["label"], EV["top5"][q]
        if t5[0] == g:
            continue
        sc = EV["scores"][q]
        errors.append({"query_id": gv[q]["openverse_id"], "ground_truth": g, "E_top1": t5[0], "E_top5": t5,
                       "correct_in_top5": g in t5, "rank_of_correct": int(np.where(np.argsort(-sc) == cls.index(g))[0][0]) + 1,
                       "confusion_pair": pair_key(g, t5[0]), "E_score_correct": round(float(sc[cls.index(g)]), 4),
                       "E_score_winner": round(float(sc[cls.index(t5[0])]), 4)})
    pairs = defaultdict(list)
    for e in errors:
        pairs[e["confusion_pair"]].append(e)
    table = []
    for k, es in pairs.items():
        rec = [e for e in es if e["correct_in_top5"]]
        table.append({"pair": k, "total_errors": len(es), "recoverable_errors": len(rec),
                      "nonrecoverable_errors": len(es) - len(rec), "correct_in_top5_rate": round(len(rec) / len(es), 4),
                      "average_score_gap": round(float(np.mean([e["E_score_winner"] - e["E_score_correct"] for e in es])), 4),
                      "maximum_possible_recovery": len(rec),
                      "directions": dict(Counter(f"{e['ground_truth']} -> {e['E_top1']}" for e in es))})
    table.sort(key=lambda r: (-r["recoverable_errors"], -r["total_errors"], r["pair"]))
    n_ok = len(kn) - len(errors)
    n_rec = sum(e["correct_in_top5"] for e in errors)
    v6 = json.loads(V6_FA.read_text(encoding="utf-8"))["adapter_e_only"]["aggregate"]
    return {"test_used": False, "n_known_val": len(kn), "E_correct": n_ok, "E_errors": len(errors),
            "E_errors_recoverable": n_rec, "E_errors_nonrecoverable": len(errors) - n_rec,
            "E_top1": round(n_ok / len(kn), 4), "theoretical_v7_ceiling": round((n_ok + n_rec) / len(kn), 4),
            "theoretical_v7_ceiling_count": n_ok + n_rec,
            "required_for_success": {"abs_top1_gain": 0.02, "net_queries": int(np.ceil(0.02 * len(kn)))},
            "cross_check_v6": {"v6_absent_from_top5": v6.get("absent_from_top5"),
                               "matches": v6.get("absent_from_top5") == len(errors) - n_rec},
            "pairs": table, "errors": errors}


def select_specialists(I: dict, CA: dict) -> dict:
    ntr = Counter(i["label"] for i in I["train"])
    nva = Counter(g["label"] for g in I["gval"])
    out = []
    for r in CA["pairs"]:
        a, b = r["pair"].split(" <-> ")
        reasons = []
        if r["recoverable_errors"] < MIN_RECOVERABLE:
            reasons.append(f"recoverable_errors {r['recoverable_errors']} < {MIN_RECOVERABLE}")
        if min(ntr[a], ntr[b]) < MIN_TRAIN_PER_CLASS:
            reasons.append(f"train {a}={ntr[a]} {b}={ntr[b]} < {MIN_TRAIN_PER_CLASS}")
        if min(nva[a], nva[b]) < MIN_VAL_PER_CLASS:
            reasons.append(f"val {a}={nva[a]} {b}={nva[b]} < {MIN_VAL_PER_CLASS}")
        out.append({"pair": r["pair"], "a": a, "b": b, "recoverable_errors": r["recoverable_errors"],
                    "total_errors": r["total_errors"], "train": {a: ntr[a], b: ntr[b]}, "val": {a: nva[a], b: nva[b]},
                    "qualifies": not reasons, "reasons": reasons})
    q = [o for o in out if o["qualifies"]]
    sel = q[:MAX_SPECIALISTS]
    return {"test_used": False, "criteria": {"min_recoverable_errors": MIN_RECOVERABLE, "min_train_per_class": MIN_TRAIN_PER_CLASS,
                                             "min_val_per_class": MIN_VAL_PER_CLASS, "max_specialists": MAX_SPECIALISTS,
                                             "order": "recoverable_errors DESC"},
            "selected": [o["pair"] for o in sel], "n_qualifying": len(q), "candidates": out}


# ------------------------------------------------------------- Phase 3-6: specialists
def head(kind: str, dim: int) -> nn.Module:
    if kind == "linear":
        return nn.Linear(dim, 1)
    return nn.Sequential(nn.Linear(dim, 128), nn.GELU(), nn.Dropout(0.2), nn.Linear(128, 1))


def bin_metrics(p: np.ndarray, y: np.ndarray) -> dict:
    pr = (p >= 0.5).astype(int)
    tp, tn = int(((pr == 1) & (y == 1)).sum()), int(((pr == 0) & (y == 0)).sum())
    fp, fn = int(((pr == 1) & (y == 0)).sum()), int(((pr == 0) & (y == 1)).sum())
    rec_a, rec_b = tp / max(1, tp + fn), tn / max(1, tn + fp)
    return {"accuracy": round((tp + tn) / len(y), 4), "balanced_accuracy": round((rec_a + rec_b) / 2, 4),
            "precision": round(tp / max(1, tp + fp), 4), "recall": round(rec_a, 4),
            "confusion_matrix": {"A_as_A": tp, "A_as_B": fn, "B_as_A": fp, "B_as_B": tn}}


def train_specialist(kind: str, Xtr: np.ndarray, ytr: np.ndarray, Xva: np.ndarray, yva: np.ndarray, seed: int) -> dict:
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    xt = torch.from_numpy((Xtr - mu) / sd).float()
    yt = torch.from_numpy(ytr).float()
    xv = torch.from_numpy((Xva - mu) / sd).float()
    m = head(kind, Xtr.shape[1])
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=WD)
    pw = torch.tensor(float((ytr == 0).sum()) / max(1.0, float((ytr == 1).sum())))
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    hist, best, bad = [], None, 0
    for ep in range(1, MAX_EPOCHS + 1):
        m.train()
        perm = rng.permutation(len(xt))
        tot = 0.0
        for s in range(0, len(perm), BATCH):
            b = torch.from_numpy(perm[s:s + BATCH])
            loss = lossf(m(xt[b]).squeeze(-1), yt[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(b) / len(perm)
        m.eval()
        with torch.no_grad():
            pv = torch.sigmoid(m(xv).squeeze(-1)).numpy()
        bm = bin_metrics(pv, yva)
        hist.append({"epoch": ep, "train_loss": round(tot, 5), **{f"val_{k}": v for k, v in bm.items()}})
        if best is None or bm["balanced_accuracy"] > best["metrics"]["balanced_accuracy"]:
            best = {"epoch": ep, "metrics": bm, "state": {k: v.clone() for k, v in m.state_dict().items()}}
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    m.load_state_dict(best["state"])
    m.eval()
    return {"kind": kind, "model": m, "mu": mu, "sd": sd, "best_epoch": best["epoch"], "val": best["metrics"], "history": hist}


def specialist_prob(sp: dict, x: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return torch.sigmoid(sp["model"](torch.from_numpy((x - sp["mu"]) / sp["sd"]).float()).squeeze(-1)).numpy()


# ------------------------------------------------------------- Phase 8-10: gated pipeline
def apply(EV: dict, specs: list[dict], pA: dict, margins: dict) -> tuple[list[list[str]], list[list[str]]]:
    """Returns final top-5 orders and per-query list of specialists that changed the order."""
    orders, fired = [], []
    for q, t5 in enumerate(EV["top5"]):
        o, f = list(t5), []
        for sp in specs:
            a, b = sp["a"], sp["b"]
            if a not in o or b not in o:
                continue
            p = float(pA[sp["pair"]][q])
            if abs(2 * p - 1) < margins[sp["pair"]]:
                continue
            want_a_first = p >= 0.5
            ia, ib = o.index(a), o.index(b)
            if (ia < ib) != want_a_first:
                o[ia], o[ib] = o[ib], o[ia]
                f.append(sp["pair"])
        orders.append(o)
        fired.append(f)
    return orders, fired


def permuted_scores(EV: dict, orders: list[list[str]], classes: list[str]) -> np.ndarray:
    new = EV["scores"].copy()
    for q, (t5, o) in enumerate(zip(EV["top5"], orders)):
        vals = [float(EV["scores"][q, classes.index(c)]) for c in t5]
        for v, c in zip(vals, o):
            new[q, classes.index(c)] = v
    return new


def changes(I: dict, EV: dict, orders: list[list[str]]) -> dict:
    corr = reg = 0
    for q, g in enumerate(I["gval"]):
        if g["label"] not in I["known"]:
            continue
        e_ok, v_ok = EV["top5"][q][0] == g["label"], orders[q][0] == g["label"]
        corr += int(v_ok and not e_ok)
        reg += int(e_ok and not v_ok)
    return {"corrections": corr, "regressions": reg, "net": corr - reg}


def full_metrics(I: dict, sc: np.ndarray) -> dict:
    m = V6.metrics(sc, I, I["cfgs"]["adapter_e"]["adapter"]["thresholds"])[0]
    return {k: round(v, 4) for k, v in m.items()}


def learning_curve(kind: str, Xtr, ytr, gtr, Xva, yva, n_a: int, n_b: int, seed: int) -> dict:
    """Image-level subsampling of TRAIN (all views of a kept image kept together); VAL balanced accuracy."""
    pts = []
    imgs = np.unique(gtr)
    for frac in (0.4, 0.6, 0.8, 1.0):
        rng = np.random.RandomState(seed + int(frac * 100))
        keep = set(rng.choice(imgs, max(4, int(round(frac * len(imgs)))), replace=False).tolist())
        m = np.array([g in keep for g in gtr])
        if len(set(ytr[m])) < 2:
            continue
        r = train_specialist(kind, Xtr[m], ytr[m], Xva, yva, seed)
        pts.append({"frac": frac, "n_images": len(keep), "val_balanced_accuracy": r["val"]["balanced_accuracy"]})
    n = np.array([p["n_images"] for p in pts], float)
    y = np.array([p["val_balanced_accuracy"] for p in pts])
    slope, icpt = np.polyfit(np.log(n), y, 1) if len(pts) >= 2 else (0.0, float(y[-1]))
    cur = n_a + n_b
    if y[-1] >= TARGET_BAL_ACC:
        need = {"status": "target already met", "additional_per_class": 0}
    elif slope <= 1e-3:
        need = {"status": "flat learning curve: more of the same images will not separate this pair",
                "additional_per_class": None}
    else:
        n_needed = float(np.exp((TARGET_BAL_ACC - icpt) / slope))
        add = int(np.ceil(max(0.0, n_needed - cur) / 2))
        need = {"status": "log-linear extrapolation", "n_images_needed_total": int(np.ceil(n_needed)),
                "additional_per_class": add, "extrapolation_factor": round(n_needed / cur, 2)}
    return {"points": pts, "slope_per_log_n": round(float(slope), 4), "target_balanced_accuracy": TARGET_BAL_ACC, **need}


def main() -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    I = V6.load_inputs()
    assert not any(i["split"] == "test" for i in I["train"] + I["val"])
    EV = e_val(I)
    e_metrics = full_metrics(I, EV["scores"])
    v6r = json.loads(V6_RES.read_text(encoding="utf-8"))
    assert abs(e_metrics["known_identity_top1_raw"] - v6r["val_frozen_models"]["adapter_e"]["known_identity_top1_raw"]) < 1e-4

    CA = confusion_analysis(I, EV)
    ca_sha = wjson("confusion_analysis.json", CA)
    SEL = select_specialists(I, CA)
    sel_sha = wjson("specialist_selection.json", SEL)
    print("E", CA["E_top1"], "ceiling", CA["theoretical_v7_ceiling"], "errors", CA["E_errors"], "recoverable",
          CA["E_errors_recoverable"], "selected", SEL["selected"], flush=True)
    if not SEL["selected"]:
        wjson("specialist_results.json", {"test_used": False, "V7_STATUS": "STOPPED", "reason": "no qualifying pair"})
        return

    E = I["models"]["adapter_e"]
    trE = S.adapter_embed(E, I["Xtr"].reshape(-1, 768)).reshape(len(I["train"]), 8, -1)
    vaE = S.adapter_embed(E, I["Xva"])
    reps = {"adapter_e": (trE, vaE), "frozen_dinov2": (I["Xtr"], I["Xva"])}
    trlab = np.array([i["label"] for i in I["train"]])
    valab = np.array([g["label"] for g in I["gval"]])
    cand = {c["pair"]: c for c in SEL["candidates"]}
    specs, results = [], []
    for k, pair in enumerate(SEL["selected"]):
        a, b = cand[pair]["a"], cand[pair]["b"]
        ti = np.where((trlab == a) | (trlab == b))[0]
        vi = np.where((valab == a) | (valab == b))[0]
        variants = {}
        for rep, (TR, VA) in reps.items():
            Xtr = TR[ti].reshape(-1, TR.shape[-1])
            ytr = np.repeat((trlab[ti] == a).astype(float), 8)
            gtr = np.repeat(ti, 8)
            Xva, yva = VA[vi], (valab[vi] == a).astype(float)
            for kind in ("linear", "mlp"):
                r = train_specialist(kind, Xtr, ytr, Xva, yva, SEED + 10 * k)
                r.update({"rep": rep, "Xtr": Xtr, "ytr": ytr, "gtr": gtr, "Xva": Xva, "yva": yva})
                variants[f"{rep}/{kind}"] = r
        # primary representation = Adapter E; MLP kept only if it beats linear on VAL balanced accuracy
        lin, mlp = variants["adapter_e/linear"], variants["adapter_e/mlp"]
        chosen = mlp if mlp["val"]["balanced_accuracy"] > lin["val"]["balanced_accuracy"] else lin
        sp = {"pair": pair, "a": a, "b": b}
        # gated activation on full VAL (all queries; inactive unless both A and B in E top-5)
        abl = {}
        pA_all = {}
        for name, r in variants.items():
            VA = reps[r["rep"]][1]
            pA_all[name] = specialist_prob(r, VA)
            by_m = {}
            for mg in MARGINS:
                o, _ = apply(EV, [sp], {pair: pA_all[name]}, {pair: mg})
                by_m[str(mg)] = changes(I, EV, o)
            abl[name] = by_m
        active = [q for q, t5 in enumerate(EV["top5"]) if a in t5 and b in t5]
        act_known_ab = [q for q in active if valab[q] in (a, b)]
        cname = f"adapter_e/{chosen['kind']}"
        best_m = max(MARGINS, key=lambda mg: (abl[cname][str(mg)]["net"], -abl[cname][str(mg)]["regressions"], mg))
        pc = pA_all[cname]
        e_acc_active = np.mean([EV["top5"][q][0] == valab[q] for q in act_known_ab]) if act_known_ab else None
        s_acc_active = np.mean([(a if pc[q] >= 0.5 else b) == valab[q] for q in act_known_ab]) if act_known_ab else None
        lc = learning_curve(chosen["kind"], chosen["Xtr"], chosen["ytr"], chosen["gtr"], chosen["Xva"], chosen["yva"],
                            cand[pair]["train"][a], cand[pair]["train"][b], SEED + 10 * k)
        sp.update({"kind": chosen["kind"], "rep": "adapter_e", "margin": best_m, "model": chosen})
        specs.append(sp)
        results.append({
            "pair": pair, "a": a, "b": b, "train_images": cand[pair]["train"], "val_images": cand[pair]["val"],
            "E_errors_involving_pair": cand[pair]["total_errors"], "recoverable_E_errors": cand[pair]["recoverable_errors"],
            "chosen_model": cname, "mlp_kept": chosen["kind"] == "mlp", "best_epoch": chosen["best_epoch"],
            "val_binary": {n: {"best_epoch": r["best_epoch"], **r["val"]} for n, r in variants.items()},
            "history": {n: r["history"] for n, r in variants.items()},
            "control": {"n_active_val_queries": len(active), "n_active_known_gt_in_pair": len(act_known_ab),
                        "E_top1_on_active": None if e_acc_active is None else round(float(e_acc_active), 4),
                        "specialist_acc_on_active": None if s_acc_active is None else round(float(s_acc_active), 4)},
            "ablation_by_margin": abl, "selected_margin": best_m, "individual_at_margin": abl[cname][str(best_m)],
            "learning_curve": lc})
        print(pair, cname, chosen["val"]["balanced_accuracy"], "margin", best_m, abl[cname][str(best_m)],
              "active", len(active), flush=True)

    # Phase 13: combined, specialists applied in selection order (recoverable_errors DESC)
    pA = {sp["pair"]: specialist_prob(sp["model"], vaE) for sp in specs}
    margins = {sp["pair"]: sp["margin"] for sp in specs}
    useful = [sp for sp, r in zip(specs, results) if r["individual_at_margin"]["net"] > 0]
    combos = {"all_selected": specs, "only_individually_positive": useful}
    combo_out = {}
    for name, ss in combos.items():
        o, fired = apply(EV, ss, pA, margins)
        sc = permuted_scores(EV, o, I["classes"])
        ch = changes(I, EV, o)
        indiv_sum = sum(r["individual_at_margin"]["net"] for sp, r in zip(specs, results) if any(sp is s for s in ss))
        multi = [q for q, f in enumerate(fired) if len(f) > 1]
        combo_out[name] = {"pairs": [s["pair"] for s in ss], "metrics": full_metrics(I, sc), "changes": ch,
                           "sum_of_individual_net": indiv_sum, "interaction_net_loss": indiv_sum - ch["net"],
                           "queries_with_multiple_specialists_firing": multi, "_orders": o}
    final_name = max(combo_out, key=lambda n: (combo_out[n]["changes"]["net"], -combo_out[n]["changes"]["regressions"]))
    F = combo_out[final_name]
    fm = F["metrics"]
    gain = round(fm["known_identity_top1_raw"] - e_metrics["known_identity_top1_raw"], 4)
    checks = {"top1_gain_ge_0.02": gain >= 0.02 - 1e-9, "corrections_ge_5": F["changes"]["corrections"] >= 5,
              "regressions_le_2": F["changes"]["regressions"] <= 2,
              "unknown_rejection_ge_E": fm["unknown_rejection"] >= e_metrics["unknown_rejection"],
              "false_confirmation_le_E_plus_0.005": fm["false_confirmation"] <= e_metrics["false_confirmation"] + 0.005}
    status = "SUCCESS" if all(checks.values()) else "INSUFFICIENT"

    # per-query records for the final system
    recs = []
    for q, g in enumerate(I["gval"]):
        if g["label"] in I["known"] and (EV["top5"][q][0] != g["label"] or F["_orders"][q][0] != g["label"]):
            recs.append({"query_id": g["openverse_id"], "gt": g["label"], "E_top5": EV["top5"][q], "V7_top5": F["_orders"][q],
                         "E_ok": EV["top5"][q][0] == g["label"], "V7_ok": F["_orders"][q][0] == g["label"]})
    out = {"test_used": False, "V7_STATUS": status, "success_checks": checks, "top1_gain": gain,
           "E_val": e_metrics, "final_system": final_name,
           "combinations": {n: {k: v for k, v in c.items() if not k.startswith("_")} for n, c in combo_out.items()},
           "specialists": results, "final_changed_or_wrong_queries": recs,
           "theoretical_v7_ceiling": CA["theoretical_v7_ceiling"],
           "max_recoverable_by_selected_pairs": sum(r["recoverable_E_errors"] for r in results),
           "inputs": {"confusion_analysis_sha256": ca_sha, "specialist_selection_sha256": sel_sha, **I["provenance"]},
           "selection_bias_note": "model variant, epoch and margin are all chosen on the same VAL split that is scored"}
    res_sha = wjson("specialist_results.json", out)
    print("STATUS", status, "gain", gain, F["changes"], checks, "results_sha", res_sha, flush=True)
    if status == "SUCCESS":
        ck = DOCS / "v7_specialists_final.pt"
        torch.save({sp["pair"]: {"a": sp["a"], "b": sp["b"], "kind": sp["kind"], "margin": sp["margin"],
                                 "state_dict": sp["model"]["model"].state_dict(), "mu": torch.from_numpy(sp["model"]["mu"]),
                                 "sd": torch.from_numpy(sp["model"]["sd"])} for sp in combos[final_name]}, ck)
        cfg = {"config_id": "kiq_v7_specialists_v1", "frozen": True, "test_used": False, "selection_split": "val",
               "base_model": "frozen Adapter E", "pairs": [{k: sp[k] for k in ("pair", "a", "b", "kind", "margin")}
                                                          for sp in combos[final_name]],
               "gating": "specialist active only if both A and B are in E top-5; swaps A/B only if |P(A)-P(B)| >= margin",
               "decision": "E frozen thresholds; E top-5 score values permuted onto final order",
               "val_metrics": fm, "checkpoint": ck.name, "checkpoint_sha256": sha(ck), **I["provenance"]}
        cp = DOCS / "v7_specialists_frozen_config.json"
        cp.write_text(json.dumps(cfg, indent=1, sort_keys=True), encoding="utf-8")
        print("FROZEN", sha(cp), sha(ck))


if __name__ == "__main__":
    main()
