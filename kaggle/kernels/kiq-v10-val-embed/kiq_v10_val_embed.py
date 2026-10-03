"""KitchenIQ V5 stages on Kaggle. MODE is set per kernel copy (see orchestration/vision_v5/make_stage_kernels.py).

reflib : TRAIN split only -> clean, representative reference library v1 (frozen DINOv2 metadata + embeddings)
adapter: TRAIN fit, VAL selection only -> Adapter C frozen config + checkpoint (test images never opened)
test   : frozen configs only -> one-shot TEST evaluation of frozen-DINOv2 baseline vs Adapter C
"""
from __future__ import annotations

import hashlib
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageEnhance

MODE = "v10_val"
SEED = 1234
PHOTO_MANIFEST_SHA = "89ea9b20effdf40983ce670722f43fce42e44bd4378cc5810853dbba94c656e0"
LIBRARY_MANIFEST_SHA = "UNUSED"
ADAPTER_CONFIG_SHA = "UNUSED"
INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working") / f"v5_{MODE}"

# library selection criteria (documented in docs/v5/REFERENCE_LIBRARY_V1.md)
REF_MAX_PER_LABEL = 20
REF_MIN_PER_LABEL = 5
REF_DEDUP_COS = 0.90
KNN_K = 5

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def status(msg: str) -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    line = f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with (WORK / "LIVE_STATUS.txt").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def write_json(name: str, obj) -> str:
    raw = json.dumps(obj, sort_keys=True, indent=1).encode("utf-8")
    (WORK / name).write_bytes(raw)
    return sha_bytes(raw)


def find_by_sha(name: str, expected: str) -> Path:
    for p in INPUT.rglob(name):
        if sha_bytes(p.read_bytes()) == expected:
            return p
    raise SystemExit(f"{name} with sha {expected} not found — frozen artifact missing or changed")


def load_photo_set(allowed_splits: set[str]) -> tuple[list[dict], Path]:
    mp = find_by_sha("manifest.json", PHOTO_MANIFEST_SHA)
    items = [i for i in json.loads(mp.read_bytes())["items"] if i["split"] in allowed_splits]
    root = mp.parent
    for i in items:
        b = (root / i["file"]).read_bytes()
        if sha_bytes(b) != i["sha256"]:
            raise SystemExit(f"image hash mismatch {i['file']} — real_photo_set_v1 altered, abort")
    status(f"photo set verified splits={sorted(allowed_splits)} n={len(items)}")
    return items, root


# ------------------------------------------------------------- frozen DINOv2
_DINO = None


def dino():
    global _DINO
    if _DINO is None:
        _DINO = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", trust_repo=True).eval().to(DEV)
    return _DINO


def to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img.convert("RGB").resize((224, 224), Image.BILINEAR)).astype("float32") / 255.0
    arr = (arr - np.array([0.485, 0.456, 0.406], "float32")) / np.array([0.229, 0.224, 0.225], "float32")
    return torch.from_numpy(arr).permute(2, 0, 1).float()


@torch.no_grad()
def embed_batch(imgs: list[Image.Image]) -> np.ndarray:
    x = torch.stack([to_tensor(i) for i in imgs]).to(DEV)
    f = dino().forward_features(x)["x_norm_clstoken"].float()
    return F.normalize(f, dim=-1).cpu().numpy()


def augment(img: Image.Image, rng: random.Random) -> Image.Image:
    w, h = img.size
    s = rng.uniform(0.6, 1.0)
    cw, ch = int(w * s), int(h * s)
    x0, y0 = rng.randint(0, w - cw), rng.randint(0, h - ch)
    out = img.crop((x0, y0, x0 + cw, y0 + ch))
    if rng.random() < 0.5:
        out = out.transpose(Image.FLIP_LEFT_RIGHT)
    out = ImageEnhance.Brightness(out).enhance(rng.uniform(0.75, 1.25))
    out = ImageEnhance.Color(out).enhance(rng.uniform(0.75, 1.25))
    return out


def embed_items(items: list[dict], root: Path, views: int = 1) -> np.ndarray:
    """Returns [n, views, 768]; view 0 is the un-augmented image."""
    rng = random.Random(SEED)
    out = np.zeros((len(items), views, 768), np.float32)
    for start in range(0, len(items), 32):
        chunk = items[start:start + 32]
        imgs = [Image.open(root / i["file"]).convert("RGB") for i in chunk]
        for v in range(views):
            batch = imgs if v == 0 else [augment(im, rng) for im in imgs]
            out[start:start + len(chunk), v] = embed_batch(batch)
    return out


# ------------------------------------------------------------- shared retrieval + decision
def norm(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-8)


def label_scores(Q: np.ndarray, R: np.ndarray, rlab: np.ndarray, labels: list[str]) -> np.ndarray:
    """score[q, c] = 0.5 * cos(q, prototype_c) + 0.5 * mean(top-3 cos(q, refs_c))."""
    S = Q @ R.T
    out = np.zeros((len(Q), len(labels)), np.float32)
    for j, c in enumerate(labels):
        m = rlab == c
        proto = norm(R[m].mean(0))
        top = np.sort(S[:, m], axis=1)[:, ::-1][:, :3].mean(1)
        out[:, j] = 0.5 * (Q @ proto) + 0.5 * top
    return out


def decide(scores: np.ndarray, labels: list[str], fam: dict[str, str], t_id: float, m_id: float, t_fam: float) -> list[dict]:
    res = []
    for row in scores:
        order = np.argsort(-row)
        s1, s2 = float(row[order[0]]), float(row[order[1]])
        top = labels[order[0]]
        if s1 >= t_id and s1 - s2 >= m_id:
            res.append({"level": "identity", "label": top, "family": fam[top], "score": s1, "margin": s1 - s2})
            continue
        fs: dict[str, float] = {}
        for j in order:
            f = fam[labels[j]]
            fs[f] = max(fs.get(f, -9.0), float(row[j]))
        bf = max(fs, key=fs.get)
        if fs[bf] >= t_fam:
            res.append({"level": "family", "label": None, "family": bf, "score": fs[bf], "margin": s1 - s2})
        else:
            res.append({"level": "unknown", "label": None, "family": None, "score": s1, "margin": s1 - s2})
    return res


def objective(dec: list[dict], gt: list[dict], known: set[str]) -> dict:
    nk = sum(1 for g in gt if g["label"] in known)
    nu = len(gt) - nk
    k_score = u_rej = false_conf = 0.0
    for d, g in zip(dec, gt):
        is_known = g["label"] in known
        if d["level"] == "identity" and d["label"] != g["label"]:
            false_conf += 1
        if is_known:
            if d["level"] == "identity" and d["label"] == g["label"]:
                k_score += 1
            elif d["level"] == "family" and d["family"] == g["family"]:
                k_score += 0.5
        elif d["level"] != "identity":
            u_rej += 1
    obj = k_score / max(1, nk) + u_rej / max(1, nu) - 2.0 * false_conf / max(1, len(gt))
    return {"objective": round(obj, 4), "known_score": round(k_score / max(1, nk), 4),
            "unknown_rejection": round(u_rej / max(1, nu), 4), "false_confirmation": round(false_conf / max(1, len(gt)), 4)}


def calibrate(scores: np.ndarray, gt: list[dict], labels: list[str], fam: dict[str, str], known: set[str]) -> dict:
    best = None
    for t_id in np.arange(0.20, 0.97, 0.02):
        for m_id in (0.0, 0.02, 0.05, 0.1):
            for t_fam in (t_id - 0.05, t_id - 0.1, 9.0):
                o = objective(decide(scores, labels, fam, t_id, m_id, t_fam), gt, known)
                if best is None or o["objective"] > best[0]["objective"]:
                    best = (o, {"t_id": round(float(t_id), 4), "m_id": m_id, "t_fam": round(float(t_fam), 4)})
    return {"thresholds": best[1], "val": best[0]}


def detail_metrics(scores: np.ndarray, dec: list[dict], gt: list[dict], labels: list[str], fam: dict[str, str],
                   state: dict[str, str], known: set[str]) -> dict:
    kn = [i for i, g in enumerate(gt) if g["label"] in known]
    top1 = [labels[int(np.argmax(scores[i]))] for i in range(len(gt))]
    top5 = [[labels[j] for j in np.argsort(-scores[i])[:5]] for i in range(len(gt))]
    return {
        "known_identity_top1_raw": round(np.mean([top1[i] == gt[i]["label"] for i in kn]), 4),
        "known_identity_top5_raw": round(np.mean([gt[i]["label"] in top5[i] for i in kn]), 4),
        "known_family_top1_raw": round(np.mean([fam[top1[i]] == gt[i]["family"] for i in kn]), 4),
        "known_state_top1_raw": round(np.mean([state[top1[i]] == gt[i]["state"] for i in kn]), 4),
        "known_accepted_correct": round(np.mean([dec[i]["level"] == "identity" and dec[i]["label"] == gt[i]["label"] for i in kn]), 4),
        "known_coverage_identity": round(np.mean([dec[i]["level"] == "identity" for i in kn]), 4),
        **objective(dec, gt, known),
    }


def earliest_divergence(g: dict, d: dict, top1: str, fam: dict[str, str], known: set[str]) -> str:
    if g["label"] not in known:
        return "OK" if d["level"] != "identity" else "OPEN_SET_FALSE_ACCEPT"
    if (g["label"] == "non_food") != (top1 == "non_food"):
        return "FOOD_NONFOOD"
    if fam[top1] != g["family"]:
        return "FAMILY"
    if top1 != g["label"]:
        return "IDENTITY"
    if d["level"] != "identity":
        return "OPEN_SET_OVER_ABSTAIN"
    return "OK"


# ------------------------------------------------------------- adapter C
class AdapterC(nn.Module):
    def __init__(self, n_classes: int, hidden: int, dim: int, dropout: float):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, hidden), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(hidden, dim))
        self.skip = nn.Linear(768, dim, bias=False)
        self.cls = nn.Parameter(torch.randn(n_classes, dim) * 0.02)
        self.scale = nn.Parameter(torch.tensor(16.0))

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj(x) + self.skip(x), dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * self.embed(x) @ F.normalize(self.cls, dim=-1).T


@torch.no_grad()
def adapter_embed(model: AdapterC, X: np.ndarray) -> np.ndarray:
    model.eval()
    return model.embed(torch.from_numpy(X).float().to(DEV)).cpu().numpy()


# ------------------------------------------------------------- MODE reflib
def run_reflib() -> None:
    items, root = load_photo_set({"train"})
    assert all(i["split"] == "train" for i in items) and all(i["role"] == "known" for i in items)
    E = embed_items(items, root, views=1)[:, 0]
    labels = np.array([i["label"] for i in items])
    S = E @ E.T
    np.fill_diagonal(S, -9)
    label_set = sorted(set(labels))
    selected, stats = [], {}
    for c in label_set:
        idx = np.where(labels == c)[0]
        rows = []
        for i in idx:
            # leave-self-out label-consistency: own LOO centroid must be the nearest centroid, or kNN majority agrees
            cent = {}
            for d in label_set:
                m = labels == d
                if d == c:
                    m = m.copy()
                    m[i] = False
                cent[d] = float(E[i] @ norm(E[m].mean(0)))
            nn_lab = Counter(labels[np.argsort(-S[i])[:KNN_K]]).most_common(1)[0][0]
            own = cent[c]
            consistent = max(cent, key=cent.get) == c or nn_lab == c
            rows.append({"i": int(i), "own_centroid_cos": own, "nearest_centroid": max(cent, key=cent.get),
                         "knn_majority": nn_lab, "consistent": consistent})
        pool = [r for r in rows if r["consistent"]]
        low = len(pool) < REF_MIN_PER_LABEL
        if low:
            pool = sorted(rows, key=lambda r: -r["own_centroid_cos"])[:REF_MIN_PER_LABEL]
        pool.sort(key=lambda r: -r["own_centroid_cos"])
        kept, dup_removed = [], 0
        for r in pool:
            if any(float(E[r["i"]] @ E[k["i"]]) >= REF_DEDUP_COS for k in kept):
                dup_removed += 1
                continue
            kept.append(r)
        # representative cap: medoid first, then farthest-point sampling over the consistent pool
        if len(kept) > REF_MAX_PER_LABEL:
            sub = np.stack([E[r["i"]] for r in kept])
            chosen = [int(np.argmax((sub @ sub.T).mean(1)))]
            while len(chosen) < REF_MAX_PER_LABEL:
                mind = 1 - (sub @ sub[chosen].T).max(1)
                mind[chosen] = -1
                chosen.append(int(np.argmax(mind)))
            kept = [kept[j] for j in sorted(chosen)]
        for r in kept:
            it = items[r["i"]]
            selected.append({
                "ref_id": f"rpv1:{it['openverse_id']}", "sha256": it["sha256"], "file": it["file"],
                "label": it["label"], "family": it["family"], "state": it["state"], "group": it["group"],
                "provider": it["provider"], "landing_url": it["landing_url"], "licence": it["licence"],
                "licence_url": it["licence_url"], "creator": it.get("creator"),
                "selection": {"own_centroid_cos": round(r["own_centroid_cos"], 4), "knn_majority": r["knn_majority"],
                              "nearest_centroid": r["nearest_centroid"], "consistent": r["consistent"],
                              "low_consistency_fallback": low}})
        stats[c] = {"n_train": int(len(idx)), "n_consistent": sum(r["consistent"] for r in rows),
                    "n_dedup_removed": dup_removed, "n_selected": len(kept), "low_consistency_fallback": low}
        status(f"reflib {c} {stats[c]}")
    selected.sort(key=lambda r: r["ref_id"])
    assert len(selected) > 0 and len({r["sha256"] for r in selected}) == len(selected)
    manifest = {"library_id": "kiq_v5_reference_library_v1", "source_set": "kiq_v5_real_photo_v1",
                "source_manifest_sha256": PHOTO_MANIFEST_SHA, "source_split": "train",
                "criteria": {"label_consistency": "LOO own-centroid nearest OR kNN(k=5) majority",
                             "dedup_cos": REF_DEDUP_COS, "max_per_label": REF_MAX_PER_LABEL,
                             "min_per_label": REF_MIN_PER_LABEL,
                             "cap_method": "medoid + farthest-point sampling", "backbone": "DINOv2:dinov2_vitb14"},
                "n": len(selected), "label_stats": stats, "items": selected}
    lib_sha = write_json("reference_library_v1_manifest.json", manifest)
    pos = {i["sha256"]: k for k, i in enumerate(items)}
    emb = {r["ref_id"]: [round(float(x), 6) for x in E[pos[r["sha256"]]]] for r in selected}
    emb_sha = write_json("reference_library_v1_dinov2_embeddings.json",
                         {"library_manifest_sha256": lib_sha, "backbone": "DINOv2:dinov2_vitb14", "embeddings": emb})
    status(f"DONE reflib n={len(selected)} manifest_sha={lib_sha} emb_sha={emb_sha}")


# ------------------------------------------------------------- MODE adapter
def load_library() -> tuple[dict, Path]:
    p = find_by_sha("reference_library_v1_manifest.json", LIBRARY_MANIFEST_SHA)
    return json.loads(p.read_bytes()), p


def gt_rows(items: list[dict]) -> list[dict]:
    return [{"label": i["label"], "family": i["family"], "state": i["state"], "role": i["role"],
             "openverse_id": i["openverse_id"]} for i in items]


def run_adapter() -> None:
    lib, _ = load_library()
    items, root = load_photo_set({"train", "val"})
    assert not any(i["split"] == "test" for i in items)
    train = [i for i in items if i["split"] == "train"]
    val = [i for i in items if i["split"] == "val"]
    ref_sha = {r["sha256"] for r in lib["items"]}
    assert ref_sha <= {i["sha256"] for i in train}, "library contains non-train images"
    classes = sorted({i["label"] for i in train})
    cix = {c: k for k, c in enumerate(classes)}
    fam = {i["label"]: i["family"] for i in items}
    state = {i["label"]: i["state"] for i in items}
    known = set(classes)

    Xtr = embed_items(train, root, views=8)
    Xva = embed_items(val, root, views=1)[:, 0]
    ytr = np.array([cix[i["label"]] for i in train])
    tr_pos = {i["sha256"]: k for k, i in enumerate(train)}
    ref_idx = [tr_pos[r["sha256"]] for r in lib["items"]]
    Rf = Xtr[ref_idx, 0]
    rlab = np.array([r["label"] for r in lib["items"]])
    gval = gt_rows(val)
    status(f"embedded train={Xtr.shape} val={Xva.shape} refs={len(ref_idx)} classes={len(classes)}")

    # frozen DINOv2 baseline: same refs, same scorer, same calibration procedure, same val set
    base_scores = label_scores(Xva, Rf, rlab, classes)
    base_cal = calibrate(base_scores, gval, classes, fam, known)
    status(f"baseline val {base_cal}")

    grid = [{"hidden": 512, "dim": d, "dropout": p, "lr": lr, "wd": 1e-4, "epochs": 80, "label_smoothing": 0.1}
            for d in (128, 256) for p in (0.1, 0.3) for lr in (1e-3, 3e-4)]
    ck = WORK / "checkpoints"
    ck.mkdir(parents=True, exist_ok=True)
    results = []
    Xt = torch.from_numpy(Xtr.reshape(-1, 768)).float().to(DEV)
    yt = torch.from_numpy(np.repeat(ytr, Xtr.shape[1])).long().to(DEV)
    for gi, cfg in enumerate(grid):
        done = ck / f"cfg{gi}_result.json"
        if done.exists():
            results.append(json.loads(done.read_text()))
            continue
        torch.manual_seed(SEED + gi)
        model = AdapterC(len(classes), cfg["hidden"], cfg["dim"], cfg["dropout"]).to(DEV)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg["epochs"])
        best = None
        for ep in range(1, cfg["epochs"] + 1):
            model.train()
            perm = torch.randperm(len(Xt), device=DEV)
            for s in range(0, len(Xt), 256):
                b = perm[s:s + 256]
                loss = F.cross_entropy(model(Xt[b]), yt[b], label_smoothing=cfg["label_smoothing"])
                opt.zero_grad()
                loss.backward()
                opt.step()
            sched.step()
            if ep % 5 == 0:
                Rv = adapter_embed(model, Rf)
                sc = label_scores(adapter_embed(model, Xva), Rv, rlab, classes)
                cal = calibrate(sc, gval, classes, fam, known)
                if best is None or cal["val"]["objective"] > best["val"]["objective"]:
                    best = {"epoch": ep, **cal}
                    torch.save({"state_dict": model.state_dict(), "cfg": cfg, "classes": classes, "epoch": ep},
                               ck / f"cfg{gi}_best.pt")
        r = {"cfg_index": gi, "cfg": cfg, **best}
        done.write_text(json.dumps(r))
        results.append(r)
        status(f"cfg{gi} {cfg} best_epoch={best['epoch']} val={best['val']}")

    winner = max(results, key=lambda r: (r["val"]["objective"], -r["cfg_index"]))
    blob = torch.load(ck / f"cfg{winner['cfg_index']}_best.pt", map_location=DEV)
    final = WORK / "adapter_c_final.pt"
    torch.save(blob, final)
    model = AdapterC(len(classes), winner["cfg"]["hidden"], winner["cfg"]["dim"], winner["cfg"]["dropout"]).to(DEV)
    model.load_state_dict(blob["state_dict"])
    t = winner["thresholds"]
    sc = label_scores(adapter_embed(model, Xva), adapter_embed(model, Rf), rlab, classes)
    adapter_val = detail_metrics(sc, decide(sc, classes, fam, t["t_id"], t["m_id"], t["t_fam"]), gval, classes, fam, state, known)
    bt = base_cal["thresholds"]
    base_val = detail_metrics(base_scores, decide(base_scores, classes, fam, bt["t_id"], bt["m_id"], bt["t_fam"]),
                              gval, classes, fam, state, known)
    config = {"config_id": "kiq_v5_adapter_c_v1", "frozen": True, "selection_split": "val", "test_used": False,
              "photo_manifest_sha256": PHOTO_MANIFEST_SHA, "library_manifest_sha256": LIBRARY_MANIFEST_SHA,
              "classes": classes, "families": fam, "states": state,
              "adapter": {"cfg": winner["cfg"], "cfg_index": winner["cfg_index"], "epoch": winner["epoch"],
                          "thresholds": t, "checkpoint": final.name,
                          "checkpoint_sha256": sha_bytes(final.read_bytes())},
              "baseline": {"name": "frozen_dinov2_hierarchical_retrieval", "thresholds": bt},
              "scorer": "0.5*cos(prototype)+0.5*mean(top3 cos); identity->family->unknown back-off",
              "selection_objective": "known_score + unknown_rejection - 2*false_confirmation (val)",
              "train_views": 8, "seed": SEED}
    cfg_sha = write_json("adapter_c_frozen_config.json", config)
    write_json("adapter_c_selection_report.json", {"grid_results": results, "winner": winner["cfg_index"],
                                                  "val_baseline": base_val, "val_adapter_c": adapter_val,
                                                  "frozen_config_sha256": cfg_sha})
    status(f"DONE adapter val_baseline={base_val} val_adapter={adapter_val} config_sha={cfg_sha}")


# ------------------------------------------------------------- MODE test
def run_test() -> None:
    cfg_path = find_by_sha("adapter_c_frozen_config.json", ADAPTER_CONFIG_SHA)
    cfg = json.loads(cfg_path.read_bytes())
    assert cfg["frozen"] and not cfg["test_used"] and cfg["library_manifest_sha256"] == LIBRARY_MANIFEST_SHA
    lib, _ = load_library()
    ck = cfg_path.parent / cfg["adapter"]["checkpoint"]
    assert sha_bytes(ck.read_bytes()) == cfg["adapter"]["checkpoint_sha256"], "checkpoint changed"
    train, root = load_photo_set({"train"})
    test, _ = load_photo_set({"test"})
    classes, fam, state = cfg["classes"], cfg["families"], cfg["states"]
    for i in test:
        fam.setdefault(i["label"], i["family"])
        state.setdefault(i["label"], i["state"])
    known = set(classes)
    tr_pos = {i["sha256"]: k for k, i in enumerate(train)}
    lib_items = [train[tr_pos[r["sha256"]]] for r in lib["items"]]
    Rf = embed_items(lib_items, root, views=1)[:, 0]
    rlab = np.array([r["label"] for r in lib["items"]])
    Xte = embed_items(test, root, views=1)[:, 0]
    gte = gt_rows(test)
    blob = torch.load(ck, map_location=DEV)
    a = cfg["adapter"]["cfg"]
    model = AdapterC(len(classes), a["hidden"], a["dim"], a["dropout"]).to(DEV)
    model.load_state_dict(blob["state_dict"])
    out, decs, scs = {}, {}, {}
    for name, X, R, t in (("baseline", Xte, Rf, cfg["baseline"]["thresholds"]),
                          ("adapter_c", adapter_embed(model, Xte), adapter_embed(model, Rf), cfg["adapter"]["thresholds"])):
        sc = label_scores(X, R, rlab, classes)
        d = decide(sc, classes, fam, t["t_id"], t["m_id"], t["t_fam"])
        out[name] = detail_metrics(sc, d, gte, classes, fam, state, known)
        decs[name], scs[name] = d, sc
    per = []
    for k, g in enumerate(gte):
        row = {"openverse_id": g["openverse_id"], "gt": g["label"], "gt_family": g["family"], "role": g["role"]}
        for name in ("baseline", "adapter_c"):
            top1 = classes[int(np.argmax(scs[name][k]))]
            d = decs[name][k]
            row[name] = {"top1": top1, "decision": d["level"], "label": d["label"], "family": d["family"],
                         "score": round(d["score"], 4), "divergence": earliest_divergence(g, d, top1, fam, known)}
        b_ok, a_ok = row["baseline"]["divergence"] == "OK", row["adapter_c"]["divergence"] == "OK"
        row["change"] = "fixed" if a_ok and not b_ok else ("regressed" if b_ok and not a_ok else ("both_ok" if a_ok else "both_fail"))
        per.append(row)
    summary = {"split": "test", "evaluated_once": True, "frozen_config_sha256": ADAPTER_CONFIG_SHA,
               "library_manifest_sha256": LIBRARY_MANIFEST_SHA, "photo_manifest_sha256": PHOTO_MANIFEST_SHA,
               "metrics": out,
               "divergence_counts": {n: dict(Counter(r[n]["divergence"] for r in per)) for n in ("baseline", "adapter_c")},
               "change_counts": dict(Counter(r["change"] for r in per)),
               "open_set": {n: {"n": sum(1 for r in per if r["role"] == "open_set_unknown"),
                                "rejected": sum(1 for r in per if r["role"] == "open_set_unknown" and r[n]["decision"] != "identity"),
                                "family_backoff": sum(1 for r in per if r["role"] == "open_set_unknown" and r[n]["decision"] == "family")}
                            for n in ("baseline", "adapter_c")}}
    write_json("test_results.json", summary)
    write_json("test_per_item.json", per)
    status(f"DONE test {json.dumps(summary)}")



# ===================== kiq_v10_regions.py (verbatim)
"""KitchenIQ V10-VR region proposals, matched-budget control crops, multi-view crops and arm layout.

Shared, unchanged, by the Kaggle VAL embedding kernel and the local 41-scene diagnostic. Nothing here scores or
decides; it only says WHICH image boxes become embedding queries. Every arm issues exactly PROTOCOL["N"] queries.
"""

import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PROTOCOL = {
    "protocol_id": "kiq_v10_m1_m3_protocol_v1",
    "N": 8,
    "topk_per_view": 20,
    "union_limit": 20,
    "seed": 1234,
    "control": {"side_fraction": [0.4, 0.9], "aspect": [0.75, 1.3333], "rng_key": "seed:image_sha256"},
    "proposals": {
        "max_per_source": 100, "nms_iou": 0.5, "min_side_px": 16, "class_agnostic": True,
        "frcnn": {"builder": "torchvision.models.detection.fasterrcnn_resnet50_fpn_v2",
                  "weights": "FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1", "box_score_thresh": 0.05,
                  "box_detections_per_img": 300, "preprocessing": "weights.transforms() (ToTensor, no resize; model resizes min 800 / max 1333)"},
        "owlv2": {"model": "google/owlv2-base-patch16-ensemble", "revision": "cfd3195ba4ea9592eec887ded089f4c08eff231d",
                  "licence": "apache-2.0", "score_thresh": 0.01, "pre_nms_topk": 300,
                  "preprocessing": "Owlv2Processor (pad to square bottom/right, resize 960x960); boxes decoded vs padded square side",
                  "queries": ["food", "a dish of food", "a plate of food", "a bowl of food", "fruit", "a vegetable",
                              "a packaged food product", "a jar", "a bottle", "a carton", "a can", "bread", "meat",
                              "a snack", "a dessert"]},
        "sam3": {"model": "facebook/sam3", "revision": "3c879f39826c281e95690f02c7821c4de09afae7",
                 "licence": "Meta SAM License (custom, 'other')", "status": "UNAVAILABLE",
                 "reason": "Hugging Face repo is gated (gated='manual': Meta approval form); no approved token in this environment. Not substituted."},
    },
    "multiview": {"pad": 0.15, "context": 0.5,
                  "layout": "full + top1 x {orig, padded, square, context} + top2 x {orig, padded, context}"},
    "padding_rule": "if a source yields fewer boxes than its arm needs, the remaining slots take the CONTROL crops in order (counted and reported)",
    "arms": {
        "E_FULL": "1 query: full image (frozen Adapter E single-view reference; NOT budget-matched)",
        "CONTROL": "full + 7 seeded random crops",
        "FRCNN": "full + top-7 Faster R-CNN boxes",
        "OWLV2": "full + top-7 OWLv2 boxes",
        "FRCNN_MV": "full + Faster R-CNN top-2 multi-view",
        "OWLV2_MV": "full + OWLv2 top-2 multi-view",
    },
    "matched_arms": ["CONTROL", "FRCNN", "OWLV2", "FRCNN_MV", "OWLV2_MV"],
    "v10_arms": ["FRCNN", "OWLV2", "FRCNN_MV", "OWLV2_MV"],
    "v10_primary_rule": "highest VAL Recall@5 among v10_arms; ties -> higher Recall@10, then arm order",
    "union_rule": "per view top-20 labels; candidate score = max over views listing it; order by score desc, then view_support desc, then label; keep 20",
    "decision_rule": "frozen V5 decide() with frozen Adapter E thresholds on the union score vector (labels in no view top-20 = -9)",
    "scene_region_match": {"iou": 0.5, "box": "view region_box (the proposal / crop the view derives from)",
                           "region_candidates": "union over the arm's views whose region_box IoU >= 0.5 with the GT box"},
    "region_recall_ious": [0.25, 0.5, 0.75],
    "spurious_iou": 0.25,
    "bootstrap": {"n": 10000, "seed": 1234, "ci": 0.95, "unit": "query id (VAL); scene id (41-scene)"},
    "gates": {
        "A_recall5_min_gain_pp": 5.0, "A_baseline": "stronger of E_FULL and CONTROL (VAL Recall@5)",
        "B_min_recovered_top20": 8, "B_population": "12 historical Adapter E VAL errors with correct label outside top-5",
        "C_max_top1_drop_pp": 2.0, "C_reference": "E_FULL VAL raw top-1",
        "D_unknown_rejection_min": "E_FULL VAL unknown rejection", "D_max_false_confirmation_rise_pp": 0.5,
        "E_photo_scene_complete_min": "6/22 (verified Adapter E GT-region reference)",
    },
    "coverage": {"R_min": 15, "S_min": 10, "D_min": 8, "diversity_cluster_cos": 0.70, "near_dup_cos": 0.85,
                 "change_note": "diversity_cluster_cos set 0.80 -> 0.70 before freeze: at 0.80 cluster count equalled reference count for all 27 labels (library already dedups at 0.90), so it exposed no diversity limit"},
    "test_split": "NOT OPENED",
}


def protocol_sha() -> str:
    return hashlib.sha256(json.dumps(PROTOCOL, sort_keys=True).encode()).hexdigest()


def file_sha(p: Path | str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------- box geometry
def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def clip(b, W, H) -> list[float]:
    return [max(0.0, min(float(W), b[0])), max(0.0, min(float(H), b[1])),
            max(0.0, min(float(W), b[2])), max(0.0, min(float(H), b[3]))]


def nms(boxes: list, scores: list, thr: float) -> list[int]:
    order = sorted(range(len(boxes)), key=lambda i: -scores[i])
    keep = []
    for i in order:
        if all(iou(boxes[i], boxes[k]) < thr for k in keep):
            keep.append(i)
    return keep


def expand(b, r: float, W, H) -> list[float]:
    w, h = b[2] - b[0], b[3] - b[1]
    return clip([b[0] - r * w, b[1] - r * h, b[2] + r * w, b[3] + r * h], W, H)


def square(b, W, H) -> list[float]:
    s = min(max(b[2] - b[0], b[3] - b[1]), W, H)
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    x0 = min(max(0.0, cx - s / 2), W - s)
    y0 = min(max(0.0, cy - s / 2), H - s)
    return [x0, y0, x0 + s, y0 + s]


def control_crops(W: int, H: int, key: str, n: int) -> list[list[float]]:
    c = PROTOCOL["control"]
    rng = random.Random(f"{PROTOCOL['seed']}:{key}")
    out = []
    for _ in range(n):
        s = rng.uniform(*c["side_fraction"])
        a = rng.uniform(*c["aspect"])
        w = min(float(W), s * W * a ** 0.5)
        h = min(float(H), s * H / a ** 0.5)
        x0 = rng.uniform(0, W - w)
        y0 = rng.uniform(0, H - h)
        out.append([x0, y0, x0 + w, y0 + h])
    return out


def _postprocess(boxes, scores, labels, W, H) -> list[dict]:
    p = PROTOCOL["proposals"]
    boxes = [clip(b, W, H) for b in boxes]
    ok = [i for i, b in enumerate(boxes) if b[2] - b[0] >= p["min_side_px"] and b[3] - b[1] >= p["min_side_px"]]
    boxes, scores, labels = [boxes[i] for i in ok], [scores[i] for i in ok], [labels[i] for i in ok]
    keep = nms(boxes, scores, p["nms_iou"])[: p["max_per_source"]]
    return [{"box": [round(v, 2) for v in boxes[i]], "confidence": round(float(scores[i]), 5),
             "class_hint": labels[i], "rank": r} for r, i in enumerate(keep)]


def duplicate_rate(boxes: list, scores: list, thr: float = 0.5) -> float:
    if not boxes:
        return 0.0
    order = sorted(range(len(boxes)), key=lambda i: -scores[i])
    dup = sum(1 for n, i in enumerate(order) if any(iou(boxes[i], boxes[j]) >= thr for j in order[:n]))
    return dup / len(boxes)


# ------------------------------------------------------------- proposal sources
class FasterRCNNSource:
    name = "frcnn"

    def __init__(self, dev: torch.device):
        import torchvision
        from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights, fasterrcnn_resnet50_fpn_v2
        cfg = PROTOCOL["proposals"]["frcnn"]
        self.dev = dev
        self.w = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
        self.model = fasterrcnn_resnet50_fpn_v2(weights=self.w, box_score_thresh=cfg["box_score_thresh"],
                                                box_detections_per_img=cfg["box_detections_per_img"]).eval().to(dev)
        self.tf = self.w.transforms()
        self.cats = self.w.meta["categories"]
        ck = Path(torch.hub.get_dir()) / "checkpoints" / os.path.basename(self.w.url)
        self.info = {"model": "Faster R-CNN ResNet-50 FPN v2", "source_repository": "pytorch/vision",
                     "revision": f"torchvision {torchvision.__version__}", "checkpoint": self.w.url,
                     "checkpoint_sha256": file_sha(ck) if ck.is_file() else None,
                     "licence": "BSD-3-Clause (torchvision code); weights trained on COCO 2017",
                     "licence_url": "https://github.com/pytorch/vision/blob/main/LICENSE",
                     "weights_available": True, "environment_compatible": True, **cfg}
        self.raw_dup = []

    @torch.no_grad()
    def __call__(self, img: Image.Image) -> list[dict]:
        W, H = img.size
        out = self.model([self.tf(img).to(self.dev)])[0]
        b = out["boxes"].cpu().tolist()
        s = out["scores"].cpu().tolist()
        lab = [self.cats[i] for i in out["labels"].cpu().tolist()]
        self.raw_dup.append(duplicate_rate(b[:300], s[:300]))
        return _postprocess(b, s, lab, W, H)


class OWLv2Source:
    name = "owlv2"

    def __init__(self, dev: torch.device):
        import transformers
        from huggingface_hub import hf_hub_download
        from transformers import Owlv2ForObjectDetection, Owlv2Processor
        cfg = PROTOCOL["proposals"]["owlv2"]
        self.dev = dev
        self.proc = Owlv2Processor.from_pretrained(cfg["model"], revision=cfg["revision"])
        self.model = Owlv2ForObjectDetection.from_pretrained(cfg["model"], revision=cfg["revision"]).eval().to(dev)
        wpath = hf_hub_download(cfg["model"], "model.safetensors", revision=cfg["revision"])
        self.text = self.proc(text=[cfg["queries"]], return_tensors="pt")
        self.info = {"model": "OWLv2 base patch16 ensemble", "source_repository": f"huggingface.co/{cfg['model']}",
                     "revision": cfg["revision"], "checkpoint": "model.safetensors",
                     "checkpoint_sha256": file_sha(wpath), "licence": cfg["licence"],
                     "licence_url": "https://www.apache.org/licenses/LICENSE-2.0",
                     "transformers_version": transformers.__version__,
                     "weights_available": True, "environment_compatible": True, **cfg}
        self.raw_dup = []

    @torch.no_grad()
    def __call__(self, img: Image.Image) -> list[dict]:
        cfg = PROTOCOL["proposals"]["owlv2"]
        W, H = img.size
        side = float(max(W, H))
        pix = self.proc(images=img, return_tensors="pt")["pixel_values"].to(self.dev)
        out = self.model(input_ids=self.text["input_ids"].to(self.dev),
                         attention_mask=self.text["attention_mask"].to(self.dev), pixel_values=pix)
        prob = torch.sigmoid(out.logits[0]).max(-1)
        sc, qi = prob.values.cpu(), prob.indices.cpu()
        cxcywh = out.pred_boxes[0].cpu()
        top = torch.argsort(sc, descending=True)[: cfg["pre_nms_topk"]]
        top = [int(i) for i in top if float(sc[i]) >= cfg["score_thresh"]]
        boxes, scores, labs = [], [], []
        for i in top:
            cx, cy, w, h = (float(v) * side for v in cxcywh[i])
            boxes.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
            scores.append(float(sc[i]))
            labs.append(cfg["queries"][int(qi[i])])
        self.raw_dup.append(duplicate_rate([clip(b, W, H) for b in boxes], scores))
        return _postprocess(boxes, scores, labs, W, H)


def sam3_registry() -> dict:
    s = PROTOCOL["proposals"]["sam3"]
    return {"model": "SAM 3", "source_repository": f"huggingface.co/{s['model']}", "revision": s["revision"],
            "checkpoint": "sam3.pt / model.safetensors", "checkpoint_sha256": None, "licence": s["licence"],
            "licence_url": f"https://huggingface.co/{s['model']}/blob/main/LICENSE", "weights_available": False,
            "environment_compatible": None, "evaluated": False, "status": s["status"], "reason": s["reason"]}


# ------------------------------------------------------------- views + matched-budget arms
def build_views(W: int, H: int, key: str, props: dict[str, list[dict]]) -> tuple[list[dict], dict[str, list[int]], dict]:
    """Unique views (crop boxes) and, per arm, the list of N view indices. Returns (views, arms, padding counts)."""
    N, mv = PROTOCOL["N"], PROTOCOL["multiview"]
    views: list[dict] = []
    index: dict[tuple, int] = {}

    def add(crop, region, kind, source, rank, conf) -> int:
        k = (tuple(round(v, 1) for v in crop), kind, source)
        if k not in index:
            index[k] = len(views)
            views.append({"view_id": len(views), "crop_box": [round(v, 2) for v in crop],
                          "region_box": [round(v, 2) for v in region], "kind": kind, "source": source,
                          "proposal_rank": rank, "confidence": conf})
        return index[k]

    full = [0.0, 0.0, float(W), float(H)]
    f = add(full, full, "full", "full_image", 0, 1.0)
    ctrl = [add(b, b, "control_crop", "control", r, None) for r, b in enumerate(control_crops(W, H, key, N - 1))]
    arms = {"E_FULL": [f], "CONTROL": [f] + ctrl}
    padding = {}
    for src in ("frcnn", "owlv2"):
        P = props.get(src) or []
        reg = [add(p["box"], p["box"], "orig", src, p["rank"], p["confidence"]) for p in P[: N - 1]]
        padding[src.upper()] = N - 1 - len(reg)
        arms[src.upper()] = [f] + reg + ctrl[: N - 1 - len(reg)]
        m = []
        for r, kinds in ((0, ("orig", "padded", "square", "context")), (1, ("orig", "padded", "context"))):
            if r >= len(P):
                continue
            b = P[r]["box"]
            for kd in kinds:
                crop = {"orig": b, "padded": expand(b, mv["pad"], W, H), "square": square(b, W, H),
                        "context": expand(b, mv["context"], W, H)}[kd]
                m.append(add(crop, b, kd, src, P[r]["rank"], P[r]["confidence"]))
        padding[src.upper() + "_MV"] = N - 1 - len(m)
        arms[src.upper() + "_MV"] = [f] + m + ctrl[: N - 1 - len(m)]
    for a in PROTOCOL["matched_arms"]:
        assert len(arms[a]) == N, (a, len(arms[a]))
    return views, arms, padding


def crop(img: Image.Image, box) -> Image.Image:
    return img.crop(tuple(int(round(v)) for v in box))


def timed(fn, *a):
    t = time.perf_counter()
    r = fn(*a)
    return r, time.perf_counter() - t

# ===================== V10-VR Milestones 2-3: VAL region proposals + matched-budget view embeddings
# Appended to the unchanged V5 shared code (load_photo_set / embed_batch / frozen DINOv2) and kiq_v10_regions.py.
# Opens VAL only. TEST is never loaded. No labels are used here; scoring happens locally on frozen Adapter E.
import platform


def run_v10_val_embed() -> None:
    t_start = time.time()
    val, root = load_photo_set({"val"})
    assert all(i["split"] == "val" for i in val)
    status(f"protocol {PROTOCOL['protocol_id']} sha={protocol_sha()} N={PROTOCOL['N']}")
    registry = {}
    sources = {}
    for cls in (FasterRCNNSource, OWLv2Source):
        try:
            src = cls(DEV)
            sources[src.name] = src
            registry[src.name] = {**src.info, "evaluated": True, "reason": None}
        except Exception as e:  # fail loudly for this model; no substitution
            registry[cls.name] = {"model": cls.name, "weights_available": False, "evaluated": False,
                                  "status": "BLOCKED", "reason": f"{type(e).__name__}: {e}"}
            status(f"BLOCKED {cls.name}: {type(e).__name__}: {e}")
    registry["sam3"] = sam3_registry()
    if not sources:
        write_json("v10_region_model_registry.json", registry)
        raise SystemExit("IMPLEMENTATION_BLOCKED: no proposal source could be loaded")
    if DEV.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    per_image, emb_rows, runtime = [], [], {k: [] for k in sources}
    t_embed = 0.0
    for n, it in enumerate(val):
        img = Image.open(root / it["file"]).convert("RGB")
        W, H = img.size
        props = {}
        for k, src in sources.items():
            props[k], dt = timed(src, img)
            runtime[k].append(dt)
        views, arms, padding = build_views(W, H, it["sha256"], props)
        t0 = time.perf_counter()
        E = embed_batch([crop(img, v["crop_box"]) for v in views])
        t_embed += time.perf_counter() - t0
        base = len(emb_rows)
        emb_rows.extend(E)
        per_image.append({"sha256": it["sha256"], "openverse_id": it["openverse_id"], "W": W, "H": H,
                          "emb_offset": base, "views": views, "arms": arms, "padding": padding, "proposals": props})
        if n % 25 == 0:
            status(f"val {n}/{len(val)} views={len(views)}")
    X = np.stack(emb_rows).astype(np.float32)
    np.save(WORK / "v10_val_view_dinov2.npy", X)
    emb_sha = sha_bytes((WORK / "v10_val_view_dinov2.npy").read_bytes())
    for k, src in sources.items():
        rt = np.array(runtime[k])
        registry[k]["runtime_s_per_image"] = {"mean": round(float(rt.mean()), 4), "median": round(float(np.median(rt)), 4)}
        registry[k]["raw_duplicate_rate_iou0.5_mean"] = round(float(np.mean(src.raw_dup)), 4)
    env = {"python": platform.python_version(), "torch": torch.__version__,
           "torchvision": __import__("torchvision").__version__,
           "transformers": __import__("transformers").__version__,
           "cuda": torch.version.cuda, "device": str(DEV),
           "gpu": torch.cuda.get_device_name(0) if DEV.type == "cuda" else None,
           "peak_gpu_mem_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1) if DEV.type == "cuda" else None,
           "dinov2_embed_s_total": round(t_embed, 2), "wall_s": round(time.time() - t_start, 1)}
    reg_sha = write_json("v10_region_model_registry.json", registry)
    out_sha = write_json("v10_val_views.json", {
        "protocol": PROTOCOL, "protocol_sha256": protocol_sha(), "photo_manifest_sha256": PHOTO_MANIFEST_SHA,
        "split": "val", "test_opened": False, "backbone": "DINOv2:dinov2_vitb14 (frozen, x_norm_clstoken, L2)",
        "embedding_file": "v10_val_view_dinov2.npy", "embedding_sha256": emb_sha, "n_images": len(val),
        "n_views": int(X.shape[0]), "environment": env, "registry_sha256": reg_sha, "images": per_image})
    status(f"DONE v10_val_embed images={len(val)} views={X.shape[0]} emb_sha={emb_sha} views_sha={out_sha}")


if __name__ == "__main__":
    status(f"START mode={MODE} device={DEV}")
    run_v10_val_embed()
