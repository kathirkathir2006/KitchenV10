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

MODE = "test_de"
SEED = 1234
PHOTO_MANIFEST_SHA = "89ea9b20effdf40983ce670722f43fce42e44bd4378cc5810853dbba94c656e0"
LIBRARY_MANIFEST_SHA = "7e95ee5636af0a14b2839b1a242699bc02635b5d0956c6d6e89a5b9019fc0a02"
ADAPTER_CONFIG_SHA = "2d1121dd428e9e8fcb06c4438988ea86cb1b8f0464b38896d2b082c18c3e561c"
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



# =====================================================================================================
# KitchenIQ V5 Adapter D / E. make_stage_kernels.py appends this file to the unchanged shared code of
# kiq_v5_stage.py (everything above its __main__ block), so photo-set loading, frozen DINOv2 embedding,
# label_scores, decide, objective, calibrate, detail_metrics, earliest_divergence and AdapterC are the
# exact historical functions.
#
# adapter_d : TRAIN fit (SupCon + train-only hard negatives), VAL selection -> frozen Adapter D
# adapter_e : Adapter D architecture + episodic pseudo-unknown open-set loss, VAL safety-gated selection
# test_de   : frozen C/D/E configs only -> one-shot TEST evaluation of baseline, C, D, E
# selftest  : unit tests only (no data); runs locally
# =====================================================================================================
import ast
import inspect
import shutil
import sys

C_CONFIG_SHA = "2d1121dd428e9e8fcb06c4438988ea86cb1b8f0464b38896d2b082c18c3e561c"
D_CONFIG_SHA = "987c29889356db4ff8055ebe8f39e95a4be9a0bd778e86e3463e8de62a0e4951"
E_CONFIG_SHA = "ab26699426b4eb71b52166c99b38c0d2376098bde8a43dcb5b45d075e49020cc"
SCORER_SRC_SHA = "ec2dd8a44f48246bd810d619b3495acb9a4a4ecc40994d92809ce20767e9a270"
EXPECTED_LIBRARY_SHA = "7e95ee5636af0a14b2839b1a242699bc02635b5d0956c6d6e89a5b9019fc0a02"
EXPECTED_PHOTO_SHA = "89ea9b20effdf40983ce670722f43fce42e44bd4378cc5810853dbba94c656e0"
BACKBONE = "dinov2_vitb14"
SCORER_FUNCS = ("label_scores", "decide", "objective", "calibrate", "detail_metrics", "earliest_divergence")

VIEWS = 8
BATCH_IMAGES = 32
LABELS_PER_BATCH = 8
HARD_K = 5
EVAL_EVERY = 5
T0 = time.time()

D_GRID = [{"hidden": 512, "dim": d, "dropout": p, "lr": lr, "wd": 1e-4, "epochs": 60, "temperature": 0.07,
           "margin": 0.20, "lambda_hard": 0.5, "lambda_open": 0.0, "open_margin": 0.10}
          for d in (128, 256) for p in (0.1, 0.3) for lr in (1e-3, 3e-4)]
E_LAMBDA_OPEN = (0.10, 0.25, 0.50)


def gpu_name() -> str:
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"


def live(adapter: str, gi: int, ep: int, loss: float, vobj) -> None:
    status(f"LIVE adapter={adapter} cfg={gi} epoch={ep} loss={loss:.4f} val_objective={vobj} "
           f"device={DEV}/{gpu_name()} elapsed_s={int(time.time() - T0)}")


# ------------------------------------------------------------- adapters D / E (retrieval embedding only)
class AdapterD(nn.Module):
    """LayerNorm->Linear->GELU->Dropout->Linear plus linear skip; L2-normalised embedding. No class vectors."""

    def __init__(self, hidden: int, dim: int, dropout: float):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, hidden), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(hidden, dim))
        self.skip = nn.Linear(768, dim, bias=False)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj(x) + self.skip(x), dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embed(x)


class AdapterE(AdapterD):
    """Identical architecture to Adapter D; differs only in the training objective (open-set episodes)."""


# ------------------------------------------------------------- losses
def supcon_masks(y: torch.Tensor, views: int) -> tuple[torch.Tensor, torch.Tensor]:
    lab = y.repeat_interleave(views)
    n = lab.numel()
    self_mask = torch.eye(n, dtype=torch.bool, device=y.device)
    same = lab[:, None] == lab[None, :]
    return same & ~self_mask, ~same


def supcon_loss(Z: torch.Tensor, y: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """Z [B, V, d]; positives = every other view with the same label (same image or same-label image)."""
    B, V, _ = Z.shape
    z = F.normalize(Z.reshape(B * V, -1), dim=-1)
    pos, _ = supcon_masks(y, V)
    self_mask = torch.eye(B * V, dtype=torch.bool, device=z.device)
    logits = (z @ z.T / temperature).masked_fill(self_mask, float("-inf"))
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    npos = pos.sum(1)
    valid = npos > 0
    per_anchor = -log_prob.masked_fill(~pos, 0.0).sum(1)[valid] / npos[valid]
    return per_anchor.mean()


def hard_negative_loss(za: torch.Tensor, zp: torch.Tensor, zn: torch.Tensor, margin: float = 0.20) -> torch.Tensor:
    """za, zp [B, d]; zn [B, K, d]; mean relu(margin + s(a,n) - s(a,p)) over all available pairs."""
    s_ap = (za * zp).sum(-1)
    s_an = torch.einsum("bd,bkd->bk", za, zn)
    return F.relu(margin + s_an - s_ap[:, None]).mean()


def episode_prototypes(Zall: torch.Tensor, y: torch.Tensor, u: int, n_classes: int) -> tuple[torch.Tensor, list[int]]:
    """Prototypes of every TRAIN label except the pseudo-unknown u."""
    cls = [c for c in range(n_classes) if c != u and bool((y == c).any())]
    protos = torch.stack([F.normalize(Zall[y == c].mean(0), dim=-1) for c in cls])
    return protos, cls


def open_set_loss(zu: torch.Tensor, protos: torch.Tensor, open_margin: float = 0.10) -> torch.Tensor:
    return F.relu((zu @ protos.T).max(1).values - open_margin).mean()


# ------------------------------------------------------------- integrity helpers
def scorer_source_sha() -> str:
    try:
        parts = [inspect.getsource(globals()[n]).strip() for n in SCORER_FUNCS]
    except (OSError, TypeError):
        tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
        src = Path(__file__).read_text(encoding="utf-8")
        seg = {n.name: ast.get_source_segment(src, n) for n in tree.body if isinstance(n, ast.FunctionDef)}
        parts = [seg[n].strip() for n in SCORER_FUNCS]
    return sha_bytes("\n\n".join(parts).encode("utf-8"))


def reference_label_scores(Q, R, rlab, labels):
    out = np.zeros((len(Q), len(labels)), np.float32)
    for j, c in enumerate(labels):
        Rc = R[rlab == c]
        p = Rc.mean(0)
        p = p / (np.linalg.norm(p) + 1e-8)
        cos = Q @ Rc.T
        k = min(3, Rc.shape[0])
        out[:, j] = 0.5 * (Q @ p) + 0.5 * np.sort(cos, 1)[:, -k:].mean(1)
    return out


def run_unit_tests() -> dict:
    g = torch.Generator().manual_seed(7)
    res = {}
    x = torch.randn(10, 768, generator=g)
    d, e = AdapterD(512, 128, 0.1), AdapterE(512, 256, 0.3)
    res["01_adapter_d_shape"] = tuple(d(x).shape) == (10, 128)
    res["02_adapter_e_shape"] = tuple(e(x).shape) == (10, 256)
    res["01b_no_class_vectors"] = not any(n.startswith(("cls", "scale")) for n, _ in d.named_parameters())
    y = torch.tensor([0, 0, 1, 1, 2, 2])
    Xv = torch.randn(6, 4, 768, generator=g)
    Z = d.embed(Xv.reshape(-1, 768)).reshape(6, 4, -1)
    l_sup = supcon_loss(Z, y, 0.07)
    hn = torch.randn(6, HARD_K, 768, generator=g)
    l_hard = hard_negative_loss(Z[:, 0], Z[:, 1], d.embed(hn.reshape(-1, 768)).reshape(6, HARD_K, -1), 0.2)
    protos, cls = episode_prototypes(Z[:, 0], y, 1, 3)
    l_open = open_set_loss(Z[y == 1][:, 0], protos, 0.10)
    res["03_supcon_finite"] = bool(torch.isfinite(l_sup))
    res["04_hard_finite"] = bool(torch.isfinite(l_hard))
    res["05_open_finite"] = bool(torch.isfinite(l_open))
    d.zero_grad()
    (l_sup + 0.5 * l_hard + 0.25 * l_open).backward()
    res["06_grad_nonzero"] = all(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in d.parameters())
    pos, neg = supcon_masks(torch.tensor([0, 0, 1]), 2)  # rows: img0v0 img0v1 img1v0 img1v1 img2v0 img2v1
    res["07_same_image_views_positive"] = bool(pos[0, 1]) and bool(pos[4, 5]) and not bool(pos[0, 0])
    res["08_same_label_diff_image_positive"] = bool(pos[0, 2]) and bool(pos[1, 3])
    res["09_diff_label_negative"] = bool(neg[0, 4]) and not bool(pos[0, 4]) and not bool(neg[0, 2])
    res["10_pseudo_unknown_excluded"] = 1 not in cls and cls == [0, 2] and protos.shape[0] == 2
    rng = np.random.RandomState(3)
    Q = norm(rng.randn(12, 32).astype("float32"))
    R = norm(rng.randn(40, 32).astype("float32"))
    rl = np.array([f"l{k % 5}" for k in range(40)])
    labs = [f"l{k}" for k in range(5)]
    res["17_scorer_matches_reference_impl"] = bool(np.allclose(label_scores(Q, R, rl, labs),
                                                              reference_label_scores(Q, R, rl, labs), atol=1e-5))
    src_sha = scorer_source_sha()
    res["17b_scorer_source_sha"] = src_sha
    res["17c_scorer_source_unchanged"] = SCORER_SRC_SHA.startswith("__") or src_sha == SCORER_SRC_SHA
    res["16_backbone_is_dinov2_vitb14"] = f'"{BACKBONE}"' in inspect.getsource(dino)
    res["14_library_sha_constant"] = LIBRARY_MANIFEST_SHA in (EXPECTED_LIBRARY_SHA, "__LIBRARY_SHA__")
    res["15_photo_sha_constant"] = PHOTO_MANIFEST_SHA == EXPECTED_PHOTO_SHA
    res["all_pass"] = all(v for k, v in res.items() if k != "17b_scorer_source_sha")
    return res


# ------------------------------------------------------------- data (TRAIN + VAL only)
def find_dir_in_input(rel: str) -> Path | None:
    for p in INPUT.rglob(rel.split("/")[-1]):
        if p.is_dir() and str(p).replace("\\", "/").endswith(rel):
            return p
    return None


def prepare_train_val() -> dict:
    lib, _ = load_library()
    items, root = load_photo_set({"train", "val"})
    splits = {i["split"] for i in items}
    assert splits <= {"train", "val"}, "test image loaded in training mode"
    train = [i for i in items if i["split"] == "train"]
    val = [i for i in items if i["split"] == "val"]
    assert all(i["role"] == "known" for i in train), "open-set identities present in train"
    tr_sha = {i["sha256"] for i in train}
    assert {r["sha256"] for r in lib["items"]} <= tr_sha, "library contains non-train images"
    classes = sorted({i["label"] for i in train})
    cix = {c: k for k, c in enumerate(classes)}
    fam = {i["label"]: i["family"] for i in items}
    state = {i["label"]: i["state"] for i in items}
    cache = WORK / "emb_cache.npz"
    if cache.exists():
        z = np.load(cache)
        Xtr, Xva = z["Xtr"], z["Xva"]
    else:
        Xtr = embed_items(train, root, views=VIEWS)
        Xva = embed_items(val, root, views=1)[:, 0]
        np.savez(cache, Xtr=Xtr, Xva=Xva)
    assert Xtr.shape == (len(train), VIEWS, 768)
    tr_pos = {i["sha256"]: k for k, i in enumerate(train)}
    ref_idx = [tr_pos[r["sha256"]] for r in lib["items"]]
    status(f"data train={Xtr.shape} val={Xva.shape} refs={len(ref_idx)} classes={len(classes)} splits={sorted(splits)}")
    return {"lib": lib, "train": train, "val": val, "classes": classes, "fam": fam, "state": state,
            "known": set(classes), "Xtr": Xtr, "Xva": Xva, "ytr": np.array([cix[i["label"]] for i in train]),
            "Rf": Xtr[ref_idx, 0], "rlab": np.array([r["label"] for r in lib["items"]]), "gval": gt_rows(val)}


def mine_hard_negatives(D: dict, name: str) -> tuple[np.ndarray, str, dict]:
    train, E0 = D["train"], D["Xtr"][:, 0]
    labels = np.array([i["label"] for i in train])
    shas = np.array([i["sha256"] for i in train])
    S = E0 @ E0.T
    idx = np.zeros((len(train), HARD_K), np.int64)
    pairs = []
    for a in range(len(train)):
        s = S[a].copy()
        s[labels == labels[a]] = -9.0
        s[shas == shas[a]] = -9.0
        order = np.argsort(-s)[:HARD_K]
        idx[a] = order
        for n in order:
            pairs.append({"anchor_sha256": shas[a], "anchor_label": labels[a], "negative_sha256": shas[n],
                          "negative_label": labels[n], "cosine_similarity": round(float(S[a, n]), 6)})
    cos = np.array([p["cosine_similarity"] for p in pairs])
    conf = Counter(tuple(sorted((p["anchor_label"], p["negative_label"]))) for p in pairs)
    tr_sha, va_sha = set(shas.tolist()), {i["sha256"] for i in D["val"]}
    used = {p["anchor_sha256"] for p in pairs} | {p["negative_sha256"] for p in pairs}
    assert used <= tr_sha and not (used & va_sha), "hard-negative manifest contains non-train images"
    assert all(p["anchor_label"] != p["negative_label"] for p in pairs)
    summary = {"n_anchors": len(train), "k": HARD_K, "n_pairs": len(pairs),
               "cosine": {"mean": round(float(cos.mean()), 4), "min": round(float(cos.min()), 4),
                          "p50": round(float(np.median(cos)), 4), "p90": round(float(np.percentile(cos, 90)), 4),
                          "max": round(float(cos.max()), 4)},
               "top_confusions": [{"labels": list(k), "pairs": v} for k, v in conf.most_common(25)],
               "only_train_sha256": True, "val_overlap": 0, "test_loaded": False}
    sha = write_json(name, {"source_split": "train", "backbone": f"DINOv2:{BACKBONE}",
                            "photo_manifest_sha256": PHOTO_MANIFEST_SHA, "embedding": "frozen view-0 cls token",
                            "exclusions": ["same label", "same image (sha256)"], "summary": summary, "pairs": pairs})
    status(f"hard negatives n_pairs={len(pairs)} cos={summary['cosine']} sha={sha}")
    return idx, sha, summary


def val_rates(dec: list[dict], gt: list[dict], known: set[str]) -> dict:
    kn = [i for i, g in enumerate(gt) if g["label"] in known]
    return {"unknown_rate_all": round(np.mean([d["level"] == "unknown" for d in dec]), 4),
            "abstention_rate_known": round(np.mean([dec[i]["level"] != "identity" for i in kn]), 4),
            "family_backoff_rate_known": round(np.mean([dec[i]["level"] == "family" for i in kn]), 4)}


def evaluate_val(D: dict, X: np.ndarray, R: np.ndarray, thresholds: dict | None = None) -> dict:
    """VAL only. thresholds=None -> calibrate on VAL (selection); otherwise use the frozen thresholds."""
    sc = label_scores(X, R, D["rlab"], D["classes"])
    assert len(sc) == len(D["gval"])
    t = thresholds or calibrate(sc, D["gval"], D["classes"], D["fam"], D["known"])["thresholds"]
    dec = decide(sc, D["classes"], D["fam"], t["t_id"], t["m_id"], t["t_fam"])
    m = detail_metrics(sc, dec, D["gval"], D["classes"], D["fam"], D["state"], D["known"])
    m.update(val_rates(dec, D["gval"], D["known"]))
    return {"thresholds": t, "metrics": m}


def balanced_batches(y: np.ndarray, rng: np.random.RandomState) -> list[list[int]]:
    """Image-level, class-balanced batches over TRAIN indices: LABELS_PER_BATCH labels x (32/8) images."""
    labs = np.unique(y)
    pools = {int(c): rng.permutation(np.where(y == c)[0]).tolist() for c in labs}
    ptr = {c: 0 for c in pools}
    per = BATCH_IMAGES // LABELS_PER_BATCH
    out = []
    for _ in range(int(np.ceil(len(y) / BATCH_IMAGES))):
        b = []
        for c in rng.choice(labs, LABELS_PER_BATCH, replace=False):
            c = int(c)
            for _ in range(per):
                if ptr[c] >= len(pools[c]):
                    pools[c], ptr[c] = rng.permutation(pools[c]).tolist(), 0
                b.append(pools[c][ptr[c]])
                ptr[c] += 1
        out.append(b)
    return out


def train_config(adapter: str, gi: int, cfg: dict, D: dict, hn_idx: np.ndarray) -> dict:
    ckd = WORK / "checkpoints"
    ckd.mkdir(parents=True, exist_ok=True)
    done, last = ckd / f"cfg{gi}_result.json", ckd / f"cfg{gi}_last.pt"
    if done.exists():
        status(f"resume: cfg{gi} already complete")
        return json.loads(done.read_text())
    Xt = torch.from_numpy(D["Xtr"]).float().to(DEV)
    yt = torch.from_numpy(D["ytr"]).long().to(DEV)
    y = D["ytr"]
    N, V, _ = D["Xtr"].shape
    C = len(D["classes"])
    pools = {int(c): np.where(y == c)[0] for c in np.unique(y)}
    hn_t = torch.from_numpy(hn_idx).long().to(DEV)
    cls_model = AdapterE if adapter == "adapter_e" else AdapterD
    torch.manual_seed(cfg["seed"])
    model = cls_model(cfg["hidden"], cfg["dim"], cfg["dropout"]).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg["epochs"])
    start, history = 1, []
    if last.exists():
        blob = torch.load(last, map_location=DEV, weights_only=False)
        model.load_state_dict(blob["state_dict"])
        opt.load_state_dict(blob["optimizer"])
        sched.load_state_dict(blob["scheduler"])
        start, history = blob["epoch"] + 1, blob["history"]
        status(f"resume: cfg{gi} from epoch {blob['epoch']}")
    episode_order = np.random.RandomState(cfg["seed"]).permutation(C)
    for ep in range(start, cfg["epochs"] + 1):
        model.train()
        torch.manual_seed(cfg["seed"] * 1000 + ep)
        rng = np.random.RandomState(cfg["seed"] * 1000 + ep)
        batches = balanced_batches(y, rng)
        tot = {"loss": 0.0, "supcon": 0.0, "hard": 0.0, "open": 0.0}
        for bi, b in enumerate(batches):
            ba = np.array(b)
            bt = torch.from_numpy(ba).long().to(DEV)
            B = len(b)
            Z = model.embed(Xt[bt].reshape(-1, 768)).reshape(B, V, -1)
            l_sup = supcon_loss(Z, yt[bt], cfg["temperature"])
            pj = np.array([rng.choice(pools[int(y[i])]) for i in ba])
            pv = np.where(pj == ba, rng.randint(1, V, B), rng.randint(0, V, B))
            zp = model.embed(Xt[torch.from_numpy(pj).to(DEV), torch.from_numpy(pv).to(DEV)])
            nv = torch.from_numpy(rng.randint(0, V, (B, HARD_K))).long().to(DEV)
            zn = model.embed(Xt[hn_t[bt], nv].reshape(-1, 768)).reshape(B, HARD_K, -1)
            l_hard = hard_negative_loss(Z[:, 0], zp, zn, cfg["margin"])
            loss = l_sup + cfg["lambda_hard"] * l_hard
            l_open = torch.zeros((), device=DEV)
            if cfg["lambda_open"] > 0:
                u = int(episode_order[((ep - 1) * len(batches) + bi) % C])
                av = torch.from_numpy(rng.randint(0, V, N)).long().to(DEV)
                Zall = model.embed(Xt[torch.arange(N, device=DEV), av])
                protos, cls = episode_prototypes(Zall, yt, u, C)
                assert u not in cls
                l_open = open_set_loss(Zall[yt == u], protos, cfg["open_margin"])
                loss = loss + cfg["lambda_open"] * l_open
            opt.zero_grad()
            loss.backward()
            opt.step()
            for k, v in (("loss", loss), ("supcon", l_sup), ("hard", l_hard), ("open", l_open)):
                tot[k] += float(v.detach()) / len(batches)
        sched.step()
        vobj = None
        if ep % EVAL_EVERY == 0 or ep == cfg["epochs"]:
            ev = evaluate_val(D, adapter_embed(model, D["Xva"]), adapter_embed(model, D["Rf"]))
            vobj = ev["metrics"]["objective"]
            cp = ckd / f"cfg{gi}_ep{ep}.pt"
            torch.save({"state_dict": model.state_dict(), "cfg": cfg, "epoch": ep, "adapter": adapter}, cp)
            history.append({"cfg_index": gi, "epoch": ep, "losses": {k: round(v, 5) for k, v in tot.items()},
                            "checkpoint": cp.name, "checkpoint_sha256": sha_bytes(cp.read_bytes()), **ev})
            torch.save({"state_dict": model.state_dict(), "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                        "epoch": ep, "cfg": cfg, "seed": cfg["seed"], "history": history}, last)
        live(adapter, gi, ep, tot["loss"], vobj)
    r = {"cfg_index": gi, "cfg": cfg, "history": history}
    done.write_text(json.dumps(r))
    return r


def restore_checkpoints_from_input() -> None:
    ckd = WORK / "checkpoints"
    if ckd.exists() and any(ckd.iterdir()):
        return
    src = find_dir_in_input(f"v5_{MODE}/checkpoints")
    if src is not None:
        shutil.copytree(src, ckd, dirs_exist_ok=True)
        status(f"resume: restored checkpoints from {src}")


def frozen_c_val(D: dict) -> dict:
    cp = find_by_sha("adapter_c_frozen_config.json", C_CONFIG_SHA)
    cfg = json.loads(cp.read_bytes())
    assert cfg["classes"] == D["classes"]
    ck = cp.parent / cfg["adapter"]["checkpoint"]
    assert sha_bytes(ck.read_bytes()) == cfg["adapter"]["checkpoint_sha256"]
    a = cfg["adapter"]["cfg"]
    m = AdapterC(len(D["classes"]), a["hidden"], a["dim"], a["dropout"]).to(DEV)
    m.load_state_dict(torch.load(ck, map_location=DEV, weights_only=False)["state_dict"])
    return {"baseline": evaluate_val(D, D["Xva"], D["Rf"], cfg["baseline"]["thresholds"]),
            "adapter_c": evaluate_val(D, adapter_embed(m, D["Xva"]), adapter_embed(m, D["Rf"]), cfg["adapter"]["thresholds"])}


def data_checks(D: dict, hn_sha: str) -> dict:
    return {"11_no_test_loaded": {i["split"] for i in D["train"] + D["val"]} <= {"train", "val"},
            "12_13_hard_negatives_train_only": True, "hard_negative_manifest_sha256": hn_sha,
            "14_library_sha": LIBRARY_MANIFEST_SHA == EXPECTED_LIBRARY_SHA,
            "15_photo_sha": PHOTO_MANIFEST_SHA == EXPECTED_PHOTO_SHA,
            "18_thresholds_calibrated_on_val_only": True, "19_test_not_used_for_selection": True}


def freeze(adapter: str, winner: dict, D: dict, extra: dict) -> tuple[str, str]:
    src = WORK / "checkpoints" / winner["checkpoint"]
    assert sha_bytes(src.read_bytes()) == winner["checkpoint_sha256"]
    final = WORK / f"{adapter}_final.pt"
    shutil.copyfile(src, final)
    ck_sha = sha_bytes(final.read_bytes())
    config = {"config_id": f"kiq_v5_{adapter}_v1", "frozen": True, "selection_split": "val", "test_used": False,
              "photo_manifest_sha256": PHOTO_MANIFEST_SHA, "library_manifest_sha256": LIBRARY_MANIFEST_SHA,
              "backbone": f"DINOv2:{BACKBONE} (frozen)", "classes": D["classes"], "families": D["fam"],
              "states": D["state"], "scorer_source_sha256": scorer_source_sha(),
              "adapter": {"class": "AdapterE" if adapter == "adapter_e" else "AdapterD", "cfg": winner["cfg"],
                          "cfg_index": winner["cfg_index"], "epoch": winner["epoch"],
                          "thresholds": winner["thresholds"], "val_metrics": winner["metrics"],
                          "checkpoint": final.name, "checkpoint_sha256": ck_sha},
              "scorer": "0.5*cos(prototype)+0.5*mean(top3 cos); identity->family->unknown back-off (unchanged)",
              "train_views": VIEWS, "batch_images": BATCH_IMAGES, "labels_per_batch": LABELS_PER_BATCH,
              "hard_negatives_k": HARD_K, "seed_base": SEED, **extra}
    return write_json(f"{adapter}_frozen_config.json", config), ck_sha


def start_mode(adapter: str) -> dict:
    ut = run_unit_tests()
    write_json(f"{adapter}_unit_tests.json", ut)
    status(f"unit tests {ut}")
    if not ut["all_pass"]:
        raise SystemExit("unit tests failed â€” training not started")
    restore_checkpoints_from_input()
    return ut


# ------------------------------------------------------------- MODE adapter_d
def run_adapter_d() -> None:
    ut = start_mode("adapter_d")
    D = prepare_train_val()
    hn_idx, hn_sha, hn_sum = mine_hard_negatives(D, "adapter_d_train_hard_negatives.json")
    ref = frozen_c_val(D)
    status(f"val baseline={ref['baseline']['metrics']} adapter_c={ref['adapter_c']['metrics']}")
    results = [train_config("adapter_d", gi, {**cfg, "seed": SEED + 200 + gi}, D, hn_idx) for gi, cfg in enumerate(D_GRID)]
    cands = [{**h, "cfg": r["cfg"]} for r in results for h in r["history"]]
    winner = max(cands, key=lambda c: (c["metrics"]["objective"], -c["cfg_index"], -c["epoch"]))
    cfg_sha, ck_sha = freeze("adapter_d", winner, D, {
        "selection_rule": "max val objective (known_score + unknown_rejection - 2*false_confirmation) over all "
                          "(cfg, epoch) checkpoints; ties -> lower cfg index, earlier epoch",
        "hard_negative_manifest_sha256": hn_sha, "losses": "SupCon(T) + lambda_hard * hard-negative margin"})
    write_json("adapter_d_selection_report.json", {
        "test_used": False, "grid": D_GRID, "candidates": cands, "winner": {k: winner[k] for k in ("cfg_index", "epoch")},
        "val_baseline": ref["baseline"], "val_adapter_c": ref["adapter_c"], "val_adapter_d": winner["metrics"],
        "hard_negative_summary": hn_sum, "unit_tests": ut, "data_checks": data_checks(D, hn_sha),
        "frozen_config_sha256": cfg_sha, "checkpoint_sha256": ck_sha})
    status(f"DONE adapter_d winner=cfg{winner['cfg_index']}@ep{winner['epoch']} val={winner['metrics']} "
           f"config_sha={cfg_sha} ckpt_sha={ck_sha}")


# ------------------------------------------------------------- MODE adapter_e
def run_adapter_e() -> None:
    ut = start_mode("adapter_e")
    dp = find_by_sha("adapter_d_frozen_config.json", D_CONFIG_SHA)
    dcfg = json.loads(dp.read_bytes())
    assert dcfg["frozen"] and not dcfg["test_used"] and dcfg["library_manifest_sha256"] == LIBRARY_MANIFEST_SHA
    D = prepare_train_val()
    assert D["classes"] == dcfg["classes"]
    hn_idx, hn_sha, hn_sum = mine_hard_negatives(D, "adapter_e_train_hard_negatives.json")
    assert hn_sha == dcfg["hard_negative_manifest_sha256"], "hard negatives differ from Adapter D's"
    ref = frozen_c_val(D)
    d_ref = dcfg["adapter"]["val_metrics"]
    base = {k: v for k, v in dcfg["adapter"]["cfg"].items() if k != "seed"}
    grid = [{**base, "lambda_open": lo} for lo in E_LAMBDA_OPEN]
    results = [train_config("adapter_e", gi, {**cfg, "seed": SEED + 300 + gi}, D, hn_idx) for gi, cfg in enumerate(grid)]
    cands = []
    for r in results:
        for h in r["history"]:
            m = h["metrics"]
            gate = m["unknown_rejection"] >= d_ref["unknown_rejection"] and m["false_confirmation"] <= d_ref["false_confirmation"]
            cands.append({**h, "cfg": r["cfg"], "safety_gate": gate})
    passing = [c for c in cands if c["safety_gate"]]
    if passing:
        winner = max(passing, key=lambda c: (c["metrics"]["known_score"], c["metrics"]["objective"], -c["cfg_index"], -c["epoch"]))
        rule = "safety gate passed; max val known_score among gated candidates"
    else:
        winner = max(cands, key=lambda c: (c["metrics"]["objective"], -c["cfg_index"], -c["epoch"]))
        rule = "NO candidate passed the safety gate; fallback = max val objective (documented trade-off)"
    gate_info = {"reference": "frozen Adapter D val", "unknown_rejection_min": d_ref["unknown_rejection"],
                 "false_confirmation_max": d_ref["false_confirmation"], "n_candidates": len(cands),
                 "n_passing": len(passing), "safety_gate_passed": bool(passing), "rule_applied": rule}
    cfg_sha, ck_sha = freeze("adapter_e", winner, D, {
        "safety_gate": gate_info, "adapter_d_config_sha256": D_CONFIG_SHA, "hard_negative_manifest_sha256": hn_sha,
        "losses": "SupCon(T) + lambda_hard * hard-negative margin + lambda_open * episodic pseudo-unknown "
                  "relu(max_known_proto_cos - open_margin); whole TRAIN label held out per episode"})
    write_json("adapter_e_selection_report.json", {
        "test_used": False, "grid": grid, "candidates": cands, "winner": {k: winner[k] for k in ("cfg_index", "epoch")},
        "safety_gate": gate_info, "val_baseline": ref["baseline"], "val_adapter_c": ref["adapter_c"],
        "val_adapter_d": d_ref, "val_adapter_e": winner["metrics"], "unit_tests": ut,
        "data_checks": data_checks(D, hn_sha), "frozen_config_sha256": cfg_sha, "checkpoint_sha256": ck_sha})
    status(f"DONE adapter_e winner=cfg{winner['cfg_index']}@ep{winner['epoch']} gate={bool(passing)} "
           f"val={winner['metrics']} config_sha={cfg_sha} ckpt_sha={ck_sha}")


# ------------------------------------------------------------- MODE test_de (one-shot)
def load_frozen(name: str, sha: str):
    p = find_by_sha(name, sha)
    cfg = json.loads(p.read_bytes())
    assert cfg["frozen"] and not cfg["test_used"], f"{name} not frozen or test already used"
    assert cfg["library_manifest_sha256"] == LIBRARY_MANIFEST_SHA and cfg["photo_manifest_sha256"] == PHOTO_MANIFEST_SHA
    ck = p.parent / cfg["adapter"]["checkpoint"]
    assert sha_bytes(ck.read_bytes()) == cfg["adapter"]["checkpoint_sha256"], f"{name} checkpoint changed"
    return cfg, ck


def change(a_ok: bool, b_ok: bool) -> str:
    return "fixed" if b_ok and not a_ok else ("regressed" if a_ok and not b_ok else ("both_ok" if b_ok else "both_fail"))


def run_test_de() -> None:
    ut = run_unit_tests()
    assert ut["all_pass"]
    ccfg, cck = load_frozen("adapter_c_frozen_config.json", C_CONFIG_SHA)
    dcfg, dck = load_frozen("adapter_d_frozen_config.json", D_CONFIG_SHA)
    ecfg, eck = load_frozen("adapter_e_frozen_config.json", E_CONFIG_SHA)
    assert scorer_source_sha() == dcfg["scorer_source_sha256"] == ecfg["scorer_source_sha256"]
    lib, _ = load_library()
    train, root = load_photo_set({"train"})
    test, _ = load_photo_set({"test"})
    classes = ccfg["classes"]
    assert classes == dcfg["classes"] == ecfg["classes"]
    fam, state = dict(ccfg["families"]), dict(ccfg["states"])
    for i in test:
        fam.setdefault(i["label"], i["family"])
        state.setdefault(i["label"], i["state"])
    known = set(classes)
    tr_pos = {i["sha256"]: k for k, i in enumerate(train)}
    Rf = embed_items([train[tr_pos[r["sha256"]]] for r in lib["items"]], root, views=1)[:, 0]
    rlab = np.array([r["label"] for r in lib["items"]])
    Xte = embed_items(test, root, views=1)[:, 0]
    gte = gt_rows(test)
    a = ccfg["adapter"]["cfg"]
    mc = AdapterC(len(classes), a["hidden"], a["dim"], a["dropout"]).to(DEV)
    mc.load_state_dict(torch.load(cck, map_location=DEV, weights_only=False)["state_dict"])
    models = {"adapter_c": mc}
    for name, cfg, ck, kls in (("adapter_d", dcfg, dck, AdapterD), ("adapter_e", ecfg, eck, AdapterE)):
        c = cfg["adapter"]["cfg"]
        m = kls(c["hidden"], c["dim"], c["dropout"]).to(DEV)
        m.load_state_dict(torch.load(ck, map_location=DEV, weights_only=False)["state_dict"])
        models[name] = m
    thr = {"baseline": ccfg["baseline"]["thresholds"], "adapter_c": ccfg["adapter"]["thresholds"],
           "adapter_d": dcfg["adapter"]["thresholds"], "adapter_e": ecfg["adapter"]["thresholds"]}
    names = ["baseline", "adapter_c", "adapter_d", "adapter_e"]
    out, decs, scs = {}, {}, {}
    for n in names:
        X, R = (Xte, Rf) if n == "baseline" else (adapter_embed(models[n], Xte), adapter_embed(models[n], Rf))
        sc = label_scores(X, R, rlab, classes)
        t = thr[n]
        d = decide(sc, classes, fam, t["t_id"], t["m_id"], t["t_fam"])
        out[n] = {**detail_metrics(sc, d, gte, classes, fam, state, known), **val_rates(d, gte, known)}
        decs[n], scs[n] = d, sc
    per = []
    for k, g in enumerate(gte):
        row = {"openverse_id": g["openverse_id"], "gt": g["label"], "gt_family": g["family"], "role": g["role"]}
        for n in names:
            top1 = classes[int(np.argmax(scs[n][k]))]
            d = decs[n][k]
            row[n] = {"top1": top1, "decision": d["level"], "label": d["label"], "family": d["family"],
                      "score": round(d["score"], 4), "divergence": earliest_divergence(g, d, top1, fam, known)}
        per.append(row)
    ok = lambda r, n: r[n]["divergence"] == "OK"
    pairs = {"baseline->adapter_d": ("baseline", "adapter_d"), "baseline->adapter_e": ("baseline", "adapter_e"),
             "adapter_c->adapter_d": ("adapter_c", "adapter_d"), "adapter_c->adapter_e": ("adapter_c", "adapter_e"),
             "adapter_d->adapter_e": ("adapter_d", "adapter_e"), "baseline->adapter_c": ("baseline", "adapter_c")}
    changes = {p: dict(Counter(change(ok(r, a_), ok(r, b_)) for r in per)) for p, (a_, b_) in pairs.items()}
    for r in per:
        r["change"] = {p: change(ok(r, a_), ok(r, b_)) for p, (a_, b_) in pairs.items()}
    unk = [r for r in per if r["role"] == "open_set_unknown"]
    open_set = {n: {"n": len(unk), "rejected": sum(r[n]["decision"] != "identity" for r in unk),
                    "identity_false_accepts": sum(r[n]["decision"] == "identity" for r in unk),
                    "family_backoff": sum(r[n]["decision"] == "family" for r in unk)} for n in names}
    div = {n: dict(Counter(r[n]["divergence"] for r in per)) for n in names}
    common = {"split": "test", "evaluated_once": True, "photo_manifest_sha256": PHOTO_MANIFEST_SHA,
              "library_manifest_sha256": LIBRARY_MANIFEST_SHA, "adapter_c_config_sha256": C_CONFIG_SHA,
              "adapter_d_config_sha256": D_CONFIG_SHA, "adapter_e_config_sha256": E_CONFIG_SHA,
              "thresholds_recalibrated": False, "weights_updated": False, "n_items": len(per)}
    for tag, keep in (("adapter_d", ["baseline", "adapter_c", "adapter_d"]), ("adapter_e", ["baseline", "adapter_c", "adapter_d", "adapter_e"])):
        write_json(f"{tag}_test_results.json", {**common, "metrics": {n: out[n] for n in keep},
                                                "divergence_counts": {n: div[n] for n in keep},
                                                "open_set": {n: open_set[n] for n in keep},
                                                "change_counts": {p: v for p, v in changes.items() if p.endswith(tag) or p == "baseline->adapter_c"}})
        write_json(f"{tag}_test_per_item.json", [{**{k: r[k] for k in ("openverse_id", "gt", "gt_family", "role")},
                                                  **{n: r[n] for n in keep},
                                                  "change": {p: v for p, v in r["change"].items() if p.endswith(tag)}} for r in per])
    write_json("de_test_results.json", {**common, "metrics": out, "divergence_counts": div, "open_set": open_set,
                                        "change_counts": changes})
    status(f"DONE test_de {json.dumps({'metrics': out, 'open_set': open_set, 'changes': changes})}")


def run_selftest() -> None:
    ut = run_unit_tests()
    print(json.dumps(ut, indent=1))
    rng = np.random.RandomState(0)
    C, per = 27, 6
    centers = norm(rng.randn(C, 768).astype("float32"))
    ytr = np.repeat(np.arange(C), per)
    Xtr = norm(centers[ytr][:, None, :] + 0.6 * rng.randn(len(ytr), VIEWS, 768).astype("float32"))
    yva = np.arange(C).repeat(2)
    Xva = norm(centers[yva] + 0.6 * rng.randn(len(yva), 768).astype("float32"))
    classes = [f"c{k:02d}" for k in range(C)]
    fam = {c: f"f{k % 5}" for k, c in enumerate(classes)}
    fam["unk"] = "f9"
    state = {c: "raw" for c in list(fam)}
    gval = [{"label": classes[k] if j % 4 else "unk", "family": fam[classes[k]] if j % 4 else "f9", "state": "raw",
             "role": "known", "openverse_id": str(j)} for j, k in enumerate(yva)]
    D = {"train": [{"sha256": f"t{k}", "label": classes[y], "split": "train"} for k, y in enumerate(ytr)],
         "val": [{"sha256": f"v{k}", "split": "val"} for k in range(len(yva))], "classes": classes, "fam": fam,
         "state": state, "known": set(classes), "Xtr": Xtr, "Xva": Xva, "ytr": ytr, "Rf": Xtr[::2, 0],
         "rlab": np.array([classes[y] for y in ytr[::2]]), "gval": gval}
    global WORK
    WORK = WORK / "smoke"
    shutil.rmtree(WORK, ignore_errors=True)
    WORK.mkdir(parents=True)
    hn, _, _ = mine_hard_negatives(D, "smoke_hn.json")
    cfg = {**D_GRID[0], "epochs": 6, "lambda_open": 0.25, "seed": 1}
    first = train_config("adapter_e", 0, {**cfg, "epochs": 5}, D, hn)
    (WORK / "checkpoints" / "cfg0_result.json").unlink()
    resumed = train_config("adapter_e", 0, cfg, D, hn)
    smoke = {"trained_epochs": [h["epoch"] for h in resumed["history"]], "resume_ok": [h["epoch"] for h in resumed["history"]] == [5, 6],
             "val_objective": [h["metrics"]["objective"] for h in resumed["history"]],
             "losses": resumed["history"][-1]["losses"], "first_run_epochs": [h["epoch"] for h in first["history"]]}
    print(json.dumps(smoke, indent=1))
    sys.exit(0 if ut["all_pass"] and smoke["resume_ok"] else 1)


if __name__ == "__main__":
    status(f"START mode={MODE} device={DEV}/{gpu_name()}")
    {"reflib": run_reflib, "adapter": run_adapter, "test": run_test, "adapter_d": run_adapter_d,
     "adapter_e": run_adapter_e, "test_de": run_test_de, "selftest": run_selftest}[MODE]()
