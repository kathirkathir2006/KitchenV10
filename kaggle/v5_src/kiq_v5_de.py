
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
D_CONFIG_SHA = "__D_CONFIG_SHA__"
E_CONFIG_SHA = "__E_CONFIG_SHA__"
SCORER_SRC_SHA = "__SCORER_SRC_SHA__"
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
