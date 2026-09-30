"""KitchenIQ V10 — ONE controlled targeted identity-classifier remediation cycle.

Production-eligible remediation manifests ONLY (no experiment-only mix).
DINOv2 ViT-B/14 frozen (mode A). Detector unchanged. Holdout untouched.
41-scene acceptance is EVAL-ONLY and is NOT used here.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

SEED = 42
EPOCHS = 6
BATCH = 16
LR = 8e-4
WD = 1e-4
IMG_SIZE = 224
DOWNLOAD_WORKERS = 12

WORK = Path("/kaggle/working/v10-targeted-clf")
IMG_DIR = WORK / "images"
WEIGHTS = WORK / "weights"
REPORTS = WORK / "reports"
CKPT = WORK / "checkpoints"
STATUS = WORK / "LIVE_STATUS.txt"

VISUAL_CLASS_LABELS = (
    "raw_ingredient",
    "packaged_food",
    "prepared_meal",
    "recipe_document",
    "non_food",
    "mixed",
)
FOOD_STATE_LABELS = ("raw", "prepared", "packaged", "processed", "unknown")
KIND_LABELS = ("ingredient", "meal", "non_food", "document", "packaged")

PREPARED = {
    "fried_rice",
    "biryani",
    "omelette",
    "lasagna",
    "salad",
    "pizza",
    "pad_thai",
    "bibimbap",
    "spring_rolls",
    "guacamole",
    "hummus",
    "donuts",
}
PACKAGED = {"packaged_cheese", "cheese", "yogurt", "cream", "tomato_paste", "beer", "milk"}
CRITICAL = {
    "tomato",
    "garlic",
    "ginger",
    "rice",
    "fried_rice",
    "biryani",
    "yogurt",
    "cheese",
    "packaged_cheese",
    "tomato_paste",
    "cream",
    "non_food",
    "salad",
    "omelette",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def status(msg: str) -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    line = f"[{now()}] {msg}"
    print(line, flush=True)
    with STATUS.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def find_input(*names: str) -> Path | None:
    root = Path("/kaggle/input")
    if not root.is_dir():
        return None
    for n in names:
        p = root / n
        if p.exists():
            return p
    for p in root.iterdir():
        for n in names:
            if n.lower() in p.name.lower():
                return p
    return None


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def heads_for(label: str, *, hardneg: bool) -> dict[str, str]:
    lab = (label or "food").strip().lower().replace(" ", "_").replace("-", "_")
    if hardneg or lab in {"non_food", "empty_plate", "hand", "human_hand"}:
        return {
            "fine_label": "non_food",
            "visual_class": "non_food",
            "food_state": "unknown",
            "kind": "non_food",
        }
    if lab in PREPARED:
        return {
            "fine_label": lab,
            "visual_class": "prepared_meal",
            "food_state": "prepared",
            "kind": "meal",
        }
    if lab in PACKAGED:
        return {
            "fine_label": lab,
            "visual_class": "packaged_food",
            "food_state": "packaged",
            "kind": "packaged",
        }
    return {
        "fine_label": lab,
        "visual_class": "raw_ingredient",
        "food_state": "raw",
        "kind": "ingredient",
    }


def download_one(url: str, dest: Path, timeout: int = 45) -> bool:
    if dest.is_file() and dest.stat().st_size > 1000:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        req = Request(url, headers={"User-Agent": "KitchenIQ-V10-Targeted/1.0"})
        with urlopen(req, timeout=timeout) as resp, tmp.open("wb") as out:
            out.write(resp.read())
        tmp.replace(dest)
        return dest.stat().st_size > 1000
    except (HTTPError, URLError, TimeoutError, OSError) as e:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        status(f"DL_FAIL {url[:80]} :: {e}")
        return False


def resolve_image_url(rec: dict[str, Any]) -> str | None:
    if rec.get("image_url"):
        return str(rec["image_url"])
    # Open Images fallback by image_id
    iid = rec.get("image_id")
    if iid and isinstance(iid, str) and len(iid) >= 8 and "/" not in iid:
        # try validation then train buckets commonly used previously
        return f"https://open-images-dataset.s3.amazonaws.com/validation/{iid}.jpg"
    return None


def build_records() -> list[dict[str, Any]]:
    man_root = find_input("kiq-v10-targeted-manifests") or Path("/kaggle/input")
    prod_paths = list(Path("/kaggle/input").rglob("production_eligible_manifest.jsonl"))
    hn_paths = list(Path("/kaggle/input").rglob("hard_negative_manifest.jsonl"))
    if not prod_paths:
        raise FileNotFoundError("production_eligible_manifest.jsonl missing in /kaggle/input")
    prod = load_jsonl(prod_paths[0])
    hn = load_jsonl(hn_paths[0]) if hn_paths else []
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for rec in prod + hn:
        if rec.get("data_zone") == "EXPERIMENT_ONLY":
            continue
        if rec.get("training_use_status") not in {None, "LICENSE_OK", "OK"} and rec.get(
            "licence_verification_status"
        ) not in {"VERIFIED", None}:
            # keep VERIFIED / LICENSE_OK only
            if rec.get("training_use_status") != "LICENSE_OK":
                continue
        hardneg = bool(rec.get("is_hard_negative"))
        label = str(rec.get("label") or rec.get("target_class") or "unknown")
        heads = heads_for(label, hardneg=hardneg)
        ph = str(rec.get("provenance_hash") or rec.get("record_id") or "")
        if not ph or ph in seen:
            continue
        url = resolve_image_url(rec)
        if not url:
            continue
        seen.add(ph)
        out.append(
            {
                "id": ph,
                "url": url,
                "path": str(IMG_DIR / f"{ph[:40]}.jpg"),
                "licence": rec.get("licence"),
                "zone": "PRODUCTION_ELIGIBLE",
                "hardneg": hardneg,
                **heads,
            }
        )
    status(f"records_built n={len(out)} unique={len(seen)} prod={len(prod)} hn={len(hn)}")
    return out


def download_all(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    ok_rows: list[dict[str, Any]] = []

    def job(r: dict[str, Any]) -> dict[str, Any] | None:
        if download_one(r["url"], Path(r["path"])):
            return r
        return None

    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as ex:
        futs = [ex.submit(job, r) for r in rows]
        for i, fut in enumerate(as_completed(futs), 1):
            got = fut.result()
            if got:
                ok_rows.append(got)
            if i % 50 == 0:
                status(f"download_progress {i}/{len(rows)} ok={len(ok_rows)}")
    status(f"download_done ok={len(ok_rows)}/{len(rows)}")
    return ok_rows


def split_rows(rows: list[dict[str, Any]], val_frac: float = 0.2) -> tuple[list, list]:
    by = defaultdict(list)
    for r in rows:
        by[r["fine_label"]].append(r)
    train, val = [], []
    rng = random.Random(SEED)
    for lab, items in by.items():
        rng.shuffle(items)
        n_val = max(1, int(round(len(items) * val_frac))) if len(items) >= 5 else 0
        val.extend(items[:n_val])
        train.extend(items[n_val:] or items)
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


class CropDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], label_vocab: dict[str, int]):
        self.rows = rows
        self.label_vocab = label_vocab
        self.vis = {n: i for i, n in enumerate(VISUAL_CLASS_LABELS)}
        self.state = {n: i for i, n in enumerate(FOOD_STATE_LABELS)}
        self.kind = {n: i for i, n in enumerate(KIND_LABELS)}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        r = self.rows[idx]
        img = Image.open(r["path"]).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        import numpy as np

        arr = np.asarray(img).astype("float32") / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype="float32")
        std = np.array([0.229, 0.224, 0.225], dtype="float32")
        arr = (arr - mean) / std
        x = torch.from_numpy(arr).permute(2, 0, 1).float()
        y_label = self.label_vocab[r["fine_label"]]
        y_vis = self.vis.get(r["visual_class"], self.vis["raw_ingredient"])
        y_state = self.state.get(r["food_state"], self.state["unknown"])
        y_kind = self.kind.get(r["kind"], self.kind["ingredient"])
        return x, y_label, y_vis, y_state, y_kind


class FoodVisionV2(nn.Module):
    def __init__(self, n_labels: int, backbone):
        super().__init__()
        self.backbone = backbone
        for p in self.backbone.parameters():
            p.requires_grad = False
        dim = 768
        self.fc_visual = nn.Linear(dim, len(VISUAL_CLASS_LABELS))
        self.fc_state = nn.Linear(dim, len(FOOD_STATE_LABELS))
        self.fc_kind = nn.Linear(dim, len(KIND_LABELS))
        self.fc_label = nn.Linear(dim, max(1, n_labels))

    def forward(self, x):
        with torch.no_grad():
            h = self.backbone(x)
        if isinstance(h, (tuple, list)):
            h = h[0]
        if h.ndim > 2:
            h = h.mean(dim=1)
        h = h.float()
        return {
            "visual_class": self.fc_visual(h),
            "food_state": self.fc_state(h),
            "kind": self.fc_kind(h),
            "label": self.fc_label(h),
        }


def load_backbone():
    return torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")


def load_or_init_model(n_labels: int, label_vocab: dict[str, int], device: torch.device):
    backbone = load_backbone()
    model = FoodVisionV2(n_labels, backbone)
    # optional warm-start from production heads if present
    wroot = find_input("kiq-fv-production-weights-v1", "kiq-fv-production-weights")
    if wroot:
        cands = list(wroot.rglob("model.pt"))
        if cands:
            blob = torch.load(cands[0], map_location="cpu", weights_only=False)
            sd = blob.get("state_dict") or blob
            # load matching head weights where shapes agree
            msd = model.state_dict()
            loaded = 0
            for k, v in sd.items():
                kk = k.replace("module.", "")
                if kk in msd and msd[kk].shape == v.shape:
                    msd[kk] = v
                    loaded += 1
            model.load_state_dict(msd, strict=False)
            status(f"warm_start_loaded_tensors={loaded} from {cands[0]}")
    model.to(device)
    return model


def train_one(
    model: nn.Module,
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
    label_vocab: dict[str, int],
    device: torch.device,
) -> dict[str, Any]:
    CKPT.mkdir(parents=True, exist_ok=True)
    ds_tr = CropDataset(train_rows, label_vocab)
    ds_va = CropDataset(val_rows, label_vocab)
    counts = Counter(r["fine_label"] for r in train_rows)
    weights = [1.0 / max(1, counts[r["fine_label"]]) for r in train_rows]
    # upsample critical
    for i, r in enumerate(train_rows):
        if r["fine_label"] in CRITICAL:
            weights[i] *= 2.5
    sampler = WeightedRandomSampler(weights, num_samples=len(train_rows), replacement=True)
    dl_tr = DataLoader(ds_tr, batch_size=BATCH, sampler=sampler, num_workers=2, pin_memory=True)
    dl_va = DataLoader(ds_va, batch_size=BATCH, shuffle=False, num_workers=2)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=WD)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    inv = {v: k for k, v in label_vocab.items()}

    start_epoch = 0
    history = []
    best_val = -1.0
    best_path = WEIGHTS / "identity_specialist_best.pt"
    last_ckpt = CKPT / "last.pt"
    if last_ckpt.is_file():
        ck = torch.load(last_ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["state_dict"], strict=False)
        opt.load_state_dict(ck["opt"])
        start_epoch = int(ck.get("epoch", 0)) + 1
        history = list(ck.get("history") or [])
        best_val = float(ck.get("best_val", -1))
        status(f"RESUMED epoch={start_epoch} best_val={best_val}")

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        # keep backbone frozen
        for p in model.backbone.parameters():
            p.requires_grad = False
        loss_sum = 0.0
        n = 0
        for x, y_lab, y_vis, y_state, y_kind in dl_tr:
            x = x.to(device, non_blocking=True)
            y_lab = y_lab.to(device)
            y_vis = y_vis.to(device)
            y_state = y_state.to(device)
            y_kind = y_kind.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                out = model(x)
                loss = (
                    F.cross_entropy(out["label"], y_lab)
                    + 0.5 * F.cross_entropy(out["visual_class"], y_vis)
                    + 0.35 * F.cross_entropy(out["food_state"], y_state)
                    + 0.25 * F.cross_entropy(out["kind"], y_kind)
                )
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            loss_sum += float(loss.item()) * x.size(0)
            n += x.size(0)
        train_loss = loss_sum / max(1, n)

        # val
        model.eval()
        correct = 0
        total = 0
        conf_sum = 0.0
        per = defaultdict(lambda: {"tp": 0, "sup": 0})
        with torch.no_grad():
            for x, y_lab, y_vis, y_state, y_kind in dl_va:
                x = x.to(device, non_blocking=True)
                y_lab = y_lab.to(device)
                out = model(x)
                probs = F.softmax(out["label"], dim=-1)
                conf, pred = probs.max(dim=-1)
                correct += int((pred == y_lab).sum().item())
                total += int(y_lab.numel())
                conf_sum += float(conf.sum().item())
                for p, t in zip(pred.tolist(), y_lab.tolist()):
                    lab = inv[t]
                    per[lab]["sup"] += 1
                    if p == t:
                        per[lab]["tp"] += 1
        val_acc = correct / max(1, total)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_acc": val_acc})
        status(f"epoch={epoch} train_loss={train_loss:.4f} val_acc={val_acc:.4f}")
        payload = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "opt": opt.state_dict(),
            "history": history,
            "best_val": best_val,
            "label_vocab": label_vocab,
            "architecture": "FoodVisionV2",
            "train_mode": "A",
            "n_labels": len(label_vocab),
            "seed": SEED,
            "programme": "v10-targeted-classifier",
            "production_promotion": False,
            "detector_changed": False,
            "holdout_touched": False,
            "experiment_only_mixed": False,
        }
        torch.save(payload, last_ckpt)
        if val_acc >= best_val:
            best_val = val_acc
            payload["best_val"] = best_val
            WEIGHTS.mkdir(parents=True, exist_ok=True)
            torch.save(payload, best_path)
            status(f"NEW_BEST val_acc={best_val:.4f}")

    per_class = {
        k: {
            "support": v["sup"],
            "recall": round(v["tp"] / max(1, v["sup"]), 4),
        }
        for k, v in per.items()
    }
    return {
        "best_val_acc": best_val,
        "history": history,
        "per_class_val": per_class,
        "best_weights": str(best_path),
        "best_sha256": sha256_file(best_path) if best_path.is_file() else None,
        "mean_val_confidence": round(conf_sum / max(1, total), 4),
    }


def main() -> None:
    set_seed(SEED)
    for d in (WORK, IMG_DIR, WEIGHTS, REPORTS, CKPT):
        d.mkdir(parents=True, exist_ok=True)
    status("START v10 targeted identity classifier — ONE cycle")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    status(f"device={device} cuda={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        status(f"gpu={[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}")

    rows = build_records()
    rows = download_all(rows)
    if len(rows) < 20:
        raise RuntimeError(f"insufficient downloaded images: {len(rows)}")

    # dedupe by file hash
    seen_hash: set[str] = set()
    deduped = []
    for r in rows:
        h = sha256_file(Path(r["path"]))
        if h in seen_hash:
            continue
        seen_hash.add(h)
        r["file_sha256"] = h
        deduped.append(r)
    status(f"deduped {len(rows)}->{len(deduped)}")
    rows = deduped

    labels = sorted({r["fine_label"] for r in rows})
    label_vocab = {lab: i for i, lab in enumerate(labels)}
    train_rows, val_rows = split_rows(rows)
    write_json(
        REPORTS / "dataset_manifest.json",
        {
            "n_total": len(rows),
            "n_train": len(train_rows),
            "n_val": len(val_rows),
            "label_counts": dict(Counter(r["fine_label"] for r in rows)),
            "zone": "PRODUCTION_ELIGIBLE_ONLY",
            "experiment_only_mixed": False,
            "acceptance_scenes_used": False,
            "seed": SEED,
        },
    )
    ds_hash = hashlib.sha256(
        json.dumps(
            sorted([(r["id"], r["fine_label"], r.get("file_sha256")) for r in rows]),
            sort_keys=True,
        ).encode()
    ).hexdigest()

    model = load_or_init_model(len(label_vocab), label_vocab, device)
    metrics = train_one(model, train_rows, val_rows, label_vocab, device)
    summary = {
        "programme": "v10-targeted-classifier",
        "generated_at": now(),
        "dataset_hash": ds_hash,
        "n_train": len(train_rows),
        "n_val": len(val_rows),
        "label_vocab": label_vocab,
        "metrics": metrics,
        "immutable_baseline_strict": 0.2439,
        "specialist_ensemble_strict": 0.2195,
        "detector_changed": False,
        "holdout_touched": False,
        "epochs": EPOCHS,
        "seed": SEED,
        "one_cycle_only": True,
    }
    write_json(REPORTS / "training_summary.json", summary)
    status(f"DONE best_val={metrics.get('best_val_acc')} sha={metrics.get('best_sha256')}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        status("FATAL\n" + traceback.format_exc())
        raise
