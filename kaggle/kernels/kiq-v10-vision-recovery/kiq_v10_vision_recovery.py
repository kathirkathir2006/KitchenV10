"""KitchenIQ V10 Vision Recovery — acquire missing production-eligible images + train FULL required∪OI vocabulary.

ONE controlled cycle. Not the failed 12-label experiment.
Detector unchanged. Holdout untouched. Experiment-only excluded.
Images stay on Kaggle (no Windows bulk mirror).
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
from urllib.parse import quote
from urllib.request import Request, urlopen

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

SEED = 42
EPOCHS = 8
BATCH = 16
LR = 5e-4
WD = 1e-4
IMG = 224
DL_WORKERS = 8
MAX_ACQUIRE_PER_LABEL = 50

WORK = Path("/kaggle/working/v10-vision-recovery")
IMG_DIR = WORK / "images"
WEIGHTS = WORK / "weights"
REPORTS = WORK / "reports"
CKPT = WORK / "checkpoints"
STATUS = WORK / "LIVE_STATUS.txt"
ACQ = WORK / "acquisition"

VISUAL = ("raw_ingredient", "packaged_food", "prepared_meal", "recipe_document", "non_food", "mixed")
STATE = ("raw", "prepared", "packaged", "processed", "unknown")
KIND = ("ingredient", "meal", "non_food", "document", "packaged")

# Open Images Mid class name hints for acquisition (CC-BY)
OI_CLASS_HINTS = {
    "tomato": "Tomato",
    "cheese": "Cheese",
    "pizza": "Pizza",
    "salad": "Salad",
    "doughnut": "Doughnut",
    "sandwich": "Sandwich",
    "egg": "Egg (Food)",
    "fried_egg": "Egg (Food)",
    "bread": "Bread",
    "milk": "Milk",
    "orange": "Orange",
    "banana": "Banana",
    "apple": "Apple",
    "carrot": "Carrot",
    "broccoli": "Broccoli",
    "mushroom": "Mushroom",
    "potato": "Potato",
    "lemon": "Lemon",
    "strawberry": "Strawberry",
    "grape": "Grape",
    "hamburger": "Hamburger",
    "hot_dog": "Hot dog",
    "french_fries": "French fries",
    "pasta": "Pasta",
    "sushi": "Sushi",
    "taco": "Taco",
    "burrito": "Burrito",
    "pancake": "Pancake",
    "waffle": "Waffle",
    "cookie": "Cookie",
    "cake": "Cake",
    "ice_cream": "Ice cream",
    "popcorn": "Popcorn",
    "coffee": "Coffee",
    "wine": "Wine",
    "beer": "Beer",
}

OPENVERSE_QUERIES = {
    "garlic": "garlic cloves food",
    "ginger": "fresh ginger root",
    "biryani": "biryani rice dish",
    "cooked_rice": "fried rice plate",
    "yogurt": "yogurt bowl",
    "cream": "heavy cream dairy",
    "tomato_paste": "tomato paste can",
    "packaged_cheese": "packaged cheese supermarket",
    "empty_plate": "empty white plate table",
    "human_hand": "human hand empty no food",
    "non_food_object": "kitchen utensil empty counter",
    "doughnut": "doughnut glazed",
    "sandwich": "club sandwich",
    "salad": "garden salad bowl",
    "pizza": "pizza plated",
    "tomato": "fresh tomato",
    "cheese": "cheese block",
    "rice": "uncooked rice bowl",
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


def find_pack() -> Path:
    for p in Path("/kaggle/input").iterdir():
        if (p / "production_identity_training_manifest_v1.json").is_file() or (
            p / "required_identity_vocabulary_v1.json"
        ).is_file():
            return p
    # fallback search
    for p in Path("/kaggle/input").rglob("production_identity_training_manifest_v1.json"):
        return p.parent
    raise FileNotFoundError("recovery pack not found")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def download(url: str, dest: Path, timeout: int = 40) -> bool:
    if dest.is_file() and dest.stat().st_size > 1500:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        req = Request(url, headers={"User-Agent": "KitchenIQ-V10-Recovery/1.0"})
        with urlopen(req, timeout=timeout) as resp, tmp.open("wb") as out:
            out.write(resp.read())
        # validate image
        try:
            im = Image.open(tmp)
            im.verify()
            w, h = Image.open(tmp).size
            if min(w, h) < 64 or w * h > 40_000_000:
                tmp.unlink(missing_ok=True)
                return False
        except Exception:
            tmp.unlink(missing_ok=True)
            return False
        tmp.replace(dest)
        return True
    except Exception as e:
        tmp.unlink(missing_ok=True)
        status(f"DL_FAIL {url[:90]} :: {e}")
        return False


def openverse_search(query: str, page: int = 1, page_size: int = 20) -> list[dict[str, Any]]:
    # CC0 / CC-BY only
    url = (
        "https://api.openverse.org/v1/images/"
        f"?q={quote(query)}&license=cc0,by&page={page}&page_size={page_size}&format=json"
    )
    try:
        req = Request(url, headers={"User-Agent": "KitchenIQ-V10-Recovery/1.0"})
        with urlopen(req, timeout=45) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="ignore"))
        return list(data.get("results") or [])
    except Exception as e:
        status(f"openverse_fail q={query} :: {e}")
        return []


def acquire_label(label: str, n_needed: int, existing_hashes: set[str]) -> list[dict[str, Any]]:
    ACQ.mkdir(parents=True, exist_ok=True)
    out: list[dict[str, Any]] = []
    q = OPENVERSE_QUERIES.get(label, label.replace("_", " "))
    page = 1
    while len(out) < n_needed and page <= 5:
        results = openverse_search(q, page=page)
        if not results:
            break
        for r in results:
            if len(out) >= n_needed:
                break
            lic = str(r.get("license") or "").lower().replace(" ", "")
            if lic not in {"cc0", "by", "ccby", "cc-by", "cc_by"} and "by" not in lic and lic != "cc0":
                # openverse returns license like "cc0" or "by"
                if r.get("license") not in {"cc0", "by"}:
                    continue
            url = r.get("url") or r.get("thumbnail")
            if not url:
                continue
            rid = str(r.get("id") or hashlib.md5(url.encode()).hexdigest())
            dest = IMG_DIR / f"acq_{label}_{rid[:16]}.jpg"
            if not download(str(url), dest):
                continue
            h = sha256_file(dest)
            if h in existing_hashes:
                dest.unlink(missing_ok=True)
                continue
            existing_hashes.add(h)
            out.append(
                {
                    "id": f"acq:{label}:{rid}",
                    "path": str(dest),
                    "fine_label": label,
                    "licence": r.get("license"),
                    "source": "openverse",
                    "source_url": r.get("foreign_landing_url") or url,
                    "zone": "PRODUCTION_ELIGIBLE",
                    "file_sha256": h,
                    "hardneg": label in {"empty_plate", "human_hand", "non_food_object"},
                }
            )
        page += 1
        time.sleep(0.4)
    status(f"acquired {label}: {len(out)}/{n_needed}")
    return out


def heads_for(label: str, hardneg: bool = False) -> dict[str, str]:
    if hardneg or label in {"non_food_object", "empty_plate", "human_hand", "non_food"}:
        return {"fine_label": "non_food", "visual_class": "non_food", "food_state": "unknown", "kind": "non_food"}
    prepared = {
        "pizza",
        "salad",
        "sandwich",
        "cooked_rice",
        "biryani",
        "fried_egg",
        "hamburger",
        "hot_dog",
        "pasta",
        "sushi",
        "burrito",
        "taco",
        "doughnut",
        "pancake",
        "waffle",
        "french_fries",
        "ice_cream",
    }
    packaged = {"packaged_cheese", "cheese", "yogurt", "cream", "tomato_paste", "milk", "beer", "wine"}
    if label in prepared:
        return {"fine_label": label, "visual_class": "prepared_meal", "food_state": "prepared", "kind": "meal"}
    if label in packaged:
        return {"fine_label": label, "visual_class": "packaged_food", "food_state": "packaged", "kind": "packaged"}
    return {"fine_label": label, "visual_class": "raw_ingredient", "food_state": "raw", "kind": "ingredient"}


def build_base_rows(pack: Path) -> list[dict[str, Any]]:
    prod = load_jsonl(pack / "production_eligible_manifest.jsonl")
    hn = load_jsonl(pack / "hard_negative_manifest.jsonl")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for rec in prod + hn:
        if rec.get("data_zone") == "EXPERIMENT_ONLY":
            continue
        if rec.get("training_use_status") not in {None, "LICENSE_OK", "OK"}:
            if rec.get("licence_verification_status") != "VERIFIED":
                continue
        hardneg = bool(rec.get("is_hard_negative"))
        label = str(rec.get("label") or rec.get("target_class") or "").strip().lower().replace(" ", "_")
        # map fried_rice alias already in remediation as fried_rice → keep; also map to cooked_rice duplicate row? keep label as-is then remap via training vocab aliases
        if label == "fried_rice":
            label = "cooked_rice"
        if label in {"hand", "human_hand"}:
            label = "human_hand"
        if label == "non_food":
            label = "non_food_object"
        url = rec.get("image_url")
        if not url and rec.get("image_id"):
            url = f"https://open-images-dataset.s3.amazonaws.com/validation/{rec['image_id']}.jpg"
        if not url:
            continue
        ph = str(rec.get("provenance_hash") or rec.get("record_id") or "")
        if not ph or ph in seen:
            continue
        seen.add(ph)
        dest = IMG_DIR / f"{ph[:40]}.jpg"
        heads = heads_for(label, hardneg=hardneg)
        rows.append(
            {
                "id": ph,
                "url": url,
                "path": str(dest),
                "licence": rec.get("licence"),
                "source": rec.get("source"),
                "zone": "PRODUCTION_ELIGIBLE",
                "hardneg": hardneg,
                "file_sha256": None,
                **heads,
            }
        )
    return rows


def download_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ok = []

    def job(r):
        if download(r["url"], Path(r["path"])):
            r["file_sha256"] = sha256_file(Path(r["path"]))
            return r
        return None

    with ThreadPoolExecutor(max_workers=DL_WORKERS) as ex:
        futs = [ex.submit(job, r) for r in rows]
        for i, fut in enumerate(as_completed(futs), 1):
            got = fut.result()
            if got:
                ok.append(got)
            if i % 40 == 0:
                status(f"download_progress {i}/{len(rows)} ok={len(ok)}")
    status(f"download_done {len(ok)}/{len(rows)}")
    return ok


class CropDS(Dataset):
    def __init__(self, rows, label_vocab):
        self.rows = rows
        self.label_vocab = label_vocab
        self.vis = {n: i for i, n in enumerate(VISUAL)}
        self.state = {n: i for i, n in enumerate(STATE)}
        self.kind = {n: i for i, n in enumerate(KIND)}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        import numpy as np

        r = self.rows[idx]
        img = Image.open(r["path"]).convert("RGB").resize((IMG, IMG), Image.BILINEAR)
        arr = np.asarray(img).astype("float32") / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype="float32")
        std = np.array([0.229, 0.224, 0.225], dtype="float32")
        arr = (arr - mean) / std
        x = torch.from_numpy(arr).permute(2, 0, 1).float()
        lab = r["fine_label"]
        if lab == "non_food_object":
            lab = "non_food"
        y = self.label_vocab[lab]
        return (
            x,
            y,
            self.vis.get(r["visual_class"], 0),
            self.state.get(r["food_state"], 4),
            self.kind.get(r["kind"], 0),
        )


class FoodVisionV2(nn.Module):
    def __init__(self, n_labels: int, backbone):
        super().__init__()
        self.backbone = backbone
        for p in self.backbone.parameters():
            p.requires_grad = False
        dim = 768
        self.fc_visual = nn.Linear(dim, len(VISUAL))
        self.fc_state = nn.Linear(dim, len(STATE))
        self.fc_kind = nn.Linear(dim, len(KIND))
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


def main() -> None:
    set_seed(SEED)
    for d in (WORK, IMG_DIR, WEIGHTS, REPORTS, CKPT, ACQ):
        d.mkdir(parents=True, exist_ok=True)
    status("START vision recovery acquire+train FULL vocab")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    status(f"device={device} gpus={torch.cuda.device_count() if torch.cuda.is_available() else 0}")

    pack = find_pack()
    status(f"pack={pack}")
    train_manifest = json.loads((pack / "production_identity_training_manifest_v1.json").read_text(encoding="utf-8"))
    acq_plan = json.loads((pack / "production_identity_acquisition_plan_v1.json").read_text(encoding="utf-8"))
    prod_vocab = json.loads((pack / "production_label_vocab.json").read_text(encoding="utf-8"))
    prod_vocab = {str(k).lower().replace(" ", "_"): int(v) for k, v in prod_vocab.items()}

    trainable = [
        t["canonical_identity"]
        for t in train_manifest["labels"]
        if t.get("include_in_training")
    ]
    # FULL vocab = production OI-62 ∪ trainable required (not 12-label toy set)
    label_names = sorted(set(prod_vocab.keys()) | set(trainable) | {"non_food"})
    # normalize non_food_object → non_food in vocab
    label_names = sorted({("non_food" if x == "non_food_object" else x) for x in label_names})
    label_vocab = {n: i for i, n in enumerate(label_names)}
    status(f"label_vocab_n={len(label_vocab)} (prod62∪required trainable)")

    rows = build_base_rows(pack)
    rows = download_rows(rows)
    hashes = {r["file_sha256"] for r in rows if r.get("file_sha256")}

    # acquisition for under-supported
    acquired_all = []
    for item in acq_plan.get("items") or []:
        if item.get("action") != "ACQUIRE_PRODUCTION_ELIGIBLE":
            continue
        lab = item["canonical_identity"]
        if lab == "non_food_object":
            lab_key = "non_food_object"
        else:
            lab_key = lab
        need = min(MAX_ACQUIRE_PER_LABEL, int(item.get("additional_images_needed") or 0))
        if need <= 0:
            continue
        # map to heads label
        got = acquire_label(lab_key, need, hashes)
        for g in got:
            # unify non_food
            if g["fine_label"] in {"non_food_object", "empty_plate", "human_hand"}:
                # keep fine label for diversity but map empty/hand separately in vocab if present
                pass
            heads = heads_for(g["fine_label"], hardneg=g.get("hardneg", False))
            g.update(heads)
            if g["fine_label"] == "non_food_object":
                g["fine_label"] = "non_food"
            acquired_all.append(g)
    write_json(ACQ / "acquired_manifest.json", {"n": len(acquired_all), "rows": acquired_all})
    rows.extend(acquired_all)

    # keep only rows whose fine_label is in vocab
    filtered = []
    for r in rows:
        lab = r["fine_label"]
        if lab == "non_food_object":
            lab = "non_food"
            r["fine_label"] = "non_food"
        if lab in label_vocab:
            filtered.append(r)
    rows = filtered
    status(f"rows_for_train={len(rows)} labels_present={len({r['fine_label'] for r in rows})}")

    # split
    by = defaultdict(list)
    for r in rows:
        by[r["fine_label"]].append(r)
    train, val = [], []
    rng = random.Random(SEED)
    for lab, items in by.items():
        rng.shuffle(items)
        n_val = max(1, int(round(len(items) * 0.2))) if len(items) >= 5 else 0
        val.extend(items[:n_val])
        train.extend(items[n_val:] or items)
    write_json(
        REPORTS / "dataset_manifest.json",
        {
            "n_total": len(rows),
            "n_train": len(train),
            "n_val": len(val),
            "n_acquired": len(acquired_all),
            "label_counts": dict(Counter(r["fine_label"] for r in rows)),
            "label_vocab": label_vocab,
            "experiment_only_mixed": False,
            "twelve_label_forbidden_ok": len(label_vocab) > 20,
        },
    )
    ds_hash = hashlib.sha256(
        json.dumps(sorted([(r["id"], r["fine_label"], r.get("file_sha256")) for r in rows])).encode()
    ).hexdigest()

    backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    model = FoodVisionV2(len(label_vocab), backbone)
    # warm-start overlapping heads from production weights if present
    wroot = None
    for p in Path("/kaggle/input").iterdir():
        if "production-weights" in p.name or "production_weights" in p.name:
            wroot = p
            break
    if wroot:
        cands = list(wroot.rglob("model.pt"))
        if cands:
            blob = torch.load(cands[0], map_location="cpu", weights_only=False)
            old_vocab = {str(k).lower().replace(" ", "_"): int(v) for k, v in (blob.get("label_vocab") or {}).items()}
            sd = blob.get("state_dict") or {}
            # copy non-label heads fully; label head partially by name intersection is hard — copy visual/state/kind
            msd = model.state_dict()
            loaded = 0
            for k, v in sd.items():
                kk = k.replace("module.", "")
                if kk.startswith("fc_label"):
                    continue
                if kk in msd and msd[kk].shape == v.shape:
                    msd[kk] = v
                    loaded += 1
            model.load_state_dict(msd, strict=False)
            status(f"warm_start_nonlabel_tensors={loaded} old_vocab={len(old_vocab)}")

    model.to(device)
    ds_tr = CropDS(train, label_vocab)
    ds_va = CropDS(val, label_vocab)
    counts = Counter(r["fine_label"] for r in train)
    weights = [1.0 / max(1, counts[r["fine_label"]]) for r in train]
    for i, r in enumerate(train):
        if r["fine_label"] in set(trainable) | {"non_food", "tomato", "garlic", "ginger", "biryani", "cooked_rice"}:
            weights[i] *= 2.0
    sampler = WeightedRandomSampler(weights, num_samples=len(train), replacement=True)
    dl_tr = DataLoader(ds_tr, batch_size=BATCH, sampler=sampler, num_workers=2, pin_memory=True)
    dl_va = DataLoader(ds_va, batch_size=BATCH, shuffle=False, num_workers=2)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=WD)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    inv = {v: k for k, v in label_vocab.items()}

    start_epoch = 0
    history = []
    best_val = -1.0
    best_path = WEIGHTS / "identity_coverage_best.pt"
    last_ckpt = CKPT / "last.pt"
    if last_ckpt.is_file() and last_ckpt.stat().st_size > 1000:
        ck = torch.load(last_ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["state_dict"], strict=False)
        start_epoch = int(ck.get("epoch", -1)) + 1
        history = list(ck.get("history") or [])
        best_val = float(ck.get("best_val", -1))
        status(f"RESUMED epoch={start_epoch}")

    for epoch in range(start_epoch, EPOCHS):
        model.train()
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

        model.eval()
        correct = total = 0
        per = defaultdict(lambda: {"tp": 0, "sup": 0})
        with torch.no_grad():
            for x, y_lab, y_vis, y_state, y_kind in dl_va:
                x = x.to(device, non_blocking=True)
                y_lab = y_lab.to(device)
                out = model(x)
                pred = out["label"].argmax(dim=-1)
                correct += int((pred == y_lab).sum().item())
                total += int(y_lab.numel())
                for p, t in zip(pred.tolist(), y_lab.tolist()):
                    lab = inv[t]
                    per[lab]["sup"] += 1
                    if p == t:
                        per[lab]["tp"] += 1
        val_acc = correct / max(1, total)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_acc": val_acc})
        status(f"epoch={epoch} loss={train_loss:.4f} val_acc={val_acc:.4f}")
        payload = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "history": history,
            "best_val": best_val,
            "label_vocab": label_vocab,
            "architecture": "FoodVisionV2",
            "train_mode": "A",
            "n_labels": len(label_vocab),
            "seed": SEED,
            "programme": "v10-vision-recovery",
            "production_promotion": False,
            "detector_changed": False,
            "holdout_touched": False,
            "experiment_only_mixed": False,
            "dataset_hash": ds_hash,
        }
        torch.save(payload, last_ckpt)
        if val_acc >= best_val:
            best_val = val_acc
            payload["best_val"] = best_val
            torch.save(payload, best_path)
            status(f"NEW_BEST {best_val:.4f}")

    summary = {
        "programme": "v10-vision-recovery",
        "generated_at": now(),
        "dataset_hash": ds_hash,
        "n_train": len(train),
        "n_val": len(val),
        "n_acquired": len(acquired_all),
        "label_vocab": label_vocab,
        "n_labels": len(label_vocab),
        "best_val_acc": best_val,
        "best_sha256": sha256_file(best_path) if best_path.is_file() else None,
        "history": history,
        "per_class_val": {k: {"support": v["sup"], "recall": round(v["tp"] / max(1, v["sup"]), 4)} for k, v in per.items()},
        "baseline_strict": 0.2439,
        "specialist_strict": 0.2195,
        "failed_targeted_strict": 0.0732,
    }
    write_json(REPORTS / "training_summary.json", summary)
    status(f"DONE n_labels={len(label_vocab)} best_val={best_val} sha={summary['best_sha256']}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        status("FATAL\n" + traceback.format_exc())
        raise
