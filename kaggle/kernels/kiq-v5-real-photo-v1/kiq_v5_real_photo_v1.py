"""KitchenIQ V5 real-photo set v1 builder (run ONCE; v1 is immutable after creation).

Guarantees enforced in code (see docs/v5/REAL_PHOTO_SET_V1_SPEC.md):
  provenance/licence  : Openverse record per image; licence in {cc0, by}; licence_url + landing page required
  no train/ref overlap: reject by id/url, sha256, dHash, DINOv2 near-duplicate vs every image/manifest attached
  no benchmark leakage: reject by dHash / DINOv2 near-duplicate vs 41-scene scene + GT-crop fingerprints
  no internal dupes   : near-duplicates inside the set are dropped (first kept, deterministic order)
  roles               : split by creator group (sha256 bucket) -> train/val/test; open-set identities val/test only
  fixed               : manifest with per-image sha256 + manifest sha; consumers must verify before use
"""
from __future__ import annotations

import hashlib
import io
import json
import re
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import quote

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

SET_ID = "kiq_v5_real_photo_v1"
WORK = Path("/kaggle/working") / SET_ID
IMG_DIR = WORK / "images"
INPUT = Path("/kaggle/input")
UA = {"User-Agent": "KitchenIQ-V5-dataset-builder/1.0 (research; CC attribution recorded)"}

LICENCES = {"cc0", "by"}
QUOTA = {"train": 30, "val": 10, "test": 10}
MAX_CANDIDATES = 300
PAGES = 12
DHASH_LEAK, COS_LEAK = 8, 0.92
DHASH_DUP, COS_DUP = 6, 0.95

# identity -> (Openverse queries, title/tag keywords, family, state)
KNOWN = {
    "tomato": (["tomato", "fresh tomatoes"], ["tomato"], "produce", "raw"),
    "garlic": (["garlic bulb", "garlic cloves"], ["garlic"], "produce", "raw"),
    "ginger": (["ginger root"], ["ginger"], "produce", "raw"),
    "cheese": (["cheese block", "cheddar cheese"], ["cheese", "cheddar"], "dairy", "raw"),
    "packaged_cheese": (["packaged cheese", "cheese package supermarket"], ["cheese"], "packaged_dairy", "packaged"),
    "yogurt": (["yogurt bowl", "yoghurt"], ["yogurt", "yoghurt"], "dairy", "raw"),
    "cream": (["whipped cream", "fresh cream"], ["cream"], "dairy", "raw"),
    "tomato_paste": (["tomato paste", "tomato puree"], ["tomato paste", "tomato puree", "paste"], "sauce", "packaged"),
    "rice": (["cooked white rice", "steamed rice bowl"], ["rice"], "rice_grain", "cooked"),
    "fried_rice": (["fried rice"], ["fried rice"], "rice_grain", "plated"),
    "biryani": (["biryani"], ["biryani", "biriyani"], "rice_grain", "plated"),
    "pasta": (["pasta dish", "spaghetti"], ["pasta", "spaghetti", "penne"], "grain_starch", "plated"),
    "lasagna": (["lasagna"], ["lasagna", "lasagne"], "grain_starch", "plated"),
    "pad_thai": (["pad thai"], ["pad thai", "padthai"], "noodle", "plated"),
    "pizza": (["pizza"], ["pizza"], "prepared_meal", "plated"),
    "salad": (["salad bowl", "green salad"], ["salad"], "salad", "plated"),
    "caesar_salad": (["caesar salad"], ["caesar"], "salad", "plated"),
    "club_sandwich": (["club sandwich"], ["club sandwich", "sandwich"], "sandwich", "plated"),
    "omelette": (["omelette", "omelet"], ["omelette", "omelet"], "prepared_egg", "plated"),
    "french_fries": (["french fries"], ["fries", "chips"], "prepared_side", "plated"),
    "spring_rolls": (["spring rolls"], ["spring roll"], "prepared_side", "plated"),
    "guacamole": (["guacamole"], ["guacamole"], "dip", "plated"),
    "hummus": (["hummus"], ["hummus", "houmous"], "dip", "plated"),
    "bibimbap": (["bibimbap"], ["bibimbap"], "rice_grain", "plated"),
    "donuts": (["donuts", "doughnuts"], ["donut", "doughnut"], "prepared_sweet", "plated"),
    "ice_cream": (["ice cream"], ["ice cream", "gelato"], "prepared_sweet", "plated"),
    "non_food": (["kitchen utensils", "empty plate", "cutting board kitchen", "kitchen counter"],
                 ["utensil", "plate", "board", "kitchen", "counter", "knife", "spoon"], "non_food", "none"),
}
# open-set identities: never in train; val/test only, scored as UNKNOWN
OPEN_SET = {
    "red_bell_pepper": (["red bell pepper"], ["pepper", "capsicum"], "produce", "raw"),
    "flatbread": (["flatbread", "naan bread"], ["flatbread", "naan", "pita"], "bread", "cooked"),
    "beef_stew": (["beef stew"], ["stew"], "stew", "plated"),
    "dumplings": (["dumplings"], ["dumpling", "gyoza"], "prepared_side", "plated"),
    "pancakes": (["pancakes"], ["pancake"], "prepared_sweet", "plated"),
}
CLIP_NAMES = {k: k.replace("_", " ") for k in {**KNOWN, **OPEN_SET}}
CLIP_NAMES["non_food"] = "kitchen objects with no food"


def status(msg: str) -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    line = f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with (WORK / "LIVE_STATUS.txt").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def dhash(img: Image.Image) -> int:
    g = np.asarray(img.convert("L").resize((9, 8), Image.BILINEAR), dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def ham(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def preprocess(img: Image.Image, dev) -> torch.Tensor:
    arr = np.asarray(img.convert("RGB").resize((224, 224), Image.BILINEAR)).astype("float32") / 255.0
    arr = (arr - np.array([0.485, 0.456, 0.406], "float32")) / np.array([0.229, 0.224, 0.225], "float32")
    return torch.from_numpy(arr).permute(2, 0, 1).float().unsqueeze(0).to(dev)


class Dino:
    def __init__(self, dev):
        self.dev = dev
        self.m = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", trust_repo=True).eval().to(dev)

    @torch.no_grad()
    def __call__(self, img: Image.Image) -> np.ndarray:
        f = self.m.forward_features(preprocess(img, self.dev))["x_norm_clstoken"]
        return F.normalize(f.float(), dim=-1)[0].cpu().numpy()


class Clip:
    """Open-weights CLIP used ONLY for label curation (never as the scanner)."""

    def __init__(self, dev):
        from transformers import CLIPModel, CLIPProcessor

        self.dev = dev
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval().to(dev)
        self.proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.labels = sorted(CLIP_NAMES)
        self.prompts = [f"a photo of {CLIP_NAMES[l]}" for l in self.labels]

    @torch.no_grad()
    def rank(self, img: Image.Image) -> list[tuple[str, float]]:
        x = self.proc(text=self.prompts, images=img, return_tensors="pt", padding=True).to(self.dev)
        p = self.model(**x).logits_per_image.softmax(-1)[0].cpu().numpy()
        order = np.argsort(-p)
        return [(self.labels[i], float(p[i])) for i in order[:5]]


def openverse(query: str, page: int) -> list[dict]:
    url = (f"https://api.openverse.org/v1/images/?q={quote(query)}&license=cc0,by"
           f"&page={page}&page_size=20&mature=false&format=json")
    for attempt in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
                return json.loads(r.read().decode("utf-8")).get("results") or []
        except Exception as e:  # noqa: BLE001
            status(f"openverse retry q={query} p={page} a={attempt} :: {e}")
            time.sleep(5 * (attempt + 1))
    return []


def fetch(url: str) -> bytes | None:
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
            data = r.read()
        return data if len(data) > 5000 else None
    except Exception:  # noqa: BLE001
        return None


def norm_creator(c: str | None, rid: str) -> str:
    c = re.sub(r"\s+", " ", (c or "").strip().lower())
    return f"creator:{c}" if c else f"record:{rid}"


def split_for(group: str) -> str:
    """One split per creator group for every identity, so a group can never span splits."""
    b = int(sha(f"{SET_ID}|{group}".encode())[:8], 16) % 100
    return "train" if b < 60 else ("val" if b < 80 else "test")


# ------------------------------------------------------------- exclusion sets
def collect_exclusions(dino: Dino) -> dict:
    strings: set[str] = set()
    for p in list(INPUT.rglob("*.jsonl")) + list(INPUT.rglob("*.json")):
        if "benchmark_fingerprints" in p.name or p.stat().st_size > 200_000_000:
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except Exception:  # noqa: BLE001
            continue
        strings.update(re.findall(r"https?://[^\s\"']+", text))
        strings.update(re.findall(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", text))
    hashes, dh, embs = set(), [], []
    imgs = [p for p in INPUT.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
    for p in imgs:
        try:
            b = p.read_bytes()
            im = Image.open(io.BytesIO(b)).convert("RGB")
        except Exception:  # noqa: BLE001
            continue
        hashes.add(sha(b))
        dh.append(dhash(im))
        embs.append(dino(im))
    lib_embs = []
    for p in INPUT.rglob("reference_library.json"):
        for it in json.loads(p.read_text(encoding="utf-8")).get("items") or []:
            hashes.add(it.get("source_hash"))
            lib_embs.append(np.asarray(it["embedding"], np.float32))
    allemb = embs + lib_embs
    E = np.stack(allemb) if allemb else np.zeros((0, 768), np.float32)
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-8) if len(E) else E
    status(f"exclusions: strings={len(strings)} images={len(imgs)} hashes={len(hashes)} embeddings={len(E)}")
    return {"strings": strings, "hashes": hashes, "dhash": dh, "emb": E, "n_images": len(imgs)}


def load_benchmark() -> dict:
    p = next(INPUT.rglob("benchmark_fingerprints.json"))
    d = json.loads(p.read_text(encoding="utf-8"))
    fp = d["fingerprints"]
    E = np.stack([np.asarray(f["embedding"], np.float32) for f in fp])
    return {"catalog_sha256": d["catalog_sha256"], "sha": {f.get("sha256") for f in fp if f.get("sha256")},
            "dhash": [int(f["dhash"], 16) for f in fp], "emb": E, "ids": [f"{f['scene_id']}:{f['kind']}" for f in fp]}


# ------------------------------------------------------------- main
def main() -> None:
    if (WORK / "manifest.json").exists():
        raise SystemExit("v1 already built in this session — refusing to overwrite an immutable set")
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    status(f"START {SET_ID} device={dev}")
    dino, clip = Dino(dev), Clip(dev)
    probe = Image.new("RGB", (256, 256), (200, 40, 40))
    status(f"selftest dino_dim={dino(probe).shape[0]} clip_top={clip.rank(probe)[0]}")
    bench = load_benchmark()
    excl = collect_exclusions(dino)
    rejects: Counter = Counter()
    kept: list[dict] = []
    kept_emb: list[np.ndarray] = []
    kept_dh: list[int] = []
    filled: dict[str, Counter] = defaultdict(Counter)

    vocab = [(k, v, False) for k, v in KNOWN.items()] + [(k, v, True) for k, v in OPEN_SET.items()]
    for label, (queries, keywords, family, state), is_open in vocab:
        quota = {"val": 10, "test": 10} if is_open else QUOTA
        seen = 0
        for q in queries:
            for page in range(1, PAGES + 1):
                if seen >= MAX_CANDIDATES or all(filled[label][s] >= n for s, n in quota.items()):
                    break
                results = openverse(q, page)
                time.sleep(1.0)
                if not results:
                    break
                for r in results:
                    if seen >= MAX_CANDIDATES or all(filled[label][s] >= n for s, n in quota.items()):
                        break
                    seen += 1
                    rid, url = str(r.get("id") or ""), r.get("url")
                    lic = str(r.get("license") or "").lower()
                    if lic not in LICENCES or not r.get("license_url") or not r.get("foreign_landing_url") or not url:
                        rejects["licence_or_provenance"] += 1
                        continue
                    if (r.get("width") or 999) < 224 or (r.get("height") or 999) < 224:
                        rejects["too_small"] += 1
                        continue
                    if rid in excl["strings"] or url in excl["strings"] or r["foreign_landing_url"] in excl["strings"]:
                        rejects["overlap_train_ref_id_url"] += 1
                        continue
                    group = norm_creator(r.get("creator"), rid)
                    split = split_for(group)
                    if is_open and split == "train":
                        rejects["open_set_group_in_train"] += 1
                        continue
                    if filled[label][split] >= quota.get(split, 0):
                        rejects["split_quota_full"] += 1
                        continue
                    data = fetch(url)
                    if data is None:
                        rejects["download_failed"] += 1
                        continue
                    h = sha(data)
                    try:
                        img = Image.open(io.BytesIO(data)).convert("RGB")
                    except Exception:  # noqa: BLE001
                        rejects["decode_failed"] += 1
                        continue
                    if h in bench["sha"]:
                        rejects["benchmark_exact"] += 1
                        continue
                    if h in excl["hashes"]:
                        rejects["overlap_train_ref_sha"] += 1
                        continue
                    d, e = dhash(img), dino(img)
                    bcos = bench["emb"] @ e
                    bham = min(ham(d, x) for x in bench["dhash"])
                    if bham <= DHASH_LEAK or float(bcos.max()) >= COS_LEAK:
                        rejects["benchmark_near_duplicate"] += 1
                        status(f"LEAK-REJECT {label} {rid} ham={bham} cos={float(bcos.max()):.3f} "
                               f"nearest={bench['ids'][int(bcos.argmax())]}")
                        continue
                    if len(excl["emb"]) and float((excl["emb"] @ e).max()) >= COS_LEAK:
                        rejects["overlap_train_ref_near_duplicate"] += 1
                        continue
                    if excl["dhash"] and min(ham(d, x) for x in excl["dhash"]) <= DHASH_LEAK:
                        rejects["overlap_train_ref_dhash"] += 1
                        continue
                    if kept_emb and (float((np.stack(kept_emb) @ e).max()) >= COS_DUP
                                     or min(ham(d, x) for x in kept_dh) <= DHASH_DUP):
                        rejects["internal_near_duplicate"] += 1
                        continue
                    ranked = clip.rank(img)
                    text = " ".join([str(r.get("title") or "")] + [str(t.get("name")) for t in r.get("tags") or []]).lower()
                    kw = any(k in text for k in keywords)
                    top1 = ranked[0][0] == label
                    top3 = label in [l for l, _ in ranked[:3]]
                    ok = (kw and top3) if split == "train" else (kw and top1)
                    if not ok:
                        rejects[f"label_unverified_{split}"] += 1
                        continue
                    name = f"{label}_{h[:16]}.jpg"
                    dest = IMG_DIR / split / label / name
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(data)
                    kept_emb.append(e)
                    kept_dh.append(d)
                    filled[label][split] += 1
                    kept.append({
                        "file": f"images/{split}/{label}/{name}", "sha256": h, "dhash": f"{d:016x}",
                        "label": label, "family": family, "state": state,
                        "role": "open_set_unknown" if is_open else "known",
                        "split": split, "group": group,
                        "source": "openverse", "openverse_id": rid, "provider": r.get("provider"),
                        "landing_url": r.get("foreign_landing_url"), "file_url": url,
                        "title": r.get("title"), "creator": r.get("creator"), "creator_url": r.get("creator_url"),
                        "licence": lic, "licence_version": r.get("license_version"), "licence_url": r.get("license_url"),
                        "attribution": r.get("attribution"),
                        "verification": {"keyword_match": kw, "clip_top1": top1, "clip_top3": top3,
                                         "clip_ranked": [[l, round(p, 4)] for l, p in ranked[:3]],
                                         "benchmark_min_dhash": bham, "benchmark_max_cos": round(float(bcos.max()), 4)},
                        "width": img.size[0], "height": img.size[1],
                    })
        status(f"label={label} open_set={is_open} filled={dict(filled[label])} seen={seen}")

    # creator groups must not span splits (assignment is per group, so this is an invariant check)
    gsplits = defaultdict(set)
    for k in kept:
        gsplits[k["group"]].add(k["split"])
    assert all(len(v) == 1 for v in gsplits.values()), "group spans splits"
    assert not any(k["role"] == "open_set_unknown" and k["split"] == "train" for k in kept)
    assert len(kept) > 0

    kept.sort(key=lambda k: k["file"])
    body = {"set_id": SET_ID, "immutable": True, "benchmark_catalog_sha256": bench["catalog_sha256"],
            "thresholds": {"dhash_leak": DHASH_LEAK, "cos_leak": COS_LEAK, "dhash_dup": DHASH_DUP, "cos_dup": COS_DUP},
            "licences_allowed": sorted(LICENCES), "label_verification":
                "keyword(title/tags) AND open-weights CLIP (train: top-3, val/test: top-1); curation only",
            "n": len(kept), "items": kept}
    raw = json.dumps(body, sort_keys=True, indent=1)
    (WORK / "manifest.json").write_text(raw, encoding="utf-8")
    manifest_sha = sha(raw.encode())
    counts = defaultdict(Counter)
    for k in kept:
        counts[k["split"]][k["label"]] += 1
    report = {"set_id": SET_ID, "manifest_sha256": manifest_sha, "n": len(kept),
              "per_split": {s: dict(c) for s, c in counts.items()},
              "split_totals": {s: sum(c.values()) for s, c in counts.items()},
              "rejects": dict(rejects), "n_groups": len(gsplits),
              "exclusion_images_fingerprinted": excl["n_images"], "exclusion_embeddings": int(len(excl["emb"]))}
    (WORK / "build_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (WORK / "ATTRIBUTION.md").open("w", encoding="utf-8") as f:
        f.write(f"# {SET_ID} attribution\n\n")
        for k in kept:
            f.write(f"- `{k['file']}`: {k.get('attribution') or k.get('title')} - {k['licence_url']} - {k['landing_url']}\n")
    status(f"DONE n={len(kept)} manifest_sha={manifest_sha} rejects={dict(rejects)}")


if __name__ == "__main__":
    main()
