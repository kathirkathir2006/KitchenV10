"""V5 readiness probe: compute, library structure, training data, benchmark hashes."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from collections import Counter
from pathlib import Path

import psutil
import torch

ROOT = Path(r"C:\Projects\KitchenIQ-V10-AI")
LIB = ROOT / "backend/instance/dev_experiments/v10-vision-v4/reference_library/reference_library.json"
CATALOG = ROOT / "backend/instance/dev_experiments/v10-specialist-ensemble/acceptance/fixtures/catalog/fixture_catalog.json"
OUT = ROOT / "docs/v5/readiness_probe.json"


def sha(p: Path) -> str | None:
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None


def main() -> None:
    info: dict = {}
    info["git"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    info["compute"] = {
        "cuda": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1) if torch.cuda.is_available() else 0,
        "cpu_cores": psutil.cpu_count(logical=True),
        "ram_gb": round(psutil.virtual_memory().total / 1e9, 1),
        "disk_free_gb": round(shutil.disk_usage("C:/").free / 1e9, 1),
        "torch": torch.__version__,
    }
    lib = json.loads(LIB.read_text(encoding="utf-8"))
    items = lib.get("items") or []
    info["library"] = {
        "n": len(items),
        "item_keys": sorted(items[0].keys()) if items else [],
        "labels": dict(Counter(i.get("canonical_identity") for i in items)),
        "families": dict(Counter(i.get("family") for i in items)),
        "states": dict(Counter(i.get("food_state") for i in items)),
        "has_source_url": sum(1 for i in items if i.get("source_url") or i.get("image_url")),
        "has_license": sum(1 for i in items if i.get("license")),
        "has_image_sha": sum(1 for i in items if i.get("image_sha256") or i.get("sha256")),
        "sha256_file": sha(LIB),
    }
    cat = json.loads(CATALOG.read_text(encoding="utf-8"))
    scenes = cat.get("scenes") or []
    gt = Counter()
    fixture_hashes = {}
    for s in scenes:
        for inst in s.get("instances") or []:
            gt[inst.get("canonical_label") or inst.get("label") or inst.get("raw_label")] += 1
        p = Path((s.get("image") or {}).get("path") or "")
        if p.is_file():
            fixture_hashes[s.get("scene_id")] = sha(p)
    info["benchmark"] = {
        "catalog": str(CATALOG),
        "catalog_sha256": sha(CATALOG),
        "n_scenes": len(scenes),
        "n_instances": sum(gt.values()),
        "gt_labels": dict(gt.most_common()),
        "fixture_images_hashed": len(fixture_hashes),
        "instance_keys": sorted((scenes[0].get("instances") or [{}])[0].keys()) if scenes else [],
    }
    lib_urls = {i.get("source_url") or i.get("image_url") for i in items}
    lib_hashes = {i.get("image_sha256") for i in items if i.get("image_sha256")}
    info["leakage"] = {
        "fixture_hash_in_library": sorted(k for k, v in fixture_hashes.items() if v in lib_hashes),
        "fixture_path_in_library_urls": sorted(
            k for k in fixture_hashes if any(k in str(u or "") for u in lib_urls)
        ),
    }
    for name in [
        "docs/vision/production_identity_training_manifest_v1.json",
        "docs/vision/production_identity_coverage_v1.json",
    ]:
        p = ROOT / name
        if p.is_file():
            d = json.loads(p.read_text(encoding="utf-8"))
            info[name] = {
                "sha256": sha(p),
                "top_keys": list(d.keys())[:15] if isinstance(d, dict) else type(d).__name__,
                "n_identities": len(d.get("identities") or []) if isinstance(d, dict) else None,
                "n_rows": len(d.get("rows") or d.get("images") or []) if isinstance(d, dict) else None,
            }
            if "identities" in d:
                st = Counter(r.get("status") for r in d["identities"])
                info[name]["status_counts"] = dict(st)
                info[name]["total_production_images"] = sum(
                    int(r.get("production_image_count") or 0) for r in d["identities"]
                )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
    print(json.dumps(info, indent=2, default=str)[:6000])


if __name__ == "__main__":
    main()
