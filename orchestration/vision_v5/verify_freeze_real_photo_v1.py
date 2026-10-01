"""Independently verify the real-photo v1 manifest, then freeze it in docs/v5 (refuses to overwrite)."""
from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(r"C:\Projects\KitchenIQ-V10-AI")
SRC = ROOT / "kaggle/outputs/kiq-v5-real-photo-v1/kiq_v5_real_photo_v1"
DST = ROOT / "docs/v5/real_photo_set_v1"
LIB = ROOT / "backend/instance/dev_experiments/v10-vision-v4/reference_library/reference_library.json"
CATALOG_SHA = "38fc3eaee75c76fa5fd1dc3d4177af4abde2cfeb3e410d290fb524f9e30a1523"


def main() -> None:
    raw = (SRC / "manifest.json").read_bytes()
    msha = hashlib.sha256(raw).hexdigest()
    report = json.loads((SRC / "build_report.json").read_text(encoding="utf-8"))
    m = json.loads(raw)
    items = m["items"]
    checks: dict[str, bool] = {}
    checks["manifest_sha_matches_build_report"] = msha == report["manifest_sha256"]
    checks["benchmark_catalog_sha_matches"] = m["benchmark_catalog_sha256"] == CATALOG_SHA
    checks["n_matches"] = len(items) == m["n"] == report["n"]
    checks["licences_only_cc0_by"] = all(i["licence"] in {"cc0", "by"} for i in items)
    checks["every_item_has_licence_url_landing_creator_field"] = all(
        i.get("licence_url") and i.get("landing_url") and "creator" in i for i in items)
    checks["unique_sha256"] = len({i["sha256"] for i in items}) == len(items)
    checks["unique_openverse_id"] = len({i["openverse_id"] for i in items}) == len(items)
    g = defaultdict(set)
    for i in items:
        g[i["group"]].add(i["split"])
    checks["no_group_spans_splits"] = all(len(v) == 1 for v in g.values())
    checks["no_open_set_in_train"] = not any(i["role"] == "open_set_unknown" and i["split"] == "train" for i in items)
    checks["benchmark_max_cos_below_0.92"] = all(i["verification"]["benchmark_max_cos"] < 0.92 for i in items)
    checks["benchmark_min_dhash_above_8"] = all(i["verification"]["benchmark_min_dhash"] > 8 for i in items)
    lib_hashes = {it.get("source_hash") for it in json.loads(LIB.read_text(encoding="utf-8"))["items"]}
    checks["no_sha_overlap_with_108_library"] = not ({i["sha256"] for i in items} & lib_hashes)
    checks["val_test_clip_top1_and_keyword"] = all(
        i["verification"]["clip_top1"] and i["verification"]["keyword_match"] for i in items if i["split"] != "train")
    failed = [k for k, v in checks.items() if not v]
    summary = {
        "set_id": m["set_id"], "manifest_sha256": msha, "n": len(items),
        "split_totals": dict(Counter(i["split"] for i in items)),
        "licences": dict(Counter(i["licence"] for i in items)),
        "providers": dict(Counter(i.get("provider") for i in items).most_common(10)),
        "n_groups": len(g), "checks": checks, "failed": failed,
        "images_location": "Kaggle kernel output kathiresannatarajan/kiq-v5-real-photo-v1 (version 3)",
        "consumer_rule": "re-hash every image against manifest sha256 and abort on any mismatch; never rebuild v1",
    }
    print(json.dumps(summary, indent=2))
    if failed:
        raise SystemExit(f"VERIFY FAILED: {failed}")
    if (DST / "FROZEN.json").exists():
        frozen = json.loads((DST / "FROZEN.json").read_text(encoding="utf-8"))
        assert frozen["manifest_sha256"] == msha, "a different v1 is already frozen — refusing to overwrite"
        print("already frozen with identical manifest")
        return
    DST.mkdir(parents=True, exist_ok=True)
    for name in ("manifest.json", "build_report.json", "ATTRIBUTION.md"):
        shutil.copy2(SRC / name, DST / name)
    (DST / "FROZEN.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("FROZEN", msha)


if __name__ == "__main__":
    main()
