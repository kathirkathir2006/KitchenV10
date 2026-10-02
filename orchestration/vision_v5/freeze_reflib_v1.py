"""Verify and freeze reference library v1 into docs/v5/reference_library_v1 (refuses to overwrite)."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(r"C:\Projects\KitchenIQ-V10-AI")
SRC = ROOT / "kaggle/outputs/kiq-v5-reflib-v1/v5_reflib"
DST = ROOT / "docs/v5/reference_library_v1"
PHOTO = ROOT / "docs/v5/real_photo_set_v1/manifest.json"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    man_p = SRC / "reference_library_v1_manifest.json"
    lib = json.loads(man_p.read_bytes())
    photo = json.loads(PHOTO.read_bytes())
    by_sha = {i["sha256"]: i for i in photo["items"]}
    checks = {
        "source_manifest_sha": lib["source_manifest_sha256"] == sha(PHOTO),
        "all_refs_in_photo_set": all(r["sha256"] in by_sha for r in lib["items"]),
        "all_refs_train_split": all(by_sha[r["sha256"]]["split"] == "train" for r in lib["items"]),
        "no_open_set_refs": all(by_sha[r["sha256"]]["role"] == "known" for r in lib["items"]),
        "unique_sha": len({r["sha256"] for r in lib["items"]}) == lib["n"],
        "unique_ref_id": len({r["ref_id"] for r in lib["items"]}) == lib["n"],
        "metadata_matches_photo_set": all(
            r["label"] == by_sha[r["sha256"]]["label"] and r["licence"] == by_sha[r["sha256"]]["licence"]
            for r in lib["items"]),
    }
    emb = json.loads((SRC / "reference_library_v1_dinov2_embeddings.json").read_bytes())
    checks["embeddings_bound_to_manifest"] = emb["library_manifest_sha256"] == sha(man_p)
    checks["embeddings_cover_all_refs"] = set(emb["embeddings"]) == {r["ref_id"] for r in lib["items"]}
    print(json.dumps(checks, indent=2))
    if not all(checks.values()):
        raise SystemExit("VERIFY FAILED")
    if (DST / "FROZEN.json").exists():
        assert json.loads((DST / "FROZEN.json").read_text())["manifest_sha256"] == sha(man_p), "different v1 already frozen"
        print("already frozen")
        return
    DST.mkdir(parents=True, exist_ok=True)
    (DST / ".gitattributes").write_text("* -text\n", encoding="utf-8")
    for n in ("reference_library_v1_manifest.json", "reference_library_v1_dinov2_embeddings.json"):
        shutil.copy2(SRC / n, DST / n)
    (DST / "FROZEN.json").write_text(json.dumps({
        "library_id": lib["library_id"], "manifest_sha256": sha(man_p),
        "embeddings_sha256": sha(SRC / "reference_library_v1_dinov2_embeddings.json"), "n": lib["n"],
        "n_labels": len(lib["label_stats"]), "kaggle_run": "kathiresannatarajan/kiq-v5-reflib-v1 version 1",
        "checks": checks}, indent=2), encoding="utf-8")
    print("FROZEN", sha(man_p))


if __name__ == "__main__":
    main()
