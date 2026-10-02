"""Freeze Adapter D or E config + checkpoint into docs/v5/adapter_<x>_v1 (refuses to overwrite).

usage: python freeze_de_v1.py d|e <kaggle_version>
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(r"C:\Projects\KitchenIQ-V10-AI")
LIB_SHA = "7e95ee5636af0a14b2839b1a242699bc02635b5d0956c6d6e89a5b9019fc0a02"
PHOTO_SHA = "89ea9b20effdf40983ce670722f43fce42e44bd4378cc5810853dbba94c656e0"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    x, version = sys.argv[1], sys.argv[2]
    name = f"adapter_{x}"
    src = ROOT / f"kaggle/outputs/kiq-v5-adapter-{x}/v5_{name}"
    dst = ROOT / f"docs/v5/{name}_v1"
    cfg_p = src / f"{name}_frozen_config.json"
    cfg = json.loads(cfg_p.read_bytes())
    ck = src / cfg["adapter"]["checkpoint"]
    rep = json.loads((src / f"{name}_selection_report.json").read_bytes())
    checks = {"frozen_flag": cfg["frozen"] is True, "test_not_used": cfg["test_used"] is False,
              "report_test_not_used": rep["test_used"] is False, "selection_on_val": cfg["selection_split"] == "val",
              "library_sha": cfg["library_manifest_sha256"] == LIB_SHA, "photo_sha": cfg["photo_manifest_sha256"] == PHOTO_SHA,
              "checkpoint_sha": sha(ck) == cfg["adapter"]["checkpoint_sha256"],
              "report_points_to_config": rep["frozen_config_sha256"] == sha(cfg_p),
              "unit_tests_pass": rep["unit_tests"]["all_pass"] is True,
              "no_test_loaded": rep["data_checks"]["11_no_test_loaded"] is True}
    print(json.dumps(checks, indent=2))
    if not all(checks.values()):
        raise SystemExit("VERIFY FAILED")
    if (dst / "FROZEN.json").exists():
        assert json.loads((dst / "FROZEN.json").read_text())["config_sha256"] == sha(cfg_p), "different config frozen"
        print("already frozen")
        return
    dst.mkdir(parents=True, exist_ok=True)
    (dst / ".gitattributes").write_text("* -text\n", encoding="utf-8")
    files = [f"{name}_frozen_config.json", f"{name}_selection_report.json", cfg["adapter"]["checkpoint"],
             f"{name}_train_hard_negatives.json", f"{name}_unit_tests.json"]
    for n in files:
        shutil.copy2(src / n, dst / n)
    (dst / "FROZEN.json").write_text(json.dumps({
        "config_id": cfg["config_id"], "config_sha256": sha(cfg_p), "checkpoint_sha256": sha(ck),
        "library_manifest_sha256": LIB_SHA, "photo_manifest_sha256": PHOTO_SHA,
        "kaggle_run": f"kathiresannatarajan/kiq-v5-adapter-{x} version {version}",
        "winner": {k: cfg["adapter"][k] for k in ("cfg", "cfg_index", "epoch", "thresholds", "val_metrics")},
        "safety_gate": cfg.get("safety_gate"), "checks": checks}, indent=2), encoding="utf-8")
    print("FROZEN", sha(cfg_p), "CKPT", sha(ck))


if __name__ == "__main__":
    main()
