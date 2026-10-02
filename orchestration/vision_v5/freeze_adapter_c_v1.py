"""Freeze Adapter C config + checkpoint into docs/v5/adapter_c_v1 (refuses to overwrite)."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(r"C:\Projects\KitchenIQ-V10-AI")
SRC = ROOT / "kaggle/outputs/kiq-v5-adapter-c/v5_adapter"
DST = ROOT / "docs/v5/adapter_c_v1"
LIB_SHA = "7e95ee5636af0a14b2839b1a242699bc02635b5d0956c6d6e89a5b9019fc0a02"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    cfg_p = SRC / "adapter_c_frozen_config.json"
    cfg = json.loads(cfg_p.read_bytes())
    ck = SRC / cfg["adapter"]["checkpoint"]
    checks = {"frozen_flag": cfg["frozen"] is True, "test_not_used": cfg["test_used"] is False,
              "selection_on_val": cfg["selection_split"] == "val",
              "library_sha": cfg["library_manifest_sha256"] == LIB_SHA,
              "checkpoint_sha": sha(ck) == cfg["adapter"]["checkpoint_sha256"]}
    print(json.dumps(checks, indent=2))
    if not all(checks.values()):
        raise SystemExit("VERIFY FAILED")
    if (DST / "FROZEN.json").exists():
        assert json.loads((DST / "FROZEN.json").read_text())["config_sha256"] == sha(cfg_p), "different config frozen"
        print("already frozen")
        return
    DST.mkdir(parents=True, exist_ok=True)
    (DST / ".gitattributes").write_text("* -text\n", encoding="utf-8")
    for n in ("adapter_c_frozen_config.json", "adapter_c_selection_report.json", cfg["adapter"]["checkpoint"]):
        shutil.copy2(SRC / n, DST / n)
    (DST / "FROZEN.json").write_text(json.dumps({
        "config_id": cfg["config_id"], "config_sha256": sha(cfg_p), "checkpoint_sha256": sha(ck),
        "library_manifest_sha256": LIB_SHA, "kaggle_run": "kathiresannatarajan/kiq-v5-adapter-c version 1",
        "winner_cfg": cfg["adapter"], "baseline_thresholds": cfg["baseline"]["thresholds"], "checks": checks},
        indent=2), encoding="utf-8")
    print("FROZEN", sha(cfg_p))


if __name__ == "__main__":
    main()
