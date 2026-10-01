"""Generate per-stage Kaggle kernel folders from kaggle/v5_src/kiq_v5_stage.py.

usage: python make_stage_kernels.py <mode> [library_sha] [adapter_config_sha]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(r"C:\Projects\KitchenIQ-V10-AI")
SRC = ROOT / "kaggle/v5_src/kiq_v5_stage.py"
SOURCES = {
    "reflib": ["kathiresannatarajan/kiq-v5-real-photo-v1"],
    "adapter": ["kathiresannatarajan/kiq-v5-real-photo-v1", "kathiresannatarajan/kiq-v5-reflib-v1"],
    "test": ["kathiresannatarajan/kiq-v5-real-photo-v1", "kathiresannatarajan/kiq-v5-reflib-v1",
             "kathiresannatarajan/kiq-v5-adapter-c"],
}
SLUG = {"reflib": "kiq-v5-reflib-v1", "adapter": "kiq-v5-adapter-c", "test": "kiq-v5-adapter-c-test"}


def main() -> None:
    mode = sys.argv[1]
    lib_sha = sys.argv[2] if len(sys.argv) > 2 else "UNSET"
    cfg_sha = sys.argv[3] if len(sys.argv) > 3 else "UNSET"
    if mode in {"adapter", "test"}:
        assert len(lib_sha) == 64, "frozen library sha required"
    if mode == "test":
        assert len(cfg_sha) == 64, "frozen adapter config sha required"
    code = SRC.read_text(encoding="utf-8").replace("__MODE__", mode).replace(
        "__LIBRARY_SHA__", lib_sha).replace("__ADAPTER_CONFIG_SHA__", cfg_sha)
    out = ROOT / "kaggle/kernels" / SLUG[mode]
    out.mkdir(parents=True, exist_ok=True)
    (out / "kiq_v5_stage.py").write_text(code, encoding="utf-8")
    (out / "kernel-metadata.json").write_text(json.dumps({
        "id": f"kathiresannatarajan/{SLUG[mode]}", "title": SLUG[mode], "code_file": "kiq_v5_stage.py",
        "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": True,
        "enable_internet": True, "dataset_sources": [], "kernel_sources": SOURCES[mode],
        "competition_sources": [], "model_sources": []}, indent=2), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
