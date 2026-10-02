"""Generate per-stage Kaggle kernel folders from kaggle/v5_src/kiq_v5_stage.py.

usage: python make_stage_kernels.py <mode> [library_sha] [adapter_config_sha] [d_config_sha] [e_config_sha]

Historical modes (reflib, adapter, test) are generated exactly as before. D/E modes (adapter_d, adapter_e,
test_de, selftest) = kiq_v5_stage.py shared code (everything above its __main__ block, unchanged) +
kaggle/v5_src/kiq_v5_de.py.
"""
from __future__ import annotations

import ast
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(r"C:\Projects\KitchenIQ-V10-AI")
SRC = ROOT / "kaggle/v5_src/kiq_v5_stage.py"
DE_SRC = ROOT / "kaggle/v5_src/kiq_v5_de.py"
PHOTO = "kathiresannatarajan/kiq-v5-real-photo-v1"
REFLIB = "kathiresannatarajan/kiq-v5-reflib-v1"
C_RUN = "kathiresannatarajan/kiq-v5-adapter-c"
SOURCES = {
    "reflib": [PHOTO],
    "adapter": [PHOTO, REFLIB],
    "test": [PHOTO, REFLIB, C_RUN],
    "adapter_d": [PHOTO, REFLIB, C_RUN],
    "adapter_e": [PHOTO, REFLIB, C_RUN, "kathiresannatarajan/kiq-v5-adapter-d"],
    "test_de": [PHOTO, REFLIB, C_RUN, "kathiresannatarajan/kiq-v5-adapter-d", "kathiresannatarajan/kiq-v5-adapter-e"],
}
SLUG = {"reflib": "kiq-v5-reflib-v1", "adapter": "kiq-v5-adapter-c", "test": "kiq-v5-adapter-c-test",
        "adapter_d": "kiq-v5-adapter-d", "adapter_e": "kiq-v5-adapter-e", "test_de": "kiq-v5-adapter-de-test"}
DE_MODES = {"adapter_d", "adapter_e", "test_de", "selftest"}
SCORER_FUNCS = ("label_scores", "decide", "objective", "calibrate", "detail_metrics", "earliest_divergence")
MAIN = 'if __name__ == "__main__":'


def scorer_sha(src: str) -> str:
    tree = ast.parse(src)
    seg = {n.name: ast.get_source_segment(src, n) for n in tree.body if isinstance(n, ast.FunctionDef)}
    return hashlib.sha256("\n\n".join(seg[n].strip() for n in SCORER_FUNCS).encode("utf-8")).hexdigest()


def build_de(mode: str, lib_sha: str, cfg_sha: str, d_sha: str, e_sha: str) -> str:
    base = SRC.read_text(encoding="utf-8")
    shared = base[:base.index(MAIN)]
    code = shared.replace("__MODE__", mode).replace("__LIBRARY_SHA__", lib_sha).replace("__ADAPTER_CONFIG_SHA__", cfg_sha)
    code += DE_SRC.read_text(encoding="utf-8").replace("__D_CONFIG_SHA__", d_sha).replace(
        "__E_CONFIG_SHA__", e_sha).replace("__SCORER_SRC_SHA__", scorer_sha(base))
    assert scorer_sha(code) == scorer_sha(base), "shared scorer code changed while assembling"
    return code


def main() -> None:
    mode = sys.argv[1]
    lib_sha = sys.argv[2] if len(sys.argv) > 2 else "UNSET"
    cfg_sha = sys.argv[3] if len(sys.argv) > 3 else "UNSET"
    d_sha = sys.argv[4] if len(sys.argv) > 4 else "UNSET"
    e_sha = sys.argv[5] if len(sys.argv) > 5 else "UNSET"
    if mode in {"adapter", "test", "adapter_d", "adapter_e", "test_de"}:
        assert len(lib_sha) == 64, "frozen library sha required"
    if mode in {"test", "test_de"}:
        assert len(cfg_sha) == 64, "frozen adapter config sha required"
    if mode in {"adapter_e", "test_de"}:
        assert len(d_sha) == 64, "frozen Adapter D config sha required"
    if mode == "test_de":
        assert len(e_sha) == 64, "frozen Adapter E config sha required"
    if mode == "selftest":
        code = build_de(mode, "__LIBRARY_SHA__", cfg_sha, d_sha, e_sha).replace(
            'Path("/kaggle/working")', 'Path(__file__).resolve().parent / "work"')
        out = ROOT / "kaggle/v5_selftest"
        out.mkdir(parents=True, exist_ok=True)
        (out / "kiq_v5_stage.py").write_text(code, encoding="utf-8")
        print(out)
        return
    if mode in DE_MODES:
        code = build_de(mode, lib_sha, cfg_sha, d_sha, e_sha)
    else:
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
