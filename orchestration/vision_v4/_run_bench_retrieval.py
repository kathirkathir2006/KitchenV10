"""Reliable V4 retrieval benchmark with crash capture."""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

NEW_ROOT = Path(r"C:\Projects\KitchenIQ-V10-AI")
sys.path.insert(0, str(NEW_ROOT / "src"))
sys.path.insert(0, str(NEW_ROOT))

LOG = NEW_ROOT / "backend/instance/dev_experiments/v10-vision-v4/reports/bench5_crash.txt"
OUT = NEW_ROOT / "backend/instance/dev_experiments/v10-vision-v4/reports/bench5_progress.txt"


def prog(msg: str) -> None:
    line = msg + "\n"
    print(msg, flush=True)
    with OUT.open("a", encoding="utf-8") as f:
        f.write(line)


def main() -> None:
    OUT.write_text("", encoding="utf-8")
    try:
        prog("start")
        from orchestration.vision_v4.run_benchmark_and_finalize import find_library, main as bench_main

        lib = find_library()
        prog(f"library={lib}")
        if lib is None:
            raise RuntimeError("library missing; refuse meaningless no-retrieval rerun")
        import json

        d = json.loads(lib.read_text(encoding="utf-8"))
        n = int(d.get("n") or len(d.get("items") or []))
        prog(f"n_items={n}")
        assert n > 0
        prog("calling bench_main")
        bench_main()
        prog("bench_main done")
    except Exception:
        tb = traceback.format_exc()
        LOG.write_text(tb, encoding="utf-8")
        prog("CRASH")
        print(tb, flush=True)
        raise


if __name__ == "__main__":
    main()
