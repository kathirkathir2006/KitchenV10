@echo off
set PYTHONIOENCODING=utf-8
cd /d C:\Projects\KitchenIQ-V10-AI
python -c "from pathlib import Path; p=Path('kaggle/v5_src/kiq_v5_de.py'); b=p.read_bytes(); p.write_bytes(b[3:] if b[:3]==b'\xef\xbb\xbf' else b)"
python orchestration\vision_v5\make_stage_kernels.py selftest || exit /b 1
python -m py_compile kaggle\v5_selftest\kiq_v5_stage.py || exit /b 1
python kaggle\v5_selftest\kiq_v5_stage.py
