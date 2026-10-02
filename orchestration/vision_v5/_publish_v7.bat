@echo off
cd /d C:\Projects\KitchenIQ-V10-AI
if not exist docs\v7\.gitattributes echo * -text> docs\v7\.gitattributes
python -c "import hashlib,pathlib;d=pathlib.Path('docs/v7');[print(hashlib.sha256(p.read_bytes()).hexdigest(),p.name) for p in sorted(d.iterdir()) if p.is_file()]"
git add -f docs\v7 kaggle\v7_src orchestration\vision_v5\_publish_v7.bat orchestration\vision_v5\_v7_summary.py
git commit -q -m "V7: pairwise specialists over frozen Adapter E (train/val only; test not opened)"
git push -q origin HEAD
git log -1 --format=%%H
