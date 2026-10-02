@echo off
cd /d C:\Projects\KitchenIQ-V10-AI
if not exist docs\v7\test\.gitattributes echo * -text> docs\v7\test\.gitattributes
python -c "import hashlib,pathlib;d=pathlib.Path('docs/v7/test');[print(hashlib.sha256(p.read_bytes()).hexdigest(),p.name) for p in sorted(d.iterdir()) if p.is_file()]"
git add -f docs\v7\test kaggle\v7_src orchestration\vision_v5\_publish_v7_test.bat docs\v5\RUN_REGISTRY.log
git commit -q -m "V7: one-shot test - NOT CONFIRMED (+0.0043 top-1, 3 fixed / 2 broken)"
git push -q origin HEAD
git log -1 --format=%%H
