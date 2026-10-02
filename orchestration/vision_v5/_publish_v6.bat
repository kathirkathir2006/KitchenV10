@echo off
cd /d C:\Projects\KitchenIQ-V10-AI
python -c "import hashlib,pathlib;d=pathlib.Path('docs/v6/reranker_v1');[print(hashlib.sha256(p.read_bytes()).hexdigest(),p.name) for p in sorted(d.iterdir()) if p.is_file()]"
git add -f docs\v6 kaggle\v6_src orchestration\vision_v5\_publish_v6.bat
git commit -q -m "V6: candidate reranker over frozen Adapter E top-5 (train/val only; test not opened)"
git push -q origin HEAD
git log -1 --format=%%H
