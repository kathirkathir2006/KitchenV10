@echo off
cd /d C:\Projects\KitchenIQ-V10-AI
set D=docs\v5\adapter_de_v1_eval
if not exist %D% mkdir %D%
copy /y kaggle\outputs\kiq-v5-adapter-de-test\v5_test_de\*.json %D%\ >nul
copy /y backend\instance\dev_experiments\v10-vision-v5\v5_exp_DE_41scene_001\results.json %D%\41scene_results.json >nul
copy /y backend\instance\dev_experiments\v10-vision-v5\v5_exp_DE_41scene_001\per_region.json %D%\41scene_per_region.json >nul
git add docs\v5 orchestration\vision_v5 kaggle\v5_src kaggle\kernels\kiq-v5-adapter-de-test
git commit -q -m "V5: Adapter D/E frozen, one-shot test + 41-scene comparison report"
git push -q origin HEAD
git log -1 --format=%%H
