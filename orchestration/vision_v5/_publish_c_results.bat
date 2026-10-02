@echo off
cd /d C:\Projects\KitchenIQ-V10-AI
set D=docs\v5\adapter_c_v1_eval
if not exist %D% mkdir %D%
copy /y kaggle\outputs\kiq-v5-adapter-c-test\v5_test\test_results.json %D%\test_results.json >nul
copy /y kaggle\outputs\kiq-v5-adapter-c-test\v5_test\test_per_item.json %D%\test_per_item.json >nul
copy /y backend\instance\dev_experiments\v10-vision-v5\v5_exp_C_41scene_001\results.json %D%\41scene_results.json >nul
copy /y backend\instance\dev_experiments\v10-vision-v5\v5_exp_C_41scene_001\per_region.json %D%\41scene_per_region.json >nul
git add docs\v5\V5_ADAPTER_C_RESULTS.md %D% orchestration\vision_v5\eval_41_frozen.py orchestration\vision_v5\_publish_c_results.bat
git commit -q -m "V5: Adapter C v1 test + 41-scene comparison report"
git push -q origin HEAD
git log -1 --format=%%H
