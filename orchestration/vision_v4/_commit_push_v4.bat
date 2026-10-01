@echo off
cd /d C:\Projects\KitchenIQ-V10-AI
git add docs/v4 orchestration/vision_v4 src/kitcheniq_v10/v4 kaggle/kernels/kiq-v10-vision-v4
git add backend/instance/dev_experiments/v10-vision-v4/reference_library/reference_library.json
git add backend/instance/dev_experiments/v10-vision-v4/reports/reference_library_summary.json
git add backend/instance/dev_experiments/v10-vision-v4/reports/KITCHENIQ_VISION_V4_FINAL_STATE.json
git add backend/instance/dev_experiments/v10-vision-v4/reports/KITCHENIQ_VISION_V4_FINAL_REPORT.md
git add backend/instance/dev_experiments/v10-vision-v4/acceptance/candidate_v4_retrieval/v4_open_world_scene_retrieval_acceptance.json
git add backend/instance/dev_experiments/v10-vision-v4/acceptance/v4_summary.json
git add backend/instance/dev_experiments/v10-vision-v4/acceptance/candidate_catalog.json
git add kaggle/outputs/kathiresannatarajan__kiq-v10-vision-v4/v10-vision-v4/reference_library.json
git add kaggle/outputs/kathiresannatarajan__kiq-v10-vision-v4/v10-vision-v4/reports/reference_library_summary.json
git add kaggle/outputs/kathiresannatarajan__kiq-v10-vision-v4/v10-vision-v4/LIVE_STATUS.txt
git status --short > backend\instance\dev_experiments\v10-vision-v4\reports\staged.txt
git commit -m "Vision V4 forensics, open-world scene engine, and blocked 41-scene benchmark."
if errorlevel 1 exit /b 1
git rev-parse HEAD > backend\instance\dev_experiments\v10-vision-v4\reports\commit.txt
git push origin HEAD > backend\instance\dev_experiments\v10-vision-v4\reports\push.txt 2>&1
type backend\instance\dev_experiments\v10-vision-v4\reports\commit.txt
type backend\instance\dev_experiments\v10-vision-v4\reports\push.txt
exit /b 0
