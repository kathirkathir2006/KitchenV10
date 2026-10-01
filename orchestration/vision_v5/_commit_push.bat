@echo off
cd /d C:\Projects\KitchenIQ-V10-AI
git add docs/v5 orchestration/vision_v5 backend/instance/dev_experiments/v10-vision-v5
git commit -q -m "V5 readiness: experiments A/B and benchmark domain diagnosis."
if errorlevel 1 exit /b 1
git push -q origin HEAD
git rev-parse --short HEAD
