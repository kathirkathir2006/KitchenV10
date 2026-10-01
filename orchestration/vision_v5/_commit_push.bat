@echo off
cd /d C:\Projects\KitchenIQ-V10-AI
git add docs/v5 orchestration/vision_v5 kaggle/kernels/kiq-v5-real-photo-v1 kaggle/datasets/kiq-v5-benchmark-fingerprints
git commit -q -m "V5: frozen real-photo set v1 with provenance, leakage and split guarantees."
if errorlevel 1 exit /b 1
git push -q origin HEAD
git rev-parse --short HEAD
