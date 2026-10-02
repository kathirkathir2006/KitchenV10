@echo off
rem usage: pull_stage_light.bat <slug>   (top-level json/txt/log + *_final.pt; skips per-epoch checkpoints and caches)
set PYTHONIOENCODING=utf-8
cd /d C:\Projects\KitchenIQ-V10-AI
C:\Users\kathi\AppData\Roaming\Python\Python313\Scripts\kaggle.exe kernels output kathiresannatarajan/%1 -p kaggle\outputs\%1 --file-pattern "^(v5_[a-z_]+/[^/]+\.(json|txt|log)|v5_[a-z_]+/[a-z_]+_final\.pt|[^/]+\.log)$" > kaggle\outputs\pull_%1.txt 2>&1
type kaggle\outputs\pull_%1.txt | findstr /v /c:"Output file downloaded"
dir /s /b kaggle\outputs\%1
