@echo off
set PYTHONIOENCODING=utf-8
cd /d C:\Projects\KitchenIQ-V10-AI
C:\Users\kathi\AppData\Roaming\Python\Python313\Scripts\kaggle.exe kernels output kathiresannatarajan/kiq-v5-real-photo-v1 -p kaggle\outputs\kiq-v5-real-photo-v1 --file-pattern ".*\.(json|md|txt)$" > kaggle\outputs\v5rp_pull.txt 2>&1
dir /s /b kaggle\outputs\kiq-v5-real-photo-v1
