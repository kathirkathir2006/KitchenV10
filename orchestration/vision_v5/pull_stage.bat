@echo off
rem usage: pull_stage.bat <slug>   (pulls only json/md/txt/log/pt; never images)
set PYTHONIOENCODING=utf-8
cd /d C:\Projects\KitchenIQ-V10-AI
C:\Users\kathi\AppData\Roaming\Python\Python313\Scripts\kaggle.exe kernels output kathiresannatarajan/%1 -p kaggle\outputs\%1 --file-pattern ".*\.(json|md|txt|log|pt)$" > kaggle\outputs\pull_%1.txt 2>&1
dir /s /b kaggle\outputs\%1
