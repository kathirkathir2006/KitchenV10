@echo off
rem usage: _stage_push.bat <mode> [library_sha] [adapter_config_sha]
set PYTHONIOENCODING=utf-8
cd /d C:\Projects\KitchenIQ-V10-AI
python -m py_compile kaggle\v5_src\kiq_v5_stage.py || exit /b 1
python orchestration\vision_v5\make_stage_kernels.py %1 %2 %3 || exit /b 1
git add kaggle/v5_src kaggle/kernels orchestration/vision_v5 docs/v5
git commit -q -m "V5 stage %1: kernel source for Kaggle run."
git push -q origin HEAD
for /f %%c in ('git rev-parse HEAD') do set COMMIT=%%c
if "%1"=="reflib" set SLUG=kiq-v5-reflib-v1
if "%1"=="adapter" set SLUG=kiq-v5-adapter-c
if "%1"=="test" set SLUG=kiq-v5-adapter-c-test
C:\Users\kathi\AppData\Roaming\Python\Python313\Scripts\kaggle.exe kernels push -p kaggle\kernels\%SLUG% > kaggle\outputs\push_%SLUG%.txt 2>&1
type kaggle\outputs\push_%SLUG%.txt
echo %date% %time% mode=%1 slug=%SLUG% commit=%COMMIT% lib=%2 cfg=%3>> docs\v5\RUN_REGISTRY.log
type kaggle\outputs\push_%SLUG%.txt>> docs\v5\RUN_REGISTRY.log
