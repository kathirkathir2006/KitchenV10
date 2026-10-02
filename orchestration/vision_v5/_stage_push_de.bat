@echo off
rem usage: _stage_push_de.bat <adapter_d|adapter_e|test_de> <library_sha> <c_config_sha> [d_config_sha] [e_config_sha]
set PYTHONIOENCODING=utf-8
cd /d C:\Projects\KitchenIQ-V10-AI
rem run orchestration\vision_v5\_selftest_de.bat first (unit tests + synthetic train/resume smoke)
python orchestration\vision_v5\make_stage_kernels.py %1 %2 %3 %4 %5 || exit /b 1
if "%1"=="adapter_d" set SLUG=kiq-v5-adapter-d
if "%1"=="adapter_e" set SLUG=kiq-v5-adapter-e
if "%1"=="test_de" set SLUG=kiq-v5-adapter-de-test
python -m py_compile kaggle\kernels\%SLUG%\kiq_v5_stage.py || exit /b 1
git add kaggle/v5_src kaggle/kernels/%SLUG% orchestration/vision_v5 docs/v5
git commit -q -m "V5 stage %1: kernel source for Kaggle run."
git push -q origin HEAD
for /f %%c in ('git rev-parse HEAD') do set COMMIT=%%c
C:\Users\kathi\AppData\Roaming\Python\Python313\Scripts\kaggle.exe kernels push -p kaggle\kernels\%SLUG% > kaggle\outputs\push_%SLUG%.txt 2>&1
type kaggle\outputs\push_%SLUG%.txt
echo %date% %time% mode=%1 slug=%SLUG% commit=%COMMIT% lib=%2 c_cfg=%3 d_cfg=%4 e_cfg=%5>> docs\v5\RUN_REGISTRY.log
type kaggle\outputs\push_%SLUG%.txt>> docs\v5\RUN_REGISTRY.log
