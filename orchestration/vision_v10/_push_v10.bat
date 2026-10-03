@echo off
set PYTHONIOENCODING=utf-8
cd /d C:\Projects\KitchenIQ-V10-AI
python orchestration\vision_v10\make_v10_kernel.py || exit /b 1
python -m py_compile kaggle\kernels\kiq-v10-val-embed\kiq_v10_val_embed.py || exit /b 1
for /f %%p in ('python -c "import sys;sys.path.insert(0,'kaggle/v10_src');import kiq_v10_regions as r;print(r.protocol_sha())"') do set PSHA=%%p
git add kaggle/v10_src kaggle/kernels/kiq-v10-val-embed orchestration/vision_v10 artifacts/v10
git commit -q -m "V10-VR M1-M3: frozen protocol %PSHA% (val-only region embedding kernel, coverage audit)"
git push -q origin HEAD
for /f %%c in ('git rev-parse HEAD') do set COMMIT=%%c
C:\Users\kathi\AppData\Roaming\Python\Python313\Scripts\kaggle.exe kernels push -p kaggle\kernels\kiq-v10-val-embed > kaggle\outputs\push_kiq-v10-val-embed.txt 2>&1
type kaggle\outputs\push_kiq-v10-val-embed.txt
echo %date% %time% mode=v10_val_embed slug=kiq-v10-val-embed commit=%COMMIT% protocol=%PSHA%>> docs\v5\RUN_REGISTRY.log
type kaggle\outputs\push_kiq-v10-val-embed.txt>> docs\v5\RUN_REGISTRY.log
