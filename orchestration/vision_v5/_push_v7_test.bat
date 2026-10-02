@echo off
set PYTHONIOENCODING=utf-8
cd /d C:\Projects\KitchenIQ-V10-AI
python orchestration\vision_v5\make_v7_test_kernel.py || exit /b 1
python -m py_compile kaggle\kernels\kiq-v7-test-embed\kiq_v5_stage.py || exit /b 1
git add kaggle/v7_src kaggle/kernels/kiq-v7-test-embed orchestration/vision_v5
git commit -q -m "V7: one-shot test embedding kernel (V7 frozen c29079096ed2)"
git push -q origin HEAD
for /f %%c in ('git rev-parse HEAD') do set COMMIT=%%c
C:\Users\kathi\AppData\Roaming\Python\Python313\Scripts\kaggle.exe kernels push -p kaggle\kernels\kiq-v7-test-embed > kaggle\outputs\push_kiq-v7-test-embed.txt 2>&1
type kaggle\outputs\push_kiq-v7-test-embed.txt
echo %date% %time% mode=v7_test_embed slug=kiq-v7-test-embed commit=%COMMIT% v7_cfg=c29079096ed20ecd78d90bf0b29d86201241f705757d152f1dcef915c1b297b5>> docs\v5\RUN_REGISTRY.log
type kaggle\outputs\push_kiq-v7-test-embed.txt>> docs\v5\RUN_REGISTRY.log
