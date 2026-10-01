@echo off
cd /d C:\Projects\KitchenIQ-V10-AI
git add docs/v5/real_photo_set_v1/.gitattributes
git add --renormalize docs/v5/real_photo_set_v1
git commit -q -m "Keep frozen real-photo v1 manifest byte-exact across checkouts."
if errorlevel 1 exit /b 1
git push -q origin HEAD
git rev-parse --short HEAD
git show HEAD:docs/v5/real_photo_set_v1/manifest.json > %TEMP%\kiq_manifest_check.json
certutil -hashfile %TEMP%\kiq_manifest_check.json SHA256
