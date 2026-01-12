@echo off
cd /d "%~dp0"

call D:\miniconda3\Scripts\activate.bat D:\miniconda3
call conda activate ocr

if exist "build" rmdir /s /q "build"
if exist "dist\ocrapp_pureray.exe" del /q "dist\ocrapp_pureray.exe"

python setup.py build_ext --inplace

pyinstaller ocrapp_pureray.spec

if exist "settings" copy /y "settings" "dist\settings"

call conda deactivate

pause
