@echo off
setlocal
cd /d "%~dp0"

set "CONDA_ROOT=D:\miniconda3"
set "CONDA_ENV=ocr"

if exist "%CONDA_ROOT%\Scripts\activate.bat" (
  call "%CONDA_ROOT%\Scripts\activate.bat" "%CONDA_ROOT%"
  call conda activate "%CONDA_ENV%"
)

if exist "build" rmdir /s /q "build"
if exist "dist\ocrapp_pureray.exe" del /q "dist\ocrapp_pureray.exe"
if exist "dist\ocrapp_pureray" rmdir /s /q "dist\ocrapp_pureray"

python setup.py build_ext --inplace
pyinstaller ocrapp_pureray.spec

if exist "settings" copy /y "settings" "dist\settings"

call conda deactivate
pause
