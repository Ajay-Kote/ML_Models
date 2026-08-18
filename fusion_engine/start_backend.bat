@echo off
REM start_backend.bat — double-click this to start the Fusion API.
REM Place this file inside the fusion_engine folder.

cd /d "%~dp0"

echo Starting Adaptive Risk Fusion API...
echo.

REM --- EDIT THIS LINE if your venv is somewhere else ---
call ..\module4_image\venv\Scripts\activate.bat

python -m uvicorn api:app --reload

pause
