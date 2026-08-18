@echo off
REM start_all.bat — double-click this ONCE to launch both the backend API
REM and the Streamlit frontend, each in its own window.
REM Place this file inside the fusion_engine folder, alongside
REM start_backend.bat and start_frontend.bat.

cd /d "%~dp0"

echo Launching backend API in a new window...
start "Fusion API (backend)" cmd /k start_backend.bat

echo Waiting 45 seconds for all 5 models to load before starting the UI...
echo (The backend loads torch, transformers, PaddleOCR etc. the first time --
echo  this takes a while. Subsequent starts may be faster.)
timeout /t 45 /nobreak

echo Launching Streamlit frontend in a new window...
start "Fusion Demo (frontend)" cmd /k start_frontend.bat

echo.
echo Both are starting. The browser tab for the demo should open automatically.
echo If it doesn't, go to: http://localhost:8501
pause
