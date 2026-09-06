@echo off
REM start_frontend.bat — double-click this to start the Streamlit demo UI.
REM Place this file inside the fusion_engine folder.
REM NOTE: Make sure start_backend.bat is already running before using this
REM       (or use start_all.bat which launches both together).

cd /d "%~dp0"

echo Starting Streamlit frontend...
echo.

REM --- EDIT THIS LINE if your venv is somewhere else ---
call venv\Scripts\activate.bat

streamlit run app.py

pause