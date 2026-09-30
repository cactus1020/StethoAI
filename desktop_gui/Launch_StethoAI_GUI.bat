@echo off
title StethoAI Desktop Workstation (V3 Deep CNN)
echo ========================================================
echo   Launching StethoAI Digital Stethoscope Workstation...
echo ========================================================
cd /d "%~dp0"
python stetho_gui.py
if errorlevel 1 (
    echo.
    echo Press any key to exit...
    pause >nul
)

