@echo off
title StudySync Application Launcher
cls
echo ========================================================
echo             STUDYSYNC APPLICATION LAUNCHER
echo ========================================================
echo.
echo Checking Python environment...

if exist ".venv\Scripts\python.exe" (
    set "PYTHON_EXE=.venv\Scripts\python.exe"
) else (
    set "PYTHON_EXE=python"
)

echo Starting StudySync server at http://127.0.0.1:5000 ...
echo Opening your web browser...
start "" "http://127.0.0.1:5000"

%PYTHON_EXE% app.py

pause
