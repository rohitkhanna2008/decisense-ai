@echo off
setlocal
echo ========================================================
echo Starting DeciSense AI Mobile Field Capture UI...
echo Collector: Saipalaniappan T ^& Field Survey Team
echo ========================================================

cd /d "%~dp0"

set "PY_CMD=python"
where python >nul 2>&1
if errorlevel 1 (
    where py >nul 2>&1
    if errorlevel 1 (
        echo [ERROR] Python is not installed or not on PATH.
        pause
        exit /b 1
    )
    set "PY_CMD=py -3"
)

%PY_CMD% -m streamlit --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Streamlit is not installed. Run: %PY_CMD% -m pip install -r requirements.txt
    pause
    exit /b 1
)

echo Launching Field Logger on http://localhost:8502 ...
%PY_CMD% -m streamlit run src/data_pipeline/field_capture_ui.py --server.port=8502

pause
