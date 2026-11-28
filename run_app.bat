@echo off
setlocal
echo ========================================================
echo Starting DeciSense AI Main Dashboard...
echo Aligned with UN SDG 11.6 and SDG 3 (RIT Chennai)
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

echo Checking Python environment...
%PY_CMD% --version
%PY_CMD% -m streamlit --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Streamlit is not installed. Run: %PY_CMD% -m pip install -r requirements.txt
    pause
    exit /b 1
)
echo.

echo Launching Streamlit Dashboard on http://localhost:8501 ...
%PY_CMD% -m streamlit run app.py --server.port=8501

pause
