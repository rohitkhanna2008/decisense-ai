@echo off
setlocal
echo ========================================================
echo Starting DeciSense AI Main Dashboard...
echo Aligned with UN SDG 11.6 and SDG 3 (RIT Chennai)
echo ========================================================

REM Add local Python install to current PATH if not present
set "PATH=C:\Users\Rohit Khanna\AppData\Local\Python\bin;%PATH%"

cd /d "C:\Users\Rohit Khanna\.gemini\antigravity\scratch\quietgrid-ai"

echo Checking Python environment...
python --version
echo.

echo Launching Streamlit Dashboard on http://localhost:8501 ...
python -m streamlit run app.py --server.port=8501

pause
