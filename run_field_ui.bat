@echo off
setlocal
echo ========================================================
echo Starting DeciSense AI Mobile Field Capture UI...
echo Collector: Saipalaniappan T ^& Field Survey Team
echo ========================================================

REM Add local Python install to current PATH if not present
set "PATH=C:\Users\Rohit Khanna\AppData\Local\Python\bin;%PATH%"

cd /d "C:\Users\Rohit Khanna\.gemini\antigravity\scratch\quietgrid-ai"

echo Launching Field Logger on http://localhost:8502 ...
python -m streamlit run src/data_pipeline/field_capture_ui.py --server.port=8502

pause
