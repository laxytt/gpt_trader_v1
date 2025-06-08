@echo off
echo Starting Simple Trading Dashboard...
echo =====================================
echo This dashboard does not include ML features
echo =====================================

cd /d "%~dp0\.."
call venv\Scripts\activate
python scripts\run_simple_dashboard.py

pause