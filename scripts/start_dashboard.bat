@echo off
echo ========================================
echo Starting Comprehensive Trading Dashboard
echo ========================================
echo.

REM Activate virtual environment
call ..\venv\Scripts\activate.bat

REM Set Python path
set PYTHONPATH=%cd%\..

REM Launch dashboard
python run_dashboard.py

pause