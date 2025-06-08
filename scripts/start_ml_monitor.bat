@echo off
echo Starting ML Performance Monitor...
echo ================================

REM Activate virtual environment if it exists
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate
)

REM Set Python path
set PYTHONPATH=%CD%

REM Run the monitor with all outputs
python scripts/ml_performance_monitor.py --terminal --html --plot --days 30

echo.
echo ML Performance Monitor completed!
echo Check the reports folder for generated files.
echo.

pause