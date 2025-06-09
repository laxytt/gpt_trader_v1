@echo off
echo ========================================
echo GPT Trading System - Production Startup
echo ========================================
echo.

REM Set project path
set PROJECT_PATH=D:\gpt_trader_v1
set PYTHONPATH=%PROJECT_PATH%

REM Change to project directory
cd /d %PROJECT_PATH%

REM Check if virtual environment exists
if exist "venv\Scripts\python.exe" (
    set PYTHON_PATH=venv\Scripts\python.exe
    echo Using virtual environment Python
) else if exist "venv_wsl\Scripts\python.exe" (
    set PYTHON_PATH=venv_wsl\Scripts\python.exe
    echo Using WSL virtual environment Python
) else (
    set PYTHON_PATH=python
    echo Using system Python
)

echo.
echo Checking environment setup...

REM Verify .env file exists
if not exist ".env" (
    echo ERROR: .env file not found!
    echo Please create a .env file with your configuration.
    echo See .env.example for reference.
    pause
    exit /b 1
)

REM Quick environment test
echo Testing environment variables...
%PYTHON_PATH% -c "from dotenv import load_dotenv; import os; load_dotenv(); print(f'MT5_FILES_DIR: {os.getenv(\"MT5_FILES_DIR\", \"NOT SET\")}')"

echo.
echo Starting essential services in visible windows...
echo.

REM Start Trading Loop (visible window)
echo [1/2] Starting Trading Loop...
start "Trading Loop" cmd /c "cd /d %PROJECT_PATH% && set PYTHONPATH=%PROJECT_PATH% && %PYTHON_PATH% trading_loop.py || pause"

REM Wait for trading loop to initialize
timeout /t 10 /nobreak

REM Start ML Scheduler (visible window)
echo [2/2] Starting ML Scheduler...
start "ML Scheduler" cmd /c "cd /d %PROJECT_PATH% && set PYTHONPATH=%PROJECT_PATH% && %PYTHON_PATH% scripts\ml_scheduler.py daemon || pause"

echo.
echo ========================================
echo All essential services started!
echo ========================================
echo.
echo You should see 2 new command windows:
echo - Trading Loop (Main trading system)
echo - ML Scheduler (Model monitoring and updates)
echo.
echo If a window closes immediately, check for errors.
echo.
echo To stop services, close their windows or press Ctrl+C in each.
echo.
pause