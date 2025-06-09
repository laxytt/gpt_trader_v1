@echo off
setlocal enabledelayedexpansion

echo Starting GPT Trading System Services...
echo.

REM Check if virtual environment exists
if not exist "D:\gpt_trader_v1\venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found at D:\gpt_trader_v1\venv
    echo Please create the virtual environment first.
    pause
    exit /b 1
)

REM Activate virtual environment
echo Activating virtual environment...
call D:\gpt_trader_v1\venv\Scripts\activate.bat

REM Check if python exists in venv
if not exist "D:\gpt_trader_v1\venv\Scripts\python.exe" (
    echo ERROR: Python not found in virtual environment
    pause
    exit /b 1
)

REM Ensure logs directory exists
if not exist "D:\gpt_trader_v1\logs" (
    mkdir "D:\gpt_trader_v1\logs"
)

REM Check if main script exists
if not exist "D:\gpt_trader_v1\trading_loop.py" (
    echo ERROR: trading_loop.py not found
    pause
    exit /b 1
)

REM Start main trading system with logging
echo Starting main trading system...
start "GPT Trading System" cmd /k ^
powershell -Command "D:\gpt_trader_v1\venv\Scripts\python.exe D:\gpt_trader_v1\trading_loop.py 2>&1 | Tee-Object -FilePath D:\gpt_trader_v1\logs\trading_loop.log"

REM Wait a bit before starting other services
echo Waiting 10 seconds before starting other services...
timeout /t 10

REM Check if control panel exists
if exist "D:\gpt_trader_v1\scripts\control_panel.py" (
    echo Starting control panel...
    start "Control Panel" cmd /k ^
    powershell -Command "D:\gpt_trader_v1\venv\Scripts\python.exe D:\gpt_trader_v1\scripts\control_panel.py status 2>&1 | Tee-Object -FilePath D:\gpt_trader_v1\logs\control_panel.log"
) else (
    echo WARNING: control_panel.py not found, skipping...
)

echo.
echo All services started!
echo.
echo Press any key to close this window...
pause >nul
