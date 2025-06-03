@echo off
echo Creating Windows Scheduled Tasks for GPT Trading System...
echo.

REM Set paths
set PROJECT_PATH=D:\gpt_trader_v1
set PYTHON_PATH=%PROJECT_PATH%\venv\Scripts\python.exe

REM Create main trading task (runs at startup)
echo Creating main trading task...
schtasks /create /tn "GPT Trading System Main" /tr "\"%PYTHON_PATH%\" \"%PROJECT_PATH%\trading_loop.py\"" /sc onstart /ru %USERNAME% /f
if %errorlevel% equ 0 (
    echo [OK] Main trading task created
) else (
    echo [FAILED] Could not create main trading task
)
echo.

REM Create daily backup task (2:00 AM)
echo Creating daily backup task...
schtasks /create /tn "GPT Trading Backup" /tr "\"%PYTHON_PATH%\" \"%PROJECT_PATH%\scripts\control_panel.py\" backup" /sc daily /st 02:00 /ru %USERNAME% /f
if %errorlevel% equ 0 (
    echo [OK] Backup task created
) else (
    echo [FAILED] Could not create backup task
)
echo.

REM Create daily report task (11:00 PM)
echo Creating daily report task...
schtasks /create /tn "GPT Trading Report" /tr "\"%PYTHON_PATH%\" \"%PROJECT_PATH%\scripts\control_panel.py\" report" /sc daily /st 23:00 /ru %USERNAME% /f
if %errorlevel% equ 0 (
    echo [OK] Report task created
) else (
    echo [FAILED] Could not create report task
)
echo.

echo.
echo Setup complete!
echo.
echo To view your tasks, run: schtasks /query /tn "GPT Trading*"
echo To delete a task, run: schtasks /delete /tn "TASK_NAME" /f
echo.
pause