@echo off
REM Activate virtual environment and run trading loop

cd /d D:\gpt_trader_v1

REM Activate the virtual environment
call venv\Scripts\activate.bat

REM Set Python path
set PYTHONPATH=D:\gpt_trader_v1

REM Run the trading loop
python trading_loop.py

REM Keep window open if there's an error
pause