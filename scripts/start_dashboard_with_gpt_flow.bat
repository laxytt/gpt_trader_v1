@echo off
echo Starting Trading Dashboard with GPT Flow Visualization...
echo.

cd /d D:\gpt_trader_v1

echo Activating virtual environment...
call venv\Scripts\activate

echo Setting PYTHONPATH...
set PYTHONPATH=D:\gpt_trader_v1

echo Starting Dashboard...
streamlit run scripts/trading_dashboard_simple.py

pause