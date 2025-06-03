# GPT Trading System

A modular and intelligent intraday trading system powered by GPT-4 and Volume Spread Analysis (VSA), using real-time market data from MetaTrader 5 (MT5).

---

## 🚀 Features

* **Signal Generation**

  * Multi-timeframe (H1 entry, H4 background)
  * Momentum breakout + strict VSA confirmation
  * GPT-4 vision with screenshot + indicators

* **Trade Management**

  * Real-time GPT trade manager with HOLD / MOVE\_SL / CLOSE\_NOW logic
  * Auto SL/TP updates and breakeven logic
  * GPT reflection on trade outcomes

* **News Filtering**

  * Per-symbol macro event blacklists
  * Optional whitelist override
  * Real-time JSON calendar parsing

* **Historical Memory**

  * RAG-like memory using FAISS + SentenceTransformer
  * Stores trade cases with result, signal, context

* **MT5 Integration**

  * Candle fetch (H1/H4) with indicators
  * Real screenshots via `trigger.txt`
  * Order send, SL/TP modify, position sync

---

## 🛠️ Project Structure

```
gpt_trader_v1/
├── core/
│   ├── chart_utils.py
│   ├── gpt_interface.py
│   ├── gpt_trade_manager.py
│   ├── trade_cycle.py
│   ├── trade_manager.py
│   ├── market_data.py
│   ├── rag_memory.py
│   ├── news_filter.py
│   ├── news_utils.py
│   ├── database.py
│   ├── paths.py
│   └── utils.py
├── data/               # SQLite DB, news cache, trade logs
├── screenshots/        # MT5-generated images
├── logs/
├── scripts/            # Optional utility runners
├── .env
├── config.py
└── trading_loop.py     # Main entrypoint
```

---

## ⚙️ Setup Instructions

### 1. Clone & Create Environment

```bash
git clone https://github.com/yourname/gpt_trader_v1.git
cd gpt_trader_v1
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure `.env`

```env
OPENAI_API_KEY=your-key
MT5_FILES_DIR=C:/.../MetaQuotes/.../MQL5/Files
TELEGRAM_TOKEN=your-token
TELEGRAM_CHAT_ID=your-chat-id
```

### 3. Launch Main Loop

```bash
$env:PYTHONPATH = "D:\gpt_trader_v1"
python trading_loop.py
```

---

## 🧠 Strategy Logic

### Entry Signal

* **Momentum:** breakout to 10-bar high/low, impulse candle, strong trend (EMA+RSI)
* **VSA:** No Demand/Supply, Test + Confirm, Shakeout, Upthrust etc.
* **Volume & ATR**: must confirm
* **Time & News Filters**: only during active sessions, skip high-impact events

### Trade Management

* `HOLD` if context consistent
* `MOVE_SL` after +1R or new structure
* `CLOSE_NOW` on reversal, volatility spike, news in 2 min

---

## 🧪 Developer Utilities

* `core/debug_utils.py` – pretty printing
* `core/resync_logger.py` – file-based logs for trade state resync
* `core/statistics.py` – win/loss streak & stats

---

## 📊 Data Requirements

* MetaTrader 5 with access to symbols like EURUSD, US30.cash, XAUUSD etc.
* Installed Python packages: `MetaTrader5`, `openai`, `sentence-transformers`, `faiss`, `mplfinance`, `ta`, `pandas`

---

## ✅ TODO / Roadmap

*

---

## 🧠 Credits

Developed by Wiktor Kukulski with modular GPT-4 integration and a deep respect for market context. ☕


🚀 How to Use the GPT Trading System

  Overview

  This is an automated trading system that combines GPT-4 AI analysis with machine learning to make trading decisions on MetaTrader 5. It
  monitors multiple currency pairs, analyzes market conditions, and executes trades automatically.

  Prerequisites

  1. MetaTrader 5 installed and logged into your broker account
  2. Python 3.8+ installed
  3. OpenAI API key with GPT-4 access
  4. Windows OS (required for MetaTrader 5 integration)

  Initial Setup

  1. Environment Configuration

  Create a .env file in the project root:
  # OpenAI Configuration
  OPENAI_API_KEY=your-openai-api-key-here

  # MetaTrader 5 Configuration
  MT5_FILES_DIR=C:/Users/YourName/AppData/Roaming/MetaQuotes/Terminal/YourBrokerID/MQL5/Files

  # Optional: Telegram Notifications
  TELEGRAM_TOKEN=your-bot-token
  TELEGRAM_CHAT_ID=your-chat-id

  2. Install Dependencies

  # Create virtual environment
  python -m venv venv

  # Activate it (Windows)
  venv\Scripts\activate

  # Install packages
  pip install -r requirements.txt

  Running the System

  1. Live Trading Mode

  # Set Python path (PowerShell)
  $env:PYTHONPATH = "D:\gpt_trader_v1"

  # Run the main trading loop
  python trading_loop.py

  What happens:
  - System connects to MT5
  - Monitors configured symbols (EURUSD, GBPUSD, etc.)
  - Every hour on the hour:
    - Analyzes market conditions
    - Generates trading signals using GPT-4 + ML
    - Executes trades if opportunities found
    - Manages open positions

  2. Backtesting Mode

  # Run historical backtest
  python run_backtest.py

  What happens:
  - Tests strategy on historical data
  - Generates performance reports
  - Saves results to backtest_results/

  Key Features & How They Work

  1. Signal Generation Flow

  Market Data → Technical Analysis → ML Model (if available) → GPT-4 Validation → Trading Signal

  - ML-First: If ML model is deployed and confident (>70%), it makes the decision
  - GPT Validation: Low confidence ML signals are validated by GPT-4
  - Pure GPT: Falls back to GPT-only if no ML model available

  2. Risk Management

  The system enforces multiple risk controls:
  - Per-Trade Risk: Default 1.5% of account per trade
  - Portfolio Limits: Max 3 open trades, max 10% total exposure
  - Correlation Check: Prevents opening correlated positions
  - Drawdown Control: Reduces activity during drawdowns

  3. Trade Management

  Once a trade is open:
  - Continuous Monitoring: Checks positions every cycle
  - Dynamic Decisions: GPT analyzes whether to HOLD, MOVE_SL, or CLOSE_NOW
  - News Awareness: Closes trades before high-impact news events

  Configuration Options

  Trading Settings (config/settings.py)

  # Trading hours (UTC)
  start_hour = 7  # 7 AM UTC
  end_hour = 23   # 11 PM UTC

  # Risk settings
  risk_per_trade_percent = 1.5
  max_open_trades = 3

  # Symbols to trade
  symbols = ["EURUSD", "GBPUSD", "USDCAD"]  # Or use symbol groups

  Symbol Groups (config/symbols.py)

  - Conservative: Major pairs only (EURUSD, GBPUSD, USDJPY)
  - Moderate: Includes minor pairs
  - Aggressive: Adds commodities and indices

  Monitoring & Logs

  1. Console Output

  Shows real-time activity:
  2024-01-20 10:00:00 | Signal generated for EURUSD: BUY (Risk: A)
  2024-01-20 10:00:05 | Trade executed: TRADE_123456
  2024-01-20 11:00:00 | Managing trade TRADE_123456: MOVE_SL to breakeven

  2. Log Files

  Detailed logs saved to logs/trading_system.log

  3. Telegram Notifications (if configured)

  - Trade executions
  - System status updates
  - Error alerts

  Database & Memory

  The system maintains several databases:
  - trades.db: All trade history and performance
  - Memory System: Learns from past trades using FAISS vector search
  - Model Storage: Trained ML models in models/ directory

  Advanced Usage

  1. Deploy ML Models

  # After training a model via backtesting
  # Models meeting performance criteria are auto-deployed

  2. A/B Testing

  The system can run experiments comparing strategies:
  - GPT-only vs ML-driven
  - Different risk parameters
  - Various technical indicators

  3. Portfolio Analysis

  # Check portfolio risk status
  GET /api/portfolio/risk  # If API enabled

  Safety Features

  1. Circuit Breakers: Temporarily disable failing services
  2. Max Loss Limits: Stops trading after significant losses
  3. News Filtering: Avoids trading during major events
  4. Validation: Multiple checks before executing trades

  Typical Daily Operation

  1. Start System: Launch before market open
  2. Automatic Operation: System runs autonomously
  3. Monitor: Check logs/Telegram for activity
  4. End of Day: System continues running or can be stopped

  Troubleshooting

  MT5 Connection Issues:
  - Ensure MT5 is running and logged in
  - Check MT5_FILES_DIR path is correct
  - Verify algo trading is enabled in MT5

  No Trades Executing:
  - Check market hours settings
  - Verify symbols are available in MT5
  - Review risk settings (may be too restrictive)
  - Check logs for GPT API errors

  Performance Issues:
  - Reduce number of symbols
  - Increase analysis cycle time
  - Check API rate limits

  Best Practices

  1. Start Small: Test with minimal risk settings first
  2. Monitor Closely: Especially during first week
  3. Review Daily: Check trade logs and performance
  4. Update News Data: Keep economic calendar current
  5. Backtest Changes: Test any strategy modifications

  The system is designed to run continuously during market hours, making intelligent trading decisions while managing risk at both trade and
  portfolio levels.



  🎉 Workflow Implementation Complete!

  I've successfully implemented a complete workflow automation system for your GPT Trading System. Here's what has       
  been created:

  📁 Structure Created:

  scripts/
  ├── workflow_orchestrator.py      # Main coordinator
  ├── control_panel.py             # Manual control interface
  ├── setup_windows_tasks.ps1      # Windows automation setup
  └── automation/
      ├── __init__.py
      ├── health_check.py          # System monitoring
      ├── ml_updater.py           # ML model training
      ├── backtest_runner.py      # Automated backtesting
      ├── database_backup.py      # Database management
      ├── performance_analytics.py # Reporting & analysis
      └── news_updater.py         # Economic calendar updates

  🚀 How to Use:

  1. Manual Control (Recommended for testing):

  # Check system status
  python scripts/control_panel.py status

  # Create backup
  python scripts/control_panel.py backup

  # Generate report
  python scripts/control_panel.py report

  # Update news
  python scripts/control_panel.py news

  # Run quick backtest
  python scripts/control_panel.py backtest --days 30

  2. Automated Setup (For production):

  # Run as Administrator in PowerShell
  cd D:\gpt_trader_v1\scripts
  .\setup_windows_tasks.ps1

  This creates scheduled tasks for:
  - Continuous: Main trading system
  - Hourly: Health checks, news updates (every 4h)
  - Daily: ML updates (3 AM), backups (2 AM), reports (11 PM)
  - Weekly: Comprehensive backtests (Saturday noon)

  3. Daily Workflow:

  # Morning check
  python scripts/control_panel.py status

  # Evening review
  python scripts/workflow_orchestrator.py --task daily

  🔧 Key Features:

  1. Health Monitoring: Checks MT5, database, API connectivity, disk space
  2. Smart ML Updates: Only deploys better-performing models
  3. Automated Backups: With rotation and compression
  4. Performance Analytics: Daily/weekly reports with charts
  5. News Integration: Mock data for now, ready for real feeds
  6. Emergency Controls: Quick stop functionality

  📊 Monitoring Dashboard:

  The control panel provides a unified view:
  - System health status
  - Component availability
  - Recent performance metrics
  - Upcoming news events
  - Quick actions

  🔐 Safety Features:

  - Graceful error handling
  - Backup before updates
  - Model validation before deployment
  - Alert cooldowns to prevent spam
  - Comprehensive logging

  🎯 Next Steps:

  1. Test manually using the control panel
  2. Run setup script to automate (as Administrator)
  3. Monitor logs in the logs/ directory
  4. Customize schedules in Task Scheduler if needed
  5. Add real news sources when ready

  The system is now ready for both manual operation and full automation!