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
