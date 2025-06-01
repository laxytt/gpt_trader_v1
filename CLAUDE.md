# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Core Commands

### Running the Trading System
```bash
# Set Python path (Windows PowerShell)
$env:PYTHONPATH = "D:\gpt_trader_v1"

# Run main trading loop
python trading_loop.py

# Run backtesting
python run_backtest.py
```

### Dependency Management
```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Environment Configuration
Create `.env` file with:
```
OPENAI_API_KEY=your-key
MT5_FILES_DIR=C:/.../MetaQuotes/.../MQL5/Files
TELEGRAM_TOKEN=your-token (optional)
TELEGRAM_CHAT_ID=your-chat-id (optional)
```

## High-Level Architecture

### Domain-Driven Design Structure
The system follows DDD principles with clear separation:

1. **Core Domain** (`core/domain/`)
   - `models.py`: Core entities (Trade, Signal, MarketData)
   - `enums/`: Trading constants and enumerations
   - `exceptions.py`: Custom exception hierarchy

2. **Infrastructure Layer** (`core/infrastructure/`)
   - `mt5/`: MetaTrader 5 integration (client, data provider, order manager)
   - `gpt/`: OpenAI GPT integration (signal generation, trade management)
   - `database/`: SQLite repositories for trades, signals, memory cases
   - `notifications/`: Telegram notification service

3. **Services Layer** (`core/services/`)
   - `trading_orchestrator.py`: Main coordination service
   - `signal_service.py`: Signal generation with VSA analysis
   - `trade_service.py`: Trade lifecycle management
   - `market_service.py`: Market data aggregation
   - `memory_service.py`: RAG-based historical trade memory
   - `news_service.py`: Economic news filtering
   - `backtesting_service.py`: Historical strategy testing

4. **Machine Learning** (`core/ml/`)
   - Feature engineering and model training infrastructure
   - Continuous improvement system

### Key Architectural Patterns

1. **Dependency Injection**: `trading_loop.py` uses `DependencyContainer` for clean component initialization

2. **Async/Await**: Entire system is async-first for concurrent symbol processing

3. **Error Handling**: Comprehensive error context tracking via `ErrorContext` wrapper

4. **Configuration**: Pydantic-based settings in `config/settings.py` with environment variable support

5. **Trading Strategy**:
   - Multi-timeframe analysis (H1 entry, H4 background)
   - GPT-4 vision for chart analysis with technical indicators
   - Volume Spread Analysis (VSA) confirmation
   - Real-time trade management with GPT decisions

### Data Flow
1. MT5 → Data Provider → Market Data
2. Market Data + Screenshots → GPT Signal Generator → Trading Signal
3. Trading Signal → Trade Service → MT5 Order Manager
4. Trade Results → Memory Service (FAISS) → Future Signal Context

### Key Integration Points
- **MetaTrader 5**: Real-time data and order execution
- **OpenAI GPT-4**: Signal generation and trade management
- **FAISS + SentenceTransformer**: Similar trade case retrieval
- **Screenshots**: MT5 generates charts, system reads from `screenshots/` directory

## Testing and Validation

Currently no formal test suite. Use:
- `run_backtest.py` for historical validation
- `core/services/offline_validator.py` for signal validation without live trading
- Monitor logs in `logs/trading_system.log`

## Common Development Tasks

### Adding New Trading Symbols
Edit `config/symbols.py` and add to appropriate group (conservative, moderate, aggressive)

### Modifying Trading Strategy
1. Signal generation logic: `core/infrastructure/gpt/signal_generator.py`
2. VSA rules: `config/prompts/system_prompt.txt`
3. Trade management: `config/prompts/management_prompt.txt`

### Database Schema Changes
Run migrations in `core/infrastructure/database/migrations.py`

### Debugging Data Issues
Use `core/utils/data_diagnostics.py` for data availability reports