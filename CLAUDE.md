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

# Train ML models
python scripts/train_ml_models.py

# Test ML performance
python scripts/test_ml_backtest.py
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
ML_ENABLED=false (set to true to enable ML)
ML_CONFIDENCE_THRESHOLD=0.7
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
   - `council_signal_service.py`: Multi-agent Trading Council signal generation
   - `signal_service.py`: Legacy single-agent signal generation (deprecated)
   - `trade_service.py`: Trade lifecycle management
   - `market_service.py`: Market data aggregation
   - `memory_service.py`: RAG-based historical trade memory
   - `news_service.py`: Economic news filtering
   - `backtesting_service.py`: Historical strategy testing

4. **Trading Council Agents** (`core/agents/`)
   - `base_agent.py`: Abstract base class for all agents
   - `technical_analyst.py`: Chart patterns and indicators specialist
   - `fundamental_analyst.py`: News and economic analysis
   - `sentiment_reader.py`: Market psychology expert
   - `risk_manager.py`: Capital preservation (has veto power)
   - `momentum_trader.py`: Trend following specialist
   - `contrarian_trader.py`: Reversal and fade trader
   - `head_trader.py`: Decision synthesizer and moderator
   - `council.py`: Orchestrates the multi-agent debate process

5. **Machine Learning** (`core/ml/`)
   - Feature engineering and model training infrastructure
   - Continuous improvement system

### Key Architectural Patterns

1. **Dependency Injection**: `trading_loop.py` uses `DependencyContainer` for clean component initialization

2. **Async/Await**: Entire system is async-first for concurrent symbol processing

3. **Error Handling**: Comprehensive error context tracking via `ErrorContext` wrapper

4. **Configuration**: Pydantic-based settings in `config/settings.py` with environment variable support

5. **Trading Strategy**:
   - Multi-timeframe analysis (H1 entry, H4 background)
   - Multi-Agent Trading Council with 7 specialized agents
   - Three-round debate process for consensus building
   - Hybrid confidence scoring (70% LLM consensus, 30% ML)
   - Risk manager with veto power for capital preservation
   - Real-time trade management with GPT decisions

### Data Flow
1. MT5 → Data Provider → Market Data
2. Market Data → Trading Council (7 Agents) → Debate → Consensus → Trading Signal
3. Trading Signal → Trade Service → MT5 Order Manager
4. Trade Results → Memory Service (FAISS) → Future Signal Context

### Key Integration Points
- **MetaTrader 5**: Real-time data and order execution
- **OpenAI GPT-4**: Powers all 7 Trading Council agents
- **FAISS + SentenceTransformer**: Similar trade case retrieval
- **Screenshots**: MT5 generates charts, system reads from `screenshots/` directory
- **Trading Council**: Multi-agent debate system for robust decisions

## Testing and Validation

Currently no formal test suite. Use:
- `run_backtest.py` for historical validation
- `test_council.py` for testing the Trading Council multi-agent system
- `scripts/visualize_council.py` for visualizing council debates
- `core/services/offline_validator.py` for signal validation without live trading
- Monitor logs in `logs/trading_system.log`

## Common Development Tasks

### Adding New Trading Symbols
Edit `config/symbols.py` and add to appropriate group (conservative, moderate, aggressive)

### Modifying Trading Strategy
1. Trading Council agents: `core/agents/` (modify agent behaviors)
2. Council configuration: `config/settings.py` (TradingSettings class)
3. Agent prompts: Within each agent class in `core/agents/`
4. Trade management: `config/prompts/management_prompt.txt`

### Database Schema Changes
Run migrations in `core/infrastructure/database/migrations.py`

### Debugging Data Issues
Use `core/utils/data_diagnostics.py` for data availability reports