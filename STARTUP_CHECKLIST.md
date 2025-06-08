# Trading System Startup Checklist

## ✅ System Status

### Core Components
- [x] MarketAux Integration
- [x] Enhanced News Service
- [x] Trading Council (7 Agents)
- [x] GPT Flow Visualization Dashboard
- [x] ML Performance Monitoring

### Recent Updates
1. **MarketAux Integration**: 
   - Fixed keywords parsing
   - Optimized country filtering
   - Enhanced search strategy for economic news

2. **GPT Flow Dashboard**:
   - Real-time request monitoring
   - Trading Council visualization
   - Token usage tracking
   - Cost analytics

## 🚀 To Start Trading

### 1. Main Trading Loop
```powershell
cd D:\gpt_trader_v1
.\venv\Scripts\activate
python trading_loop.py
```

### 2. Monitoring Dashboard
Open a second terminal:
```powershell
cd D:\gpt_trader_v1
.\venv\Scripts\activate
streamlit run scripts/comprehensive_trading_dashboard.py
```

Then navigate to:
- **GPT Flow Visualization** - See real-time GPT requests and Trading Council decisions
- **ML Performance** - Track ML model performance
- **Open Trades** - Monitor active positions
- **Performance Metrics** - View trading statistics

### 3. Optional Services

**Update News Data** (if needed):
```powershell
python scripts/control_panel.py news
```

**ML Model Updates** (if ML enabled):
```powershell
python scripts/ml_continuous_learning.py
```

## ⚙️ Configuration

### MarketAux Settings (.env)
```
MARKETAUX_ENABLED=true
MARKETAUX_API_TOKEN=your_token
MARKETAUX_DAILY_LIMIT=100
```

### Trading Settings
- Symbols: EURUSD, GBPUSD, USDCAD, AUDUSD
- Risk per trade: 1.5%
- Max open trades: 3
- Trading hours: 07:00 - 23:00 UTC

## 📊 What You'll See

1. **Trading Loop Output**:
   - Market analysis cycles
   - Trading Council debates
   - Signal generation
   - Trade execution

2. **Dashboard Features**:
   - Live GPT request flow
   - Agent voting visualization
   - Token usage and costs
   - Performance metrics
   - ML predictions (if enabled)

## 🔍 Troubleshooting

1. **If MarketAux returns 0 articles**:
   - Check API token is valid
   - Verify daily limit not exceeded
   - News will still work via ForexFactory

2. **If GPT Flow page doesn't show**:
   - Refresh the dashboard
   - Check browser console for errors
   - Ensure trading loop is running

3. **Memory Service Issues**:
   - Ensure torch, sentence-transformers, faiss-cpu are installed
   - Check virtual environment is activated

## 📈 Ready to Trade!

The system is configured and ready. Start with the trading loop and dashboard to monitor your automated trading system with full visibility into the AI decision-making process!