# Comprehensive Trading Dashboard

## Overview

The Comprehensive Trading Dashboard is an enhanced web-based interface that combines ML metrics with live trading data, providing a complete view of your trading system's performance and allowing real-time trade management.

## Features

### 1. **Dashboard Overview**
- Real-time open trades count and P&L
- Today's profit/loss tracking
- Active symbols and ML status
- Account balance monitoring
- System health indicators
- 24-hour trading summary
- Active signals display
- P&L trend visualization

### 2. **Open Trades**
- Live P&L updates for all open positions
- Trade duration tracking
- Quick actions (close profitable, update S/L to breakeven)
- Export functionality
- Color-coded profit/loss display

### 3. **Trade Management**
- Individual trade modification
- Stop Loss and Take Profit updates
- Partial position closing
- Full position closing
- Real-time price monitoring

### 4. **Historical Analysis**
- Date range filtering
- Symbol-specific analysis
- Profit distribution visualization
- Monthly performance breakdown
- Comprehensive trade history
- Win rate and profit metrics

### 5. **Council Decisions**
- View Trading Council agent debates
- Individual agent opinions and reasoning
- Consensus analysis
- Confidence scores
- Trade results linked to signals

### 6. **Signal History**
- Signal generation tracking
- Execution rate monitoring
- Confidence level analysis
- Performance over time
- Signal-to-trade correlation

### 7. **ML Performance**
- Model performance metrics
- Win rate and profit tracking
- Model status monitoring
- Retraining controls
- Feature importance (planned)
- Performance trends

### 8. **Performance Metrics**
- Net profit and win rate
- Profit factor and Sharpe ratio
- Maximum drawdown analysis
- Recovery factor
- Risk metrics (VaR, Expected Shortfall)
- Symbol-by-symbol breakdown
- Equity curve visualization

### 9. **Database Statistics**
- Table record counts
- Data growth monitoring
- Recent activity tracking
- Database maintenance tools
- Backup functionality
- Optimization controls

## Installation

1. Ensure all requirements are installed:
```bash
pip install -r ../requirements.txt
```

2. Verify MT5 is installed and configured properly

3. Ensure your `.env` file has all required settings

## Running the Dashboard

### Comprehensive Dashboard (Full Features including ML)

#### Option 1: Using the fixed launch script (Recommended for torch compatibility)
```bash
python scripts/run_dashboard_fixed.py
```

#### Option 2: Using the standard launch script
```bash
python scripts/run_dashboard.py
```

#### Option 3: Using the batch file (Windows)
```batch
cd scripts
start_dashboard.bat
```

#### Option 4: Direct Streamlit command
```bash
streamlit run scripts/comprehensive_trading_dashboard.py
```

### Simple Dashboard (No ML Dependencies)

If you encounter compatibility issues with the comprehensive dashboard, use the simple version:

#### Option 1: Using the launch script
```bash
python scripts/run_simple_dashboard.py
```

#### Option 2: Using the batch file (Windows)
```batch
cd scripts
start_simple_dashboard.bat
```

#### Option 3: Direct Streamlit command
```bash
streamlit run scripts/trading_dashboard_simple.py
```

The simple dashboard includes all core trading features but excludes ML performance metrics.

## Configuration

The dashboard uses the same configuration as your main trading system from `config/settings.py`. Ensure your settings are properly configured before launching.

## Usage Tips

1. **Auto-refresh**: Enable the auto-refresh option for live monitoring (refreshes every 30 seconds)

2. **MT5 Connection**: The dashboard will attempt to connect to MT5 on startup. Some features require an active MT5 connection.

3. **Trade Management**: Be cautious when using trade management features. Always double-check before closing or modifying positions.

4. **Performance Analysis**: Use different time periods to analyze performance trends and identify patterns.

5. **Council Decisions**: Review agent debates to understand the reasoning behind trading decisions.

## Troubleshooting

### MT5 Connection Issues
- Ensure MT5 terminal is running
- Check your MT5 credentials in the `.env` file
- Verify the MT5 path in settings

### Missing Data
- Check database path in settings
- Ensure the trading system has been running and generating data
- Verify database integrity using the maintenance tools

### Performance Issues
- For large datasets, use date filters to limit the data range
- Consider running database optimization from the Database Statistics page

## Security Note

This dashboard provides full access to your trading system. Ensure it's only accessible on your local machine or properly secured if exposed to a network.

## Future Enhancements

- Real-time chart integration
- Advanced ML feature analysis
- Automated report generation
- Mobile-responsive design
- Alert and notification system
- Multi-account support

## Support

For issues or questions, refer to the main project documentation or check the logs in the `logs/` directory.