# MT5 Connection and Order Management Test Guide

## Overview
This guide helps you test and verify your MT5 connection and order management settings before running the trading system.

## Test Scripts Available

### 1. Simple Connection Test (`test_mt5_simple.py`)
Quick test to verify MT5 connection and basic functionality.

**What it tests:**
- MT5 connection
- Account information
- Symbol availability
- Market data retrieval
- Current positions and orders

**How to run:**
```bash
python test_mt5_simple.py
```

### 2. Comprehensive Test Suite (`test_mt5_connection.py`)
Full test suite including order placement (demo accounts only).

**What it tests:**
1. MT5 Connection
2. Account Information
3. Market Data Fetching
4. Symbol Information
5. Order Calculations (profit, margin, lot size)
6. Order Placement (demo only)
7. Order Modification (SL/TP changes)
8. Position Closure
9. Historical Data Retrieval

**How to run:**
```bash
python test_mt5_connection.py
```

## Pre-Test Checklist

### 1. MT5 Terminal Setup
- [ ] MT5 terminal is installed
- [ ] You're logged into your account (preferably demo)
- [ ] Auto-trading is enabled (Tools → Options → Expert Advisors → Allow automated trading)
- [ ] DLL imports are allowed if needed

### 2. Environment Configuration (.env file)
```env
# MT5 Settings
MT5_LOGIN=your_demo_login
MT5_PASSWORD=your_password
MT5_SERVER=your_broker_server
MT5_PATH=C:/Program Files/MetaTrader 5/terminal64.exe
```

### 3. Python Dependencies
Ensure you have the required packages:
```bash
pip install MetaTrader5
```

## Expected Test Results

### Successful Connection Test Output
```
MT5 Connection Test
==================================================

✅ MT5 initialized successfully

Terminal Info:
  - Connected: Yes
  - Company: YourBroker
  - Version: 500.3426

Account Information:
  - Login: 12345678
  - Balance: $10000.00
  - Account Type: DEMO

✅ This is a DEMO account - safe for testing
```

### Common Issues and Solutions

#### 1. "Failed to initialize MT5"
**Possible causes:**
- MT5 terminal not running
- Wrong path in .env file
- Incorrect login credentials

**Solution:**
- Start MT5 terminal manually
- Verify MT5_PATH in .env file
- Check login credentials

#### 2. "Symbol not available"
**Possible causes:**
- Symbol not offered by broker
- Different symbol naming (e.g., "GOLD" vs "XAUUSD")
- Market closed

**Solution:**
- Check symbol list in MT5 terminal
- Use correct symbol names for your broker

#### 3. "Trade not allowed"
**Possible causes:**
- Auto-trading disabled
- Account restrictions
- Market closed

**Solution:**
- Enable auto-trading in MT5
- Check account permissions
- Test during market hours

## Order Management Testing (Demo Only)

The comprehensive test includes order placement tests that will:
1. Place a small test order (0.01 lot)
2. Modify its SL/TP
3. Close the position

**Important:** These tests should only be run on DEMO accounts!

### Test Order Parameters
- Symbol: EURUSD
- Volume: 0.01 lots (minimum)
- Stop Loss: 20 pips
- Take Profit: 30 pips
- Risk: ~$2-3 on 0.01 lot

## Verifying Your Setup

After running the tests, verify:

1. **Connection**: All connection tests pass
2. **Data Access**: Can retrieve current prices and historical data
3. **Account Info**: Balance, leverage, and margins are correct
4. **Symbol Access**: All your trading symbols are available
5. **Order Calculations**: Profit/loss calculations are accurate
6. **Order Execution** (demo): Orders can be placed and managed

## Risk Management Verification

The test suite also verifies risk management calculations:
- Lot size calculation based on risk percentage
- Margin requirements
- Profit/loss calculations

Example output:
```
Order Calculations:
✅ Profit calculation - 10 pips profit on 0.01 lot = $1.00
✅ Margin calculation - Required margin for 0.01 lot = $20.00
✅ Lot size calculation - For $100.00 risk with 20 pip SL = 0.050 lots
```

## Next Steps

Once all tests pass:
1. Review your risk settings in `config/settings.py`
2. Ensure ML models are loaded (if using ML)
3. Verify news data is up-to-date
4. Start with small position sizes
5. Monitor the first few trades carefully

## Safety Tips

1. **Always test on demo first** - Run the system on a demo account for at least a week
2. **Start small** - Use minimum lot sizes initially
3. **Monitor actively** - Watch the first few trading sessions closely
4. **Check logs** - Review logs in `logs/trading_system.log` for any issues
5. **Set appropriate risk** - Default is 1.5% per trade, adjust as needed

## Troubleshooting Commands

Check MT5 connection status:
```python
import MetaTrader5 as mt5
print(mt5.initialize())
print(mt5.terminal_info())
print(mt5.last_error())
```

List available symbols:
```python
symbols = mt5.symbols_get()
for s in symbols[:10]:
    print(s.name, s.description)
```

Check account permissions:
```python
account = mt5.account_info()
print(f"Trade allowed: {account.trade_allowed}")
print(f"Trade expert: {account.trade_expert}")
```