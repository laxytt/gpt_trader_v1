# MT5 Test Script Fixes

## Issues Fixed

### 1. Terminal Info Attribute Error
```
AttributeError: 'TerminalInfo' object has no attribute 'version'
```

**Fix**: Added checks for different attribute names as they vary by MT5 version:
- Some versions have `version`
- Some versions have `build`
- Now checks for both

### 2. Account Info Dictionary Access
MT5 returns account info as a named tuple, which the MT5Client converts to a dictionary using `._asdict()`.

**Fix**: Changed all attribute access to dictionary access:
```python
# Wrong
account_info.balance

# Correct
account_info.get('balance', 0)
```

### 3. Server Attribute Missing
MT5Settings doesn't store server/login info. The system connects to whatever MT5 terminal is already running.

**Fix**: Removed all references to `settings.mt5.server`

## How to Run Tests

1. **Start MT5 Terminal**
   - Open MetaTrader 5
   - Login to your account (preferably demo)
   - Enable automated trading: Tools → Options → Expert Advisors → Allow automated trading

2. **Run Simple Test**
   ```bash
   python test_mt5_simple.py
   ```
   This gives you a quick overview of connection status and account info.

3. **Run Comprehensive Test**
   ```bash
   python test_mt5_connection.py
   ```
   This runs all tests including order management (demo accounts only).

4. **Debug Info** (if needed)
   ```bash
   python debug_mt5_info.py
   ```
   This shows all available attributes for terminal_info and account_info.

## Expected Output

### Successful Connection:
```
MT5 Connection Test
==================================================
✅ MT5 initialized successfully

Terminal Info:
  - Connected: Yes
  - Company: MetaQuotes Ltd.
  - Build: 3640

Account Information:
  - Login: 12345678
  - Balance: $10000.00
  - Account Type: DEMO

✅ This is a DEMO account - safe for testing
```

### Key Points
- The system uses the MT5 terminal that's already running
- No login credentials needed in Python code
- Account type detection works by checking `trade_mode` (0 = demo)
- All MT5 data is returned as dictionaries, not objects