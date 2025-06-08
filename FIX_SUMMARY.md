# Fix Summary - MT5 Test Errors

## Error 1: Import Error
```
ImportError: cannot import name 'SignalType' from 'core.domain.enums'
```

### Solution
Changed import statement in `test_mt5_connection.py`:
- `SignalType` is in `core.domain.models` not `core.domain.enums`

## Error 2: AttributeError
```
AttributeError: 'MT5Settings' object has no attribute 'server'
```

### Solution
The MT5 configuration in this system is different:
- MT5Settings only has: `files_dir`, `magic_number`, `timeout_seconds`, `max_retries`
- No login credentials are stored - it uses the already logged-in MT5 terminal
- Account info is retrieved via `mt5.account_info()` after connection

### Changes Made:
1. Removed references to `settings.mt5.server`
2. Changed from attribute access to dictionary access for account_info
3. Updated demo account detection to use account_info from MT5

### Example:
```python
# Before
account_info.balance
account_info.server

# After  
account_info.get('balance', 0)
account_info.get('server', 'N/A')
```

## How MT5 Connection Works
1. You manually login to MT5 terminal
2. The Python script connects to the running terminal
3. No credentials needed in the code
4. Uses `mt5.initialize()` to connect to the terminal

The test scripts should now run without errors.