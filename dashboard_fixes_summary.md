# Dashboard Fixes Summary

## Issues Fixed

### 1. Missing `get_all_signals` Method in SignalRepository
**Problem**: The GPT Flow Dashboard was trying to call `signal_repo.get_all_signals()` but this method didn't exist.

**Solution**: Added the method to `core/infrastructure/database/repositories.py` at line 526:
```python
@handle_database_errors
def get_all_signals(self, limit: int = 100) -> List[Dict[str, Any]]:
    """Get all signals for dashboard display"""
    # Returns signals with proper formatting for dashboard
```

### 2. Request Logger Not Filtering by Time
**Problem**: The dashboard showed old requests (15:00:17) instead of recent ones because `get_recent_requests` wasn't filtering by time.

**Solution**: Updated `core/infrastructure/gpt/request_logger.py` to add `hours_back` parameter:
```python
def get_recent_requests(
    self, 
    limit: int = 100,
    hours_back: int = 24,  # Added this parameter
    ...
):
    query = """
        SELECT * FROM gpt_requests
        WHERE timestamp > datetime('now', '-{} hours')
    """.format(hours_back)
```

### 3. Incorrect Timestamp Display
**Problem**: Timestamps weren't being parsed correctly, showing wrong times.

**Solution**: Updated `scripts/gpt_flow_dashboard.py` to properly parse and convert timestamps:
```python
# Parse ISO timestamp
if 'T' in timestamp_str:
    timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
else:
    timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')

# Convert to local time for display
if timestamp.tzinfo is None:
    timestamp = timestamp.replace(tzinfo=tz.utc)
local_ts = timestamp.astimezone()
time_str = local_ts.strftime('%H:%M:%S')
```

### 4. Signal Object vs Dictionary Handling
**Problem**: Dashboard expected signal objects but repository returns dictionaries.

**Solution**: Updated dashboard to use dictionary access methods:
```python
# Changed from: signal.analysis
# To: signal.get('analysis')
```

## How to Verify the Fixes

1. Restart the GPT Flow Dashboard:
   ```bash
   python scripts/gpt_flow_dashboard.py
   ```

2. Check that:
   - Recent requests show current times (not 15:00:17)
   - No more "'SignalRepository' object has no attribute 'get_all_signals'" error
   - Council Decisions section loads without errors
   - Request Payloads shows requests from the selected time range

## Files Modified

1. `/core/infrastructure/database/repositories.py` - Added `get_all_signals` method
2. `/core/infrastructure/gpt/request_logger.py` - Added `hours_back` parameter
3. `/scripts/gpt_flow_dashboard.py` - Fixed timestamp parsing and signal handling