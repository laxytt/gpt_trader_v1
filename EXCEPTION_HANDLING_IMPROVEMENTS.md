# Exception Handling Improvements in TradingOrchestrator

## Summary
Replaced broad `except Exception:` clauses with specific exception types throughout the `trading_orchestrator.py` file to improve error handling precision and debugging.

## Changes Made

### 1. Import Updates
- Added `ValidationError` to the imports from `core.domain.exceptions`

### 2. Symbol Processing (`process_symbol` method)
- **Before**: Single broad `except Exception` clause
- **After**: Three specific handlers:
  - `(ValueError, ValidationError)` for symbol validation issues
  - `ServiceError` for service-related failures
  - `Exception` as fallback for unexpected errors

### 3. Main Trading Loop (`run` method)
- Added `asyncio.CancelledError` handling to properly propagate cancellation
- Separated `(ServiceError, TradingSystemError)` from general exceptions
- Improved error messages to distinguish between error types

### 4. System Initialization (`_initialize_system` method)
- `ConfigurationError` is now re-raised as-is (important configuration issues)
- `(ConnectionError, TimeoutError)` wrapped as `ServiceError` with context
- General exceptions wrapped as `ServiceError` with proper error chaining

### 5. Configuration Validation (`_validate_configuration` method)
- Changed from `Exception` to `(ValueError, ValidationError)` for symbol validation

### 6. Trading Cycle (`_execute_trading_cycle` method)
- Health check: `(ServiceError, asyncio.TimeoutError)` for expected failures
- News fetching: Added `ConnectionError` to expected network issues
- Market overview: `(ServiceError, asyncio.TimeoutError)` for service failures
- Added `asyncio.CancelledError` propagation

### 7. Market Intelligence (`_get_active_symbols` method)
- `ServiceError` for service-specific issues
- `asyncio.TimeoutError` for timeout scenarios
- Separate logging for each error type

### 8. Symbol Processing Loop (`_process_symbols` method)
- Added `asyncio.CancelledError` propagation (important for clean shutdown)
- `(ServiceError, TradingSystemError)` for known trading errors
- Fallback for unexpected errors

### 9. System Shutdown (`_shutdown_system` method)
- `asyncio.CancelledError` is now expected and silently handled
- Other exceptions still logged but don't prevent shutdown completion

## Benefits
1. **Better Error Diagnosis**: Specific exception types make it easier to identify the root cause
2. **Proper Async Handling**: `asyncio.CancelledError` is now properly propagated for clean shutdowns
3. **Error Categorization**: Different error types are handled appropriately based on severity
4. **Improved Logging**: Error messages now indicate the type of failure more clearly
5. **Graceful Degradation**: Service errors don't crash the system but allow fallback behavior

## Exception Hierarchy Used
- `ValidationError`: Input validation failures
- `ConfigurationError`: System configuration issues (critical)
- `ServiceError`: Service layer failures (recoverable)
- `TradingSystemError`: Trading-specific errors
- `ConnectionError`, `TimeoutError`: Network-related issues
- `asyncio.TimeoutError`: Async operation timeouts
- `asyncio.CancelledError`: Task cancellation (must propagate)