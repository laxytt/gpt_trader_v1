# MT5 Order Management Test Suite Guide

## Overview

The comprehensive MT5 order management test suite (`test_mt5_order_management.py`) tests all order operations that the GPT trading system can perform. This includes opening, closing, modifying, and managing positions in various scenarios.

## Test Suite Features

### 1. Single Order Operations
- Open BUY/SELL positions
- Close positions completely
- Verify order execution and position details

### 2. Order Modifications
- Modify stop loss and take profit levels
- Verify modifications are applied correctly
- Handle modification edge cases

### 3. Partial Position Closure
- Close portions of larger positions
- Maintain remaining position with updated parameters
- Calculate partial P/L correctly

### 4. Multiple Positions
- Open multiple positions simultaneously
- Track aggregate risk across positions
- Close all positions efficiently

### 5. Stop Loss Management
- Implement trailing stop logic
- Move stops to reduce risk as price moves favorably
- Protect profits with dynamic stop adjustments

### 6. Take Profit Management
- Partial profit taking at intermediate levels
- Move stops to breakeven after partial TP
- Maximize profit potential while reducing risk

### 7. Break-even Management
- Move stop loss to entry price when in profit
- Lock in risk-free trades
- Automated break-even logic

### 8. GPT Signal Execution
- Execute complex trading signals as GPT would generate
- Risk-based position sizing
- Complete signal metadata tracking

### 9. Risk Management
- Enforce maximum position size limits
- Calculate aggregate portfolio risk
- Prevent excessive exposure

## Running the Tests

### Prerequisites

1. **MT5 Terminal**: Must be running and logged into a DEMO account
2. **Python Environment**: Activate your virtual environment
3. **Settings**: Ensure `.env` file is configured

### Basic Usage

```bash
# Run the comprehensive test suite
python test_mt5_order_management.py
```

The test will:
1. Verify you're on a demo account (safety check)
2. Run through all test scenarios
3. Clean up any remaining positions
4. Provide a detailed summary

### Test Output Example

```
================================================================================
MT5 ORDER MANAGEMENT TEST SUITE
================================================================================
Time: 2025-01-06 21:30:00
⚠️  WARNING: This will place real orders on your account!
            Only run on DEMO accounts!
================================================================================

VERIFYING ACCOUNT TYPE...
Account: 1510893367
Server: FTMO-Demo
Balance: $100000.00
Type: DEMO

============================================================
TEST 1: SINGLE ORDER PLACEMENT
============================================================

Placing BUY order on EURUSD:
  Entry: 1.13994
  SL: 1.13794
  TP: 1.14294

✅ PASS - Single order - placement
     Details: Order #123456789 placed, lot size: 0.50

Position details:
  Profit: $-1.50
  Volume: 0.50
  Current price: 1.13992

Closing position...

✅ PASS - Single order - closure
     Details: Position closed successfully
```

## Test Scenarios Explained

### 1. Single Order Test
Tests basic order placement and closure, verifying all parameters are correctly set.

### 2. Order Modification Test
Places an order then modifies its SL/TP, ensuring the system can adjust positions dynamically.

### 3. Partial Close Test
Opens a larger position and closes 50%, useful for taking partial profits.

### 4. Multiple Positions Test
Opens positions on different symbols to test portfolio management capabilities.

### 5. Trailing Stop Test
Simulates dynamic stop loss adjustment as price moves favorably.

### 6. Take Profit Management Test
Demonstrates partial profit taking and moving stops to breakeven.

### 7. Break-even Test
Shows how to lock in risk-free trades by moving stops to entry.

### 8. GPT Signal Test
Executes a complex signal with full reasoning and indicator context, as the AI would generate.

### 9. Risk Management Test
Verifies position sizing limits and aggregate risk calculations.

## Safety Features

1. **Demo Account Check**: Refuses to run on live accounts
2. **Cleanup**: Automatically closes all test positions
3. **Error Handling**: Comprehensive try-catch blocks
4. **Position Tracking**: Maintains list of open positions

## Common Issues and Solutions

### "No tick data"
- Ensure the symbol is available and market is open
- Check if symbol is in Market Watch

### "Position not found"
- Position may have been stopped out
- Check if position was manually closed

### "Failed to modify"
- Price may be too close to current level
- Market may be moving too fast

### "Partial close failed"
- Volume may be too small (minimum 0.01 lots)
- Position may not be large enough to split

## Integration with GPT Trading

The order manager tested here is the same one used by the GPT trading system:

1. **Signal Execution**: `execute_signal()` converts GPT signals to MT5 orders
2. **Position Management**: `modify_position()` adjusts stops/targets
3. **Risk Control**: Automatic position sizing based on risk parameters
4. **Trade Monitoring**: Real-time position tracking and P/L calculation

## Best Practices

1. **Always Test on Demo**: Never run these tests on live accounts
2. **Monitor Execution**: Watch the MT5 terminal during tests
3. **Check Results**: Verify all test results match expectations
4. **Clean Environment**: Ensure no manual positions are open before testing

## Extended Testing

For production readiness, also test:
- Different market conditions (trending, ranging)
- Various symbols (forex, indices, commodities)
- Edge cases (minimum lots, maximum exposure)
- Network interruptions and reconnection
- Slippage and requotes handling