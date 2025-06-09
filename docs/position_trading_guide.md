# Position Trading Data Support Guide

## Overview

The MT5 data provider has been enhanced to support daily (D1) and weekly (W1) timeframes for position trading strategies. These longer timeframes are essential for identifying major trends and making strategic position trades that can last from weeks to months.

## Key Features

### 1. Extended Timeframe Support

The system now fully supports:
- **Daily (D1)**: Minimum 100 bars for proper trend analysis
- **Weekly (W1)**: Minimum 52 bars for year-long trend context
- All existing intraday timeframes (M1, M5, M15, M30, H1, H4)

### 2. Position Trading Specific Indicators

New indicators have been added specifically for position trading:
- **EMA20**: Fast moving average for trend confirmation
- **SMA50 & SMA200**: Classic position trading moving averages
- **ATR Percentage**: ATR as a percentage of price for volatility comparison
- **Weekly Range**: 5-day price range for volatility analysis
- **True Range**: Individual bar volatility measurement

### 3. Automatic Bar Count Adjustment

When requesting daily or weekly data, the system automatically ensures minimum bars:
- Daily requests with < 100 bars are adjusted to 100 bars
- Weekly requests with < 52 bars are adjusted to 52 bars

## Usage Examples

### Basic Daily/Weekly Data Retrieval

```python
from core.infrastructure.mt5.data_provider import MT5DataProvider
from core.domain.enums import TimeFrame

# Get daily data
daily_data = await data_provider.get_market_data(
    symbol="EURUSD",
    timeframe=TimeFrame.D1,
    bars=100  # Will fetch at least 100 daily bars
)

# Get weekly data
weekly_data = await data_provider.get_market_data(
    symbol="EURUSD",
    timeframe=TimeFrame.W1,
    bars=52  # Will fetch at least 52 weekly bars
)
```

### Position Trading Data (Combined)

The new `get_position_trading_data` method fetches both daily and weekly data:

```python
# Get comprehensive position trading data
position_data = await data_provider.get_position_trading_data(
    symbol="EURUSD",
    include_weekly=True  # Include weekly timeframe
)

# Access the data
daily_data = position_data['daily']   # 100+ daily bars
weekly_data = position_data['weekly']  # 52+ weekly bars (if available)
```

### Position Sizing Calculations

Calculate position sizes based on daily ATR:

```python
# Calculate position sizing metrics
sizing = data_provider.calculate_position_size_metrics(
    market_data=daily_data,
    account_balance=10000,
    risk_percentage=1.0  # Risk 1% per trade
)

# Returns:
# {
#     'current_price': 1.0850,
#     'current_atr': 0.0065,
#     'atr_percentage': 0.60,
#     'suggested_stop_distance': 0.0130,  # 2x ATR
#     'suggested_stop_price_long': 1.0720,
#     'suggested_stop_price_short': 1.0980,
#     'position_size': 76923,
#     'position_value': 83461,
#     'lots': 0.77,
#     'risk_amount': 100.0
# }
```

## Available Indicators

### Standard Indicators (All Timeframes)
- **EMA50, EMA200**: Exponential moving averages
- **RSI14**: Relative Strength Index with slope
- **ATR14**: Average True Range
- **Volume Ratio**: Current vs average volume

### Position Trading Indicators (Enhanced for D1/W1)
- **EMA20**: Fast trend indicator
- **SMA50, SMA200**: Simple moving averages for major trends
- **ATR Percentage**: Volatility as % of price
- **Weekly Range**: 5-period high-low range
- **True Range**: Individual bar volatility

## Data Provider Configuration

The MT5DataProvider includes position trading specific parameters:

```python
class MT5DataProvider:
    def __init__(self, mt5_client, chart_generator=None):
        # ... existing initialization ...
        
        # Position trading specific parameters
        self.position_trading_min_daily_bars = 100   # For trend analysis
        self.position_trading_min_weekly_bars = 52   # For yearly context
```

## Best Practices

### 1. Data Validation
Always validate that sufficient data is available:

```python
# Validate daily data availability
is_valid = await data_provider.validate_symbol_data(
    symbol="EURUSD",
    timeframe=TimeFrame.D1
)
```

### 2. Multi-Timeframe Analysis
Combine multiple timeframes for comprehensive analysis:

```python
# Get data for multiple timeframes
timeframes = [TimeFrame.H4, TimeFrame.D1, TimeFrame.W1]
multi_tf_data = await data_provider.get_multi_timeframe_data(
    symbol="EURUSD",
    timeframes=timeframes,
    bars=100
)
```

### 3. Position Sizing
Always use ATR-based position sizing for longer-term trades:
- Use 2x ATR for initial stop loss (default)
- Adjust position size based on account risk percentage
- Consider ATR percentage for volatility comparison

## Technical Implementation Details

### UnifiedDataProvider Enhancements

The UnifiedDataProvider automatically adjusts buffer bars for longer timeframes:
- **Daily (D1)**: Fetches `num_bars + 200` (minimum 300) for indicator calculation
- **Weekly (W1)**: Fetches `num_bars + 52` (minimum 104) for 2 years of data

### Indicator Calculation

All indicators are calculated in the UnifiedDataProvider with proper error handling:
- Each indicator calculation is wrapped in try-except
- Failed indicators log errors but don't stop data retrieval
- NaN values are handled gracefully in the Candle model

### Data Caching

The UnifiedDataProvider includes intelligent caching:
- Historical data is cached with 5-minute TTL
- Recent bars always fetch fresh data
- Cache is keyed by symbol, timeframe, and date range

## Error Handling

The system handles various edge cases:
- Insufficient data warnings for indicators requiring more history
- Automatic symbol resolution for broker compatibility
- Graceful fallback when weekly data is unavailable

## Testing

Use the provided test script to verify position trading data:

```bash
python test_position_trading_data.py
```

This will test:
- Daily and weekly data retrieval
- Indicator calculations
- Position sizing metrics
- Multi-timeframe data fetching

## Future Enhancements

Potential improvements for position trading:
1. Monthly (MN1) timeframe support
2. Additional position trading indicators (Ichimoku, Bollinger Bands)
3. Automated position size adjustment based on volatility
4. Integration with position trading strategies in the Trading Council