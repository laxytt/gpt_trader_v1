# Position Trading Mode Implementation

## Overview
The trading agents have been updated to seamlessly support position trading strategies on daily (D1) and weekly (W1) timeframes. The system automatically detects when these timeframes are provided and switches to position trading mode with appropriate analysis and risk management.

## Key Updates

### 1. Technical Analyst (`core/agents/technical_analyst.py`)
- **Position Mode Detection**: `_is_position_trading_mode()` detects D1/W1 timeframes
- **Daily Structure Analysis**: `_analyze_daily_structure()` identifies higher highs/lows on daily
- **Weekly Bias**: `_get_weekly_bias()` provides directional bias from weekly timeframe
- **Major Levels**: `_find_major_levels()` identifies key support/resistance for position trades
- **Pattern Recognition**: `_identify_daily_pattern()` detects major daily patterns (triangles, flags)
- **Breakout Analysis**: `_analyze_breakout_potential()` assesses breakout probability
- **Multiple Targets**: Returns 3 targets with expected holding timeframes (e.g., T1: 5 days, T2: 10 days)

### 2. Momentum Trader (`core/agents/momentum_trader.py`)
- **Multi-Period Momentum**: Calculates ROC for 5/20/50/100 days
- **Weekly Momentum**: 4-week, 12-week, and 26-week momentum analysis
- **Trend Duration**: `_calculate_trend_duration()` measures how long current trend has lasted
- **Trend Quality**: `_assess_position_trend_quality()` evaluates smoothness vs choppiness
- **Divergence Detection**: `_check_momentum_divergence()` identifies momentum divergences
- **Volume Trend**: `_analyze_position_volume_trend()` confirms momentum with volume
- **Trend Stage**: Identifies early/middle/late/exhaustion stages
- **Continuation Probability**: Assesses likelihood of multi-week trend continuation

### 3. Risk Manager (`core/agents/risk_manager.py`)
- **Position Stop Calculation**: `_calculate_position_stop_loss()` uses 3-5x daily ATR
- **Multi-Day VaR**: `_calculate_multi_day_var()` computes 5-day Value at Risk
- **Maximum Drawdown**: `_calculate_max_drawdown()` tracks 100-day max drawdown
- **Correlation Risk**: `_assess_position_correlation_risk()` evaluates portfolio correlation
- **Gap Risk**: `_assess_gap_risk()` analyzes weekend/overnight gap exposure
- **Time Decay**: `_assess_time_decay_risk()` evaluates position decay over time
- **Position Sizing**: Kelly Criterion adapted for longer holding periods
- **Scaling Plans**: Provides entry/exit scaling recommendations

## Usage

### Automatic Mode Detection
The system automatically detects position trading mode when daily or weekly timeframes are present:

```python
# Position trading mode (automatic)
market_data = {
    'd1': daily_data,    # Daily timeframe
    'w1': weekly_data,   # Weekly timeframe (optional)
    'h4': h4_data        # H4 for refined entry timing (optional)
}

# Standard scalping/day trading mode
market_data = {
    'h1': hourly_data,   # Hourly timeframe
    'h4': h4_data        # 4-hour timeframe
}
```

### Key Differences in Position Mode

1. **Stop Loss Placement**
   - Position Mode: 3-5x daily ATR (wider stops for volatility)
   - Standard Mode: 1-2x hourly ATR (tighter stops)

2. **Profit Targets**
   - Position Mode: Multiple targets with timeframes (T1: 5 days, T2: 10 days, T3: 20 days)
   - Standard Mode: Single target based on risk/reward ratio

3. **Analysis Focus**
   - Position Mode: Major trends, weekly bias, breakout patterns, trend quality
   - Standard Mode: Intraday patterns, momentum, short-term support/resistance

4. **Risk Management**
   - Position Mode: Multi-day VaR, correlation risk, gap risk, time stops
   - Standard Mode: Intraday risk, spread impact, immediate volatility

5. **Entry Strategy**
   - Position Mode: Breakouts, pullbacks to major levels, accumulation zones
   - Standard Mode: Momentum entries, quick reversals, scalping setups

## Testing

Run the test script to verify position mode detection:

```bash
# Simple detection test
python3 test_position_mode_simple.py

# Full test with mock data (no API calls)
python3 test_position_trading.py --mock

# Full test with real API calls
python3 test_position_trading.py
```

## Integration Notes

1. **No Code Changes Required**: The trading loop and council system work unchanged
2. **Backward Compatible**: Existing H1/H4 strategies continue to work normally
3. **Flexible Timeframes**: Can mix timeframes (e.g., D1 + H4 for position entry refinement)
4. **Council Consensus**: All agents adapt their debate style for position trading context

## Example Position Trade Analysis

When in position mode, agents provide analysis like:

**Technical Analyst**:
- "Strong bullish trend on daily, confirmed by weekly structure"
- "Major resistance at 1.2500, support at 1.2100"
- "Ascending triangle pattern, high breakout potential"
- "Entry on pullback to 1.2200-1.2220 zone"

**Momentum Trader**:
- "Accelerating momentum, ROC(20d): +3.5%"
- "Trend in middle stage, high continuation probability"
- "Volume confirming upward momentum"
- "Trail stop using 20-day EMA"

**Risk Manager**:
- "Position risk: Medium, 5-day VaR: 2.5%"
- "Position size: 0.3 lots (30% Kelly)"
- "Stop: 100 pips (4x daily ATR)"
- "Max holding: 20 days before time stop"

## Benefits

1. **Capture Larger Moves**: Position trades target 200-500+ pip moves vs 20-50 pip scalps
2. **Lower Transaction Costs**: Fewer trades mean less spread/commission impact
3. **Less Screen Time**: Set and monitor vs constant watching
4. **Better Risk/Reward**: Wider stops allow for larger targets
5. **Portfolio Approach**: Multiple positions across uncorrelated pairs