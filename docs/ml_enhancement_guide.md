# ML Enhancement and Continuous Learning Guide

This guide explains how to use the ML enhancement system to improve model performance automatically.

## Overview

The ML enhancement system consists of several components:

1. **Continuous Learning System** - Monitors performance and triggers retraining
2. **Performance Analytics** - Comprehensive performance tracking and analysis
3. **ML Scheduler** - Automates regular checks and updates
4. **ML Dashboard** - Interactive monitoring interface

## Quick Start

### 1. Enable ML in Settings

Edit `config/settings.py` or set environment variables:

```python
# In settings.py
ml:
    enabled: true
    confidence_threshold: 0.7
    update_frequency_days: 30  # How often to check for updates
```

### 2. Run Initial Training

Train models for your symbols:

```bash
python scripts/train_ml_production.py --symbols EURUSD GBPUSD USDCAD AUDUSD
```

### 3. Start Continuous Learning

#### Option A: Run as Service
```bash
# Run continuous learning daemon
python scripts/ml_continuous_learning.py
```

#### Option B: Use Scheduler
```bash
# Run scheduler daemon (recommended)
python scripts/ml_scheduler.py daemon

# Or run specific tasks
python scripts/ml_scheduler.py check   # Performance check only
python scripts/ml_scheduler.py update  # Model update only
```

#### Option C: Windows Task Scheduler
Create scheduled tasks for automated execution:

```powershell
# Create daily performance check task
schtasks /create /tn "ML_Performance_Check" /tr "python D:\gpt_trader_v1\scripts\ml_scheduler.py check" /sc daily /st 06:00

# Create monthly model update task
schtasks /create /tn "ML_Model_Update" /tr "python D:\gpt_trader_v1\scripts\ml_scheduler.py update" /sc monthly /d 1 /st 02:00
```

### 4. Monitor Performance

#### Using Dashboard (requires streamlit)
```bash
pip install streamlit
streamlit run scripts/ml_improvement_dashboard.py
```

#### Using Performance Analytics
```bash
python scripts/performance_analytics.py
```

## How It Works

### Automatic Model Improvement

1. **Performance Monitoring**
   - Tracks win rate, profitability, and signal quality
   - Evaluates each symbol independently
   - Triggers retraining when performance drops

2. **Retraining Criteria**
   - Win rate < 45%
   - Negative average profit
   - Low signal execution rate (< 30%)
   - Insufficient trades (< 10 in evaluation period)

3. **Validation Process**
   - New models are backtested on recent 30 days
   - Only deployed if performance improves
   - Old models are archived for rollback

### Performance Analytics

The system tracks:
- **Basic Metrics**: Win rate, profit/loss, trade count
- **Advanced Metrics**: Sharpe ratio, max drawdown, profit factor
- **Pattern Analysis**: Winning trade patterns, best trading hours
- **Signal Quality**: Execution rate, signal distribution

### Notifications

If Telegram is configured, you'll receive notifications for:
- Models needing attention
- Successful model improvements
- System errors

## Configuration Options

### ML Settings

```python
MLSettings:
    enabled: bool = True                      # Enable/disable ML
    confidence_threshold: float = 0.7         # Min confidence for ML signals
    fallback_to_gpt: bool = True             # Use GPT when ML confidence is low
    update_frequency_days: int = 30          # Days between model checks
    min_training_samples: int = 1000         # Min samples for training
    use_ensemble: bool = False               # Use ensemble models
```

### Offline Validation

Reduce GPT costs by increasing validation threshold:

```python
TradingSettings:
    offline_validation_threshold: float = 0.7  # Skip GPT if score < threshold
```

## Best Practices

### 1. Regular Monitoring
- Check performance weekly
- Review improvement history monthly
- Adjust retraining criteria based on results

### 2. Data Quality
- Ensure sufficient historical data (6+ months)
- Clean outliers before retraining
- Verify data continuity

### 3. Model Management
- Keep last 3 model versions archived
- Test new models thoroughly before deployment
- Document significant changes

### 4. Performance Optimization
- Start with conservative thresholds
- Gradually adjust based on results
- Monitor resource usage

## Troubleshooting

### Models Not Updating
1. Check if ML is enabled in settings
2. Verify sufficient trades for evaluation
3. Check logs in `logs/ml_scheduler.log`

### Poor Model Performance
1. Increase training data period
2. Review feature engineering
3. Adjust label creation criteria
4. Consider market regime changes

### High Resource Usage
1. Increase update frequency interval
2. Reduce number of symbols
3. Use smaller feature sets

## Advanced Usage

### Custom Retraining Criteria

Edit `_should_retrain()` in `ml_continuous_learning.py`:

```python
def _should_retrain(self, metrics: Dict[str, Any]) -> bool:
    reasons = []
    
    # Add custom criteria
    if metrics['sharpe_ratio'] < 0.5:
        reasons.append("Low Sharpe ratio")
    
    if metrics['max_drawdown'] > 0.2:
        reasons.append("High drawdown")
    
    # ... existing criteria ...
    
    return len(reasons) > 0
```

### A/B Testing

Compare model versions:

```python
# In trading_loop.py, implement A/B testing
if random.random() < 0.5:
    # Use new model
    signal = await signal_service.generate_signal_with_ml(symbol)
else:
    # Use old model
    signal = await signal_service.generate_signal(symbol)

# Track performance separately
```

### Ensemble Models

Enable ensemble in settings and modify signal generation:

```python
# Multiple models vote on signals
predictions = []
for model in ensemble_models:
    pred = model.predict(features)
    predictions.append(pred)

# Majority vote or weighted average
final_signal = aggregate_predictions(predictions)
```

## Monitoring Checklist

Daily:
- [ ] Check for system errors
- [ ] Verify trades are executing
- [ ] Monitor win rate trends

Weekly:
- [ ] Run performance analytics
- [ ] Review model recommendations
- [ ] Check resource usage

Monthly:
- [ ] Review improvement history
- [ ] Analyze pattern changes
- [ ] Update retraining criteria if needed
- [ ] Archive old models

## Summary

The ML enhancement system enables:
1. **Automatic Performance Monitoring** - Continuous tracking of model effectiveness
2. **Smart Retraining** - Updates models only when performance degrades
3. **Validation & Safety** - Ensures new models are better before deployment
4. **Cost Optimization** - Reduces GPT calls through offline validation
5. **Comprehensive Analytics** - Deep insights into trading performance

By following this guide, your trading system will continuously improve and adapt to changing market conditions automatically.