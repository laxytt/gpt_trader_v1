# Machine Learning Integration Guide

This guide explains how to enable and use Machine Learning (ML) models in the GPT Trading System.

## Overview

The ML integration enhances the trading system by using trained models to predict trading signals based on historical patterns. When enabled, ML models work alongside GPT-4 vision analysis to make more informed trading decisions.

## Quick Start

### 1. Install ML Dependencies

```bash
pip install scikit-learn joblib
```

### 2. Train ML Models

Run the training script to create models for your symbols:

```bash
python scripts/train_ml_models.py
```

This will:
- Fetch historical data for each symbol
- Engineer features (technical indicators, price patterns, volume analysis)
- Train Random Forest models with optimized hyperparameters
- Save models that meet performance thresholds
- Automatically deploy the best models

### 3. Enable ML in Configuration

Add to your `.env` file:

```env
ML_ENABLED=true
ML_CONFIDENCE_THRESHOLD=0.7
ML_FALLBACK_TO_GPT=true
```

Or modify `config/settings.py` defaults.

### 4. Run Trading with ML

The system will automatically use ML when enabled:

```bash
python trading_loop.py
```

## Configuration Options

| Setting | Default | Description |
|---------|---------|-------------|
| `ML_ENABLED` | `false` | Enable ML-based signal generation |
| `ML_CONFIDENCE_THRESHOLD` | `0.7` | Minimum confidence for ML signals (0.0-1.0) |
| `ML_FALLBACK_TO_GPT` | `true` | Use GPT when ML confidence is low |
| `ML_MODEL_UPDATE_DAYS` | `30` | Days between model retraining |
| `ML_MIN_TRAINING_SAMPLES` | `1000` | Minimum samples required for training |
| `ML_FEATURE_LOOKBACK_PERIODS` | `[5,10,20,50]` | Lookback periods for features |
| `ML_USE_ENSEMBLE` | `false` | Use ensemble of models |

## How It Works

### Signal Generation Flow

1. **ML Enabled + High Confidence (≥0.7)**
   - ML model makes the primary decision
   - Signal generated directly from ML prediction
   - Metadata includes ML confidence and model ID

2. **ML Enabled + Low Confidence (<0.7)**
   - ML prediction passed to GPT for validation
   - GPT makes final decision with ML context
   - Combined approach leverages both systems

3. **ML Disabled**
   - Standard GPT-4 vision analysis
   - Technical indicators + VSA patterns
   - No ML involvement

### Feature Engineering

The system creates 40+ features including:

- **Price Features**: Returns, momentum, volatility
- **Volume Features**: Volume ratios, OBV, VWAP
- **Technical Indicators**: RSI, MACD, Bollinger Bands, ATR, ADX
- **Market Microstructure**: Spreads, bar statistics, efficiency ratios
- **Time Features**: Hour of day, day of week, trading sessions
- **Multi-timeframe**: H4 context features for H1 predictions

### Model Training Process

1. **Data Collection**: 6-12 months of historical data
2. **Feature Engineering**: Transform raw OHLCV to ML features
3. **Grid Search**: Optimize hyperparameters
4. **Validation**: Time-series cross-validation
5. **Deployment**: Models with F1 score > 0.5

## Testing ML Performance

### Compare ML vs Non-ML

Run the comparison script:

```bash
python scripts/test_ml_backtest.py
```

This will:
- Run backtests with ML enabled and disabled
- Compare key metrics (return, Sharpe, win rate)
- Show performance improvement

### Monitor Active Models

Check which models are deployed:

```python
from scripts.train_ml_models import MLTrainingOrchestrator

orchestrator = MLTrainingOrchestrator()
for symbol in ['EURUSD', 'GBPUSD']:
    result = orchestrator.model_repository.get_active_model(f"signal_predictor_{symbol}")
    if result:
        metadata, path = result
        print(f"{symbol}: Model {metadata.model_id}, F1={metadata.training_metrics['f1_score']:.3f}")
```

## Model Management

### Manual Model Deployment

Deploy a specific model:

```python
from core.services.model_management_service import ModelManagementService

model_service = ModelManagementService(Path("models"), model_repository)
await model_service.deploy_model("model_id_here")
```

### Model Rollback

Rollback to previous version:

```python
await model_service.rollback_model("signal_predictor_EURUSD", target_version="1.0.0")
```

## Performance Expectations

Based on testing:

- **Offline Mode Average**: -9.84% return, 35.1% win rate
- **Full GPT Mode**: +2.84% return, 26.4% win rate
- **ML Mode (Expected)**: Better risk-adjusted returns, lower drawdowns

ML models excel at:
- Filtering out low-probability trades
- Consistent risk management
- Pattern recognition in technical setups

## Troubleshooting

### No Models Found

If the system reports no active models:

1. Check the `models/` directory exists
2. Run `python scripts/train_ml_models.py`
3. Verify models meet deployment criteria (F1 > 0.5)

### Low ML Confidence

If ML confidence is consistently low:

1. Check feature engineering is working correctly
2. Ensure sufficient training data (>1000 samples)
3. Consider retraining with recent data

### Performance Issues

If ML slows down the system:

1. Reduce feature lookback periods
2. Use simpler models (fewer trees)
3. Cache feature calculations

## Best Practices

1. **Regular Retraining**: Update models monthly with new data
2. **Monitor Performance**: Track ML predictions vs actual outcomes
3. **A/B Testing**: Compare ML vs non-ML performance regularly
4. **Feature Selection**: Remove low-importance features periodically
5. **Risk Management**: Don't increase position sizes just because ML is enabled

## Advanced Usage

### Custom Feature Engineering

Modify `core/ml/feature_engineering.py` to add custom features:

```python
def _add_custom_features(self, features: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
    # Add your custom indicators
    features['custom_indicator'] = your_calculation(data)
    return features
```

### Alternative Models

Replace Random Forest with other algorithms in `train_ml_models.py`:

```python
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

# Use XGBoost instead
base_model = XGBClassifier(n_estimators=100, max_depth=5)
```

### Ensemble Models

Enable ensemble mode for combining multiple models:

```env
ML_USE_ENSEMBLE=true
```

## Next Steps

1. Train models for your symbols
2. Run backtests to verify performance
3. Enable ML in production gradually
4. Monitor and retrain regularly

For more details, see the implementation in:
- `core/ml/` - ML components
- `scripts/train_ml_models.py` - Training pipeline
- `core/services/signal_service.py` - ML integration