# ML Performance Monitor

A comprehensive monitoring solution for ML model performance in the GPT Trader system. This tool provides visualizations and reports without requiring streamlit, avoiding PyTorch compatibility issues.

## Features

- **Terminal Output**: Real-time performance statistics displayed in the console
- **HTML Reports**: Interactive web-based dashboards with Plotly visualizations
- **Static Plots**: Matplotlib-generated PNG images for easy sharing
- **No Streamlit Dependencies**: Avoids PyTorch compatibility issues

## Usage

### Quick Start (Windows)
```bash
# Run with all outputs
scripts\start_ml_monitor.bat
```

### Command Line Options
```bash
# Basic usage - shows terminal summary
python scripts/ml_performance_monitor.py

# Generate all outputs
python scripts/ml_performance_monitor.py --terminal --html --plot

# Specific options
python scripts/ml_performance_monitor.py --html --days 60  # HTML report for last 60 days
python scripts/ml_performance_monitor.py --plot            # Only generate matplotlib plots
python scripts/ml_performance_monitor.py --terminal        # Only show terminal summary
```

### Parameters
- `--db-path`: Path to the database (default: `data/trades.db`)
- `--models-dir`: Directory containing ML models (default: `models`)
- `--days`: Number of days to analyze (default: 30)
- `--html`: Generate interactive HTML report
- `--plot`: Generate static matplotlib plots
- `--terminal`: Show summary in terminal (default if no other output specified)

## Output Files

1. **HTML Report**: `reports/ml_performance_report.html`
   - Interactive Plotly charts
   - Model performance metrics
   - Recent predictions table
   - Confusion matrices

2. **Static Plots**: `reports/ml_performance_plots.png`
   - 4-panel matplotlib figure
   - Daily accuracy trends
   - Symbol-wise performance
   - Confidence distributions
   - Model comparisons

## Metrics Tracked

### Model Metrics
- Accuracy, Precision, Recall, F1 Score
- Training vs validation performance
- Model version history
- Active/inactive status

### Prediction Metrics
- Overall accuracy rate
- Accuracy by symbol
- Accuracy by signal type (BUY/SELL/HOLD)
- Confidence level analysis
- Prediction volume over time

### Performance Analysis
- Confusion matrices
- Feature importance (when available)
- Model comparison charts
- Confidence vs accuracy correlation

## Terminal Output Example
```
================================================================================
ML PERFORMANCE MONITOR - SUMMARY STATISTICS
================================================================================

📊 ACTIVE MODELS:

  Model: pattern_trader_EURUSD (v4.0.0)
  Created: 2025-01-06 14:30
  Accuracy: 0.752
  Precision: 0.748
  F1 Score: 0.745

📈 PREDICTION PERFORMANCE (Last 30 days):

  Overall Accuracy: 73.5%

  Accuracy by Symbol:
    EURUSD: 75.2% (1,240 predictions)
    GBPUSD: 72.8% (1,180 predictions)
    USDJPY: 71.9% (1,150 predictions)

  Accuracy by Signal Type:
    BUY: 74.8% (892 predictions)
    SELL: 73.1% (845 predictions)
    HOLD: 72.7% (1,833 predictions)

  Confidence Analysis:
    High Confidence (≥80%): 82.3% accuracy (712 predictions)
    Medium Confidence (60-80%): 71.5% accuracy (1,854 predictions)
    Low Confidence (<60%): 64.2% accuracy (1,004 predictions)

================================================================================
```

## Integration with Trading System

The monitor reads from:
- `model_metadata` table: Trained model information
- `ml_predictions` table: Real-time prediction results (if available)
- Model files in the `models/` directory

If the `ml_predictions` table doesn't exist, the monitor will generate sample data for demonstration purposes.

## Troubleshooting

1. **No data displayed**: Check if the database path is correct and contains model metadata
2. **Import errors**: Ensure all requirements are installed: `pip install plotly tabulate matplotlib pandas numpy`
3. **HTML report won't open**: Use the full file path displayed in the console

## Future Enhancements

- Real-time monitoring mode
- Email report generation
- Performance alerts
- A/B testing comparisons
- Model drift detection