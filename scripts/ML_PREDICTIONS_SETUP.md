# ML Predictions Table Setup

## Summary

Successfully created the `ml_predictions` table for tracking ML model predictions and their accuracy over time.

## Table Structure

The `ml_predictions` table includes the following columns:

- `id` (INTEGER PRIMARY KEY AUTOINCREMENT) - Unique identifier
- `created_at` (TEXT) - Timestamp when the prediction was recorded
- `symbol` (TEXT NOT NULL) - Trading symbol (e.g., EURUSD)
- `predicted_signal` (TEXT NOT NULL) - ML model's predicted signal (BUY/SELL/HOLD)
- `ml_confidence` (REAL NOT NULL) - Model's confidence score (0.0-1.0)
- `actual_signal` (TEXT) - The actual signal that was executed (for comparison)
- `was_correct` (BOOLEAN) - Whether the prediction matched the actual outcome
- `model_version` (TEXT NOT NULL) - Version of the ML model used
- `features_used` (TEXT NOT NULL) - JSON string of features used for prediction
- `market_conditions` (TEXT) - Additional market context
- `prediction_timestamp` (TEXT NOT NULL) - When the prediction was made
- `signal_id` (TEXT) - Foreign key reference to the signals table

## Indexes

Created indexes for efficient querying:
- `idx_ml_predictions_symbol` - Fast lookup by symbol
- `idx_ml_predictions_timestamp` - Time-based queries
- `idx_ml_predictions_model_version` - Filter by model version
- `idx_ml_predictions_was_correct` - Performance analysis queries

## Migration Details

- Added as Migration #6 to the database
- Database version updated from 5 to 6
- Table created with 0 initial records

## Usage

This table will be used by:
1. The ML continuous improvement system to track prediction accuracy
2. Performance analytics to measure ML model effectiveness
3. A/B testing to compare different model versions
4. Model retraining to identify areas for improvement

## Scripts Created

1. **migrate_ml_predictions.py** - Standalone migration script
   - Can be run with `--yes` flag for auto-confirmation
   - Creates the table and updates migration records

2. **db_info.py** - Database information script
   - Shows all tables, row counts, and structure
   - Specifically checks for ml_predictions table

3. **run_migration.py** - General migration runner (requires full environment)
4. **check_database.py** - Database status checker (requires full environment)

## Next Steps

To use this table in your ML system:

1. Update the ML prediction code to save predictions to this table
2. Implement a comparison system to mark predictions as correct/incorrect after outcomes are known
3. Create analytics to track model performance over time
4. Use the data to continuously improve model accuracy