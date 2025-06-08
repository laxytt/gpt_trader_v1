"""
ML Prediction Logger - Logs ML predictions for performance monitoring
"""

import sqlite3
import json
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class MLPredictionLogger:
    """Logs ML predictions to database for performance monitoring"""
    
    def __init__(self, db_path: str = "data/trades.db"):
        self.db_path = Path(db_path)
        self._ensure_table_exists()
    
    def _ensure_table_exists(self):
        """Create ML predictions table if it doesn't exist"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS ml_predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    symbol TEXT NOT NULL,
                    predicted_signal TEXT NOT NULL,
                    ml_confidence REAL NOT NULL,
                    actual_signal TEXT,
                    was_correct BOOLEAN,
                    model_version TEXT,
                    features_used TEXT,
                    execution_time_ms REAL,
                    notes TEXT
                )
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_ml_predictions_created 
                ON ml_predictions(created_at DESC)
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_ml_predictions_symbol 
                ON ml_predictions(symbol, created_at DESC)
            ''')
    
    def log_prediction(
        self,
        symbol: str,
        predicted_signal: str,
        ml_confidence: float,
        model_version: str,
        features_used: Dict[str, Any],
        execution_time_ms: Optional[float] = None,
        notes: Optional[str] = None
    ) -> int:
        """
        Log an ML prediction
        
        Args:
            symbol: Trading symbol (e.g., 'EURUSD')
            predicted_signal: Predicted signal ('BUY', 'SELL', 'HOLD')
            ml_confidence: Confidence level (0.0 to 1.0)
            model_version: Version of the model used
            features_used: Dictionary of features used for prediction
            execution_time_ms: Time taken for prediction in milliseconds
            notes: Optional notes
            
        Returns:
            ID of the logged prediction
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                INSERT INTO ml_predictions 
                (symbol, predicted_signal, ml_confidence, model_version, 
                 features_used, execution_time_ms, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                symbol,
                predicted_signal,
                ml_confidence,
                model_version,
                json.dumps(features_used),
                execution_time_ms,
                notes
            ))
            
            prediction_id = cursor.lastrowid
            logger.debug(f"Logged ML prediction {prediction_id} for {symbol}: {predicted_signal} ({ml_confidence:.1%})")
            
            return prediction_id
    
    def update_actual_result(
        self,
        prediction_id: int,
        actual_signal: str,
        was_correct: Optional[bool] = None
    ):
        """
        Update a prediction with actual result
        
        Args:
            prediction_id: ID of the prediction to update
            actual_signal: Actual signal that occurred
            was_correct: Whether prediction was correct (auto-calculated if None)
        """
        with sqlite3.connect(self.db_path) as conn:
            if was_correct is None:
                # Auto-calculate correctness
                cursor = conn.execute(
                    'SELECT predicted_signal FROM ml_predictions WHERE id = ?',
                    (prediction_id,)
                )
                row = cursor.fetchone()
                if row:
                    was_correct = row[0] == actual_signal
            
            conn.execute('''
                UPDATE ml_predictions 
                SET actual_signal = ?, was_correct = ?
                WHERE id = ?
            ''', (actual_signal, was_correct, prediction_id))
            
            logger.debug(f"Updated prediction {prediction_id} with actual result: {actual_signal} (correct: {was_correct})")
    
    def get_recent_accuracy(self, symbol: Optional[str] = None, hours: int = 24) -> Dict[str, Any]:
        """
        Get accuracy statistics for recent predictions
        
        Args:
            symbol: Filter by symbol (None for all)
            hours: Number of hours to look back
            
        Returns:
            Dictionary with accuracy statistics
        """
        with sqlite3.connect(self.db_path) as conn:
            # Base query
            query = '''
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN was_correct = 1 THEN 1 ELSE 0 END) as correct,
                    AVG(ml_confidence) as avg_confidence
                FROM ml_predictions
                WHERE created_at >= datetime('now', ?)
                AND actual_signal IS NOT NULL
            '''
            params = [f'-{hours} hours']
            
            if symbol:
                query += ' AND symbol = ?'
                params.append(symbol)
            
            cursor = conn.execute(query, params)
            row = cursor.fetchone()
            
            if row and row[0] > 0:
                return {
                    'total_predictions': row[0],
                    'correct_predictions': row[1],
                    'accuracy': row[1] / row[0] if row[0] > 0 else 0,
                    'avg_confidence': row[2],
                    'period_hours': hours,
                    'symbol': symbol
                }
            else:
                return {
                    'total_predictions': 0,
                    'correct_predictions': 0,
                    'accuracy': 0,
                    'avg_confidence': 0,
                    'period_hours': hours,
                    'symbol': symbol
                }


# Example usage in trading system:
"""
# In your signal generation service:

from core.infrastructure.database.ml_prediction_logger import MLPredictionLogger

class MLEnhancedSignalService:
    def __init__(self):
        self.ml_logger = MLPredictionLogger()
    
    async def generate_signal(self, symbol: str, market_data: Dict) -> Signal:
        # Extract features
        features = self.feature_engineer.extract_features(market_data)
        
        # Make prediction
        start_time = time.time()
        ml_prediction = self.model.predict(features)
        ml_confidence = self.model.predict_proba(features).max()
        execution_time = (time.time() - start_time) * 1000  # ms
        
        # Log the prediction
        prediction_id = self.ml_logger.log_prediction(
            symbol=symbol,
            predicted_signal=ml_prediction,
            ml_confidence=ml_confidence,
            model_version=self.model_version,
            features_used=features,
            execution_time_ms=execution_time
        )
        
        # Store prediction_id for later update
        signal.ml_prediction_id = prediction_id
        
        return signal
    
    async def on_trade_closed(self, trade: Trade, signal: Signal):
        # Update prediction with actual result
        if hasattr(signal, 'ml_prediction_id'):
            actual_signal = 'BUY' if trade.profit > 0 else 'SELL'
            self.ml_logger.update_actual_result(
                prediction_id=signal.ml_prediction_id,
                actual_signal=actual_signal
            )
"""