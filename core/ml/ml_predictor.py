"""
ML Predictor for Trading Signals
Handles model loading, feature extraction, and predictions
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import pandas as pd
from datetime import datetime, timezone

from core.domain.models import MarketData, Candle
from core.domain.enums import SignalType

logger = logging.getLogger(__name__)


class MLPredictor:
    """
    ML prediction service that loads and uses trained models
    """
    
    def __init__(self, models_dir: Path = Path("models")):
        self.models_dir = models_dir
        self.loaded_models: Dict[str, Dict[str, Any]] = {}
        self._load_available_models()
        
    def _load_available_models(self):
        """Load all available ML models on initialization"""
        if not self.models_dir.exists():
            logger.warning(f"Models directory not found: {self.models_dir}")
            return
            
        # Load individual model packages
        model_files = sorted(self.models_dir.glob("*_ml_package_*.pkl"))
        
        for model_file in model_files:
            try:
                symbol = model_file.name.split('_')[0]
                
                # Skip if we already have a model for this symbol
                if symbol in self.loaded_models:
                    continue
                    
                with open(model_file, 'rb') as f:
                    model_package = pickle.load(f)
                    
                self.loaded_models[symbol] = {
                    'package': model_package,
                    'file_path': model_file,
                    'loaded_at': datetime.now(timezone.utc)
                }
                
                logger.info(f"Loaded ML model for {symbol} from {model_file.name}")
                
            except Exception as e:
                logger.error(f"Failed to load model {model_file}: {e}")
                
    def has_model(self, symbol: str) -> bool:
        """Check if ML model exists for symbol"""
        return symbol in self.loaded_models
        
    async def get_ml_prediction(
        self,
        symbol: str,
        market_data: Dict[str, MarketData]
    ) -> Dict[str, Any]:
        """
        Get ML prediction for the given symbol and market data
        
        Returns:
            Dict containing:
                - ml_enabled: bool
                - ml_signal: SignalType or None
                - ml_confidence: float (0-100) or None
                - ml_metadata: Additional information
        """
        try:
            # Check if model exists
            if not self.has_model(symbol):
                logger.debug(f"No ML model found for {symbol}")
                return {
                    'ml_enabled': False,
                    'ml_signal': None,
                    'ml_confidence': None,
                    'ml_metadata': {'reason': 'No model found'}
                }
                
            # Get the model package
            model_info = self.loaded_models[symbol]
            model_package = model_info['package']
            
            # Extract components
            pipeline = model_package.get('pipeline')
            feature_engineer = model_package.get('feature_engineer')
            feature_names = model_package.get('feature_names', [])
            
            if not pipeline or not feature_engineer:
                logger.error(f"Invalid model package for {symbol}")
                return {
                    'ml_enabled': False,
                    'ml_signal': None,
                    'ml_confidence': None,
                    'ml_metadata': {'reason': 'Invalid model package'}
                }
                
            # Convert market data to DataFrame
            df = self._market_data_to_dataframe(market_data['h1'])
            
            if df is None or len(df) < 100:  # Need sufficient data for features
                return {
                    'ml_enabled': False,
                    'ml_signal': None,
                    'ml_confidence': None,
                    'ml_metadata': {'reason': 'Insufficient market data'}
                }
                
            # Engineer features
            try:
                features = feature_engineer.transform(df)
                
                # Ensure we have the right number of features
                if features.shape[1] != len(feature_names):
                    logger.warning(
                        f"Feature mismatch for {symbol}: "
                        f"expected {len(feature_names)}, got {features.shape[1]}"
                    )
                    # Try to align if possible
                    features = features[:, :len(feature_names)]
                    
            except Exception as e:
                logger.error(f"Feature engineering failed for {symbol}: {e}")
                return {
                    'ml_enabled': False,
                    'ml_signal': None,
                    'ml_confidence': None,
                    'ml_metadata': {'reason': f'Feature engineering error: {str(e)}'}
                }
                
            # Get the latest features (most recent data point)
            latest_features = features[-1:] if len(features.shape) == 2 else features.reshape(1, -1)
            
            # Make prediction
            try:
                # Get prediction and probability
                prediction = pipeline.predict(latest_features)[0]
                
                # Get probability if available
                if hasattr(pipeline, 'predict_proba'):
                    probabilities = pipeline.predict_proba(latest_features)[0]
                    confidence = float(np.max(probabilities) * 100)
                else:
                    # Use a default confidence based on model performance
                    model_performance = model_package.get('performance', {})
                    confidence = model_performance.get('precision', 0.7) * 100
                    
                # Convert prediction to signal type
                signal = self._prediction_to_signal(prediction)
                
                # Get additional metadata
                metadata = {
                    'model_type': model_package.get('model_type', 'unknown'),
                    'training_date': model_package.get('training_date', 'unknown'),
                    'model_performance': model_package.get('performance', {}),
                    'prediction_raw': int(prediction),
                    'feature_importances': self._get_top_features(pipeline, feature_names)
                }
                
                logger.info(
                    f"ML prediction for {symbol}: {signal.value} "
                    f"with confidence {confidence:.1f}%"
                )
                
                return {
                    'ml_enabled': True,
                    'ml_signal': signal,
                    'ml_confidence': confidence,
                    'ml_metadata': metadata
                }
                
            except Exception as e:
                logger.error(f"Prediction failed for {symbol}: {e}")
                return {
                    'ml_enabled': False,
                    'ml_signal': None,
                    'ml_confidence': None,
                    'ml_metadata': {'reason': f'Prediction error: {str(e)}'}
                }
                
        except Exception as e:
            logger.error(f"ML prediction error for {symbol}: {e}", exc_info=True)
            return {
                'ml_enabled': False,
                'ml_signal': None,
                'ml_confidence': None,
                'ml_metadata': {'reason': f'Unexpected error: {str(e)}'}
            }
            
    def _market_data_to_dataframe(self, market_data: MarketData) -> Optional[pd.DataFrame]:
        """Convert MarketData to pandas DataFrame"""
        try:
            if not market_data or not market_data.candles:
                return None
                
            # Extract candle data
            data = []
            for candle in market_data.candles:
                data.append({
                    'open': candle.open,
                    'high': candle.high,
                    'low': candle.low,
                    'close': candle.close,
                    'volume': candle.volume,
                    'time': candle.time
                })
                
            df = pd.DataFrame(data)
            
            # Sort by time
            df = df.sort_values('time').reset_index(drop=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to convert market data to DataFrame: {e}")
            return None
            
    def _prediction_to_signal(self, prediction: int) -> SignalType:
        """Convert ML prediction to SignalType"""
        # Assuming: 0 = WAIT/NEUTRAL, 1 = BUY, -1 or 2 = SELL
        if prediction == 1:
            return SignalType.BUY
        elif prediction in [-1, 2]:
            return SignalType.SELL
        else:
            return SignalType.WAIT
            
    def _get_top_features(
        self,
        pipeline: Any,
        feature_names: List[str],
        top_n: int = 5
    ) -> List[Dict[str, float]]:
        """Get top feature importances if available"""
        try:
            # Try to get the actual model from pipeline
            if hasattr(pipeline, 'named_steps'):
                # It's a sklearn Pipeline
                model = pipeline.named_steps.get('model') or pipeline.named_steps.get('classifier')
            else:
                model = pipeline
                
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                
                # Get top features
                indices = np.argsort(importances)[::-1][:top_n]
                
                top_features = []
                for idx in indices:
                    if idx < len(feature_names):
                        top_features.append({
                            'name': feature_names[idx],
                            'importance': float(importances[idx])
                        })
                        
                return top_features
                
        except Exception as e:
            logger.debug(f"Could not extract feature importances: {e}")
            
        return []
        
    def get_model_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get information about loaded model for symbol"""
        if symbol not in self.loaded_models:
            return None
            
        model_info = self.loaded_models[symbol]
        package = model_info['package']
        
        return {
            'symbol': symbol,
            'loaded_at': model_info['loaded_at'].isoformat(),
            'file_path': str(model_info['file_path']),
            'model_type': package.get('model_type', 'unknown'),
            'training_date': package.get('training_date', 'unknown'),
            'performance': package.get('performance', {}),
            'feature_count': len(package.get('feature_names', []))
        }
        
    def reload_models(self):
        """Reload all models (useful if new models are deployed)"""
        logger.info("Reloading ML models...")
        self.loaded_models.clear()
        self._load_available_models()