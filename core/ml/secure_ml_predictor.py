"""
Secure ML predictor that uses SecureModelLoader for safe model loading.
Drop-in replacement for MLPredictor with enhanced security.
"""

import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import pandas as pd

from core.domain.models import MarketData
from core.domain.enums import RiskClass
from core.domain.exceptions import ValidationError, SecurityError
from core.ml.feature_engineering import FeatureEngineer
from core.ml.secure_model_loader import SecureModelLoader, create_secure_loader
from core.infrastructure.database.ml_prediction_logger import MLPredictionLogger

logger = logging.getLogger(__name__)


class SecureMLPredictor:
    """
    Secure ML predictor service that safely loads and uses ML models.
    Provides predictions for trading signals with security verification.
    """
    
    def __init__(
        self, 
        models_dir: str = "models",
        db_path: str = "data/trades.db",
        secret_key: Optional[str] = None,
        enable_checksum_verification: bool = True,
        enable_signature_verification: bool = True
    ):
        """
        Initialize secure ML predictor.
        
        Args:
            models_dir: Directory containing ML models
            db_path: Path to database for prediction logging
            secret_key: Secret key for model signature verification
            enable_checksum_verification: Whether to verify model checksums
            enable_signature_verification: Whether to verify model signatures
        """
        self.models_dir = Path(models_dir)
        self.enable_checksum_verification = enable_checksum_verification
        self.enable_signature_verification = enable_signature_verification
        
        # Initialize secure model loader
        self.model_loader = create_secure_loader(
            models_dir=self.models_dir,
            secret_key=secret_key
        )
        
        # Initialize prediction logger
        self.prediction_logger = MLPredictionLogger(db_path)
        
        # Cache for loaded models
        self._model_cache: Dict[str, Dict[str, Any]] = {}
        
        # Load models
        self._load_all_models()
        
        logger.info(f"SecureMLPredictor initialized with {len(self._model_cache)} models")
    
    def _load_all_models(self):
        """Load all available ML models with security verification"""
        if not self.models_dir.exists():
            logger.warning(f"Models directory not found: {self.models_dir}")
            return
        
        # Look for model package files
        model_files = list(self.models_dir.glob("*_ml_package_*.pkl"))
        
        for model_file in model_files:
            try:
                # Extract symbol from filename (e.g., "EURUSD_ml_package_v1.0.pkl")
                symbol = model_file.stem.split('_')[0]
                
                logger.info(f"Loading secure model for {symbol} from {model_file}")
                
                # Load with security verification
                model_package = self.model_loader.load_model(
                    model_file,
                    verify_checksum=self.enable_checksum_verification,
                    verify_signature=self.enable_signature_verification
                )
                
                # Validate model package structure
                required_keys = ['pipeline', 'feature_engineer', 'feature_names']
                for key in required_keys:
                    if key not in model_package:
                        raise ValidationError(f"Model package missing required key: {key}")
                
                # Validate model has expected methods
                pipeline = model_package['pipeline']
                if not hasattr(pipeline, 'predict_proba'):
                    raise ValidationError("Model pipeline missing predict_proba method")
                
                # Cache the model
                self._model_cache[symbol] = model_package
                
                # Log model info
                metadata = model_package.get('metadata', {})
                performance = model_package.get('performance', {})
                logger.info(
                    f"Loaded ML model for {symbol} - "
                    f"Version: {metadata.get('version', 'unknown')}, "
                    f"Accuracy: {performance.get('accuracy', 0):.3f}"
                )
                
            except SecurityError as e:
                logger.error(f"Security check failed for {model_file}: {e}")
                # Don't load potentially compromised models
            except Exception as e:
                logger.error(f"Failed to load model {model_file}: {e}")
    
    def predict_signal(
        self,
        symbol: str,
        market_data: MarketData,
        news_sentiment: float = 0.0,
        vsa_signal: float = 0.0
    ) -> Tuple[Optional[float], Optional[Dict[str, Any]]]:
        """
        Generate ML prediction for trading signal with security.
        
        Args:
            symbol: Trading symbol
            market_data: Current market data
            news_sentiment: News sentiment score (-1 to 1)
            vsa_signal: Volume spread analysis signal
            
        Returns:
            Tuple of (ml_confidence, prediction_details)
        """
        # Check if we have a model for this symbol
        if symbol not in self._model_cache:
            logger.warning(f"No ML model available for {symbol}")
            return None, None
        
        try:
            model_package = self._model_cache[symbol]
            pipeline = model_package['pipeline']
            feature_engineer = model_package['feature_engineer']
            feature_names = model_package['feature_names']
            
            # Validate feature engineer
            if not isinstance(feature_engineer, FeatureEngineer):
                # Try to recreate if it's a dict of parameters
                if isinstance(feature_engineer, dict):
                    feature_engineer = FeatureEngineer()
                else:
                    raise ValidationError("Invalid feature engineer in model package")
            
            # Extract features
            features_dict = feature_engineer.extract_features(
                market_data.h1_candles,
                market_data.h4_candles,
                market_data.current_price,
                news_sentiment,
                vsa_signal
            )
            
            # Create feature vector in correct order
            feature_vector = []
            for feature_name in feature_names:
                if feature_name in features_dict:
                    feature_vector.append(features_dict[feature_name])
                else:
                    # Handle missing features with defaults
                    logger.warning(f"Missing feature {feature_name}, using 0")
                    feature_vector.append(0)
            
            # Convert to numpy array
            X = np.array(feature_vector).reshape(1, -1)
            
            # Validate feature vector
            if np.any(np.isnan(X)) or np.any(np.isinf(X)):
                logger.warning(f"Invalid features detected for {symbol}")
                return None, None
            
            # Get prediction
            try:
                # Get probability predictions
                probabilities = pipeline.predict_proba(X)[0]
                
                # Get the actual prediction
                prediction = pipeline.predict(X)[0]
                
                # Calculate confidence based on prediction
                if prediction == 1:  # BUY signal
                    ml_confidence = probabilities[1]  # Probability of class 1
                elif prediction == -1:  # SELL signal
                    ml_confidence = probabilities[0]  # Probability of class -1
                else:  # WAIT signal (0)
                    ml_confidence = max(probabilities) if len(probabilities) > 2 else 0.5
                
                # Adjust confidence to -1 to 1 range for SELL signals
                if prediction == -1:
                    ml_confidence = -ml_confidence
                
                # Create prediction details
                prediction_details = {
                    'prediction': int(prediction),
                    'probabilities': probabilities.tolist(),
                    'confidence': float(ml_confidence),
                    'features_used': len(feature_names),
                    'model_version': model_package.get('metadata', {}).get('version', 'unknown'),
                    'feature_importance': self._get_feature_importance(pipeline, feature_names),
                    'timestamp': datetime.now().isoformat()
                }
                
                # Log the prediction
                self.prediction_logger.log_prediction(
                    symbol=symbol,
                    ml_prediction=int(prediction),
                    ml_confidence=float(abs(ml_confidence)),
                    features=features_dict,
                    model_version=model_package.get('metadata', {}).get('version', 'unknown')
                )
                
                return ml_confidence, prediction_details
                
            except Exception as e:
                logger.error(f"Model prediction failed for {symbol}: {e}")
                return None, None
                
        except Exception as e:
            logger.error(f"ML prediction error for {symbol}: {e}")
            return None, None
    
    def _get_feature_importance(
        self, 
        pipeline: Any, 
        feature_names: List[str]
    ) -> Optional[Dict[str, float]]:
        """Extract feature importance from the model if available"""
        try:
            # Get the actual model from the pipeline
            if hasattr(pipeline, 'named_steps'):
                # It's a sklearn Pipeline
                model = pipeline.named_steps.get('classifier') or pipeline.named_steps.get('model')
            else:
                model = pipeline
            
            # Try to get feature importance
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                return {
                    name: float(importance) 
                    for name, importance in zip(feature_names, importances)
                }
            elif hasattr(model, 'coef_'):
                # For linear models
                coef = model.coef_
                if coef.ndim > 1:
                    coef = np.abs(coef).mean(axis=0)
                return {
                    name: float(abs(c)) 
                    for name, c in zip(feature_names, coef)
                }
        except Exception as e:
            logger.debug(f"Could not extract feature importance: {e}")
        
        return None
    
    def get_model_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get information about a loaded model"""
        if symbol not in self._model_cache:
            return None
        
        model_package = self._model_cache[symbol]
        return {
            'symbol': symbol,
            'loaded': True,
            'metadata': model_package.get('metadata', {}),
            'performance': model_package.get('performance', {}),
            'feature_count': len(model_package.get('feature_names', [])),
            'feature_names': model_package.get('feature_names', [])[:10]  # First 10 features
        }
    
    def get_all_models_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all loaded models"""
        return {
            symbol: self.get_model_info(symbol)
            for symbol in self._model_cache
        }
    
    def reload_models(self):
        """Reload all models with security verification"""
        logger.info("Reloading ML models with security verification...")
        self._model_cache.clear()
        self._load_all_models()
    
    def verify_all_models(self) -> Dict[str, Dict[str, Any]]:
        """Verify all models in the models directory"""
        verification_results = {}
        
        model_files = list(self.models_dir.glob("*.pkl")) + list(self.models_dir.glob("*.joblib"))
        
        for model_file in model_files:
            verification_results[model_file.name] = self.model_loader.verify_model(model_file)
        
        return verification_results
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        # Clean up if needed
        pass


# Factory function for backward compatibility
def create_secure_ml_predictor(
    models_dir: str = "models",
    db_path: str = "data/trades.db",
    secret_key: Optional[str] = None
) -> SecureMLPredictor:
    """
    Create a secure ML predictor instance.
    
    Args:
        models_dir: Directory containing ML models
        db_path: Database path for prediction logging
        secret_key: Secret key for model signatures
        
    Returns:
        SecureMLPredictor instance
    """
    return SecureMLPredictor(
        models_dir=models_dir,
        db_path=db_path,
        secret_key=secret_key,
        enable_checksum_verification=True,
        enable_signature_verification=True
    )


# Export public API
__all__ = ['SecureMLPredictor', 'create_secure_ml_predictor']