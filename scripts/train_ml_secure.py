"""
Secure ML model training script that saves models with checksums and signatures.
Based on train_ml_production.py but with security enhancements.
"""

import os
import sys
import json
import hashlib
import argparse
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List, Tuple, Optional

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from core.ml.model_trainer import ModelTrainer
from core.ml.secure_model_loader import SecureModelLoader
from core.infrastructure.database.repositories import TradeRepository, SignalRepository
from config.settings import get_settings

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SecureModelTrainer:
    """Trains and saves ML models with security features"""
    
    def __init__(
        self,
        models_dir: str = "models",
        secret_key: Optional[str] = None,
        authorized_by: str = "system"
    ):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        # Initialize secure loader for saving
        self.secure_loader = SecureModelLoader(
            models_dir=self.models_dir,
            secret_key=secret_key or os.environ.get('MODEL_SECRET_KEY')
        )
        
        self.authorized_by = authorized_by
        
        # Initialize settings and repositories
        settings = get_settings()
        self.trade_repo = TradeRepository(settings.database.db_path)
        self.signal_repo = SignalRepository(settings.database.db_path)
        
        logger.info("SecureModelTrainer initialized")
    
    def train_model_for_symbol(
        self,
        symbol: str,
        days_back: int = 90,
        min_trades: int = 50,
        test_size: float = 0.2
    ) -> Dict[str, Any]:
        """
        Train a model for a specific symbol and save it securely.
        
        Args:
            symbol: Trading symbol
            days_back: Days of historical data to use
            min_trades: Minimum trades required for training
            test_size: Test set size for evaluation
            
        Returns:
            Training results with security metadata
        """
        logger.info(f"Training secure model for {symbol}")
        
        # Initialize trainer
        trainer = ModelTrainer(
            symbol=symbol,
            trade_repository=self.trade_repo,
            signal_repository=self.signal_repo
        )
        
        # Prepare data
        logger.info(f"Preparing data for {symbol} (last {days_back} days)")
        X_train, X_test, y_train, y_test, feature_names = trainer.prepare_data(
            days_back=days_back,
            min_trades=min_trades,
            test_size=test_size
        )
        
        if X_train is None:
            logger.warning(f"Insufficient data for {symbol}")
            return {
                'symbol': symbol,
                'status': 'failed',
                'reason': 'insufficient_data'
            }
        
        # Train model
        logger.info(f"Training model for {symbol} with {len(X_train)} samples")
        best_model, best_params, cv_scores = trainer.train_model(X_train, y_train)
        
        # Evaluate model
        performance = trainer.evaluate_model(best_model, X_test, y_test)
        logger.info(
            f"{symbol} Model Performance - "
            f"Accuracy: {performance['accuracy']:.3f}, "
            f"F1: {performance['f1_score']:.3f}"
        )
        
        # Create model package
        model_package = {
            'pipeline': best_model,
            'feature_engineer': trainer.feature_engineer,
            'feature_names': feature_names,
            'best_params': best_params,
            'cv_scores': cv_scores,
            'performance': performance,
            'metadata': {
                'symbol': symbol,
                'training_date': datetime.now(timezone.utc).isoformat(),
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'days_back': days_back,
                'data_hash': self._calculate_data_hash(X_train, y_train),
                'version': f"{datetime.now().strftime('%Y%m%d')}.1.0"
            }
        }
        
        # Save model securely
        model_name = f"{symbol}_pattern_trader"
        version = model_package['metadata']['version']
        
        try:
            model_path = self.secure_loader.save_model(
                model=model_package,
                model_name=model_name,
                version=version,
                model_type='joblib',
                metadata={
                    'feature_names': feature_names,
                    'performance_metrics': performance,
                    'training_data_hash': model_package['metadata']['data_hash']
                },
                authorized_by=self.authorized_by
            )
            
            logger.info(f"Model saved securely to {model_path}")
            
            # Verify the saved model
            verification = self.secure_loader.verify_model(model_path)
            
            return {
                'symbol': symbol,
                'status': 'success',
                'model_path': str(model_path),
                'performance': performance,
                'verification': verification,
                'metadata': model_package['metadata']
            }
            
        except Exception as e:
            logger.error(f"Failed to save model for {symbol}: {e}")
            return {
                'symbol': symbol,
                'status': 'failed',
                'reason': str(e)
            }
    
    def _calculate_data_hash(self, X_train, y_train) -> str:
        """Calculate hash of training data for integrity tracking"""
        # Create a deterministic representation of the data
        data_str = f"{X_train.shape}_{y_train.shape}_{X_train.mean():.6f}_{y_train.mean():.6f}"
        return hashlib.sha256(data_str.encode()).hexdigest()[:16]
    
    def train_all_symbols(
        self,
        symbols: List[str],
        days_back: int = 90,
        min_trades: int = 50
    ) -> Dict[str, Any]:
        """
        Train models for multiple symbols.
        
        Args:
            symbols: List of symbols to train
            days_back: Days of historical data
            min_trades: Minimum trades required
            
        Returns:
            Summary of training results
        """
        results = {
            'successful': [],
            'failed': [],
            'summary': {}
        }
        
        for symbol in symbols:
            logger.info(f"\n{'='*50}")
            logger.info(f"Training model for {symbol}")
            logger.info(f"{'='*50}")
            
            result = self.train_model_for_symbol(
                symbol=symbol,
                days_back=days_back,
                min_trades=min_trades
            )
            
            if result['status'] == 'success':
                results['successful'].append(result)
            else:
                results['failed'].append(result)
        
        # Update allowed models list
        self._update_allowed_models()
        
        # Create summary
        results['summary'] = {
            'total_symbols': len(symbols),
            'successful': len(results['successful']),
            'failed': len(results['failed']),
            'timestamp': datetime.now().isoformat()
        }
        
        # Save training report
        report_file = self.models_dir / f"training_report_secure_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"\nTraining complete. Report saved to {report_file}")
        logger.info(f"Successful: {results['summary']['successful']}/{results['summary']['total_symbols']}")
        
        return results
    
    def _update_allowed_models(self):
        """Update the allowed models list after training"""
        allowed_models = {}
        
        # Get all models with metadata
        metadata_dir = self.models_dir / '.metadata'
        if metadata_dir.exists():
            for metadata_file in metadata_dir.glob("*.json"):
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    
                    model_name = metadata_file.stem
                    allowed_models[model_name] = metadata['checksum_sha256']
                    
                except Exception as e:
                    logger.error(f"Error reading metadata {metadata_file}: {e}")
        
        # Save allowed models list
        allowed_models_file = self.models_dir / 'allowed_models.json'
        with open(allowed_models_file, 'w') as f:
            json.dump(allowed_models, f, indent=2)
        
        logger.info(f"Updated allowed models list with {len(allowed_models)} models")


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(
        description="Train ML models with security features"
    )
    parser.add_argument(
        '--symbols',
        type=str,
        nargs='+',
        default=['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD'],
        help='Symbols to train models for'
    )
    parser.add_argument(
        '--days-back',
        type=int,
        default=90,
        help='Days of historical data to use'
    )
    parser.add_argument(
        '--min-trades',
        type=int,
        default=50,
        help='Minimum trades required for training'
    )
    parser.add_argument(
        '--models-dir',
        type=str,
        default='models',
        help='Directory to save models'
    )
    parser.add_argument(
        '--authorized-by',
        type=str,
        default=os.environ.get('USER', 'system'),
        help='Who authorized this training'
    )
    
    args = parser.parse_args()
    
    # Check for secret key
    if not os.environ.get('MODEL_SECRET_KEY'):
        logger.warning(
            "MODEL_SECRET_KEY not set. Models will be saved without signatures. "
            "Set this environment variable for maximum security."
        )
    
    # Initialize trainer
    trainer = SecureModelTrainer(
        models_dir=args.models_dir,
        authorized_by=args.authorized_by
    )
    
    # Train models
    logger.info(f"Training secure models for symbols: {', '.join(args.symbols)}")
    results = trainer.train_all_symbols(
        symbols=args.symbols,
        days_back=args.days_back,
        min_trades=args.min_trades
    )
    
    # Print summary
    print(f"\n{'='*60}")
    print("TRAINING SUMMARY")
    print(f"{'='*60}")
    print(f"Total symbols: {results['summary']['total_symbols']}")
    print(f"Successful: {results['summary']['successful']}")
    print(f"Failed: {results['summary']['failed']}")
    
    if results['successful']:
        print(f"\n{'='*60}")
        print("SUCCESSFUL MODELS")
        print(f"{'='*60}")
        for result in results['successful']:
            print(f"\n{result['symbol']}:")
            print(f"  Model: {Path(result['model_path']).name}")
            print(f"  Accuracy: {result['performance']['accuracy']:.3f}")
            print(f"  F1 Score: {result['performance']['f1_score']:.3f}")
            print(f"  Checksum Valid: {result['verification']['checksum_valid']}")
            print(f"  Signature Valid: {result['verification']['signature_valid']}")
    
    if results['failed']:
        print(f"\n{'='*60}")
        print("FAILED MODELS")
        print(f"{'='*60}")
        for result in results['failed']:
            print(f"\n{result['symbol']}: {result.get('reason', 'Unknown error')}")


if __name__ == "__main__":
    main()