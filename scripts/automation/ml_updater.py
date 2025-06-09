"""
Automated ML model training and deployment for GPT Trading System.
"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Any, Optional

from config.settings import get_settings
from core.services.model_management_service import ModelManagementService
from core.ml.continuous_improvement import ContinuousImprovementEngine
from core.ml.model_trainer import TradingModelPipeline
from core.ml.model_evaluation import ModelEvaluator
from core.infrastructure.database.repositories import TradeRepository, SignalRepository

logger = logging.getLogger(__name__)


class MLUpdater:
    """Automated ML model training and deployment"""
    
    def __init__(self):
        self.settings = get_settings()
        # Initialize model repository
        from core.services.model_management_service import ModelRepository
        model_repository = ModelRepository(self.settings.database.db_path)
        
        self.model_management = ModelManagementService(
            model_repository=model_repository,
            models_dir=Path("models")
        )
        # Note: ContinuousImprovementEngine requires model_pipeline and config
        # For now, we'll handle ML updates directly without it
        # self.continuous_improvement = ContinuousImprovementEngine(model_pipeline, config)
        self.trade_repository = TradeRepository(self.settings.database.db_path)
        self.signal_repository = SignalRepository(self.settings.database.db_path)
    
    async def daily_update(self) -> Dict[str, Any]:
        """Run daily ML model update"""
        logger.info("Starting daily ML model update")
        
        try:
            # 1. Collect recent data
            logger.info("Collecting recent trading data")
            data = await self._collect_recent_data(days=7)
            
            if not data['has_sufficient_data']:
                logger.warning("Insufficient data for ML training")
                return {
                    'success': False,
                    'reason': 'Insufficient data',
                    'data_stats': data['stats']
                }
            
            # 2. Train incremental model
            logger.info("Training incremental model")
            training_result = await self._train_incremental_model(data)
            
            # 3. Evaluate model performance
            logger.info("Evaluating model performance")
            evaluation = await self._evaluate_model(
                training_result['model_id'],
                data['test_data']
            )
            
            # 4. Deploy if improved
            if evaluation['is_better_than_production']:
                logger.info("New model performs better, deploying")
                deployment_result = await self._deploy_model(
                    training_result['model_id']
                )
                
                return {
                    'success': True,
                    'model_deployed': True,
                    'model_id': training_result['model_id'],
                    'improvement': evaluation['improvement_percent'],
                    'deployment': deployment_result
                }
            else:
                logger.info("Current model performs better, keeping existing")
                return {
                    'success': True,
                    'model_deployed': False,
                    'reason': 'Current model performs better',
                    'performance_gap': evaluation['performance_gap']
                }
                
        except Exception as e:
            logger.error(f"ML update failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _collect_recent_data(self, days: int) -> Dict[str, Any]:
        """Collect recent trading data for training"""
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=days)
        
        # Get trades and signals
        trades = self.trade_repository.get_trades_by_date_range(
            start_date, end_date
        )
        signals = self.signal_repository.get_signals_by_date_range(
            start_date, end_date
        )
        
        # Calculate statistics
        stats = {
            'trade_count': len(trades),
            'signal_count': len(signals),
            'unique_symbols': len(set(t.symbol for t in trades)),
            'win_rate': sum(1 for t in trades if t.result == 'WIN') / len(trades) if trades else 0,
            'avg_pnl': sum(t.pnl_usd for t in trades) / len(trades) if trades else 0
        }
        
        # Need at least 50 trades for meaningful training
        has_sufficient_data = len(trades) >= 50 and len(signals) >= 100
        
        # Split data for training/testing
        split_idx = int(len(trades) * 0.8)
        
        return {
            'has_sufficient_data': has_sufficient_data,
            'stats': stats,
            'trades': trades,
            'signals': signals,
            'train_data': {
                'trades': trades[:split_idx],
                'signals': signals[:int(len(signals) * 0.8)]
            },
            'test_data': {
                'trades': trades[split_idx:],
                'signals': signals[int(len(signals) * 0.8):]
            }
        }
    
    async def _train_incremental_model(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Train model incrementally with recent data"""
        # Initialize training pipeline
        pipeline = TradingModelPipeline()
        
        # Prepare training data
        features, labels = await self._prepare_training_data(
            data['train_data']['trades'],
            data['train_data']['signals']
        )
        
        # Load existing model for incremental training
        current_model = await self.model_management.get_active_model('signal_classifier')
        
        # Train model
        if current_model:
            # Incremental training
            model = await pipeline.train_incremental(
                current_model[0],  # model object
                features,
                labels,
                learning_rate=0.001  # Lower LR for incremental
            )
        else:
            # Full training if no existing model
            model = await pipeline.train_full(
                features,
                labels
            )
        
        # Save model
        model_metadata = await self.model_management.save_model(
            model=model,
            model_type='signal_classifier',
            version=f"v{datetime.now().strftime('%Y%m%d_%H%M')}",
            training_metrics={
                'train_size': len(features),
                'incremental': current_model is not None
            },
            feature_names=pipeline.get_feature_names()
        )
        
        return {
            'model_id': model_metadata.model_id,
            'version': model_metadata.version,
            'incremental': current_model is not None
        }
    
    async def _prepare_training_data(self, trades, signals):
        """Prepare features and labels for training"""
        # This is a simplified version - expand based on your needs
        features = []
        labels = []
        
        for signal in signals:
            # Find corresponding trade outcome
            matching_trades = [
                t for t in trades 
                if t.symbol == signal.symbol 
                and t.timestamp > signal.timestamp
                and t.timestamp < signal.timestamp + timedelta(hours=24)
            ]
            
            if matching_trades:
                # Extract features from signal
                feature_vector = [
                    signal.confidence,
                    signal.risk_reward_ratio,
                    1 if signal.signal_type == 'BUY' else 0,
                    # Add more features as needed
                ]
                features.append(feature_vector)
                
                # Label: 1 if profitable, 0 if not
                trade = matching_trades[0]
                labels.append(1 if trade.result == 'WIN' else 0)
        
        return features, labels
    
    async def _evaluate_model(
        self, 
        model_id: str, 
        test_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evaluate model performance"""
        evaluator = ModelEvaluator()
        
        # Load model
        model, metadata = await self.model_management.load_model(model_id)
        
        # Prepare test features
        test_features, test_labels = await self._prepare_training_data(
            test_data['trades'],
            test_data['signals']
        )
        
        # Evaluate performance
        metrics = evaluator.evaluate_model(model, test_features, test_labels)
        
        # Compare with production model
        production_model = await self.model_management.get_active_model('signal_classifier')
        
        if production_model:
            prod_metrics = evaluator.evaluate_model(
                production_model[0],
                test_features,
                test_labels
            )
            
            # Compare key metrics
            improvement_percent = (
                (metrics['f1_score'] - prod_metrics['f1_score']) / 
                prod_metrics['f1_score'] * 100
            )
            
            is_better = (
                metrics['f1_score'] > prod_metrics['f1_score'] and
                metrics['precision'] >= prod_metrics['precision'] * 0.95  # Allow 5% precision drop
            )
            
            return {
                'is_better_than_production': is_better,
                'improvement_percent': improvement_percent,
                'new_metrics': metrics,
                'production_metrics': prod_metrics,
                'performance_gap': metrics['f1_score'] - prod_metrics['f1_score']
            }
        else:
            # No production model to compare
            return {
                'is_better_than_production': True,
                'improvement_percent': 100,
                'new_metrics': metrics,
                'production_metrics': None,
                'performance_gap': metrics['f1_score']
            }
    
    async def _deploy_model(self, model_id: str) -> Dict[str, Any]:
        """Deploy model to production"""
        try:
            # Deploy model
            success = await self.model_management.deploy_model(model_id)
            
            if success:
                # Run post-deployment tests
                test_result = await self._run_deployment_tests(model_id)
                
                if test_result['passed']:
                    logger.info(f"Model {model_id} deployed successfully")
                    return {
                        'success': True,
                        'model_id': model_id,
                        'deployment_time': datetime.now(timezone.utc).isoformat(),
                        'tests_passed': True
                    }
                else:
                    # Rollback on test failure
                    logger.error("Deployment tests failed, rolling back")
                    await self.model_management.rollback_model('signal_classifier')
                    return {
                        'success': False,
                        'reason': 'Deployment tests failed',
                        'failed_tests': test_result['failed_tests']
                    }
            else:
                return {
                    'success': False,
                    'reason': 'Deployment failed'
                }
                
        except Exception as e:
            logger.error(f"Deployment error: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _run_deployment_tests(self, model_id: str) -> Dict[str, Any]:
        """Run tests to verify deployed model works correctly"""
        tests_passed = []
        tests_failed = []
        
        # Test 1: Model loads correctly
        try:
            model, metadata = await self.model_management.load_model(model_id)
            tests_passed.append("model_loading")
        except Exception as e:
            tests_failed.append(f"model_loading: {e}")
        
        # Test 2: Model makes predictions
        try:
            # Create dummy features
            dummy_features = [[0.8, 2.5, 1]]  # confidence, RR, is_buy
            predictions = model.predict(dummy_features)
            if predictions is not None:
                tests_passed.append("prediction")
            else:
                tests_failed.append("prediction: No output")
        except Exception as e:
            tests_failed.append(f"prediction: {e}")
        
        # Test 3: Model integrates with signal service
        # This would test the actual integration
        
        return {
            'passed': len(tests_failed) == 0,
            'tests_passed': tests_passed,
            'failed_tests': tests_failed
        }


async def main():
    """Run ML update standalone"""
    import argparse
    
    parser = argparse.ArgumentParser(description='ML Model Updater')
    parser.add_argument('--force', action='store_true', 
                      help='Force update even with insufficient data')
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(name)s | %(levelname)s | %(message)s'
    )
    
    updater = MLUpdater()
    result = await updater.daily_update()
    
    if result['success']:
        if result.get('model_deployed'):
            print(f"✅ Model deployed successfully!")
            print(f"   Model ID: {result['model_id']}")
            print(f"   Improvement: {result['improvement']:.2f}%")
        else:
            print("✅ ML update completed - current model kept")
            print(f"   Reason: {result.get('reason')}")
    else:
        print(f"❌ ML update failed: {result.get('error', result.get('reason'))}")


if __name__ == "__main__":
    asyncio.run(main())