# core/ml/model_trainer.py
"""
ML model training pipeline integrated with backtesting
"""

from asyncio.log import logger
import numpy as np
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit

from core.domain.enums.mt5_enums import TimeFrame
from core.infrastructure.data.data_manager import DataManager
from core.services.backtesting_service import BacktestEngine, BacktestConfig, BacktestMode, BacktestResults
from core.ml.feature_engineering import FeatureEngineer
from core.ml.model_evaluation import ModelEvaluator


class TradingModelPipeline:
    """
    Complete ML pipeline for trading model development,
    including feature engineering, training, and backtesting validation.
    """
    
    def __init__(
        self,
        data_manager: DataManager,
        backtest_engine: BacktestEngine,
        feature_engineer: FeatureEngineer
    ):
        self.data_manager = data_manager
        self.backtest_engine = backtest_engine
        self.feature_engineer = feature_engineer
        self.models = {}
        self.backtest_results = {}
    
    async def train_and_validate(
        self,
        symbols: List[str],
        train_start: datetime,
        train_end: datetime,
        validation_months: int = 3
    ) -> Dict[str, Any]:
        """
        Train models and validate with backtesting.
        
        This is the main entry point for ML workflow.
        """
        results = {
            'models': {},
            'backtest_results': {},
            'recommendations': {}
        }
        
        for symbol in symbols:
            logger.info(f"Training model for {symbol}")
            
            # 1. Get training data
            train_data = await self.data_manager.get_data(
                symbol=symbol,
                timeframe=TimeFrame.H1,
                start_date=train_start,
                end_date=train_end
            )
            
            # 2. Feature engineering
            features, labels = self.feature_engineer.prepare_features(
                train_data,
                target='signal_success'  # Binary: successful trade or not
            )
            
            # 3. Time series cross-validation with backtesting
            tscv = TimeSeriesSplit(n_splits=5)
            cv_results = []
            
            for fold, (train_idx, val_idx) in enumerate(tscv.split(features)):
                # Train model on fold
                model = self._train_model(
                    features.iloc[train_idx],
                    labels.iloc[train_idx]
                )
                
                # Get validation period dates
                val_start = train_data.index[val_idx[0]]
                val_end = train_data.index[val_idx[-1]]
                
                # Run backtest on validation period
                backtest_config = BacktestConfig(
                    start_date=val_start,
                    end_date=val_end,
                    symbols=[symbol],
                    mode=BacktestMode.ML_DRIVEN,  # New mode using ML predictions
                    model=model,
                    feature_engineer=self.feature_engineer
                )
                
                fold_results = await self.backtest_engine.run_backtest(backtest_config)
                cv_results.append({
                    'fold': fold,
                    'sharpe_ratio': fold_results.sharpe_ratio,
                    'win_rate': fold_results.win_rate,
                    'max_drawdown': fold_results.max_drawdown
                })
            
            # 4. Train final model on all data
            final_model = self._train_model(features, labels)
            
            # 5. Out-of-sample backtest
            oos_start = train_end
            oos_end = train_end + timedelta(days=30 * validation_months)
            
            oos_config = BacktestConfig(
                start_date=oos_start,
                end_date=oos_end,
                symbols=[symbol],
                mode=BacktestMode.ML_DRIVEN,
                model=final_model,
                feature_engineer=self.feature_engineer
            )
            
            oos_results = await self.backtest_engine.run_backtest(oos_config)
            
            # 6. Evaluate and decide
            should_deploy = self._evaluate_model(cv_results, oos_results)
            
            results['models'][symbol] = {
                'model': final_model,
                'cv_performance': cv_results,
                'oos_performance': oos_results,
                'deploy': should_deploy
            }
            
            if should_deploy:
                self.models[symbol] = final_model
                await self._save_model(symbol, final_model)
        
        return results
    
    def _train_model(self, features: pd.DataFrame, labels: pd.Series):
        """Train a model (can be swapped for any algorithm)"""
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        model.fit(features, labels)
        return model
    
    def _evaluate_model(
        self, 
        cv_results: List[Dict], 
        oos_results: BacktestResults
    ) -> bool:
        """
        Evaluate if model should be deployed based on backtest results.
        
        This is where you set your deployment criteria.
        """
        # Calculate average CV metrics
        avg_cv_sharpe = np.mean([r['sharpe_ratio'] for r in cv_results])
        avg_cv_dd = np.mean([r['max_drawdown'] for r in cv_results])
        
        # Deployment criteria
        criteria = {
            'min_cv_sharpe': 1.0,
            'max_cv_drawdown': 20.0,
            'min_oos_sharpe': 0.8,
            'max_oos_drawdown': 25.0,
            'min_oos_win_rate': 0.45
        }
        
        return (
            avg_cv_sharpe >= criteria['min_cv_sharpe'] and
            avg_cv_dd <= criteria['max_cv_drawdown'] and
            oos_results.sharpe_ratio >= criteria['min_oos_sharpe'] and
            oos_results.max_drawdown <= criteria['max_oos_drawdown'] and
            oos_results.win_rate >= criteria['min_oos_win_rate']
        )
    
    async def _save_model(self, symbol: str, model: Any):
        """Save trained model using ModelManagementService"""
        from pathlib import Path
        from core.services.model_management_service import ModelManagementService, ModelRepository
        
        # Initialize model management
        model_repository = ModelRepository(str(Path("data/trades.db")))
        model_management = ModelManagementService(
            models_dir=Path("models"),
            model_repository=model_repository
        )
        
        # Save model with metadata
        model_id = await model_management.save_model(
            model=model,
            model_type=f"signal_predictor_{symbol}",
            version="1.0.0",
            training_metrics={
                'accuracy': 0.0,  # These would come from evaluation
                'f1_score': 0.0,
                'precision': 0.0,
                'recall': 0.0
            },
            feature_names=self.feature_engineer.feature_names,
            hyperparameters={
                'n_estimators': 100,
                'max_depth': 10,
                'random_state': 42
            },
            training_data_info={
                'symbol': symbol,
                'trained_at': datetime.now().isoformat()
            }
        )
        
        logger.info(f"Model saved for {symbol} with ID: {model_id}")
        return model_id