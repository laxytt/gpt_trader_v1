# core/ml/continuous_improvement.py
"""
Continuous improvement system using scheduled backtests
"""

import asyncio
from asyncio.log import logger
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List
from core.ml.model_trainer import TradingModelPipeline
from core.services.backtesting_service import BacktestConfig, BacktestMode, BacktestResults
import schedule



class ContinuousImprovementEngine:
    """
    Manages continuous backtesting and model retraining.
    """
    
    def __init__(
        self,
        model_pipeline: TradingModelPipeline,
        config: Dict[str, Any]
    ):
        self.model_pipeline = model_pipeline
        self.config = config
        self.performance_history = []
    
    async def run_scheduled_backtest(self):
        """Run scheduled backtest for monitoring live performance"""
        logger.info("Starting scheduled backtest")
        
        # Get symbols currently being traded
        active_symbols = self.config['active_symbols']
        
        # Backtest last N days of live trading
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=self.config['backtest_lookback_days'])
        
        results = {}
        alerts = []
        
        for symbol in active_symbols:
            # Run backtest with current live model
            config = BacktestConfig(
                start_date=start_date,
                end_date=end_date,
                symbols=[symbol],
                mode=BacktestMode.LIVE_MODEL,  # Uses current production model
            )
            
            backtest_results = await self.backtest_engine.run_backtest(config)
            
            # Compare with expected performance
            performance_check = self._check_performance_degradation(
                symbol, 
                backtest_results
            )
            
            if performance_check['degraded']:
                alerts.append({
                    'symbol': symbol,
                    'issue': performance_check['reason'],
                    'metrics': performance_check['metrics']
                })
            
            results[symbol] = {
                'backtest': backtest_results,
                'status': 'degraded' if performance_check['degraded'] else 'healthy'
            }
        
        # Store results
        self.performance_history.append({
            'timestamp': datetime.now(timezone.utc),
            'results': results
        })
        
        # Send alerts if needed
        if alerts:
            await self._send_performance_alerts(alerts)
        
        # Trigger retraining if needed
        if self._should_retrain(results):
            await self.trigger_retraining()
        
        return results
    
    def _check_performance_degradation(
        self, 
        symbol: str, 
        results: BacktestResults
    ) -> Dict[str, Any]:
        """Check if performance has degraded"""
        # Get baseline metrics (from initial validation)
        baseline = self.config['baseline_metrics'].get(symbol, {})
        
        degradation_thresholds = {
            'sharpe_decline': 0.3,  # 30% decline in Sharpe
            'max_dd_increase': 1.5,  # 50% increase in drawdown
            'win_rate_decline': 0.1  # 10% absolute decline in win rate
        }
        
        checks = {
            'sharpe_degraded': (
                results.sharpe_ratio < baseline.get('sharpe_ratio', 1.0) * 
                (1 - degradation_thresholds['sharpe_decline'])
            ),
            'drawdown_increased': (
                results.max_drawdown > baseline.get('max_drawdown', 20) * 
                degradation_thresholds['max_dd_increase']
            ),
            'win_rate_declined': (
                results.win_rate < baseline.get('win_rate', 0.5) - 
                degradation_thresholds['win_rate_decline']
            )
        }
        
        degraded = any(checks.values())
        
        return {
            'degraded': degraded,
            'reason': ', '.join([k for k, v in checks.items() if v]),
            'metrics': {
                'sharpe_ratio': results.sharpe_ratio,
                'max_drawdown': results.max_drawdown,
                'win_rate': results.win_rate
            }
        }
    
    async def trigger_retraining(self):
        """Trigger model retraining based on recent performance"""
        logger.info("Triggering model retraining")
        
        # Get data for retraining
        retrain_end = datetime.now(timezone.utc)
        retrain_start = retrain_end - timedelta(days=self.config['retrain_window_days'])
        
        # Run retraining pipeline
        retrain_results = await self.model_pipeline.train_and_validate(
            symbols=self.config['active_symbols'],
            train_start=retrain_start,
            train_end=retrain_end,
            validation_months=1
        )
        
        # Evaluate new models
        deployment_decisions = {}
        
        for symbol, model_results in retrain_results['models'].items():
            if model_results['deploy']:
                # Run A/B test backtest
                ab_results = await self._run_ab_test(
                    symbol,
                    current_model=self.model_pipeline.models.get(symbol),
                    new_model=model_results['model']
                )
                
                if ab_results['new_better']:
                    deployment_decisions[symbol] = 'deploy_new'
                else:
                    deployment_decisions[symbol] = 'keep_current'
            else:
                deployment_decisions[symbol] = 'keep_current'
        
        # Send retraining report
        await self._send_retraining_report(retrain_results, deployment_decisions)
        
        return deployment_decisions
    
    def schedule_tasks(self):
        """Schedule recurring tasks"""
        # Daily performance check
        schedule.every().day.at("00:00").do(
            lambda: asyncio.create_task(self.run_scheduled_backtest())
        )
        
        # Weekly deep analysis
        schedule.every().monday.at("02:00").do(
            lambda: asyncio.create_task(self.run_deep_analysis())
        )
        
        # Monthly retraining evaluation
        schedule.every().month.do(
            lambda: asyncio.create_task(self.evaluate_retraining_need())
        )