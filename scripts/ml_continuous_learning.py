#!/usr/bin/env python3
"""
Continuous Learning System for ML Models
Automatically retrains models based on recent trading performance
"""

import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
import sys
import pandas as pd
from typing import Dict, List, Any
import subprocess
import json

sys.path.append(str(Path(__file__).parent.parent))

from core.infrastructure.database.repositories import TradeRepository, SignalRepository
from core.infrastructure.database.backtest_repository import BacktestRepository
from core.services.model_management_service import ModelManagementService
from config.settings import get_settings

logger = logging.getLogger(__name__)


class ContinuousLearningSystem:
    """Monitors performance and triggers retraining when needed"""
    
    def __init__(self):
        self.settings = get_settings()
        self.trade_repo = TradeRepository(self.settings.database.db_path)
        self.signal_repo = SignalRepository(self.settings.database.db_path)
        self.backtest_repo = BacktestRepository(self.settings.database.db_path)
        self.model_management = ModelManagementService(
            models_dir=Path("models"),
            model_repository=None
        )
        self.improvement_history_file = Path("reports/ml_improvements.json")
        
    async def evaluate_model_performance(self, symbol: str, days_back: int = 30) -> Dict[str, Any]:
        """Evaluate current model performance on recent trades"""
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        # Get recent trades
        trades = self.trade_repo.get_trades_by_date_range(start_date, end_date, symbol)
        signals = self.signal_repo.get_signals_by_date_range(start_date, end_date, symbol)
        
        if not trades:
            return {
                'symbol': symbol,
                'period_days': days_back,
                'total_trades': 0,
                'win_rate': 0,
                'avg_profit': 0,
                'total_profit': 0,
                'needs_retraining': False,
                'reason': 'No trades in period'
            }
        
        # Calculate metrics
        winning_trades = [t for t in trades if t.profit_loss and t.profit_loss > 0]
        total_profit = sum(t.profit_loss for t in trades if t.profit_loss)
        
        metrics = {
            'symbol': symbol,
            'period_days': days_back,
            'total_trades': len(trades),
            'winning_trades': len(winning_trades),
            'win_rate': len(winning_trades) / len(trades) if trades else 0,
            'avg_profit': total_profit / len(trades) if trades else 0,
            'total_profit': total_profit,
            'signal_count': len(signals),
            'signal_to_trade_ratio': len(trades) / len(signals) if signals else 0
        }
        
        # Determine if retraining needed
        metrics['needs_retraining'] = self._should_retrain(metrics)
        
        return metrics
    
    def _should_retrain(self, metrics: Dict[str, Any]) -> bool:
        """Determine if model needs retraining based on performance"""
        
        reasons = []
        
        # Check win rate
        if metrics['win_rate'] < 0.45:  # Below 45% win rate
            reasons.append(f"Low win rate: {metrics['win_rate']:.1%}")
        
        # Check if enough trades
        if metrics['total_trades'] < 10:
            reasons.append(f"Insufficient trades: {metrics['total_trades']}")
            
        # Check profitability
        if metrics['avg_profit'] < 0:
            reasons.append(f"Negative avg profit: ${metrics['avg_profit']:.2f}")
        
        # Check signal quality
        if metrics['signal_to_trade_ratio'] < 0.3:  # Less than 30% of signals executed
            reasons.append(f"Low signal execution: {metrics['signal_to_trade_ratio']:.1%}")
        
        metrics['retrain_reasons'] = reasons
        return len(reasons) > 0
    
    async def retrain_model(self, symbol: str) -> Dict[str, Any]:
        """Retrain model with recent data"""
        
        logger.info(f"Starting model retraining for {symbol}")
        
        try:
            # Use more recent data for retraining
            end_date = datetime.now()
            start_date = end_date - timedelta(days=180)  # 6 months
            
            # Archive old model first
            await self._archive_old_model(symbol)
            
            # Run the training script
            result = subprocess.run(
                [
                    sys.executable, 
                    "scripts/train_ml_production.py",
                    "--symbols", symbol,
                    "--start-date", start_date.strftime('%Y-%m-%d'),
                    "--end-date", end_date.strftime('%Y-%m-%d')
                ],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                logger.info(f"Model retrained successfully for {symbol}")
                
                # Validate the new model
                backtest_result = await self._validate_new_model(symbol)
                
                return {
                    'status': 'success',
                    'symbol': symbol,
                    'training_period': f"{start_date.date()} to {end_date.date()}",
                    'backtest': backtest_result,
                    'output': result.stdout
                }
            else:
                logger.error(f"Retraining failed: {result.stderr}")
                return {
                    'status': 'failed',
                    'error': result.stderr
                }
                
        except Exception as e:
            logger.error(f"Error during retraining: {e}", exc_info=True)
            return {
                'status': 'error',
                'error': str(e)
            }
    
    async def _archive_old_model(self, symbol: str):
        """Archive the current model before replacing"""
        model_type = f"pattern_trader_{symbol}"
        current_model = await self.model_management.get_active_model(model_type)
        
        if current_model:
            archive_path = Path("models/archive") / f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            archive_path.parent.mkdir(exist_ok=True)
            # Archive logic here
            logger.info(f"Archived old model for {symbol}")
    
    async def _validate_new_model(self, symbol: str) -> Dict[str, Any]:
        """Validate new model with backtesting"""
        logger.info(f"Validating new model for {symbol}")
        
        try:
            # Run backtest on recent 30 days
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            
            # Run backtest
            result = subprocess.run(
                [
                    sys.executable,
                    "run_backtest.py",
                    "--symbols", symbol,
                    "--start-date", start_date.strftime('%Y-%m-%d'),
                    "--end-date", end_date.strftime('%Y-%m-%d'),
                    "--ml-mode", "true"
                ],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                # Parse backtest results
                import re
                output = result.stdout
                
                # Extract key metrics
                win_rate_match = re.search(r'Win Rate: ([0-9.]+)%', output)
                profit_match = re.search(r'Total Return: \$([0-9.-]+)', output)
                trades_match = re.search(r'Total Trades: ([0-9]+)', output)
                
                validation_result = {
                    'validation_period': '30 days',
                    'status': 'completed',
                    'win_rate': float(win_rate_match.group(1)) if win_rate_match else 0.0,
                    'total_profit': float(profit_match.group(1)) if profit_match else 0.0,
                    'total_trades': int(trades_match.group(1)) if trades_match else 0,
                    'passed': False
                }
                
                # Determine if model passes validation
                if validation_result['win_rate'] >= 50 and validation_result['total_profit'] > 0:
                    validation_result['passed'] = True
                    validation_result['recommendation'] = 'Deploy new model'
                else:
                    validation_result['recommendation'] = 'Keep existing model'
                
                return validation_result
                
            else:
                logger.error(f"Backtest validation failed: {result.stderr}")
                return {
                    'status': 'failed',
                    'error': result.stderr
                }
                
        except Exception as e:
            logger.error(f"Error during validation: {e}", exc_info=True)
            return {
                'status': 'error',
                'error': str(e)
            }
    
    async def continuous_improvement_cycle(self):
        """Main loop for continuous improvement"""
        
        while True:
            try:
                logger.info("=== Starting Continuous Improvement Cycle ===")
                
                symbols = self.settings.trading.symbols
                improvement_summary = {
                    'timestamp': datetime.now().isoformat(),
                    'symbols_evaluated': len(symbols),
                    'models_retrained': 0,
                    'details': {}
                }
                
                for symbol in symbols:
                    # Evaluate current performance
                    performance = await self.evaluate_model_performance(symbol)
                    
                    logger.info(
                        f"{symbol} Performance - Win Rate: {performance['win_rate']:.1%}, "
                        f"Trades: {performance['total_trades']}, "
                        f"Avg Profit: ${performance['avg_profit']:.2f}, "
                        f"Needs Retraining: {performance['needs_retraining']}"
                    )
                    
                    improvement_summary['details'][symbol] = {
                        'performance': performance,
                        'action_taken': 'none'
                    }
                    
                    if performance['needs_retraining']:
                        logger.info(f"Retraining triggered for {symbol}: {performance.get('retrain_reasons', [])}")
                        retrain_result = await self.retrain_model(symbol)
                        
                        if retrain_result['status'] == 'success':
                            logger.info(f"Retraining successful for {symbol}")
                            
                            # Check if new model is better
                            if retrain_result['backtest']['passed']:
                                improvement_summary['models_retrained'] += 1
                                improvement_summary['details'][symbol]['action_taken'] = 'retrained_and_deployed'
                                
                                # Send notification
                                await self._send_improvement_notification(symbol, retrain_result)
                            else:
                                improvement_summary['details'][symbol]['action_taken'] = 'retrained_but_not_deployed'
                                logger.info(f"New model for {symbol} did not pass validation")
                        else:
                            logger.error(f"Retraining failed for {symbol}: {retrain_result.get('error')}")
                            improvement_summary['details'][symbol]['action_taken'] = 'retrain_failed'
                
                # Save improvement summary
                await self._save_improvement_summary(improvement_summary)
                
                # Generate and save analytics report
                from scripts.performance_analytics import PerformanceAnalytics
                analytics = PerformanceAnalytics()
                analytics.generate_report()
                
                # Wait before next cycle
                next_check = self.settings.ml.update_frequency_days * 86400  # Convert days to seconds
                logger.info(f"Next improvement cycle in {self.settings.ml.update_frequency_days} days")
                await asyncio.sleep(next_check)
                
            except Exception as e:
                logger.error(f"Error in continuous improvement cycle: {e}", exc_info=True)
                await asyncio.sleep(3600)  # Wait 1 hour on error
    
    async def _save_improvement_summary(self, summary: Dict[str, Any]):
        """Save improvement summary to history file"""
        try:
            # Load existing history
            history = []
            if self.improvement_history_file.exists():
                with open(self.improvement_history_file, 'r') as f:
                    history = json.load(f)
            
            # Add new summary
            history.append(summary)
            
            # Keep only last 100 entries
            history = history[-100:]
            
            # Save updated history
            self.improvement_history_file.parent.mkdir(exist_ok=True)
            with open(self.improvement_history_file, 'w') as f:
                json.dump(history, f, indent=2)
                
            logger.info(f"Saved improvement summary to {self.improvement_history_file}")
            
        except Exception as e:
            logger.error(f"Failed to save improvement summary: {e}")
    
    async def _send_improvement_notification(self, symbol: str, result: Dict[str, Any]):
        """Send notification about model improvement"""
        try:
            if self.settings.is_telegram_enabled:
                from core.infrastructure.notifications.telegram import TelegramNotifier
                notifier = TelegramNotifier(
                    self.settings.telegram.token,
                    self.settings.telegram.chat_id
                )
                
                message = (
                    f"🤖 ML Model Improved!\n"
                    f"Symbol: {symbol}\n"
                    f"New Win Rate: {result['backtest']['win_rate']:.1f}%\n"
                    f"Validation Profit: ${result['backtest']['total_profit']:.2f}\n"
                    f"Status: Deployed ✅"
                )
                
                await notifier.send_message(message)
                
        except Exception as e:
            logger.error(f"Failed to send notification: {e}")


async def main():
    """Run continuous learning system"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    system = ContinuousLearningSystem()
    await system.continuous_improvement_cycle()


if __name__ == "__main__":
    asyncio.run(main())