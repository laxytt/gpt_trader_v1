#!/usr/bin/env python3
"""
ML Scheduler - Orchestrates automated ML model updates and monitoring
Can be run as a service or scheduled task
"""

import asyncio
import logging
import schedule
import time
from datetime import datetime
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from scripts.ml_continuous_learning import ContinuousLearningSystem
from scripts.performance_analytics import PerformanceAnalytics
from config.settings import get_settings

logger = logging.getLogger(__name__)


class MLScheduler:
    """Scheduler for ML-related tasks"""
    
    def __init__(self):
        self.settings = get_settings()
        self.continuous_learning = ContinuousLearningSystem()
        self.analytics = PerformanceAnalytics()
        
    def run_performance_check(self):
        """Run performance check on all models"""
        logger.info("Running scheduled performance check")
        
        try:
            # Generate performance report
            report = self.analytics.generate_report()
            
            # Check if any models need retraining
            symbols_needing_update = []
            
            for symbol in self.settings.trading.symbols:
                performance = asyncio.run(
                    self.continuous_learning.evaluate_model_performance(symbol)
                )
                
                if performance['needs_retraining']:
                    symbols_needing_update.append({
                        'symbol': symbol,
                        'reasons': performance.get('retrain_reasons', [])
                    })
            
            if symbols_needing_update:
                logger.warning(f"Models needing update: {symbols_needing_update}")
                
                # Send notification if enabled
                if self.settings.is_telegram_enabled:
                    asyncio.run(self._send_update_notification(symbols_needing_update))
            else:
                logger.info("All models performing within acceptable parameters")
                
        except Exception as e:
            logger.error(f"Error in performance check: {e}", exc_info=True)
    
    def run_model_updates(self):
        """Run model updates if needed"""
        logger.info("Running scheduled model update check")
        
        try:
            # Only run if ML is enabled
            if not self.settings.ml.enabled:
                logger.info("ML is disabled, skipping model updates")
                return
            
            # Run continuous improvement cycle once
            asyncio.run(self._run_single_improvement_cycle())
            
        except Exception as e:
            logger.error(f"Error in model updates: {e}", exc_info=True)
    
    async def _run_single_improvement_cycle(self):
        """Run a single improvement cycle"""
        logger.info("=== Starting ML Improvement Cycle ===")
        
        symbols = self.settings.trading.symbols
        models_updated = 0
        
        for symbol in symbols:
            # Evaluate current performance
            performance = await self.continuous_learning.evaluate_model_performance(symbol)
            
            if performance['needs_retraining']:
                logger.info(f"Retraining {symbol}: {performance.get('retrain_reasons', [])}")
                
                result = await self.continuous_learning.retrain_model(symbol)
                
                if result['status'] == 'success' and result['backtest']['passed']:
                    models_updated += 1
                    logger.info(f"Successfully updated model for {symbol}")
        
        logger.info(f"Improvement cycle complete. Models updated: {models_updated}")
    
    async def _send_update_notification(self, symbols_needing_update):
        """Send notification about models needing update"""
        from core.infrastructure.notifications.telegram import TelegramNotifier
        
        notifier = TelegramNotifier(
            self.settings.telegram.token,
            self.settings.telegram.chat_id
        )
        
        message = "🤖 ML Models Need Attention\n\n"
        for item in symbols_needing_update:
            message += f"{item['symbol']}: {', '.join(item['reasons'])}\n"
        
        await notifier.send_message(message)
    
    def setup_schedule(self):
        """Setup scheduled tasks"""
        # Daily performance check at 6 AM
        schedule.every().day.at("06:00").do(self.run_performance_check)
        
        # Model updates based on settings (default: every 30 days)
        # Run at 2 AM on scheduled days
        update_days = self.settings.ml.update_frequency_days
        
        if update_days == 1:
            schedule.every().day.at("02:00").do(self.run_model_updates)
        elif update_days == 7:
            schedule.every().monday.at("02:00").do(self.run_model_updates)
        elif update_days == 30:
            # Run on the 1st of each month
            schedule.every().day.at("02:00").do(self._check_monthly_update)
        else:
            # Custom interval - check daily if it's time
            schedule.every().day.at("02:00").do(self._check_custom_update)
        
        logger.info("ML scheduler configured:")
        logger.info("- Daily performance checks at 06:00")
        logger.info(f"- Model updates every {update_days} days at 02:00")
    
    def _check_monthly_update(self):
        """Check if it's the 1st of the month for monthly updates"""
        if datetime.now().day == 1:
            self.run_model_updates()
    
    def _check_custom_update(self):
        """Check if it's time for custom interval updates"""
        # This would need to track last update time
        # For now, simplified implementation
        last_update_file = Path("models/.last_update")
        
        if last_update_file.exists():
            with open(last_update_file, 'r') as f:
                last_update = datetime.fromisoformat(f.read().strip())
            
            days_since = (datetime.now() - last_update).days
            
            if days_since >= self.settings.ml.update_frequency_days:
                self.run_model_updates()
                
                # Update timestamp
                with open(last_update_file, 'w') as f:
                    f.write(datetime.now().isoformat())
        else:
            # First run
            self.run_model_updates()
            last_update_file.parent.mkdir(exist_ok=True)
            with open(last_update_file, 'w') as f:
                f.write(datetime.now().isoformat())
    
    def run_forever(self):
        """Run the scheduler forever"""
        self.setup_schedule()
        
        logger.info("ML Scheduler started. Press Ctrl+C to stop.")
        
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
        except KeyboardInterrupt:
            logger.info("ML Scheduler stopped by user")
        except Exception as e:
            logger.error(f"ML Scheduler error: {e}", exc_info=True)


def main():
    """Main entry point"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/ml_scheduler.log'),
            logging.StreamHandler()
        ]
    )
    
    # Check command line arguments
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        scheduler = MLScheduler()
        
        if command == "check":
            # Run performance check once
            scheduler.run_performance_check()
        elif command == "update":
            # Run model updates once
            scheduler.run_model_updates()
        elif command == "daemon":
            # Run as daemon
            scheduler.run_forever()
        else:
            print(f"Unknown command: {command}")
            print("Usage: python ml_scheduler.py [check|update|daemon]")
    else:
        # Default: run as daemon
        scheduler = MLScheduler()
        scheduler.run_forever()


if __name__ == "__main__":
    main()