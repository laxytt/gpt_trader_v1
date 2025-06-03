#!/usr/bin/env python3
"""
Main workflow orchestrator for GPT Trading System.
Coordinates all automated tasks and ensures smooth operation.
"""

import asyncio
import logging
import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import get_settings
from core.infrastructure.database.repositories import TradeRepository
from scripts.automation.health_check import HealthChecker
from scripts.automation.ml_updater import MLUpdater
from scripts.automation.backtest_runner import BacktestRunner
from scripts.automation.performance_analytics import PerformanceAnalyzer

logger = logging.getLogger(__name__)


class WorkflowOrchestrator:
    """Orchestrates all automated workflow tasks"""
    
    def __init__(self):
        self.settings = get_settings()
        self.health_checker = HealthChecker()
        self.ml_updater = MLUpdater()
        self.backtest_runner = BacktestRunner()
        self.performance_analyzer = PerformanceAnalyzer()
    
    async def run_daily_tasks(self):
        """Execute daily workflow tasks"""
        logger.info("Starting daily workflow tasks")
        
        try:
            # 1. Health check first
            health_status = await self.health_checker.check_all()
            if not health_status['healthy']:
                logger.error(f"System unhealthy: {health_status['issues']}")
                await self.health_checker.send_alerts(health_status['issues'])
                return
            
            # 2. Generate performance report
            logger.info("Generating daily performance report")
            await self.performance_analyzer.generate_daily_report()
            
            # 3. Update ML models if market is closed
            if self._is_market_closed():
                logger.info("Market closed, updating ML models")
                await self.ml_updater.daily_update()
            
            # 4. Backup database
            logger.info("Backing up database")
            await self._backup_database()
            
            logger.info("Daily workflow tasks completed successfully")
            
        except Exception as e:
            logger.error(f"Daily workflow failed: {e}")
            await self.health_checker.send_alert(
                "Daily Workflow Failed",
                f"Error in daily tasks: {str(e)}"
            )
    
    async def run_weekly_tasks(self):
        """Execute weekly workflow tasks"""
        logger.info("Starting weekly workflow tasks")
        
        try:
            # 1. Comprehensive backtest
            logger.info("Running weekly comprehensive backtest")
            backtest_results = await self.backtest_runner.run_weekly_backtest()
            
            # 2. Compare backtest vs live performance
            logger.info("Comparing backtest vs live performance")
            comparison = await self.performance_analyzer.compare_backtest_vs_live(
                backtest_results
            )
            
            # 3. Strategy optimization suggestions
            if comparison['optimization_needed']:
                logger.info("Generating strategy optimization suggestions")
                await self._generate_optimization_report(comparison)
            
            logger.info("Weekly workflow tasks completed successfully")
            
        except Exception as e:
            logger.error(f"Weekly workflow failed: {e}")
            await self.health_checker.send_alert(
                "Weekly Workflow Failed",
                f"Error in weekly tasks: {str(e)}"
            )
    
    def _is_market_closed(self) -> bool:
        """Check if forex market is closed"""
        now = datetime.now(timezone.utc)
        weekday = now.weekday()
        hour = now.hour
        
        # Market closed on Saturday
        if weekday == 5:
            return True
        
        # Market closed Friday after 22:00 UTC
        if weekday == 4 and hour >= 22:
            return True
        
        # Market closed Sunday before 22:00 UTC
        if weekday == 6 and hour < 22:
            return True
        
        return False
    
    async def _backup_database(self):
        """Backup trading database"""
        from scripts.automation.database_backup import DatabaseBackup
        backup = DatabaseBackup()
        await backup.create_backup()
    
    async def _generate_optimization_report(self, comparison: Dict[str, Any]):
        """Generate strategy optimization report"""
        # Implementation for strategy optimization
        pass


async def main():
    """Main entry point for workflow orchestrator"""
    import argparse
    
    parser = argparse.ArgumentParser(description='GPT Trading Workflow Orchestrator')
    parser.add_argument('--task', choices=['daily', 'weekly', 'all'], 
                      default='daily', help='Task to run')
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        handlers=[
            logging.FileHandler('logs/workflow.log'),
            logging.StreamHandler()
        ]
    )
    
    orchestrator = WorkflowOrchestrator()
    
    if args.task == 'daily':
        await orchestrator.run_daily_tasks()
    elif args.task == 'weekly':
        await orchestrator.run_weekly_tasks()
    else:
        await orchestrator.run_daily_tasks()
        await orchestrator.run_weekly_tasks()


if __name__ == "__main__":
    asyncio.run(main())