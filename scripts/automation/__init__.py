"""
Automation scripts for GPT Trading System.
"""

# Import modules individually to avoid circular dependencies
__all__ = [
    'HealthChecker',
    'MLUpdater', 
    'BacktestRunner',
    'DatabaseBackup',
    'PerformanceAnalyzer',
    'NewsUpdater'
]

# Lazy imports to avoid issues
def get_health_checker():
    from .health_check import HealthChecker
    return HealthChecker

def get_ml_updater():
    from .ml_updater import MLUpdater
    return MLUpdater

def get_backtest_runner():
    from .backtest_runner import BacktestRunner
    return BacktestRunner

def get_database_backup():
    from .database_backup import DatabaseBackup
    return DatabaseBackup

def get_performance_analyzer():
    from .performance_analytics import PerformanceAnalyzer
    return PerformanceAnalyzer

def get_news_updater():
    from .news_updater import NewsUpdater
    return NewsUpdater