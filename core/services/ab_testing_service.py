"""
A/B Testing Service for comparing trading strategies and models.
Enables systematic comparison of different approaches with statistical significance.
"""

import logging
import random
import json
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
from scipy import stats

from core.domain.models import TradingSignal, Trade, TradeStatus
from core.domain.exceptions import ServiceError, ErrorContext
from core.infrastructure.database.repositories import BaseRepository
from core.utils.error_handling import with_error_recovery

logger = logging.getLogger(__name__)


class StrategyType(Enum):
    """Types of strategies that can be tested"""
    GPT_ONLY = "gpt_only"
    ML_ONLY = "ml_only"
    ML_WITH_GPT_VALIDATION = "ml_with_gpt_validation"
    OFFLINE_VALIDATION_FIRST = "offline_validation_first"
    ENSEMBLE = "ensemble"


@dataclass
class ABTestConfig:
    """Configuration for an A/B test"""
    test_id: str
    name: str
    description: str
    strategy_a: StrategyType
    strategy_b: StrategyType
    allocation_ratio: float = 0.5  # Percentage allocated to strategy A
    min_samples_per_group: int = 30
    max_duration_days: int = 30
    confidence_level: float = 0.95
    active: bool = True
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)


@dataclass
class ABTestResult:
    """Results of an A/B test"""
    test_id: str
    strategy_a_trades: int
    strategy_b_trades: int
    strategy_a_win_rate: float
    strategy_b_win_rate: float
    strategy_a_avg_pnl: float
    strategy_b_avg_pnl: float
    strategy_a_sharpe: float
    strategy_b_sharpe: float
    p_value_win_rate: float
    p_value_pnl: float
    statistical_significance: bool
    winner: Optional[str] = None
    confidence_interval_a: Tuple[float, float] = None
    confidence_interval_b: Tuple[float, float] = None


class ABTestRepository(BaseRepository):
    """Repository for A/B test data"""
    
    def __init__(self, db_path: str):
        super().__init__(db_path)
        self._ensure_table_exists()
    
    def _ensure_table_exists(self):
        """Create A/B test tables if they don't exist"""
        with self._get_connection() as conn:
            # Test configurations
            conn.execute('''
                CREATE TABLE IF NOT EXISTS ab_test_configs (
                    test_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    strategy_a TEXT NOT NULL,
                    strategy_b TEXT NOT NULL,
                    allocation_ratio REAL DEFAULT 0.5,
                    min_samples_per_group INTEGER DEFAULT 30,
                    max_duration_days INTEGER DEFAULT 30,
                    confidence_level REAL DEFAULT 0.95,
                    active BOOLEAN DEFAULT TRUE,
                    created_at TIMESTAMP NOT NULL,
                    ended_at TIMESTAMP
                )
            ''')
            
            # Test assignments (which trades belong to which test/strategy)
            conn.execute('''
                CREATE TABLE IF NOT EXISTS ab_test_assignments (
                    trade_id TEXT PRIMARY KEY,
                    test_id TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    assigned_at TIMESTAMP NOT NULL,
                    FOREIGN KEY (test_id) REFERENCES ab_test_configs(test_id)
                )
            ''')
            
            # Test results
            conn.execute('''
                CREATE TABLE IF NOT EXISTS ab_test_results (
                    test_id TEXT PRIMARY KEY,
                    completed_at TIMESTAMP NOT NULL,
                    results_json TEXT NOT NULL,
                    FOREIGN KEY (test_id) REFERENCES ab_test_configs(test_id)
                )
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_active_tests 
                ON ab_test_configs(active)
            ''')
    
    def save_config(self, config: ABTestConfig):
        """Save A/B test configuration"""
        with self._get_connection() as conn:
            conn.execute('''
                INSERT OR REPLACE INTO ab_test_configs 
                (test_id, name, description, strategy_a, strategy_b,
                 allocation_ratio, min_samples_per_group, max_duration_days,
                 confidence_level, active, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                config.test_id,
                config.name,
                config.description,
                config.strategy_a.value,
                config.strategy_b.value,
                config.allocation_ratio,
                config.min_samples_per_group,
                config.max_duration_days,
                config.confidence_level,
                config.active,
                config.created_at
            ))
    
    def get_active_tests(self) -> List[ABTestConfig]:
        """Get all active A/B tests"""
        with self._get_connection() as conn:
            rows = conn.execute('''
                SELECT * FROM ab_test_configs WHERE active = TRUE
            ''').fetchall()
            
            configs = []
            for row in rows:
                configs.append(ABTestConfig(
                    test_id=row['test_id'],
                    name=row['name'],
                    description=row['description'],
                    strategy_a=StrategyType(row['strategy_a']),
                    strategy_b=StrategyType(row['strategy_b']),
                    allocation_ratio=row['allocation_ratio'],
                    min_samples_per_group=row['min_samples_per_group'],
                    max_duration_days=row['max_duration_days'],
                    confidence_level=row['confidence_level'],
                    active=row['active'],
                    created_at=datetime.fromisoformat(row['created_at'])
                ))
            
            return configs
    
    def save_assignment(self, trade_id: str, test_id: str, strategy: str):
        """Save trade assignment to A/B test"""
        with self._get_connection() as conn:
            conn.execute('''
                INSERT INTO ab_test_assignments 
                (trade_id, test_id, strategy, assigned_at)
                VALUES (?, ?, ?, ?)
            ''', (trade_id, test_id, strategy, datetime.now(timezone.utc)))
    
    def get_test_trades(self, test_id: str) -> Dict[str, List[str]]:
        """Get trades assigned to each strategy in a test"""
        with self._get_connection() as conn:
            rows = conn.execute('''
                SELECT trade_id, strategy 
                FROM ab_test_assignments 
                WHERE test_id = ?
            ''', (test_id,)).fetchall()
            
            trades = {'A': [], 'B': []}
            for row in rows:
                strategy_key = 'A' if row['strategy'] == 'strategy_a' else 'B'
                trades[strategy_key].append(row['trade_id'])
            
            return trades


class ABTestingService:
    """
    Service for managing A/B tests of trading strategies.
    Handles test configuration, trade assignment, and result analysis.
    """
    
    def __init__(self, repository: ABTestRepository):
        self.repository = repository
        self._active_tests_cache = {}
        self._cache_expiry = datetime.now(timezone.utc)
    
    @with_error_recovery
    async def create_test(self, config: ABTestConfig) -> str:
        """Create a new A/B test"""
        with ErrorContext("Creating A/B test", test_name=config.name):
            # Validate configuration
            if config.allocation_ratio <= 0 or config.allocation_ratio >= 1:
                raise ServiceError("Allocation ratio must be between 0 and 1")
            
            if config.strategy_a == config.strategy_b:
                raise ServiceError("Strategies must be different")
            
            # Save configuration
            self.repository.save_config(config)
            
            # Clear cache
            self._active_tests_cache = {}
            
            logger.info(f"Created A/B test: {config.test_id} - {config.name}")
            return config.test_id
    
    @with_error_recovery
    async def assign_strategy(self, symbol: str) -> Tuple[StrategyType, Optional[str]]:
        """
        Assign a strategy for a symbol based on active A/B tests.
        
        Returns:
            Tuple of (strategy_type, test_id)
        """
        # Get active tests
        active_tests = await self._get_active_tests()
        
        if not active_tests:
            # No active tests, use default strategy
            return StrategyType.GPT_ONLY, None
        
        # For simplicity, use the first active test
        # In production, could have more complex logic for multiple tests
        test = active_tests[0]
        
        # Random assignment based on allocation ratio
        use_strategy_a = random.random() < test.allocation_ratio
        
        if use_strategy_a:
            return test.strategy_a, test.test_id
        else:
            return test.strategy_b, test.test_id
    
    @with_error_recovery
    async def record_trade_assignment(
        self, 
        trade: Trade, 
        test_id: str, 
        strategy: StrategyType
    ):
        """Record which test and strategy a trade was assigned to"""
        strategy_key = 'strategy_a' if strategy in [
            StrategyType.GPT_ONLY, 
            StrategyType.ML_WITH_GPT_VALIDATION
        ] else 'strategy_b'
        
        self.repository.save_assignment(trade.id, test_id, strategy_key)
        logger.debug(f"Assigned trade {trade.id} to test {test_id}, {strategy_key}")
    
    @with_error_recovery
    async def analyze_test(self, test_id: str, trades_data: List[Trade]) -> ABTestResult:
        """
        Analyze results of an A/B test.
        
        Args:
            test_id: Test identifier
            trades_data: All trades to analyze
            
        Returns:
            ABTestResult with statistical analysis
        """
        # Get test configuration
        config = await self._get_test_config(test_id)
        if not config:
            raise ServiceError(f"Test not found: {test_id}")
        
        # Get trades assigned to this test
        test_trades = self.repository.get_test_trades(test_id)
        
        # Filter trades data
        strategy_a_trades = [
            t for t in trades_data if t.id in test_trades['A']
        ]
        strategy_b_trades = [
            t for t in trades_data if t.id in test_trades['B']
        ]
        
        # Calculate metrics for each strategy
        metrics_a = self._calculate_strategy_metrics(strategy_a_trades)
        metrics_b = self._calculate_strategy_metrics(strategy_b_trades)
        
        # Statistical tests
        p_value_win_rate = self._test_win_rate_significance(
            strategy_a_trades, strategy_b_trades
        )
        p_value_pnl = self._test_pnl_significance(
            strategy_a_trades, strategy_b_trades
        )
        
        # Determine statistical significance
        alpha = 1 - config.confidence_level
        is_significant = (
            p_value_win_rate < alpha or p_value_pnl < alpha
        ) and (
            len(strategy_a_trades) >= config.min_samples_per_group and
            len(strategy_b_trades) >= config.min_samples_per_group
        )
        
        # Determine winner if significant
        winner = None
        if is_significant:
            if metrics_a['win_rate'] > metrics_b['win_rate'] and metrics_a['avg_pnl'] > metrics_b['avg_pnl']:
                winner = 'strategy_a'
            elif metrics_b['win_rate'] > metrics_a['win_rate'] and metrics_b['avg_pnl'] > metrics_a['avg_pnl']:
                winner = 'strategy_b'
        
        # Calculate confidence intervals
        ci_a = self._calculate_confidence_interval(
            [t.realized_pnl for t in strategy_a_trades if t.realized_pnl],
            config.confidence_level
        )
        ci_b = self._calculate_confidence_interval(
            [t.realized_pnl for t in strategy_b_trades if t.realized_pnl],
            config.confidence_level
        )
        
        return ABTestResult(
            test_id=test_id,
            strategy_a_trades=len(strategy_a_trades),
            strategy_b_trades=len(strategy_b_trades),
            strategy_a_win_rate=metrics_a['win_rate'],
            strategy_b_win_rate=metrics_b['win_rate'],
            strategy_a_avg_pnl=metrics_a['avg_pnl'],
            strategy_b_avg_pnl=metrics_b['avg_pnl'],
            strategy_a_sharpe=metrics_a['sharpe_ratio'],
            strategy_b_sharpe=metrics_b['sharpe_ratio'],
            p_value_win_rate=p_value_win_rate,
            p_value_pnl=p_value_pnl,
            statistical_significance=is_significant,
            winner=winner,
            confidence_interval_a=ci_a,
            confidence_interval_b=ci_b
        )
    
    async def _get_active_tests(self) -> List[ABTestConfig]:
        """Get active tests with caching"""
        if (
            self._active_tests_cache and 
            datetime.now(timezone.utc) < self._cache_expiry
        ):
            return list(self._active_tests_cache.values())
        
        # Refresh cache
        active_tests = self.repository.get_active_tests()
        self._active_tests_cache = {t.test_id: t for t in active_tests}
        self._cache_expiry = datetime.now(timezone.utc) + timedelta(minutes=5)
        
        return active_tests
    
    async def _get_test_config(self, test_id: str) -> Optional[ABTestConfig]:
        """Get test configuration"""
        active_tests = await self._get_active_tests()
        return self._active_tests_cache.get(test_id)
    
    def _calculate_strategy_metrics(self, trades: List[Trade]) -> Dict[str, float]:
        """Calculate metrics for a strategy"""
        if not trades:
            return {
                'win_rate': 0.0,
                'avg_pnl': 0.0,
                'sharpe_ratio': 0.0
            }
        
        # Win rate
        winning_trades = [t for t in trades if t.realized_pnl and t.realized_pnl > 0]
        win_rate = len(winning_trades) / len(trades)
        
        # Average P&L
        pnls = [t.realized_pnl for t in trades if t.realized_pnl is not None]
        avg_pnl = np.mean(pnls) if pnls else 0.0
        
        # Sharpe ratio (simplified)
        if len(pnls) > 1:
            returns = np.array(pnls)
            sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-6) * np.sqrt(252)
        else:
            sharpe_ratio = 0.0
        
        return {
            'win_rate': win_rate,
            'avg_pnl': avg_pnl,
            'sharpe_ratio': sharpe_ratio
        }
    
    def _test_win_rate_significance(
        self, 
        trades_a: List[Trade], 
        trades_b: List[Trade]
    ) -> float:
        """Test if win rate difference is statistically significant"""
        if len(trades_a) < 5 or len(trades_b) < 5:
            return 1.0  # Not enough data
        
        wins_a = sum(1 for t in trades_a if t.realized_pnl and t.realized_pnl > 0)
        wins_b = sum(1 for t in trades_b if t.realized_pnl and t.realized_pnl > 0)
        
        # Chi-square test for proportions
        observed = [[wins_a, len(trades_a) - wins_a],
                   [wins_b, len(trades_b) - wins_b]]
        
        try:
            _, p_value, _, _ = stats.chi2_contingency(observed)
            return p_value
        except:
            return 1.0
    
    def _test_pnl_significance(
        self, 
        trades_a: List[Trade], 
        trades_b: List[Trade]
    ) -> float:
        """Test if P&L difference is statistically significant"""
        pnls_a = [t.realized_pnl for t in trades_a if t.realized_pnl is not None]
        pnls_b = [t.realized_pnl for t in trades_b if t.realized_pnl is not None]
        
        if len(pnls_a) < 5 or len(pnls_b) < 5:
            return 1.0
        
        # Two-sample t-test
        try:
            _, p_value = stats.ttest_ind(pnls_a, pnls_b)
            return p_value
        except:
            return 1.0
    
    def _calculate_confidence_interval(
        self, 
        data: List[float], 
        confidence_level: float
    ) -> Tuple[float, float]:
        """Calculate confidence interval for mean"""
        if len(data) < 2:
            return (0.0, 0.0)
        
        mean = np.mean(data)
        sem = stats.sem(data)
        interval = sem * stats.t.ppf((1 + confidence_level) / 2, len(data) - 1)
        
        return (mean - interval, mean + interval)
    
    async def end_test(self, test_id: str) -> ABTestResult:
        """End an A/B test and analyze final results"""
        with ErrorContext("Ending A/B test", test_id=test_id):
            # Mark test as inactive
            with self.repository._get_connection() as conn:
                conn.execute('''
                    UPDATE ab_test_configs 
                    SET active = FALSE, ended_at = ? 
                    WHERE test_id = ?
                ''', (datetime.now(timezone.utc), test_id))
            
            # Clear cache
            self._active_tests_cache = {}
            
            # Get all trades and analyze
            # This would need to be integrated with TradeRepository
            # For now, return a placeholder
            logger.info(f"Ended A/B test: {test_id}")
            
            # In production, would fetch trades and analyze
            # return await self.analyze_test(test_id, trades)
            
    async def get_test_summary(self, test_id: str) -> Dict[str, Any]:
        """Get summary of an A/B test"""
        config = await self._get_test_config(test_id)
        if not config:
            raise ServiceError(f"Test not found: {test_id}")
        
        test_trades = self.repository.get_test_trades(test_id)
        
        return {
            'test_id': test_id,
            'name': config.name,
            'description': config.description,
            'strategy_a': config.strategy_a.value,
            'strategy_b': config.strategy_b.value,
            'allocation_ratio': config.allocation_ratio,
            'created_at': config.created_at.isoformat(),
            'active': config.active,
            'trades_assigned': {
                'strategy_a': len(test_trades['A']),
                'strategy_b': len(test_trades['B'])
            }
        }


# Export
__all__ = [
    'ABTestingService', 
    'ABTestConfig', 
    'ABTestResult', 
    'StrategyType',
    'ABTestRepository'
]