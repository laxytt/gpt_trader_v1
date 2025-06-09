"""
Automated backtesting runner for GPT Trading System.
"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Any, List
import json

from config.settings import get_settings
from core.services.backtesting_service import BacktestEngine, BacktestConfig, BacktestMode
from core.infrastructure.database.backtest_repository import BacktestRepository

logger = logging.getLogger(__name__)


class BacktestRunner:
    """Automated backtest execution and analysis"""
    
    def __init__(self):
        self.settings = get_settings()
        self.backtest_repo = BacktestRepository(self.settings.database.db_path)
        self.results_dir = Path("backtest_results")
        self.results_dir.mkdir(exist_ok=True)
    
    async def run_weekly_backtest(self) -> Dict[str, Any]:
        """Run comprehensive weekly backtest"""
        logger.info("Starting weekly comprehensive backtest")
        
        try:
            # Configure backtest parameters
            end_date = datetime.now(timezone.utc)
            start_date = datetime.now(timezone.utc) - timedelta(days=90)
            
            config = BacktestConfig(
                start_date=start_date,
                end_date=end_date,
                symbols=self.settings.trading.symbols,
                mode=BacktestMode.FULL,  # Full mode with ML
                initial_balance=10000.0,
                risk_per_trade=self.settings.trading.risk_per_trade_percent,
                max_open_trades=self.settings.trading.max_open_trades,
                spread_multiplier=1.5,  # Realistic spreads
                commission_per_lot=7.0,
                use_ml_signals=True,
                parallel_processing=True
            )
            
            # Initialize required components
            from core.infrastructure.mt5.data_provider import MT5DataProvider
            from core.infrastructure.mt5.client import MT5Client
            
            mt5_client = MT5Client(self.settings.mt5)
            data_provider = MT5DataProvider(mt5_client)
            
            # Initialize backtest engine
            engine = BacktestEngine(
                data_provider=data_provider,
                db_path=self.settings.database.db_path
            )
            
            # Run backtest
            logger.info(f"Running backtest from {start_date} to {end_date}")
            results = await engine.run_backtest(config)
            
            # Save results
            report_path = await self._save_results(results, config)
            
            # Analyze results
            analysis = await self._analyze_results(results)
            
            return {
                'success': True,
                'report_path': str(report_path),
                'summary': {
                    'total_trades': results.total_trades,
                    'win_rate': results.win_rate,
                    'total_pnl': results.total_pnl,
                    'sharpe_ratio': results.sharpe_ratio,
                    'max_drawdown': results.max_drawdown
                },
                'analysis': analysis
            }
            
        except Exception as e:
            logger.error(f"Weekly backtest failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def run_strategy_comparison(
        self, 
        strategies: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Compare multiple strategy configurations"""
        logger.info(f"Comparing {len(strategies)} strategies")
        
        results = {}
        
        for i, strategy in enumerate(strategies):
            logger.info(f"Testing strategy {i+1}/{len(strategies)}: {strategy['name']}")
            
            # Create config with strategy parameters
            config = BacktestConfig(
                start_date=strategy.get('start_date', 
                    datetime.now(timezone.utc) - timedelta(days=30)),
                end_date=strategy.get('end_date', datetime.now(timezone.utc)),
                symbols=strategy.get('symbols', self.settings.trading.symbols),
                mode=BacktestMode[strategy.get('mode', 'SIGNALS_ONLY')],
                initial_balance=10000.0,
                risk_per_trade=strategy.get('risk_percent', 1.5),
                max_open_trades=strategy.get('max_trades', 3),
                **strategy.get('extra_params', {})
            )
            
            # Initialize components for backtest
            mt5_client = MT5Client(self.settings.mt5)
            data_provider = MT5DataProvider(mt5_client)
            
            # Run backtest
            engine = BacktestEngine(
                data_provider=data_provider,
                db_path=self.settings.database.db_path
            )
            result = await engine.run_backtest(config)
            
            results[strategy['name']] = {
                'config': strategy,
                'results': result,
                'score': self._calculate_strategy_score(result)
            }
        
        # Find best strategy
        best_strategy = max(
            results.items(),
            key=lambda x: x[1]['score']
        )
        
        return {
            'strategies': results,
            'best_strategy': best_strategy[0],
            'comparison': self._create_comparison_table(results)
        }
    
    async def run_parameter_optimization(
        self,
        parameter_ranges: Dict[str, List[Any]]
    ) -> Dict[str, Any]:
        """Optimize strategy parameters"""
        logger.info("Starting parameter optimization")
        
        # Generate parameter combinations
        from itertools import product
        
        param_names = list(parameter_ranges.keys())
        param_values = [parameter_ranges[name] for name in param_names]
        combinations = list(product(*param_values))
        
        logger.info(f"Testing {len(combinations)} parameter combinations")
        
        best_result = None
        best_params = None
        best_score = float('-inf')
        
        for i, combo in enumerate(combinations):
            params = dict(zip(param_names, combo))
            logger.info(f"Testing combination {i+1}/{len(combinations)}: {params}")
            
            # Create config with parameters
            config = BacktestConfig(
                start_date=datetime.now(timezone.utc) - timedelta(days=60),
                end_date=datetime.now(timezone.utc),
                symbols=self.settings.trading.symbols[:2],  # Limit symbols for speed
                mode=BacktestMode.OFFLINE_ONLY,  # Fast mode
                initial_balance=10000.0,
                **params
            )
            
            # Initialize components for backtest
            mt5_client = MT5Client(self.settings.mt5)
            data_provider = MT5DataProvider(mt5_client)
            
            # Run backtest
            engine = BacktestEngine(
                data_provider=data_provider,
                db_path=self.settings.database.db_path
            )
            result = await engine.run_backtest(config)
            
            # Calculate score
            score = self._calculate_strategy_score(result)
            
            if score > best_score:
                best_score = score
                best_result = result
                best_params = params
        
        return {
            'best_parameters': best_params,
            'best_score': best_score,
            'best_result': {
                'total_trades': best_result.total_trades,
                'win_rate': best_result.win_rate,
                'sharpe_ratio': best_result.sharpe_ratio,
                'total_pnl': best_result.total_pnl
            },
            'iterations': len(combinations)
        }
    
    def _calculate_strategy_score(self, result) -> float:
        """Calculate a composite score for strategy performance"""
        # Weighted scoring system
        score = 0
        
        # Profitability (40%)
        score += (result.total_pnl / 10000) * 40  # Normalized by initial balance
        
        # Win rate (20%)
        score += result.win_rate * 20
        
        # Risk-adjusted returns (30%)
        if result.sharpe_ratio > 0:
            score += min(result.sharpe_ratio * 10, 30)  # Cap at 30
        
        # Drawdown penalty (10%)
        drawdown_penalty = min(result.max_drawdown / 10, 10)
        score -= drawdown_penalty
        
        return score
    
    async def _save_results(
        self, 
        results: Any, 
        config: BacktestConfig
    ) -> Path:
        """Save backtest results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed report
        report_path = self.results_dir / f"backtest_{timestamp}.html"
        # TODO: Implement HTML report generation
        # For now, just save the summary
        
        # Save summary JSON
        summary_path = self.results_dir / f"backtest_{timestamp}_summary.json"
        summary = {
            'timestamp': timestamp,
            'config': {
                'start_date': str(config.start_date),
                'end_date': str(config.end_date),
                'symbols': config.symbols,
                'mode': config.mode.value,
                'risk_per_trade': config.risk_per_trade,
                'max_open_trades': config.max_open_trades
            },
            'results': {
                'total_trades': results.total_trades,
                'winning_trades': results.winning_trades,
                'losing_trades': results.losing_trades,
                'win_rate': results.win_rate,
                'total_pnl': results.total_pnl,
                'total_pnl_percent': results.total_return,
                'sharpe_ratio': results.sharpe_ratio,
                'sortino_ratio': results.sortino_ratio,
                'max_drawdown_percent': results.max_drawdown,
                'profit_factor': results.profit_factor,
                'avg_win': results.average_win,
                'avg_loss': results.average_loss,
                'avg_trade_duration': results.average_bars_held
            },
            'symbol_performance': results.statistics.get('by_symbol', {})
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save to database
        await self.backtest_repo.save_results(results, config)
        
        return report_path
    
    async def _analyze_results(self, results: Any) -> Dict[str, Any]:
        """Analyze backtest results for insights"""
        analysis = {
            'performance_rating': self._rate_performance(results),
            'risk_assessment': self._assess_risk(results),
            'consistency_analysis': self._analyze_consistency(results),
            'recommendations': []
        }
        
        # Generate recommendations
        if results.win_rate < 0.4:
            analysis['recommendations'].append(
                "Low win rate - consider reviewing entry criteria"
            )
        
        if results.max_drawdown > 20:
            analysis['recommendations'].append(
                "High drawdown - consider reducing position sizes"
            )
        
        if results.sharpe_ratio < 1.0:
            analysis['recommendations'].append(
                "Low risk-adjusted returns - strategy may need optimization"
            )
        
        # Analyze by symbol
        symbol_analysis = {}
        symbol_performance = results.statistics.get('by_symbol', {})
        for symbol, perf in symbol_performance.items():
            symbol_analysis[symbol] = {
                'profitable': perf.get('total_pnl', 0) > 0,
                'trades': perf.get('trades', 0),
                'win_rate': perf.get('win_rate', 0),
                'recommendation': self._get_symbol_recommendation(perf)
            }
        
        analysis['symbol_analysis'] = symbol_analysis
        
        return analysis
    
    def _rate_performance(self, results) -> str:
        """Rate overall performance"""
        score = self._calculate_strategy_score(results)
        
        if score >= 80:
            return "Excellent"
        elif score >= 60:
            return "Good"
        elif score >= 40:
            return "Average"
        elif score >= 20:
            return "Below Average"
        else:
            return "Poor"
    
    def _assess_risk(self, results) -> Dict[str, Any]:
        """Assess risk metrics"""
        return {
            'risk_level': 'High' if results.max_drawdown > 20 else 
                         'Medium' if results.max_drawdown > 10 else 'Low',
            'max_drawdown': results.max_drawdown,
            'risk_reward_ratio': results.average_win / abs(results.average_loss) if results.average_loss != 0 else 0,
            'consecutive_losses': 0  # TODO: Calculate from trades
        }
    
    def _analyze_consistency(self, results) -> Dict[str, Any]:
        """Analyze trading consistency"""
        # This would analyze monthly/weekly performance consistency
        # Simplified version here
        return {
            'trades_per_day': results.total_trades / 
                             max((results.config.end_date.date() - results.config.start_date.date()).days if hasattr(results.config.end_date, 'date') else (results.config.end_date - results.config.start_date).days, 1),
            'profit_factor': results.profit_factor,
            'win_rate_stability': 'Stable'  # Would calculate variance
        }
    
    def _get_symbol_recommendation(self, performance: Dict[str, Any]) -> str:
        """Get recommendation for specific symbol"""
        total_pnl = performance.get('total_pnl', 0)
        win_rate = performance.get('win_rate', 0)
        
        if total_pnl < 0 and win_rate < 0.4:
            return "Consider removing from portfolio"
        elif win_rate > 0.6 and total_pnl > 0:
            return "Strong performer - consider increasing allocation"
        else:
            return "Monitor performance"
    
    def _create_comparison_table(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create comparison table for multiple strategies"""
        table = []
        
        for name, data in results.items():
            result = data['results']
            table.append({
                'strategy': name,
                'trades': result.total_trades,
                'win_rate': f"{result.win_rate:.2%}",
                'total_pnl': f"${result.total_pnl:.2f}",
                'sharpe_ratio': f"{result.sharpe_ratio:.2f}",
                'max_drawdown': f"{result.max_drawdown:.2%}",
                'score': f"{data['score']:.2f}"
            })
        
        return sorted(table, key=lambda x: x['score'], reverse=True)


async def main():
    """Run backtest standalone"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Automated Backtesting')
    parser.add_argument('--mode', choices=['weekly', 'optimize', 'compare'],
                      default='weekly', help='Backtest mode')
    parser.add_argument('--days', type=int, default=90,
                      help='Number of days to backtest')
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(name)s | %(levelname)s | %(message)s'
    )
    
    runner = BacktestRunner()
    
    if args.mode == 'weekly':
        result = await runner.run_weekly_backtest()
        if result['success']:
            print(f"✅ Backtest completed successfully!")
            print(f"   Report: {result['report_path']}")
            print(f"   Performance: {result['analysis']['performance_rating']}")
            print(f"   Summary:")
            for key, value in result['summary'].items():
                print(f"     {key}: {value}")
    
    elif args.mode == 'optimize':
        # Example parameter optimization
        result = await runner.run_parameter_optimization({
            'risk_per_trade': [1.0, 1.5, 2.0],
            'max_open_trades': [1, 2, 3],
            'spread_multiplier': [1.0, 1.5, 2.0]
        })
        print(f"✅ Optimization completed!")
        print(f"   Best parameters: {result['best_parameters']}")
        print(f"   Best score: {result['best_score']:.2f}")
    
    elif args.mode == 'compare':
        # Example strategy comparison
        strategies = [
            {
                'name': 'Conservative',
                'risk_percent': 1.0,
                'max_trades': 1
            },
            {
                'name': 'Moderate',
                'risk_percent': 1.5,
                'max_trades': 2
            },
            {
                'name': 'Aggressive',
                'risk_percent': 2.0,
                'max_trades': 3
            }
        ]
        result = await runner.run_strategy_comparison(strategies)
        print(f"✅ Strategy comparison completed!")
        print(f"   Best strategy: {result['best_strategy']}")
        print("\n   Comparison table:")
        for row in result['comparison']:
            print(f"     {row}")


if __name__ == "__main__":
    asyncio.run(main())