#!/usr/bin/env python3
"""
Performance Analytics System
Tracks and analyzes trading performance for continuous improvement
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sys
import json
from typing import Dict, List, Any, Tuple

sys.path.append(str(Path(__file__).parent.parent))

from core.infrastructure.database.repositories import TradeRepository, SignalRepository
from core.domain.models import Trade, TradingSignal, SignalType
from config.settings import get_settings
import matplotlib.pyplot as plt
import seaborn as sns


class PerformanceAnalytics:
    """Comprehensive performance tracking and analysis"""
    
    def __init__(self):
        self.settings = get_settings()
        self.trade_repo = TradeRepository(self.settings.database.db_path)
        self.signal_repo = SignalRepository(self.settings.database.db_path)
        
    def analyze_performance(self, days_back: int = 30) -> Dict[str, Any]:
        """Comprehensive performance analysis"""
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        analysis = {
            'period': f'{days_back} days',
            'start_date': start_date.isoformat(),
            'end_date': end_date.isoformat(),
            'symbols': {}
        }
        
        # Analyze each symbol
        for symbol in self.settings.trading.symbols:
            symbol_analysis = self._analyze_symbol(symbol, start_date, end_date)
            analysis['symbols'][symbol] = symbol_analysis
        
        # Overall metrics
        analysis['overall'] = self._calculate_overall_metrics(analysis['symbols'])
        
        # Identify improvement areas
        analysis['recommendations'] = self._generate_recommendations(analysis)
        
        return analysis
    
    def _analyze_symbol(self, symbol: str, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Analyze performance for a specific symbol"""
        
        trades = self.trade_repo.get_trades_by_date_range(start_date, end_date, symbol)
        signals = self.signal_repo.get_signals_by_date_range(start_date, end_date, symbol)
        
        if not trades:
            return {
                'total_trades': 0,
                'status': 'no_trades',
                'recommendation': 'Increase signal sensitivity or review market conditions'
            }
        
        # Basic metrics
        winning_trades = [t for t in trades if t.profit_loss and t.profit_loss > 0]
        losing_trades = [t for t in trades if t.profit_loss and t.profit_loss < 0]
        
        total_profit = sum(t.profit_loss for t in trades if t.profit_loss)
        avg_win = np.mean([t.profit_loss for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.profit_loss for t in losing_trades]) if losing_trades else 0
        
        # Advanced metrics
        metrics = {
            'total_trades': len(trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / len(trades) if trades else 0,
            'total_profit': total_profit,
            'avg_profit_per_trade': total_profit / len(trades) if trades else 0,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': abs(avg_win / avg_loss) if avg_loss != 0 else float('inf'),
            'expectancy': (len(winning_trades)/len(trades) * avg_win) + (len(losing_trades)/len(trades) * avg_loss) if trades else 0,
            'signal_count': len(signals),
            'signal_execution_rate': len(trades) / len(signals) if signals else 0,
            'signal_quality': self._analyze_signal_quality(signals, trades),
            'best_hours': self._analyze_best_trading_hours(trades),
            'drawdown': self._calculate_max_drawdown(trades),
            'sharpe_ratio': self._calculate_sharpe_ratio(trades)
        }
        
        # Pattern analysis
        metrics['patterns'] = self._analyze_winning_patterns(trades, signals)
        
        return metrics
    
    def _analyze_signal_quality(self, signals: List[TradingSignal], trades: List[Trade]) -> Dict[str, Any]:
        """Analyze quality of signals"""
        
        if not signals:
            return {'status': 'no_signals'}
        
        # Group signals by type
        signal_types = {}
        for signal in signals:
            signal_type = signal.signal.value
            if signal_type not in signal_types:
                signal_types[signal_type] = 0
            signal_types[signal_type] += 1
        
        # Match signals to trades
        executed_signals = 0
        for signal in signals:
            # Simple matching by time proximity (within 1 hour)
            for trade in trades:
                time_diff = abs((trade.entry_time - signal.timestamp).total_seconds())
                if time_diff < 3600 and signal.symbol == trade.symbol:
                    executed_signals += 1
                    break
        
        return {
            'total_signals': len(signals),
            'signal_distribution': signal_types,
            'execution_rate': executed_signals / len(signals) if signals else 0,
            'wait_signal_ratio': signal_types.get('WAIT', 0) / len(signals) if signals else 0
        }
    
    def _analyze_best_trading_hours(self, trades: List[Trade]) -> Dict[int, float]:
        """Analyze profitability by hour of day"""
        
        hourly_profit = {}
        
        for trade in trades:
            if trade.entry_time and trade.profit_loss:
                hour = trade.entry_time.hour
                if hour not in hourly_profit:
                    hourly_profit[hour] = []
                hourly_profit[hour].append(trade.profit_loss)
        
        # Calculate average profit by hour
        return {
            hour: np.mean(profits) 
            for hour, profits in hourly_profit.items()
        }
    
    def _calculate_max_drawdown(self, trades: List[Trade]) -> float:
        """Calculate maximum drawdown"""
        
        if not trades:
            return 0.0
        
        cumulative_profit = []
        running_total = 0
        
        for trade in sorted(trades, key=lambda x: x.entry_time):
            if trade.profit_loss:
                running_total += trade.profit_loss
                cumulative_profit.append(running_total)
        
        if not cumulative_profit:
            return 0.0
        
        # Calculate drawdown
        peak = cumulative_profit[0]
        max_drawdown = 0
        
        for value in cumulative_profit:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak if peak > 0 else 0
            max_drawdown = max(max_drawdown, drawdown)
        
        return max_drawdown
    
    def _calculate_sharpe_ratio(self, trades: List[Trade], risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        
        if not trades or len(trades) < 2:
            return 0.0
        
        returns = [t.profit_loss for t in trades if t.profit_loss is not None]
        
        if not returns:
            return 0.0
        
        # Annualized
        avg_return = np.mean(returns) * 252  # Assuming daily returns
        std_return = np.std(returns) * np.sqrt(252)
        
        if std_return == 0:
            return 0.0
        
        return (avg_return - risk_free_rate) / std_return
    
    def _analyze_winning_patterns(self, trades: List[Trade], signals: List[TradingSignal]) -> Dict[str, Any]:
        """Identify patterns in winning trades"""
        
        winning_trades = [t for t in trades if t.profit_loss and t.profit_loss > 0]
        
        if not winning_trades:
            return {'status': 'no_winning_trades'}
        
        patterns = {
            'entry_patterns': {},
            'time_patterns': {},
            'risk_patterns': {}
        }
        
        # Analyze entry patterns
        for trade in winning_trades:
            # Find corresponding signal
            for signal in signals:
                if signal.symbol == trade.symbol and abs((signal.timestamp - trade.entry_time).total_seconds()) < 3600:
                    risk_class = signal.risk_class.value
                    patterns['entry_patterns'][risk_class] = patterns['entry_patterns'].get(risk_class, 0) + 1
                    break
        
        # Time in trade analysis
        time_in_trade = []
        for trade in winning_trades:
            if trade.exit_time:
                duration = (trade.exit_time - trade.entry_time).total_seconds() / 3600  # hours
                time_in_trade.append(duration)
        
        if time_in_trade:
            patterns['time_patterns'] = {
                'avg_duration_hours': np.mean(time_in_trade),
                'min_duration_hours': np.min(time_in_trade),
                'max_duration_hours': np.max(time_in_trade)
            }
        
        return patterns
    
    def _calculate_overall_metrics(self, symbol_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall portfolio metrics"""
        
        total_trades = sum(m.get('total_trades', 0) for m in symbol_metrics.values())
        total_profit = sum(m.get('total_profit', 0) for m in symbol_metrics.values())
        
        if total_trades == 0:
            return {'status': 'no_trades'}
        
        total_wins = sum(m.get('winning_trades', 0) for m in symbol_metrics.values())
        
        return {
            'total_trades': total_trades,
            'total_profit': total_profit,
            'overall_win_rate': total_wins / total_trades if total_trades > 0 else 0,
            'avg_profit_per_trade': total_profit / total_trades,
            'best_symbol': max(symbol_metrics.items(), key=lambda x: x[1].get('total_profit', 0))[0] if symbol_metrics else None,
            'worst_symbol': min(symbol_metrics.items(), key=lambda x: x[1].get('total_profit', 0))[0] if symbol_metrics else None
        }
    
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations"""
        
        recommendations = []
        
        # Overall recommendations
        overall = analysis.get('overall', {})
        if overall.get('overall_win_rate', 0) < 0.45:
            recommendations.append("Win rate below 45% - Consider adjusting entry criteria or risk management")
        
        # Symbol-specific recommendations
        for symbol, metrics in analysis.get('symbols', {}).items():
            if metrics.get('total_trades', 0) == 0:
                recommendations.append(f"{symbol}: No trades executed - Review signal generation parameters")
            elif metrics.get('win_rate', 0) < 0.4:
                recommendations.append(f"{symbol}: Low win rate ({metrics['win_rate']:.1%}) - Consider retraining ML model")
            elif metrics.get('drawdown', 0) > 0.2:
                recommendations.append(f"{symbol}: High drawdown ({metrics['drawdown']:.1%}) - Implement stricter risk management")
            
            # Signal quality
            signal_quality = metrics.get('signal_quality', {})
            if signal_quality.get('wait_signal_ratio', 0) > 0.8:
                recommendations.append(f"{symbol}: Too many WAIT signals - Market conditions may be unfavorable")
        
        return recommendations
    
    def generate_report(self, output_path: str = "reports/performance_analysis.json"):
        """Generate comprehensive performance report"""
        
        analysis = self.analyze_performance(days_back=30)
        
        # Save JSON report
        output_file = Path(output_path)
        output_file.parent.mkdir(exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        print(f"Performance report saved to: {output_file}")
        
        # Print summary
        print("\n=== PERFORMANCE SUMMARY ===")
        print(f"Period: {analysis['period']}")
        
        overall = analysis.get('overall', {})
        if overall.get('status') != 'no_trades':
            print(f"Total Trades: {overall.get('total_trades', 0)}")
            print(f"Total Profit: ${overall.get('total_profit', 0):.2f}")
            print(f"Win Rate: {overall.get('overall_win_rate', 0):.1%}")
            print(f"Avg Profit/Trade: ${overall.get('avg_profit_per_trade', 0):.2f}")
        
        print("\n=== RECOMMENDATIONS ===")
        for rec in analysis.get('recommendations', []):
            print(f"• {rec}")
        
        return analysis


if __name__ == "__main__":
    analytics = PerformanceAnalytics()
    analytics.generate_report()