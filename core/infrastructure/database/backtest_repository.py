"""Repository for backtest results storage"""

import json
import sqlite3
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
import uuid

from core.infrastructure.database.repositories import BaseRepository
from core.services.backtesting_service import BacktestResults, BacktestTrade
from core.domain.exceptions import RepositoryError, handle_database_errors


class BacktestRepository(BaseRepository):
    """Repository for backtest results persistence"""
    
    def _create_tables(self, conn: sqlite3.Connection):
        """Tables are created by migrations"""
        pass
    
    @handle_database_errors
    def save_backtest_run(self, results: BacktestResults) -> str:
        """
        Save complete backtest run to database.
        
        Args:
            results: BacktestResults object
            
        Returns:
            Backtest run ID
        """
        run_id = f"backtest_{uuid.uuid4().hex[:12]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        with self.get_connection() as conn:
            # Save main run data
            conn.execute("""
                INSERT INTO backtest_runs (
                    id, start_date, end_date, symbols, mode,
                    initial_balance, risk_per_trade, final_balance,
                    total_return, win_rate, sharpe_ratio, max_drawdown,
                    profit_factor, total_trades, config_json, statistics_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                run_id,
                results.config.start_date.isoformat(),
                results.config.end_date.isoformat(),
                json.dumps(results.config.symbols),
                results.config.mode.value,
                results.config.initial_balance,
                results.config.risk_per_trade,
                results.equity_curve[-1] if results.equity_curve else results.config.initial_balance,
                results.total_return,
                results.win_rate,
                results.sharpe_ratio,
                results.max_drawdown,
                results.profit_factor,
                results.total_trades,
                json.dumps(self._config_to_dict(results.config)),
                json.dumps(results.statistics)
            ))
            
            # Save individual trades
            for trade in results.trades:
                self._save_backtest_trade(conn, run_id, trade)
            
            # Save equity curve samples (every 10th point to save space)
            for i, equity in enumerate(results.equity_curve[::10]):
                timestamp = results.config.start_date + (results.config.end_date - results.config.start_date) * (i * 10 / len(results.equity_curve))
                
                # Calculate drawdown at this point
                peak = max(results.equity_curve[:i*10+1]) if i > 0 else equity
                drawdown = (peak - equity) / peak * 100 if peak > 0 else 0
                
                conn.execute("""
                    INSERT INTO backtest_equity_curves (
                        backtest_run_id, timestamp, equity, drawdown
                    ) VALUES (?, ?, ?, ?)
                """, (run_id, timestamp.isoformat(), equity, drawdown))
            
            conn.commit()
            
        return run_id
    
    def _save_backtest_trade(self, conn: sqlite3.Connection, run_id: str, trade: BacktestTrade):
        """Save individual backtest trade"""
        # Extract validation score if available from signal
        validation_score = None
        if hasattr(trade.signal, 'metadata') and trade.signal.metadata:
            validation_score = trade.signal.metadata.get('validation_score')
        
        conn.execute("""
            INSERT INTO backtest_trades (
                backtest_run_id, symbol, signal_type, entry_time, exit_time,
                entry_price, exit_price, stop_loss, take_profit, lot_size,
                pnl, pnl_points, result, exit_reason, bars_held,
                risk_reward_achieved, signal_reason, risk_class, validation_score
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            run_id,
            trade.signal.symbol,
            trade.signal.signal.value,
            trade.entry_time.isoformat(),
            trade.exit_time.isoformat() if trade.exit_time else None,
            trade.entry_price,
            trade.exit_price,
            trade.signal.stop_loss,
            trade.signal.take_profit,
            trade.lot_size,
            trade.pnl,
            trade.pnl_points,
            trade.result.value if trade.result else None,
            trade.exit_reason,
            trade.bars_held,
            trade.risk_reward_achieved,
            trade.signal.reason,
            trade.signal.risk_class.value,
            validation_score
        ))
    
    def _config_to_dict(self, config) -> Dict[str, Any]:
        """Convert BacktestConfig to dictionary"""
        return {
            'start_date': config.start_date.isoformat(),
            'end_date': config.end_date.isoformat(),
            'symbols': config.symbols,
            'timeframe': config.timeframe,
            'initial_balance': config.initial_balance,
            'risk_per_trade': config.risk_per_trade,
            'max_open_trades': config.max_open_trades,
            'commission_per_lot': config.commission_per_lot,
            'slippage_points': config.slippage_points,
            'mode': config.mode.value,
            'use_spread_filter': config.use_spread_filter,
            'use_news_filter': config.use_news_filter
        }
    
    @handle_database_errors
    def get_backtest_runs(
        self, 
        symbol: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get backtest runs, optionally filtered by symbol"""
        with self.get_connection() as conn:
            if symbol:
                cursor = conn.execute("""
                    SELECT * FROM backtest_runs 
                    WHERE symbols LIKE ? 
                    ORDER BY created_at DESC 
                    LIMIT ?
                """, (f'%"{symbol}"%', limit))
            else:
                cursor = conn.execute("""
                    SELECT * FROM backtest_runs 
                    ORDER BY created_at DESC 
                    LIMIT ?
                """, (limit,))
            
            return [dict(row) for row in cursor.fetchall()]
    
    @handle_database_errors
    def get_backtest_trades(
        self, 
        run_id: Optional[str] = None,
        symbol: Optional[str] = None,
        result: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get backtest trades with optional filters"""
        with self.get_connection() as conn:
            query = "SELECT * FROM backtest_trades WHERE 1=1"
            params = []
            
            if run_id:
                query += " AND backtest_run_id = ?"
                params.append(run_id)
            
            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)
            
            if result:
                query += " AND result = ?"
                params.append(result)
            
            query += " ORDER BY entry_time DESC"
            
            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
    
    @handle_database_errors
    def get_aggregated_stats(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Get aggregated statistics across all backtests"""
        with self.get_connection() as conn:
            # Overall stats
            if symbol:
                cursor = conn.execute("""
                    SELECT 
                        COUNT(DISTINCT backtest_run_id) as total_runs,
                        COUNT(*) as total_trades,
                        AVG(CASE WHEN result = 'win' THEN 1 ELSE 0 END) as avg_win_rate,
                        AVG(pnl) as avg_pnl,
                        AVG(risk_reward_achieved) as avg_rr,
                        AVG(bars_held) as avg_duration
                    FROM backtest_trades
                    WHERE symbol = ?
                """, (symbol,))
            else:
                cursor = conn.execute("""
                    SELECT 
                        COUNT(DISTINCT backtest_run_id) as total_runs,
                        COUNT(*) as total_trades,
                        AVG(CASE WHEN result = 'win' THEN 1 ELSE 0 END) as avg_win_rate,
                        AVG(pnl) as avg_pnl,
                        AVG(risk_reward_achieved) as avg_rr,
                        AVG(bars_held) as avg_duration
                    FROM backtest_trades
                """)
            
            overall_stats = dict(cursor.fetchone())
            
            # Performance by validation score ranges
            cursor = conn.execute("""
                SELECT 
                    CASE 
                        WHEN validation_score >= 0.8 THEN 'High (0.8+)'
                        WHEN validation_score >= 0.6 THEN 'Medium (0.6-0.8)'
                        WHEN validation_score >= 0.4 THEN 'Low (0.4-0.6)'
                        ELSE 'Very Low (<0.4)'
                    END as score_range,
                    COUNT(*) as trades,
                    AVG(CASE WHEN result = 'win' THEN 1 ELSE 0 END) as win_rate,
                    AVG(pnl) as avg_pnl
                FROM backtest_trades
                WHERE validation_score IS NOT NULL
                GROUP BY score_range
                ORDER BY score_range
            """)
            
            validation_stats = [dict(row) for row in cursor.fetchall()]
            
            return {
                'overall': overall_stats,
                'by_validation_score': validation_stats
            }