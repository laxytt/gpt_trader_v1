"""
Async database repositories for high-performance data access.
Uses aiosqlite for non-blocking database operations.
"""

import aiosqlite
import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, AsyncGenerator
from datetime import datetime, timezone
from contextlib import asynccontextmanager
import json
from pathlib import Path

from core.domain.models import (
    Trade, TradingSignal, TradeCase, TradeStatus, SignalType, 
    RiskClass, TradeResult, create_trade_id, create_case_id
)
from core.domain.exceptions import DatabaseError, RepositoryError
from core.infrastructure.database.migrations import migrate_database

logger = logging.getLogger(__name__)


class AsyncBaseRepository(ABC):
    """Base async repository class with common database operations"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._ensure_database_exists()
    
    def _ensure_database_exists(self):
        """Ensure database file and tables exist (sync operation on init)"""
        try:
            if not migrate_database(self.db_path):
                raise DatabaseError("Database migration failed")
            logger.debug(f"Async database initialized: {self.db_path}")
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise DatabaseError(f"Failed to initialize database: {str(e)}")
    
    @asynccontextmanager
    async def get_connection(self) -> AsyncGenerator[aiosqlite.Connection, None]:
        """Get async database connection with automatic cleanup"""
        conn = None
        try:
            conn = await aiosqlite.connect(self.db_path)
            conn.row_factory = aiosqlite.Row
            # Enable WAL mode for better concurrency
            await conn.execute("PRAGMA journal_mode=WAL")
            await conn.execute("PRAGMA busy_timeout=5000")
            
            yield conn
            
            await conn.commit()
        except Exception as e:
            if conn:
                await conn.rollback()
            logger.error(f"Async database error: {e}")
            raise DatabaseError(f"Async database operation failed: {str(e)}")
        finally:
            if conn:
                await conn.close()
    
    @asynccontextmanager
    async def transaction(self) -> AsyncGenerator[aiosqlite.Connection, None]:
        """Get a connection with explicit transaction control"""
        async with self.get_connection() as conn:
            yield conn


class AsyncTradeRepository(AsyncBaseRepository):
    """Async repository for trade data persistence and retrieval"""
    
    async def save_trade(self, trade: Trade) -> bool:
        """Save or update a trade asynchronously"""
        query = """
            INSERT OR REPLACE INTO trades (
                id, symbol, direction, volume, entry_price, exit_price,
                stop_loss, take_profit, profit_loss, commission, swap,
                entry_time, exit_time, duration_minutes, trade_type,
                risk_class, signal_id, reflection, max_drawdown_pips,
                max_profit_pips, management_actions, status, result,
                original_signal_json, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        async with self.get_connection() as conn:
            try:
                await conn.execute(query, (
                    trade.id,
                    trade.symbol,
                    trade.side.value,
                    trade.lot_size,
                    trade.entry_price,
                    trade.exit_price,
                    trade.stop_loss,
                    trade.take_profit,
                    trade.current_pnl,
                    0,  # commission
                    0,  # swap
                    trade.timestamp.isoformat(),
                    trade.exit_timestamp.isoformat() if trade.exit_timestamp else None,
                    trade.duration_minutes,
                    'live',  # trade_type
                    trade.risk_class.value if hasattr(trade, 'risk_class') else 'C',
                    getattr(trade, 'signal_id', None),
                    trade.reflection,
                    trade.max_drawdown_pips,
                    getattr(trade, 'max_profit_pips', 0),
                    json.dumps(trade.management_history),
                    trade.status.value,
                    trade.result.value if trade.result else None,
                    json.dumps(trade.original_signal.to_dict()) if trade.original_signal else None,
                    json.dumps(getattr(trade, 'metadata', {}))
                ))
                
                logger.info(f"Trade saved: {trade.id}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to save trade {trade.id}: {e}")
                raise RepositoryError(f"Failed to save trade: {str(e)}")
    
    async def get_trade_by_id(self, trade_id: str) -> Optional[Trade]:
        """Get a specific trade by ID asynchronously"""
        query = "SELECT * FROM trades WHERE id = ?"
        
        async with self.get_connection() as conn:
            async with conn.execute(query, (trade_id,)) as cursor:
                row = await cursor.fetchone()
                if row:
                    return self._row_to_trade(dict(row))
                return None
    
    async def get_open_trades(self) -> List[Trade]:
        """Get all currently open trades asynchronously"""
        query = "SELECT * FROM trades WHERE status = ? ORDER BY entry_time DESC"
        
        async with self.get_connection() as conn:
            trades = []
            async with conn.execute(query, (TradeStatus.OPEN.value,)) as cursor:
                async for row in cursor:
                    trade = self._row_to_trade(dict(row))
                    if trade:
                        trades.append(trade)
            return trades
    
    async def get_trades_by_date_range(
        self, 
        start_date: datetime, 
        end_date: datetime,
        symbol: Optional[str] = None
    ) -> List[Trade]:
        """Get trades within a date range asynchronously"""
        if symbol:
            query = """
                SELECT * FROM trades 
                WHERE entry_time >= ? AND entry_time <= ? AND symbol = ?
                ORDER BY entry_time DESC
            """
            params = (start_date.isoformat(), end_date.isoformat(), symbol)
        else:
            query = """
                SELECT * FROM trades 
                WHERE entry_time >= ? AND entry_time <= ?
                ORDER BY entry_time DESC
            """
            params = (start_date.isoformat(), end_date.isoformat())
        
        async with self.get_connection() as conn:
            trades = []
            async with conn.execute(query, params) as cursor:
                async for row in cursor:
                    trade = self._row_to_trade(dict(row))
                    if trade:
                        trades.append(trade)
            return trades
    
    async def get_recent_trades(self, limit: int = 10) -> List[Trade]:
        """Get most recent trades asynchronously"""
        query = "SELECT * FROM trades ORDER BY entry_time DESC LIMIT ?"
        
        async with self.get_connection() as conn:
            trades = []
            async with conn.execute(query, (limit,)) as cursor:
                async for row in cursor:
                    trade = self._row_to_trade(dict(row))
                    if trade:
                        trades.append(trade)
            return trades
    
    async def get_trade_by_symbol(self, symbol: str) -> Optional[Trade]:
        """Get the most recent open trade for a symbol asynchronously"""
        query = """
            SELECT * FROM trades 
            WHERE symbol = ? AND status = ? 
            ORDER BY entry_time DESC 
            LIMIT 1
        """
        
        async with self.get_connection() as conn:
            async with conn.execute(query, (symbol, TradeStatus.OPEN.value)) as cursor:
                row = await cursor.fetchone()
                if row:
                    return self._row_to_trade(dict(row))
                return None
    
    def _row_to_trade(self, row: Dict[str, Any]) -> Optional[Trade]:
        """Convert database row to Trade object"""
        try:
            # Parse original signal if available
            original_signal = None
            if row.get('original_signal_json'):
                signal_data = json.loads(row['original_signal_json'])
                original_signal = TradingSignal(
                    symbol=signal_data['symbol'],
                    signal=SignalType[signal_data['signal']],
                    reason=signal_data['reason'],
                    risk_class=RiskClass[signal_data['risk_class']],
                    timestamp=datetime.fromisoformat(signal_data['timestamp']),
                    entry=signal_data.get('entry'),
                    stop_loss=signal_data.get('stop_loss'),
                    take_profit=signal_data.get('take_profit'),
                    risk_reward=signal_data.get('risk_reward')
                )
            
            trade = Trade(
                id=row['id'],
                symbol=row['symbol'],
                side=SignalType[row['direction']],
                entry_price=row['entry_price'],
                stop_loss=row['stop_loss'],
                take_profit=row['take_profit'],
                status=TradeStatus[row['status']],
                timestamp=datetime.fromisoformat(row['entry_time']),
                ticket=row.get('ticket'),
                lot_size=row.get('volume'),
                max_drawdown_pips=row.get('max_drawdown_pips', 0),
                current_pnl=row.get('profit_loss', 0),
                exit_price=row.get('exit_price'),
                exit_timestamp=datetime.fromisoformat(row['exit_time']) if row.get('exit_time') else None,
                result=TradeResult[row['result']] if row.get('result') else None,
                risk_reward_ratio=row.get('risk_reward_ratio'),
                risk_amount_usd=row.get('risk_amount_usd'),
                original_signal=original_signal,
                management_history=json.loads(row.get('management_actions', '[]')),
                reflection=row.get('reflection')
            )
            
            return trade
            
        except Exception as e:
            logger.error(f"Failed to parse trade row: {e}")
            return None


class AsyncSignalRepository(AsyncBaseRepository):
    """Async repository for trading signal data"""
    
    async def save_signal(self, signal: TradingSignal) -> int:
        """Save a trading signal and return its ID asynchronously"""
        query = """
            INSERT INTO signals (
                symbol, direction, entry_price, stop_loss, take_profit,
                risk_reward, confidence, ml_confidence, analysis,
                created_at, valid_until, executed, trade_id, 
                metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        analysis_json = json.dumps(signal.market_context) if signal.market_context else None
        metadata = {
            'risk_class': signal.risk_class.value,
            'reason': signal.reason,
            'news_events': len(signal.news_events)
        }
        
        async with self.get_connection() as conn:
            try:
                cursor = await conn.execute(query, (
                    signal.symbol,
                    signal.signal.value,
                    signal.entry,
                    signal.stop_loss,
                    signal.take_profit,
                    signal.risk_reward,
                    signal.market_context.get('confidence', 0) if signal.market_context else 0,
                    signal.market_context.get('ml_confidence', 0) if signal.market_context else 0,
                    analysis_json,
                    signal.timestamp.isoformat(),
                    (signal.timestamp.replace(hour=23, minute=59, second=59)).isoformat(),
                    False,
                    None,
                    json.dumps(metadata)
                ))
                
                signal_id = cursor.lastrowid
                logger.info(f"Signal saved with ID: {signal_id}")
                return signal_id
                
            except Exception as e:
                logger.error(f"Failed to save signal: {e}")
                raise RepositoryError(f"Failed to save signal: {str(e)}")
    
    async def get_signal_by_id(self, signal_id: int) -> Optional[TradingSignal]:
        """Get a specific signal by ID asynchronously"""
        query = "SELECT * FROM signals WHERE id = ?"
        
        async with self.get_connection() as conn:
            async with conn.execute(query, (signal_id,)) as cursor:
                row = await cursor.fetchone()
                if row:
                    return self._row_to_signal(dict(row))
                return None
    
    async def get_recent_signals(self, limit: int = 10) -> List[TradingSignal]:
        """Get most recent signals asynchronously"""
        query = "SELECT * FROM signals ORDER BY created_at DESC LIMIT ?"
        
        async with self.get_connection() as conn:
            signals = []
            async with conn.execute(query, (limit,)) as cursor:
                async for row in cursor:
                    signal = self._row_to_signal(dict(row))
                    if signal:
                        signals.append(signal)
            return signals
    
    async def get_unexecuted_signals(self) -> List[TradingSignal]:
        """Get signals that haven't been executed yet asynchronously"""
        query = """
            SELECT * FROM signals 
            WHERE executed = 0 AND valid_until > ? AND direction != 'WAIT'
            ORDER BY created_at DESC
        """
        
        async with self.get_connection() as conn:
            signals = []
            async with conn.execute(query, (datetime.now(timezone.utc).isoformat(),)) as cursor:
                async for row in cursor:
                    signal = self._row_to_signal(dict(row))
                    if signal:
                        signals.append(signal)
            return signals
    
    async def mark_signal_executed(self, signal_id: int, trade_id: str) -> bool:
        """Mark a signal as executed with associated trade asynchronously"""
        query = "UPDATE signals SET executed = 1, trade_id = ? WHERE id = ?"
        
        async with self.get_connection() as conn:
            try:
                await conn.execute(query, (trade_id, signal_id))
                logger.info(f"Signal {signal_id} marked as executed with trade {trade_id}")
                return True
            except Exception as e:
                logger.error(f"Failed to mark signal as executed: {e}")
                return False
    
    def _row_to_signal(self, row: Dict[str, Any]) -> Optional[TradingSignal]:
        """Convert database row to TradingSignal object"""
        try:
            metadata = json.loads(row.get('metadata_json', '{}'))
            market_context = json.loads(row.get('analysis', '{}')) if row.get('analysis') else {}
            
            signal = TradingSignal(
                symbol=row['symbol'],
                signal=SignalType[row['direction']],
                reason=metadata.get('reason', ''),
                risk_class=RiskClass[metadata.get('risk_class', 'C')],
                timestamp=datetime.fromisoformat(row['created_at']),
                entry=row.get('entry_price'),
                stop_loss=row.get('stop_loss'),
                take_profit=row.get('take_profit'),
                risk_reward=row.get('risk_reward'),
                market_context=market_context
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"Failed to parse signal row: {e}")
            return None


# Export async repositories
__all__ = ['AsyncTradeRepository', 'AsyncSignalRepository']