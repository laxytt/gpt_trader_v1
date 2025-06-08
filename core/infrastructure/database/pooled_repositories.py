"""
Database repositories using connection pooling for better performance.
These repositories use the connection pool instead of creating new connections.
"""

import sqlite3
import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
from contextlib import contextmanager
import json

from core.domain.models import (
    Trade, TradingSignal, TradeCase, TradeStatus, SignalType, 
    RiskClass, TradeResult, create_trade_id, create_case_id
)
from core.domain.exceptions import (
    DatabaseError, RepositoryError, SerializationError, handle_database_errors,
    ErrorContext
)
from core.infrastructure.database.migrations import migrate_database
from core.infrastructure.database.connection_pool import (
    get_sync_connection_pool, SqliteConnectionPool, PoolConfig
)

logger = logging.getLogger(__name__)


class PooledBaseRepository(ABC):
    """Base repository using connection pooling"""
    
    def __init__(self, db_path: str, pool_config: Optional[PoolConfig] = None):
        self.db_path = db_path
        self._ensure_database_exists()
        
        # Get or create connection pool
        self.pool = get_sync_connection_pool(db_path, pool_config)
        logger.info(f"Repository using connection pool for {db_path}")
    
    def _ensure_database_exists(self):
        """Ensure database file and migrations are run"""
        try:
            if not migrate_database(self.db_path):
                raise DatabaseError("Database migration failed")
            logger.debug(f"Database initialized: {self.db_path}")
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise DatabaseError(f"Failed to initialize database: {str(e)}")
    
    @contextmanager
    def get_connection(self, commit=True):
        """Get database connection from pool"""
        with self.pool.get_connection() as conn:
            try:
                yield conn
                if commit:
                    conn.commit()
            except Exception as e:
                conn.rollback()
                logger.error(f"Database error: {e}")
                raise DatabaseError(f"Database operation failed: {str(e)}")
    
    @contextmanager
    def transaction(self):
        """Get a connection with explicit transaction control"""
        with self.get_connection(commit=False) as conn:
            yield conn
            conn.commit()
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics"""
        return self.pool.get_stats()


class PooledTradeRepository(PooledBaseRepository):
    """Trade repository using connection pooling"""
    
    def save_trade(self, trade: Trade, conn: Optional[sqlite3.Connection] = None) -> None:
        """Save or update a trade"""
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
        
        # Use provided connection or get from pool
        if conn:
            self._execute_save(conn, query, trade)
        else:
            with self.get_connection() as conn:
                self._execute_save(conn, query, trade)
    
    def _execute_save(self, conn: sqlite3.Connection, query: str, trade: Trade):
        """Execute the save query"""
        management_actions_json = json.dumps(trade.management_history) if trade.management_history else None
        original_signal_json = json.dumps(trade.original_signal.to_dict()) if trade.original_signal else None
        metadata_json = json.dumps({
            'risk_reward_ratio': trade.risk_reward_ratio,
            'risk_amount_usd': trade.risk_amount_usd,
            'ticket': trade.ticket,
            'max_profit_pips': getattr(trade, 'max_profit_pips', 0),
            'broker_commission': getattr(trade, 'broker_commission', 0),
            'swap': getattr(trade, 'swap', 0)
        })
        
        conn.execute(query, (
            trade.id,
            trade.symbol,
            trade.side.value,
            trade.lot_size,
            trade.entry_price,
            trade.exit_price,
            trade.stop_loss,
            trade.take_profit,
            trade.current_pnl,
            getattr(trade, 'broker_commission', 0),
            getattr(trade, 'swap', 0),
            trade.timestamp.isoformat(),
            trade.exit_timestamp.isoformat() if trade.exit_timestamp else None,
            trade.duration_minutes,
            'live',  # trade_type
            getattr(trade, 'risk_class', RiskClass.CONSERVATIVE).value,
            getattr(trade, 'signal_id', None),
            trade.reflection,
            trade.max_drawdown_pips,
            getattr(trade, 'max_profit_pips', 0),
            management_actions_json,
            trade.status.value,
            trade.result.value if trade.result else None,
            original_signal_json,
            metadata_json
        ))
        
        logger.info(f"Trade saved: {trade.id}")
    
    def get_trade_by_id(self, trade_id: str) -> Optional[Trade]:
        """Get a specific trade by ID"""
        query = "SELECT * FROM trades WHERE id = ?"
        
        with self.get_connection() as conn:
            cursor = conn.execute(query, (trade_id,))
            row = cursor.fetchone()
            if row:
                return self._row_to_trade(dict(row))
            return None
    
    def get_open_trades(self) -> List[Trade]:
        """Get all currently open trades"""
        query = "SELECT * FROM trades WHERE status = ? ORDER BY entry_time DESC"
        
        with self.get_connection() as conn:
            cursor = conn.execute(query, (TradeStatus.OPEN.value,))
            trades = []
            for row in cursor:
                trade = self._row_to_trade(dict(row))
                if trade:
                    trades.append(trade)
            return trades
    
    def get_trades_by_date_range(
        self, 
        start_date: datetime, 
        end_date: datetime,
        symbol: Optional[str] = None
    ) -> List[Trade]:
        """Get trades within a date range"""
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
        
        with self.get_connection() as conn:
            cursor = conn.execute(query, params)
            trades = []
            for row in cursor:
                trade = self._row_to_trade(dict(row))
                if trade:
                    trades.append(trade)
            return trades
    
    def get_recent_trades(self, limit: int = 10) -> List[Trade]:
        """Get most recent trades"""
        query = "SELECT * FROM trades ORDER BY entry_time DESC LIMIT ?"
        
        with self.get_connection() as conn:
            cursor = conn.execute(query, (limit,))
            trades = []
            for row in cursor:
                trade = self._row_to_trade(dict(row))
                if trade:
                    trades.append(trade)
            return trades
    
    def get_trade_by_symbol(self, symbol: str) -> Optional[Trade]:
        """Get the most recent open trade for a symbol"""
        query = """
            SELECT * FROM trades 
            WHERE symbol = ? AND status = ? 
            ORDER BY entry_time DESC 
            LIMIT 1
        """
        
        with self.get_connection() as conn:
            cursor = conn.execute(query, (symbol, TradeStatus.OPEN.value))
            row = cursor.fetchone()
            if row:
                return self._row_to_trade(dict(row))
            return None
    
    def update_trade_status(
        self, 
        trade_id: str, 
        status: TradeStatus,
        exit_price: Optional[float] = None,
        exit_time: Optional[datetime] = None,
        result: Optional[TradeResult] = None
    ) -> bool:
        """Update trade status and exit information"""
        query = """
            UPDATE trades 
            SET status = ?, exit_price = ?, exit_time = ?, result = ?
            WHERE id = ?
        """
        
        try:
            with self.get_connection() as conn:
                cursor = conn.execute(query, (
                    status.value,
                    exit_price,
                    exit_time.isoformat() if exit_time else None,
                    result.value if result else None,
                    trade_id
                ))
                
                if cursor.rowcount > 0:
                    logger.info(f"Trade {trade_id} status updated to {status.value}")
                    return True
                else:
                    logger.warning(f"Trade {trade_id} not found for status update")
                    return False
                    
        except Exception as e:
            logger.error(f"Failed to update trade status: {e}")
            raise RepositoryError(f"Failed to update trade status: {str(e)}")
    
    def get_trade_statistics(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Get trading statistics for a date range"""
        query = """
            SELECT 
                COUNT(*) as total_trades,
                SUM(CASE WHEN result = 'WIN' THEN 1 ELSE 0 END) as winning_trades,
                SUM(CASE WHEN result = 'LOSS' THEN 1 ELSE 0 END) as losing_trades,
                SUM(profit_loss) as total_pnl,
                AVG(profit_loss) as avg_pnl,
                MAX(profit_loss) as max_profit,
                MIN(profit_loss) as max_loss,
                AVG(duration_minutes) as avg_duration_minutes
            FROM trades
            WHERE entry_time >= ? AND entry_time <= ?
            AND status = 'CLOSED'
        """
        
        with self.get_connection() as conn:
            cursor = conn.execute(query, (start_date.isoformat(), end_date.isoformat()))
            row = cursor.fetchone()
            
            if row:
                stats = dict(row)
                # Calculate win rate
                total = stats['total_trades']
                if total > 0:
                    stats['win_rate'] = stats['winning_trades'] / total
                else:
                    stats['win_rate'] = 0.0
                
                return stats
            
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'total_pnl': 0.0,
                'avg_pnl': 0.0,
                'max_profit': 0.0,
                'max_loss': 0.0,
                'avg_duration_minutes': 0.0,
                'win_rate': 0.0
            }
    
    def _row_to_trade(self, row: Dict[str, Any]) -> Optional[Trade]:
        """Convert database row to Trade object"""
        try:
            # Parse JSON fields
            management_history = []
            if row.get('management_actions'):
                management_history = json.loads(row['management_actions'])
            
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
            
            # Parse metadata
            metadata = {}
            if row.get('metadata_json'):
                metadata = json.loads(row['metadata_json'])
            
            trade = Trade(
                id=row['id'],
                symbol=row['symbol'],
                side=SignalType[row['direction']],
                entry_price=row['entry_price'],
                stop_loss=row['stop_loss'],
                take_profit=row['take_profit'],
                status=TradeStatus[row['status']],
                timestamp=datetime.fromisoformat(row['entry_time']),
                ticket=metadata.get('ticket'),
                lot_size=row.get('volume'),
                max_drawdown_pips=row.get('max_drawdown_pips', 0),
                current_pnl=row.get('profit_loss', 0),
                exit_price=row.get('exit_price'),
                exit_timestamp=datetime.fromisoformat(row['exit_time']) if row.get('exit_time') else None,
                result=TradeResult[row['result']] if row.get('result') else None,
                risk_reward_ratio=metadata.get('risk_reward_ratio'),
                risk_amount_usd=metadata.get('risk_amount_usd'),
                original_signal=original_signal,
                management_history=management_history,
                reflection=row.get('reflection')
            )
            
            return trade
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in trade data: {e}")
            raise SerializationError(f"Failed to parse JSON in trade record: {str(e)}") from e
        except KeyError as e:
            logger.error(f"Missing required field in trade data: {e}")
            raise RepositoryError(f"Missing required field in trade record: {str(e)}") from e
        except ValueError as e:
            logger.error(f"Invalid value in trade data: {e}")
            raise RepositoryError(f"Invalid value in trade record: {str(e)}") from e
        except Exception as e:
            logger.error(f"Unexpected error converting trade data: {type(e).__name__}: {e}")
            raise RepositoryError(f"Failed to convert database row to Trade: {str(e)}") from e


class PooledSignalRepository(PooledBaseRepository):
    """Signal repository using connection pooling"""
    
    def save_signal_in_transaction(self, conn: sqlite3.Connection, signal: TradingSignal) -> None:
        """Save a signal within an existing transaction"""
        signal_id = f"{signal.symbol}_{signal.timestamp.strftime('%Y%m%d_%H%M%S')}"
        
        # Serialize complex fields
        market_context_json = json.dumps(signal.market_context) if signal.market_context else None
        news_events_json = json.dumps([
            {
                'timestamp': event.timestamp.isoformat(),
                'country': event.country,
                'title': event.title,
                'impact': event.impact
            }
            for event in signal.news_events
        ]) if signal.news_events else None
        
        conn.execute("""
            INSERT OR REPLACE INTO signals (
                id, symbol, signal, entry_price, stop_loss, take_profit,
                risk_reward, risk_class, reason, timestamp,
                market_context_json, news_events_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            signal_id, signal.symbol, signal.signal.value,
            signal.entry, signal.stop_loss, signal.take_profit,
            signal.risk_reward, signal.risk_class.value, signal.reason,
            signal.timestamp.isoformat(),
            market_context_json, news_events_json
        ))
    
    def save_signal(self, signal: TradingSignal) -> str:
        """Save a signal and return its ID"""
        signal_id = f"{signal.symbol}_{signal.timestamp.strftime('%Y%m%d_%H%M%S')}"
        
        with self.get_connection() as conn:
            self.save_signal_in_transaction(conn, signal)
        
        logger.info(f"Signal saved: {signal_id}")
        return signal_id
    
    def get_signals_by_date_range(
        self, 
        start_date: datetime, 
        end_date: datetime,
        symbol: Optional[str] = None
    ) -> List[TradingSignal]:
        """Get signals within a date range"""
        if symbol:
            query = """
                SELECT * FROM signals 
                WHERE timestamp >= ? AND timestamp <= ? AND symbol = ?
                ORDER BY timestamp DESC
            """
            params = (start_date.isoformat(), end_date.isoformat(), symbol)
        else:
            query = """
                SELECT * FROM signals 
                WHERE timestamp >= ? AND timestamp <= ?
                ORDER BY timestamp DESC
            """
            params = (start_date.isoformat(), end_date.isoformat())
        
        with self.get_connection() as conn:
            cursor = conn.execute(query, params)
            signals = []
            for row in cursor:
                signal = self._row_to_signal(dict(row))
                if signal:
                    signals.append(signal)
            return signals
    
    def get_recent_signals(self, limit: int = 10) -> List[TradingSignal]:
        """Get most recent signals"""
        query = "SELECT * FROM signals ORDER BY timestamp DESC LIMIT ?"
        
        with self.get_connection() as conn:
            cursor = conn.execute(query, (limit,))
            signals = []
            for row in cursor:
                signal = self._row_to_signal(dict(row))
                if signal:
                    signals.append(signal)
            return signals
    
    def get_all_signals(self, hours_back: int = 24) -> List[Dict[str, Any]]:
        """Get all signals within the specified hours"""
        cutoff_time = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=hours_back)
        
        query = """
            SELECT id, symbol, signal, entry_price, stop_loss, take_profit,
                   risk_reward, risk_class, reason, timestamp,
                   market_context_json, news_events_json
            FROM signals 
            WHERE timestamp > ?
            ORDER BY timestamp DESC
        """
        
        with self.get_connection() as conn:
            cursor = conn.execute(query, (cutoff_time.isoformat(),))
            signals = []
            
            for row in cursor:
                signal_dict = dict(row)
                
                # Parse JSON fields for display
                if signal_dict.get('market_context_json'):
                    try:
                        signal_dict['market_context'] = json.loads(signal_dict['market_context_json'])
                    except:
                        signal_dict['market_context'] = {}
                
                if signal_dict.get('news_events_json'):
                    try:
                        signal_dict['news_events'] = json.loads(signal_dict['news_events_json'])
                    except:
                        signal_dict['news_events'] = []
                
                # Remove raw JSON fields
                signal_dict.pop('market_context_json', None)
                signal_dict.pop('news_events_json', None)
                
                signals.append(signal_dict)
            
            return signals
    
    def _row_to_signal(self, row: Dict[str, Any]) -> Optional[TradingSignal]:
        """Convert database row to TradingSignal object"""
        try:
            # Parse market context
            market_context = {}
            if row.get('market_context_json'):
                market_context = json.loads(row['market_context_json'])
            
            # Parse news events
            news_events = []
            if row.get('news_events_json'):
                news_data = json.loads(row['news_events_json'])
                # Convert to NewsEvent objects if needed
                # For now, keeping as dict
            
            signal = TradingSignal(
                symbol=row['symbol'],
                signal=SignalType[row['signal']],
                reason=row['reason'],
                risk_class=RiskClass[row['risk_class']],
                timestamp=datetime.fromisoformat(row['timestamp']),
                entry=row.get('entry_price'),
                stop_loss=row.get('stop_loss'),
                take_profit=row.get('take_profit'),
                risk_reward=row.get('risk_reward'),
                market_context=market_context,
                news_events=news_events
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"Error converting row to Signal: {e}")
            return None


# Export pooled repositories
__all__ = ['PooledTradeRepository', 'PooledSignalRepository']