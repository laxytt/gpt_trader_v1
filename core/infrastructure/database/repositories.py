"""
Database repositories for data persistence and retrieval.
Implements the Repository pattern for clean data access abstraction.
"""

import sqlite3
import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
from contextlib import contextmanager
import json

from config.settings import DatabaseSettings
from core.domain.models import (
    Trade, TradingSignal, TradeCase, TradeStatus, SignalType, 
    RiskClass, TradeResult, create_trade_id, create_case_id
)
from core.domain.exceptions import (
    DatabaseError, RepositoryError, SerializationError, handle_database_errors,
    ErrorContext
)
from core.infrastructure.database.migrations import migrate_database


logger = logging.getLogger(__name__)


class BaseRepository(ABC):
    """Base repository class with common database operations"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._ensure_database_exists()
    
    def _ensure_database_exists(self):
        """Ensure database file and tables exist"""
        try:
            # Run migrations to ensure schema is up to date
            if not migrate_database(self.db_path):
                raise DatabaseError("Database migration failed")
            
            logger.debug(f"Database initialized: {self.db_path}")
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise DatabaseError(f"Failed to initialize database: {str(e)}")
    
    @abstractmethod
    def _create_tables(self, conn: sqlite3.Connection):
        """Create necessary tables for this repository"""
        pass
    
    @contextmanager
    def get_connection(self, commit=True):
        """Get database connection with automatic cleanup and transaction management
        
        Args:
            commit: Whether to auto-commit on success (default True)
        """
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row  # Enable dict-like access
            # Enable WAL mode for better concurrency
            conn.execute("PRAGMA journal_mode=WAL")
            # Set busy timeout to handle concurrent access
            conn.execute("PRAGMA busy_timeout=5000")
            
            yield conn
            
            if commit:
                conn.commit()
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database error: {e}")
            raise DatabaseError(f"Database operation failed: {str(e)}")
        finally:
            if conn:
                conn.close()
    
    @contextmanager
    def transaction(self):
        """Get a connection with explicit transaction control
        
        Usage:
            with repo.transaction() as conn:
                # Multiple operations
                conn.execute(...)
                conn.execute(...)
                # All committed together
        """
        with self.get_connection(commit=True) as conn:
            yield conn
    
    @contextmanager
    def atomic_operation(self):
        """
        Context manager for atomic database operations with automatic rollback
        
        Usage:
            with repo.atomic_operation() as conn:
                repo.save_in_transaction(conn, obj1)
                repo.save_in_transaction(conn, obj2)
                # Both saved atomically or rolled back
        """
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=5000")
            conn.execute("BEGIN EXCLUSIVE")  # Start exclusive transaction
            
            yield conn
            
            conn.commit()
            logger.debug("Atomic operation committed successfully")
        except Exception as e:
            if conn:
                conn.rollback()
                logger.error(f"Atomic operation rolled back: {e}")
            raise DatabaseError(f"Atomic operation failed: {str(e)}")
        finally:
            if conn:
                conn.close()


class TradeRepository(BaseRepository):
    """Repository for trade data persistence"""
    
    def _create_tables(self, conn: sqlite3.Connection):
        """Tables are created by migrations"""
        pass  # Tables created by migration system
    
    def save_in_transaction(self, conn: sqlite3.Connection, trade: Trade) -> None:
        """
        Save a trade within an existing transaction.
        
        Args:
            conn: Active database connection with transaction
            trade: Trade object to save
        """
        # Serialize complex fields
        original_signal_json = json.dumps(
            trade.original_signal.to_dict() if trade.original_signal else None
        )
        management_history_json = json.dumps(trade.management_history)
        
        conn.execute("""
            INSERT OR REPLACE INTO trades (
                id, symbol, side, status, entry_price, stop_loss, take_profit,
                ticket, lot_size, timestamp, exit_price, exit_timestamp, result,
                max_drawdown_pips, current_pnl, risk_reward_ratio, risk_amount_usd,
                original_signal_json, management_history_json, reflection, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            trade.id, trade.symbol, trade.side.value, trade.status.value,
            trade.entry_price, trade.stop_loss, trade.take_profit,
            trade.ticket, trade.lot_size, trade.timestamp.isoformat(),
            trade.exit_price, 
            trade.exit_timestamp.isoformat() if trade.exit_timestamp else None,
            trade.result.value if trade.result else None,
            trade.max_drawdown_pips, trade.current_pnl, trade.risk_reward_ratio,
            trade.risk_amount_usd, original_signal_json, management_history_json,
            trade.reflection, datetime.now(timezone.utc).isoformat()
        ))
        
        logger.debug(f"Trade saved in transaction: {trade.id}")
    
    @handle_database_errors
    def save(self, trade: Trade) -> None:
        """
        Save or update a trade.
        
        Args:
            trade: Trade object to save
        """
        with ErrorContext("Save trade", symbol=trade.symbol) as ctx:
            ctx.add_detail("trade_id", trade.id)
            
            with self.get_connection() as conn:
                # Serialize complex fields
                original_signal_json = json.dumps(
                    trade.original_signal.to_dict() if trade.original_signal else None
                )
                management_history_json = json.dumps(trade.management_history)
                
                conn.execute("""
                    INSERT OR REPLACE INTO trades (
                        id, symbol, side, status, entry_price, stop_loss, take_profit,
                        ticket, lot_size, timestamp, exit_price, exit_timestamp, result,
                        max_drawdown_pips, current_pnl, risk_reward_ratio, risk_amount_usd,
                        original_signal_json, management_history_json, reflection, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    trade.id, trade.symbol, trade.side.value, trade.status.value,
                    trade.entry_price, trade.stop_loss, trade.take_profit,
                    trade.ticket, trade.lot_size, trade.timestamp.isoformat(),
                    trade.exit_price, 
                    trade.exit_timestamp.isoformat() if trade.exit_timestamp else None,
                    trade.result.value if trade.result else None,
                    trade.max_drawdown_pips, trade.current_pnl, trade.risk_reward_ratio,
                    trade.risk_amount_usd, original_signal_json, management_history_json,
                    trade.reflection, datetime.now(timezone.utc).isoformat()
                ))
                conn.commit()
                
                logger.debug(f"Trade saved: {trade.id}")
    
    @handle_database_errors
    def find_by_id(self, trade_id: str) -> Optional[Trade]:
        """
        Find trade by ID.
        
        Args:
            trade_id: Trade ID to search for
            
        Returns:
            Trade object or None if not found
        """
        with self.get_connection() as conn:
            cursor = conn.execute("SELECT * FROM trades WHERE id = ?", (trade_id,))
            row = cursor.fetchone()
            
            if row:
                return self._row_to_trade(row)
            return None
    
    @handle_database_errors
    def find_by_symbol(self, symbol: str, status: Optional[TradeStatus] = None) -> List[Trade]:
        """
        Find trades by symbol and optionally status.
        
        Args:
            symbol: Trading symbol
            status: Optional status filter
            
        Returns:
            List of Trade objects
        """
        with self.get_connection() as conn:
            if status:
                cursor = conn.execute(
                    "SELECT * FROM trades WHERE symbol = ? AND status = ? ORDER BY timestamp DESC",
                    (symbol, status.value)
                )
            else:
                cursor = conn.execute(
                    "SELECT * FROM trades WHERE symbol = ? ORDER BY timestamp DESC",
                    (symbol,)
                )
            
            return [self._row_to_trade(row) for row in cursor.fetchall()]
    
    @handle_database_errors
    def find_open_trades(self) -> List[Trade]:
        """
        Find all open trades.
        
        Returns:
            List of open Trade objects
        """
        with self.get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM trades WHERE status = ? ORDER BY timestamp DESC",
                (TradeStatus.OPEN.value,)
            )
            
            return [self._row_to_trade(row) for row in cursor.fetchall()]
    
    @handle_database_errors
    def find_recent_trades(self, symbol: str, limit: int = 10) -> List[Trade]:
        """
        Find recent trades for a symbol.
        
        Args:
            symbol: Trading symbol
            limit: Maximum number of trades to return
            
        Returns:
            List of recent Trade objects
        """
        with self.get_connection() as conn:
            cursor = conn.execute("""
                SELECT * FROM trades 
                WHERE symbol = ? 
                ORDER BY timestamp DESC 
                LIMIT ?
            """, (symbol, limit))
            
            return [self._row_to_trade(row) for row in cursor.fetchall()]
    
    @handle_database_errors
    def update_status(self, trade_id: str, status: TradeStatus) -> bool:
        """
        Update trade status.
        
        Args:
            trade_id: Trade ID to update
            status: New status
            
        Returns:
            True if update successful
        """
        with self.get_connection() as conn:
            cursor = conn.execute(
                "UPDATE trades SET status = ?, updated_at = ? WHERE id = ?",
                (status.value, datetime.now(timezone.utc).isoformat(), trade_id)
            )
            conn.commit()
            
            return cursor.rowcount > 0
    
    @handle_database_errors
    def get_trades_by_date_range(
        self, 
        start_date: datetime, 
        end_date: datetime,
        symbol: Optional[str] = None
    ) -> List[Trade]:
        """
        Get trades within a date range.
        
        Args:
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            symbol: Optional symbol filter
            
        Returns:
            List of Trade objects
        """
        with self.get_connection() as conn:
            if symbol:
                cursor = conn.execute("""
                    SELECT * FROM trades 
                    WHERE timestamp >= ? AND timestamp <= ? AND symbol = ?
                    ORDER BY timestamp DESC
                """, (start_date.isoformat(), end_date.isoformat(), symbol))
            else:
                cursor = conn.execute("""
                    SELECT * FROM trades 
                    WHERE timestamp >= ? AND timestamp <= ?
                    ORDER BY timestamp DESC
                """, (start_date.isoformat(), end_date.isoformat()))
            
            return [self._row_to_trade(row) for row in cursor.fetchall()]
    
    def _row_to_trade(self, row: sqlite3.Row) -> Trade:
        """Convert database row to Trade object"""
        try:
            # Parse JSON fields
            original_signal = None
            if row['original_signal_json']:
                signal_data = json.loads(row['original_signal_json'])
                # Would need to reconstruct TradingSignal from dict
                # For now, store as dict in trade context
            
            management_history = json.loads(row['management_history_json']) if row['management_history_json'] else []
            
            trade = Trade(
                id=row['id'],
                symbol=row['symbol'],
                side=SignalType(row['side']),
                entry_price=row['entry_price'],
                stop_loss=row['stop_loss'],
                take_profit=row['take_profit'],
                status=TradeStatus(row['status']),
                timestamp=datetime.fromisoformat(row['timestamp']),
                ticket=row['ticket'],
                lot_size=row['lot_size'],
                exit_price=row['exit_price'],
                exit_timestamp=datetime.fromisoformat(row['exit_timestamp']) if row['exit_timestamp'] else None,
                result=TradeResult(row['result']) if row['result'] else None,
                max_drawdown_pips=row['max_drawdown_pips'] or 0,
                current_pnl=row['current_pnl'] or 0,
                risk_reward_ratio=row['risk_reward_ratio'],
                risk_amount_usd=row['risk_amount_usd'],
                original_signal=original_signal,
                management_history=management_history,
                reflection=row['reflection']
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


class SignalRepository(BaseRepository):
    """Repository for trading signal history"""
    
    def _create_tables(self, conn: sqlite3.Connection):
        """Tables are created by migrations"""
        pass  # Tables created by migration system
    
    def save_signal_in_transaction(self, conn: sqlite3.Connection, signal: TradingSignal) -> None:
        """
        Save a signal within an existing transaction.
        
        Args:
            conn: Active database connection with transaction
            signal: TradingSignal to save
        """
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
            signal.timestamp.isoformat(), market_context_json, news_events_json
        ))
    
    @handle_database_errors
    def save_signal(self, signal: TradingSignal) -> None:
        """
        Save a trading signal.
        
        Args:
            signal: TradingSignal to save
        """
        signal_id = f"{signal.symbol}_{signal.timestamp.strftime('%Y%m%d_%H%M%S')}"
        
        with self.get_connection() as conn:
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
                signal.timestamp.isoformat(), market_context_json, news_events_json
            ))
            conn.commit()
    
    @handle_database_errors
    def get_recent_signals(self, symbol: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent signals for a symbol.
        
        Args:
            symbol: Trading symbol
            limit: Maximum number of signals
            
        Returns:
            List of signal dictionaries
        """
        with self.get_connection() as conn:
            cursor = conn.execute("""
                SELECT * FROM signals 
                WHERE symbol = ? 
                ORDER BY timestamp DESC 
                LIMIT ?
            """, (symbol, limit))
            
            return [dict(row) for row in cursor.fetchall()]
    
    @handle_database_errors
    def get_signals_by_date_range(
        self,
        start_date: datetime,
        end_date: datetime,
        symbol: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get signals within a date range.
        
        Args:
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            symbol: Optional symbol filter
            
        Returns:
            List of signal dictionaries
        """
        with self.get_connection() as conn:
            if symbol:
                cursor = conn.execute("""
                    SELECT * FROM signals 
                    WHERE timestamp >= ? AND timestamp <= ? AND symbol = ?
                    ORDER BY timestamp DESC
                """, (start_date.isoformat(), end_date.isoformat(), symbol))
            else:
                cursor = conn.execute("""
                    SELECT * FROM signals 
                    WHERE timestamp >= ? AND timestamp <= ?
                    ORDER BY timestamp DESC
                """, (start_date.isoformat(), end_date.isoformat()))
            
            return [dict(row) for row in cursor.fetchall()]
    
    @handle_database_errors
    def get_all_signals(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get all signals for dashboard display.
        
        Args:
            limit: Maximum number of signals to return
            
        Returns:
            List of signal dictionaries
        """
        with self.get_connection() as conn:
            cursor = conn.execute("""
                SELECT 
                    id,
                    symbol,
                    signal as direction,
                    entry_price,
                    stop_loss,
                    take_profit,
                    risk_reward,
                    risk_class,
                    reason,
                    timestamp as created_at,
                    market_context_json,
                    news_events_json
                FROM signals
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,))
            
            signals = []
            for row in cursor.fetchall():
                signal = dict(row)
                
                # Add confidence field (not stored directly, calculate from risk_class)
                risk_class = signal.get('risk_class', 'MEDIUM')
                if risk_class == 'HIGH':
                    signal['confidence'] = 0.8
                elif risk_class == 'MEDIUM':
                    signal['confidence'] = 0.6
                else:
                    signal['confidence'] = 0.4
                
                # Add ML confidence (placeholder, as it's not stored)
                signal['ml_confidence'] = signal['confidence']
                
                # Parse market context if it's JSON
                if signal.get('market_context_json'):
                    try:
                        import json
                        signal['analysis'] = json.loads(signal['market_context_json'])
                    except:
                        signal['analysis'] = {}
                else:
                    signal['analysis'] = {}
                
                # Clean up fields not needed by dashboard
                signal.pop('market_context_json', None)
                signal.pop('news_events_json', None)
                
                signals.append(signal)
            
            return signals


class MemoryCaseRepository(BaseRepository):
    """Repository for RAG memory trade cases"""
    
    def _create_tables(self, conn: sqlite3.Connection):
        """Tables are created by migrations"""
        pass  # Tables created by migration system
    
    def save_case_in_transaction(self, conn: sqlite3.Connection, case: TradeCase, embedding: bytes) -> None:
        """
        Save a case within an existing transaction.
        
        Args:
            conn: Active database connection with transaction
            case: TradeCase to save
            embedding: Serialized embedding vector
        """
        conn.execute("""
            INSERT OR REPLACE INTO memory_cases (
                id, symbol, timestamp, context, entry_price, signal,
                risk_reward, result, reason, embedding
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            case.id, case.symbol, case.timestamp.isoformat(),
            case.context, case.entry_price, case.signal.value,
            case.risk_reward, case.result.value, case.reason, embedding
        ))
    
    @handle_database_errors
    def save_case(self, case: TradeCase, embedding: bytes) -> None:
        """
        Save a trade case with embedding.
        
        Args:
            case: TradeCase to save
            embedding: Serialized embedding vector
        """
        with self.get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO memory_cases (
                    id, symbol, timestamp, context, entry_price, signal,
                    risk_reward, result, reason, embedding
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                case.id, case.symbol, case.timestamp.isoformat(),
                case.context, case.entry_price, case.signal.value,
                case.risk_reward, case.result.value, case.reason, embedding
            ))
            conn.commit()
    
    @handle_database_errors
    def get_cases_for_symbol(self, symbol: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get cases for a symbol with embeddings.
        
        Args:
            symbol: Trading symbol
            limit: Maximum number of cases
            
        Returns:
            List of case dictionaries with embeddings
        """
        with self.get_connection() as conn:
            cursor = conn.execute("""
                SELECT * FROM memory_cases 
                WHERE symbol = ? 
                ORDER BY timestamp DESC 
                LIMIT ?
            """, (symbol, limit))
            
            return [dict(row) for row in cursor.fetchall()]
    
    @handle_database_errors
    def get_all_cases_with_embeddings(self) -> List[Dict[str, Any]]:
        """
        Get all cases with embeddings for similarity search.
        
        Returns:
            List of all cases with embeddings
        """
        with self.get_connection() as conn:
            cursor = conn.execute("""
                SELECT id, symbol, context, embedding 
                FROM memory_cases 
                ORDER BY timestamp DESC
            """)
            
            return [dict(row) for row in cursor.fetchall()]
    
    @handle_database_errors
    def cleanup_old_cases(self, max_cases: int = 300) -> int:
        """
        Remove oldest cases to maintain size limit.
        
        Args:
            max_cases: Maximum number of cases to keep
            
        Returns:
            Number of cases removed
        """
        with self.get_connection() as conn:
            # Count current cases
            cursor = conn.execute("SELECT COUNT(*) FROM memory_cases")
            current_count = cursor.fetchone()[0]
            
            if current_count <= max_cases:
                return 0
            
            # Remove oldest cases
            cases_to_remove = current_count - max_cases
            cursor = conn.execute("""
                DELETE FROM memory_cases 
                WHERE id IN (
                    SELECT id FROM memory_cases 
                    ORDER BY timestamp ASC 
                    LIMIT ?
                )
            """, (cases_to_remove,))
            
            conn.commit()
            return cursor.rowcount


# Export all repositories
__all__ = [
    'BaseRepository', 'TradeRepository', 'SignalRepository', 'MemoryCaseRepository'
]