"""
Database migrations for the GPT Trading System.
Manages database schema changes and version control.
"""

import sqlite3
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from pathlib import Path

from core.domain.exceptions import DatabaseError, ErrorContext


logger = logging.getLogger(__name__)


class Migration:
    """Base class for database migrations"""
    
    def __init__(self, version: int, description: str):
        self.version = version
        self.description = description
        self.timestamp = datetime.now(timezone.utc)
    
    def up(self, conn: sqlite3.Connection):
        """Apply the migration"""
        raise NotImplementedError("Subclasses must implement up() method")
    
    def down(self, conn: sqlite3.Connection):
        """Rollback the migration (optional)"""
        logger.warning(f"No rollback implemented for migration {self.version}")
    
    def __str__(self):
        return f"Migration {self.version}: {self.description}"


class InitialMigration(Migration):
    """Initial database schema migration"""
    
    def __init__(self):
        super().__init__(1, "Create initial database schema")
    
    def up(self, conn: sqlite3.Connection):
        """Create initial tables"""
        # Create migrations table first
        conn.execute("""
            CREATE TABLE IF NOT EXISTS migrations (
                version INTEGER PRIMARY KEY,
                description TEXT NOT NULL,
                applied_at TEXT NOT NULL
            )
        """)
        
        # Create trades table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                status TEXT NOT NULL,
                entry_price REAL NOT NULL,
                stop_loss REAL NOT NULL,
                take_profit REAL NOT NULL,
                ticket INTEGER,
                lot_size REAL,
                timestamp TEXT NOT NULL,
                exit_price REAL,
                exit_timestamp TEXT,
                result TEXT,
                max_drawdown_pips REAL DEFAULT 0,
                current_pnl REAL DEFAULT 0,
                risk_reward_ratio REAL,
                risk_amount_usd REAL,
                original_signal_json TEXT,
                management_history_json TEXT,
                reflection TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create signals table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                signal TEXT NOT NULL,
                entry_price REAL,
                stop_loss REAL,
                take_profit REAL,
                risk_reward REAL,
                risk_class TEXT NOT NULL,
                reason TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                market_context_json TEXT,
                news_events_json TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create memory cases table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS memory_cases (
                id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                context TEXT NOT NULL,
                entry_price REAL NOT NULL,
                signal TEXT NOT NULL,
                risk_reward REAL NOT NULL,
                result TEXT NOT NULL,
                reason TEXT NOT NULL,
                embedding BLOB,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indexes
        self._create_indexes(conn)
        
        conn.commit()
        logger.info("Initial database schema created")
    
    def _create_indexes(self, conn: sqlite3.Connection):
        """Create database indexes for better performance"""
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)",
            "CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status)",
            "CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_signals_symbol ON signals(symbol)",
            "CREATE INDEX IF NOT EXISTS idx_signals_timestamp ON signals(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_cases_symbol ON memory_cases(symbol)",
            "CREATE INDEX IF NOT EXISTS idx_cases_timestamp ON memory_cases(timestamp)",
        ]
        
        for index_sql in indexes:
            conn.execute(index_sql)


class AddTradeMetadataMigration(Migration):
    """Add additional metadata columns to trades table"""
    
    def __init__(self):
        super().__init__(2, "Add trade metadata columns")
    
    def up(self, conn: sqlite3.Connection):
        """Add new columns for enhanced trade tracking"""
        try:
            # Add new columns (SQLite doesn't support adding multiple columns at once)
            new_columns = [
                ("trade_duration_minutes", "REAL"),
                ("entry_reason", "TEXT"),
                ("exit_reason", "TEXT"),
                ("slippage_pips", "REAL"),
                ("commission_paid", "REAL"),
                ("swap_paid", "REAL")
            ]
            
            for column_name, column_type in new_columns:
                try:
                    conn.execute(f"ALTER TABLE trades ADD COLUMN {column_name} {column_type}")
                except sqlite3.OperationalError as e:
                    if "duplicate column name" not in str(e):
                        raise
                    # Column already exists, skip
                    
            conn.commit()
            logger.info("Trade metadata columns added")
            
        except Exception as e:
            logger.error(f"Failed to add trade metadata columns: {e}")
            raise


class AddPerformanceMetricsMigration(Migration):
    """Add performance tracking tables"""
    
    def __init__(self):
        super().__init__(3, "Add performance metrics tables")
    
    def up(self, conn: sqlite3.Connection):
        """Create performance tracking tables"""
        # Daily performance summary
        conn.execute("""
            CREATE TABLE IF NOT EXISTS daily_performance (
                date TEXT PRIMARY KEY,
                total_trades INTEGER DEFAULT 0,
                winning_trades INTEGER DEFAULT 0,
                losing_trades INTEGER DEFAULT 0,
                total_pnl REAL DEFAULT 0,
                max_drawdown REAL DEFAULT 0,
                win_rate REAL DEFAULT 0,
                avg_win REAL DEFAULT 0,
                avg_loss REAL DEFAULT 0,
                profit_factor REAL DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Symbol performance
        conn.execute("""
            CREATE TABLE IF NOT EXISTS symbol_performance (
                symbol TEXT PRIMARY KEY,
                total_trades INTEGER DEFAULT 0,
                winning_trades INTEGER DEFAULT 0,
                losing_trades INTEGER DEFAULT 0,
                total_pnl REAL DEFAULT 0,
                win_rate REAL DEFAULT 0,
                avg_rr REAL DEFAULT 0,
                last_updated TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        logger.info("Performance metrics tables created")


class AddNewsEventsMigration(Migration):
    """Add news events tracking"""
    
    def __init__(self):
        super().__init__(4, "Add news events tracking")
    
    def up(self, conn: sqlite3.Connection):
        """Create news events table"""
        conn.execute("""
            CREATE TABLE IF NOT EXISTS news_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                country TEXT NOT NULL,
                title TEXT NOT NULL,
                impact TEXT NOT NULL,
                actual TEXT,
                forecast TEXT,
                previous TEXT,
                affected_symbols TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Index for fast lookups
        conn.execute("CREATE INDEX IF NOT EXISTS idx_news_timestamp ON news_events(timestamp)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_news_impact ON news_events(impact)")
        
        conn.commit()
        logger.info("News events table created")

class AddBacktestResultsMigration(Migration):
    """Add backtest results storage for ML"""
    
    def __init__(self):
        super().__init__(5, "Add backtest results tables")
    
    def up(self, conn: sqlite3.Connection):
        """Create backtest results tables"""
        
        # Main backtest runs table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS backtest_runs (
                id TEXT PRIMARY KEY,
                start_date TEXT NOT NULL,
                end_date TEXT NOT NULL,
                symbols TEXT NOT NULL,
                mode TEXT NOT NULL,
                initial_balance REAL NOT NULL,
                risk_per_trade REAL NOT NULL,
                final_balance REAL NOT NULL,
                total_return REAL NOT NULL,
                win_rate REAL NOT NULL,
                sharpe_ratio REAL NOT NULL,
                max_drawdown REAL NOT NULL,
                profit_factor REAL NOT NULL,
                total_trades INTEGER NOT NULL,
                config_json TEXT NOT NULL,
                statistics_json TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Individual backtest trades table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS backtest_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                backtest_run_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                signal_type TEXT NOT NULL,
                entry_time TEXT NOT NULL,
                exit_time TEXT,
                entry_price REAL NOT NULL,
                exit_price REAL,
                stop_loss REAL NOT NULL,
                take_profit REAL NOT NULL,
                lot_size REAL NOT NULL,
                pnl REAL,
                pnl_points REAL,
                result TEXT,
                exit_reason TEXT,
                bars_held INTEGER,
                risk_reward_achieved REAL,
                signal_reason TEXT,
                risk_class TEXT,
                validation_score REAL,
                FOREIGN KEY (backtest_run_id) REFERENCES backtest_runs(id)
            )
        """)
        
        # Backtest equity curves table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS backtest_equity_curves (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                backtest_run_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                equity REAL NOT NULL,
                drawdown REAL NOT NULL,
                FOREIGN KEY (backtest_run_id) REFERENCES backtest_runs(id)
            )
        """)
        
        # Indexes for performance
        conn.execute("CREATE INDEX IF NOT EXISTS idx_backtest_runs_symbols ON backtest_runs(symbols)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_backtest_runs_created ON backtest_runs(created_at)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_backtest_trades_symbol ON backtest_trades(symbol)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_backtest_trades_result ON backtest_trades(result)")
        
        conn.commit()
        logger.info("Backtest results tables created")


class DatabaseMigrator:
    """Manages database migrations"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.migrations = self._get_migrations()
    
    def _get_migrations(self) -> List[Migration]:
        """Get all available migrations in order"""
        return [
            InitialMigration(),
            AddTradeMetadataMigration(),
            AddPerformanceMetricsMigration(),
            AddNewsEventsMigration(),
            AddBacktestResultsMigration()
        ]
    
    def get_current_version(self) -> int:
        """Get current database version"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name='migrations'
                """)
                
                if not cursor.fetchone():
                    return 0  # No migrations table, version 0
                
                cursor = conn.execute("SELECT MAX(version) FROM migrations")
                result = cursor.fetchone()
                return result[0] if result[0] is not None else 0
                
        except Exception as e:
            logger.error(f"Failed to get database version: {e}")
            return 0
    
    def get_target_version(self) -> int:
        """Get the latest migration version"""
        return max(m.version for m in self.migrations) if self.migrations else 0
    
    def migrate(self, target_version: Optional[int] = None) -> bool:
        """
        Run migrations up to target version.
        
        Args:
            target_version: Version to migrate to (latest if None)
            
        Returns:
            True if migrations successful
        """
        if target_version is None:
            target_version = self.get_target_version()
        
        current_version = self.get_current_version()
        
        if current_version >= target_version:
            logger.info(f"Database already at version {current_version}")
            return True
        
        logger.info(f"Migrating database from version {current_version} to {target_version}")
        
        with ErrorContext("Database migration"):
            try:
                with sqlite3.connect(self.db_path) as conn:
                    # Run migrations in order
                    for migration in self.migrations:
                        if migration.version <= current_version:
                            continue
                        if migration.version > target_version:
                            break
                        
                        logger.info(f"Applying {migration}")
                        migration.up(conn)
                        
                        # Record migration
                        conn.execute("""
                            INSERT OR REPLACE INTO migrations (version, description, applied_at)
                            VALUES (?, ?, ?)
                        """, (
                            migration.version,
                            migration.description,
                            datetime.now(timezone.utc).isoformat()
                        ))
                        
                        conn.commit()
                        logger.info(f"Migration {migration.version} applied successfully")
                
                logger.info(f"Database migration completed successfully")
                return True
                
            except Exception as e:
                logger.error(f"Migration failed: {e}")
                raise DatabaseError(f"Migration failed: {str(e)}")
    
    def rollback(self, target_version: int) -> bool:
        """
        Rollback migrations to target version.
        
        Args:
            target_version: Version to rollback to
            
        Returns:
            True if rollback successful
        """
        current_version = self.get_current_version()
        
        if current_version <= target_version:
            logger.info(f"Database already at or below version {target_version}")
            return True
        
        logger.warning(f"Rolling back database from version {current_version} to {target_version}")
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Run rollbacks in reverse order
                for migration in reversed(self.migrations):
                    if migration.version <= target_version:
                        break
                    if migration.version > current_version:
                        continue
                    
                    logger.info(f"Rolling back {migration}")
                    migration.down(conn)
                    
                    # Remove migration record
                    conn.execute("DELETE FROM migrations WHERE version = ?", (migration.version,))
                    conn.commit()
                    logger.info(f"Migration {migration.version} rolled back")
            
            logger.info("Database rollback completed")
            return True
            
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            raise DatabaseError(f"Rollback failed: {str(e)}")
    
    def get_migration_history(self) -> List[Dict[str, Any]]:
        """Get migration history"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT version, description, applied_at 
                    FROM migrations 
                    ORDER BY version
                """)
                
                return [
                    {
                        'version': row[0],
                        'description': row[1],
                        'applied_at': row[2]
                    }
                    for row in cursor.fetchall()
                ]
                
        except Exception as e:
            logger.error(f"Failed to get migration history: {e}")
            return []
    
    def check_schema_integrity(self) -> bool:
        """Check database schema integrity"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Check that all expected tables exist
                cursor = conn.execute("""
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name NOT LIKE 'sqlite_%'
                """)
                
                existing_tables = set(row[0] for row in cursor.fetchall())
                expected_tables = {
                    'migrations', 'trades', 'signals', 'memory_cases',
                    'daily_performance', 'symbol_performance', 'news_events'
                }
                
                missing_tables = expected_tables - existing_tables
                if missing_tables:
                    logger.error(f"Missing tables: {missing_tables}")
                    return False
                
                logger.info("Database schema integrity check passed")
                return True
                
        except Exception as e:
            logger.error(f"Schema integrity check failed: {e}")
            return False


def migrate_database(db_path: str, target_version: Optional[int] = None) -> bool:
    """
    Convenience function to migrate database.
    
    Args:
        db_path: Path to database file
        target_version: Version to migrate to
        
    Returns:
        True if successful
    """
    migrator = DatabaseMigrator(db_path)
    return migrator.migrate(target_version)


def get_database_info(db_path: str) -> Dict[str, Any]:
    """
    Get database information.
    
    Args:
        db_path: Path to database file
        
    Returns:
        Dictionary with database info
    """
    migrator = DatabaseMigrator(db_path)
    
    return {
        'db_path': db_path,
        'exists': Path(db_path).exists(),
        'current_version': migrator.get_current_version(),
        'target_version': migrator.get_target_version(),
        'migration_history': migrator.get_migration_history(),
        'schema_valid': migrator.check_schema_integrity()
    }


# Export main classes and functions
__all__ = [
    'Migration',
    'DatabaseMigrator', 
    'migrate_database',
    'get_database_info'
]