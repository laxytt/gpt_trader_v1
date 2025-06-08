"""
GPT Request Logger for Dashboard Visibility
Stores all GPT requests/responses for monitoring and debugging
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
import asyncio
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)


@dataclass
class GPTRequest:
    """GPT request data"""
    id: str
    timestamp: datetime
    model: str
    agent_type: Optional[str]
    symbol: Optional[str]
    request_type: str  # analyze, debate, synthesize
    messages: List[Dict[str, str]]
    temperature: float
    max_tokens: Optional[int]
    
    # Response data
    response_text: Optional[str] = None
    completion_tokens: Optional[int] = None
    prompt_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    cost: Optional[float] = None
    
    # Metadata
    duration_ms: Optional[int] = None
    error: Optional[str] = None
    retry_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['messages'] = json.dumps(self.messages)
        return data


class GPTRequestLogger:
    """
    Logs GPT requests and responses to SQLite for dashboard visibility
    """
    
    def __init__(self, db_path: Path = Path("data/gpt_requests.db")):
        """Initialize the logger"""
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
        self._locks = {}  # Will store locks per thread/loop
        
    def _init_database(self):
        """Initialize the database schema"""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS gpt_requests (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    model TEXT NOT NULL,
                    agent_type TEXT,
                    symbol TEXT,
                    request_type TEXT NOT NULL,
                    messages TEXT NOT NULL,
                    temperature REAL NOT NULL,
                    max_tokens INTEGER,
                    
                    -- Response
                    response_text TEXT,
                    completion_tokens INTEGER,
                    prompt_tokens INTEGER,
                    total_tokens INTEGER,
                    cost REAL,
                    
                    -- Metadata
                    duration_ms INTEGER,
                    error TEXT,
                    retry_count INTEGER DEFAULT 0,
                    
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp 
                ON gpt_requests(timestamp DESC)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_agent_type 
                ON gpt_requests(agent_type)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_symbol 
                ON gpt_requests(symbol)
            """)
            
            # Create summary view
            conn.execute("""
                CREATE VIEW IF NOT EXISTS gpt_request_summary AS
                SELECT 
                    DATE(timestamp) as date,
                    agent_type,
                    model,
                    COUNT(*) as request_count,
                    SUM(total_tokens) as total_tokens,
                    SUM(cost) as total_cost,
                    AVG(duration_ms) as avg_duration_ms,
                    SUM(CASE WHEN error IS NOT NULL THEN 1 ELSE 0 END) as error_count
                FROM gpt_requests
                GROUP BY DATE(timestamp), agent_type, model
            """)
            
    @contextmanager
    def _get_connection(self):
        """Get database connection"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()
            
    async def log_request(self, request: GPTRequest):
        """Log a GPT request"""
        # Get or create lock for current event loop/thread
        import threading
        thread_id = threading.get_ident()
        
        if thread_id not in self._locks:
            try:
                # Try to get current event loop
                loop = asyncio.get_event_loop()
                self._locks[thread_id] = asyncio.Lock()
            except RuntimeError:
                # No event loop, use threading lock
                self._locks[thread_id] = threading.Lock()
        
        lock = self._locks[thread_id]
        
        # Handle both async and sync locks
        if isinstance(lock, asyncio.Lock):
            async with lock:
                await self._do_log_request(request)
        else:
            with lock:
                await self._do_log_request(request)
                
    async def _do_log_request(self, request: GPTRequest):
        try:
            with self._get_connection() as conn:
                data = request.to_dict()
                
                conn.execute("""
                    INSERT INTO gpt_requests (
                        id, timestamp, model, agent_type, symbol, 
                        request_type, messages, temperature, max_tokens,
                        response_text, completion_tokens, prompt_tokens,
                        total_tokens, cost, duration_ms, error, retry_count
                    ) VALUES (
                        :id, :timestamp, :model, :agent_type, :symbol,
                        :request_type, :messages, :temperature, :max_tokens,
                        :response_text, :completion_tokens, :prompt_tokens,
                        :total_tokens, :cost, :duration_ms, :error, :retry_count
                    )
                """, data)
                
        except Exception as e:
            logger.error(f"Failed to log GPT request: {e}")
                
    def get_recent_requests(
        self, 
        limit: int = 100,
        hours_back: int = 24,
        agent_type: Optional[str] = None,
        symbol: Optional[str] = None,
        include_errors: bool = True
    ) -> List[Dict[str, Any]]:
        """Get recent requests for dashboard"""
        try:
            with self._get_connection() as conn:
                query = """
                    SELECT * FROM gpt_requests
                    WHERE timestamp > datetime('now', '-{} hours')
                """.format(hours_back)
                params = []
                
                if agent_type:
                    query += " AND agent_type = ?"
                    params.append(agent_type)
                    
                if symbol:
                    query += " AND symbol = ?"
                    params.append(symbol)
                    
                if not include_errors:
                    query += " AND error IS NULL"
                    
                query += " ORDER BY timestamp DESC LIMIT ?"
                params.append(limit)
                
                rows = conn.execute(query, params).fetchall()
                
                requests = []
                for row in rows:
                    req = dict(row)
                    req['messages'] = json.loads(req['messages'])
                    requests.append(req)
                    
                return requests
                
        except Exception as e:
            logger.error(f"Failed to get recent requests: {e}")
            return []
            
    def get_usage_stats(self, hours: int = 24) -> Dict[str, Any]:
        """Get usage statistics for the last N hours"""
        try:
            with self._get_connection() as conn:
                # Overall stats
                overall = conn.execute("""
                    SELECT 
                        COUNT(*) as total_requests,
                        SUM(total_tokens) as total_tokens,
                        SUM(cost) as total_cost,
                        AVG(duration_ms) as avg_duration_ms,
                        SUM(CASE WHEN error IS NOT NULL THEN 1 ELSE 0 END) as error_count
                    FROM gpt_requests
                    WHERE timestamp > datetime('now', ? || ' hours')
                """, (-hours,)).fetchone()
                
                # Per agent stats
                per_agent = conn.execute("""
                    SELECT 
                        agent_type,
                        COUNT(*) as request_count,
                        SUM(total_tokens) as tokens,
                        SUM(cost) as cost
                    FROM gpt_requests
                    WHERE timestamp > datetime('now', ? || ' hours')
                    GROUP BY agent_type
                """, (-hours,)).fetchall()
                
                # Per model stats
                per_model = conn.execute("""
                    SELECT 
                        model,
                        COUNT(*) as request_count,
                        SUM(total_tokens) as tokens,
                        SUM(cost) as cost
                    FROM gpt_requests
                    WHERE timestamp > datetime('now', ? || ' hours')
                    GROUP BY model
                """, (-hours,)).fetchall()
                
                return {
                    'overall': dict(overall) if overall else {},
                    'per_agent': [dict(row) for row in per_agent],
                    'per_model': [dict(row) for row in per_model],
                    'time_range_hours': hours
                }
                
        except Exception as e:
            logger.error(f"Failed to get usage stats: {e}")
            return {}
            
    def cleanup_old_requests(self, days: int = 7):
        """Remove requests older than N days"""
        try:
            with self._get_connection() as conn:
                deleted = conn.execute("""
                    DELETE FROM gpt_requests
                    WHERE timestamp < datetime('now', ? || ' days')
                """, (-days,))
                
                logger.info(f"Cleaned up {deleted.rowcount} old GPT requests")
                
        except Exception as e:
            logger.error(f"Failed to cleanup old requests: {e}")


# Global logger instance
_logger_instance: Optional[GPTRequestLogger] = None


def get_request_logger() -> GPTRequestLogger:
    """Get the global request logger instance"""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = GPTRequestLogger()
    return _logger_instance