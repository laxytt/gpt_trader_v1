import sqlite3
from datetime import datetime

DB_PATH = "data/trades.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            status TEXT NOT NULL,
            side TEXT,
            entry REAL,
            sl REAL,
            tp REAL,
            ticket INTEGER UNIQUE,
            timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_trade_state(trade_state: dict):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO trades (symbol, status, side, entry, sl, tp, ticket, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        trade_state["symbol"],
        trade_state["status"],
        trade_state.get("side"),
        trade_state.get("entry"),
        trade_state.get("sl"),
        trade_state.get("tp"),
        trade_state.get("ticket"),
        trade_state.get("timestamp")
    ))
    conn.commit()
    conn.close()

def load_trade_state(symbol: str) -> dict:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT symbol, status, side, entry, sl, tp, ticket, timestamp
        FROM trades WHERE symbol = ?
        ORDER BY id DESC LIMIT 1
    """, (symbol,))
    row = cursor.fetchone()
    conn.close()

    if row:
        keys = ["symbol", "status", "side", "entry", "sl", "tp", "ticket", "timestamp"]
        return dict(zip(keys, row))
    return {"symbol": symbol, "status": "idle", "timestamp": datetime.utcnow().isoformat()}

def save_gpt_signal_decision(signal: dict):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS gpt_signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT,
            timestamp TEXT,
            signal TEXT,
            entry REAL,
            sl REAL,
            tp REAL,
            rr REAL,
            risk_class TEXT,
            reason TEXT
        )
    """)
    cursor.execute("""
        INSERT INTO gpt_signals (symbol, timestamp, signal, entry, sl, tp, rr, risk_class, reason)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        signal["symbol"], datetime.utcnow().isoformat(),
        signal["signal"], signal["entry"], signal["sl"],
        signal["tp"], signal["rr"], signal["risk_class"], signal["reason"]
    ))
    conn.commit()
    conn.close()

def get_open_trades():
    """
    Zwraca wszystkie otwarte transakcje (status='open') jako listę dictów.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT symbol, status, side, entry, sl, tp, ticket, timestamp
        FROM trades WHERE status = 'open'
    """)
    rows = cursor.fetchall()
    conn.close()

    keys = ["symbol", "status", "side", "entry", "sl", "tp", "ticket", "timestamp"]
    return [dict(zip(keys, row)) for row in rows]

def get_trade_by_symbol(symbol: str):
    """
    Zwraca ostatnią otwartą transakcję dla danego symbolu.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT symbol, status, side, entry, sl, tp, ticket, timestamp
        FROM trades
        WHERE symbol = ? AND status = 'open'
        ORDER BY id DESC LIMIT 1
    """, (symbol,))
    row = cursor.fetchone()
    conn.close()

    if row:
        keys = ["symbol", "status", "side", "entry", "sl", "tp", "ticket", "timestamp"]
        return dict(zip(keys, row))
    return None

from core.rag_memory import TradeMemoryRAG
memory = TradeMemoryRAG()
