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
