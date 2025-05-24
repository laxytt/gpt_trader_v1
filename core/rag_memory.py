import sqlite3
import logging
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from datetime import datetime
from core.database import DB_PATH

logger = logging.getLogger(__name__)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384  # dla all-MiniLM-L6-v2

class TradeMemoryRAG:
    def __init__(self, max_cases: int = 300):
        self.max_cases = max_cases
        self.model = SentenceTransformer(MODEL_NAME)
        self.index = faiss.IndexFlatL2(EMBEDDING_DIM)
        self._init_db()
        self._load_cases()

    def _init_db(self):
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cases (
                id TEXT PRIMARY KEY,
                symbol TEXT,
                timestamp TEXT,
                context TEXT,
                entry REAL,
                signal TEXT,
                rr REAL,
                result TEXT,
                reason TEXT,
                embedding BLOB
            )
        """)
        conn.commit()
        conn.close()

    def _load_cases(self):
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT id, embedding FROM cases ORDER BY timestamp DESC LIMIT ?", (self.max_cases,))
        rows = cursor.fetchall()
        conn.close()

        embeddings = []
        self.case_ids = []

        for row in rows:
            case_id, embedding_blob = row
            embedding = np.frombuffer(embedding_blob, dtype=np.float32)
            embeddings.append(embedding)
            self.case_ids.append(case_id)

        if embeddings:
            embeddings_matrix = np.stack(embeddings)
            self.index.add(embeddings_matrix)
            logger.info(f"Loaded {len(self.case_ids)} embeddings into FAISS index.")

    def add_case(self, case: dict):
        required_keys = ["symbol", "context", "entry", "signal", "rr", "result", "reason"]
        if not all(k in case for k in required_keys):
            logger.warning("Case missing required keys, skipping save.")
            return

        case_id = case.get("id") or f"{case['symbol']}_{case.get('timestamp', datetime.utcnow().isoformat())}"
        embedding = self.model.encode(case["context"]).astype(np.float32)

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM cases")
        total_cases = cursor.fetchone()[0]

        if total_cases >= self.max_cases:
            cursor.execute("DELETE FROM cases WHERE id = (SELECT id FROM cases ORDER BY timestamp ASC LIMIT 1)")

        cursor.execute("""
            INSERT OR REPLACE INTO cases
            (id, symbol, timestamp, context, entry, signal, rr, result, reason, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            case_id,
            case["symbol"],
            case.get("timestamp", datetime.utcnow().isoformat()),
            case["context"],
            case["entry"],
            case["signal"],
            case["rr"],
            case["result"],
            case["reason"],
            embedding.tobytes()
        ))
        conn.commit()
        conn.close()

        self.index.add(np.array([embedding]))
        self.case_ids.append(case_id)
        logger.info(f"New case added: {case_id}")

    def query(self, context: str, k: int = 3, symbol: str = None, oversample: int = 10) -> list[dict]:
        if not self.case_ids:
            logger.warning("No cases available for querying.")
            return []

        query_embedding = self.model.encode(context).astype(np.float32)
        distances, indices = self.index.search(np.array([query_embedding]), k * oversample)

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        results = []
        for idx in indices[0]:
            if idx >= len(self.case_ids):
                continue
            case_id = self.case_ids[idx]
            cursor.execute("SELECT symbol, timestamp, context, entry, signal, rr, result, reason FROM cases WHERE id=?", (case_id,))
            row = cursor.fetchone()
            if row:
                fetched_case = {
                    "id": case_id,
                    "symbol": row[0],
                    "timestamp": row[1],
                    "context": row[2],
                    "entry": row[3],
                    "signal": row[4],
                    "rr": row[5],
                    "result": row[6],
                    "reason": row[7]
                }
                if symbol:
                    if fetched_case["symbol"].upper() == symbol.upper():
                        results.append(fetched_case)
                else:
                    results.append(fetched_case)
                if len(results) >= k:
                    break
        conn.close()
        return results

    def query_cases(self, symbol: str = "EURUSD", limit: int = 10) -> list[dict]:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, symbol, timestamp, context, entry, signal, rr, result, reason 
            FROM cases WHERE symbol=? 
            ORDER BY timestamp DESC LIMIT ?
        """, (symbol.upper(), limit))
        rows = cursor.fetchall()
        conn.close()

        return [
            {
                "id": row[0],
                "symbol": row[1],
                "timestamp": row[2],
                "context": row[3],
                "entry": row[4],
                "signal": row[5],
                "rr": row[6],
                "result": row[7],
                "reason": row[8]
            }
            for row in rows
        ]

    def get_previous_signal(self, symbol):
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT symbol, status, side, entry, sl, tp, ticket, timestamp FROM trades WHERE symbol = ? ORDER BY id DESC LIMIT 1",
            (symbol,))
        row = cursor.fetchone()
        conn.close()
        if row:
            keys = ["symbol", "status", "side", "entry", "sl", "tp", "ticket", "timestamp"]
            return dict(zip(keys, row))
        return None
