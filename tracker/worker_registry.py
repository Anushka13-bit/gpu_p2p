import sqlite3
import uuid
import time
from typing import Dict, Any, Tuple, List, Optional
import os

class WorkerRegistry:
    def __init__(self, db_path: str = "registry.db"):
        self.db_path = db_path
        self._init_db()

    def _get_conn(self):
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._get_conn() as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS worker_registry (
                    worker_id TEXT PRIMARY KEY,
                    name TEXT,
                    email TEXT UNIQUE,
                    worker_token TEXT UNIQUE,
                    status TEXT,
                    joined_at REAL,
                    shards_completed INTEGER DEFAULT 0,
                    failed_validations INTEGER DEFAULT 0,
                    consecutive_successes INTEGER DEFAULT 0,
                    ban_reason TEXT
                )
            ''')
            conn.execute('''
                CREATE TABLE IF NOT EXISTS ban_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    worker_id TEXT,
                    reason TEXT,
                    banned_at REAL,
                    banned_by TEXT
                )
            ''')
            conn.commit()

    def signup(self, name: str, email: str) -> Dict[str, Any]:
        worker_id = str(uuid.uuid4())
        worker_token = str(uuid.uuid4())
        status = "probation"
        joined_at = time.time()
        
        with self._get_conn() as conn:
            try:
                conn.execute(
                    "INSERT INTO worker_registry (worker_id, name, email, worker_token, status, joined_at) VALUES (?, ?, ?, ?, ?, ?)",
                    (worker_id, name, email, worker_token, status, joined_at)
                )
                conn.commit()
                return {
                    "worker_id": worker_id,
                    "worker_token": worker_token,
                    "status": status,
                    "message": "Welcome! You are on probation. Complete 3 valid shards to earn trusted status."
                }
            except sqlite3.IntegrityError:
                return {}

    def authenticate(self, token: str) -> Optional[Dict[str, Any]]:
        with self._get_conn() as conn:
            cur = conn.execute("SELECT * FROM worker_registry WHERE worker_token = ?", (token,))
            row = cur.fetchone()
            if row:
                return dict(row)
            return None

    def is_banned(self, worker_id: str) -> Tuple[bool, str]:
        with self._get_conn() as conn:
            cur = conn.execute("SELECT status, ban_reason FROM worker_registry WHERE worker_id = ?", (worker_id,))
            row = cur.fetchone()
            if row and row["status"] == "banned":
                return True, row["ban_reason"] or "Banned"
            return False, ""

    def record_success(self, worker_id: str) -> str:
        with self._get_conn() as conn:
            cur = conn.execute("SELECT status, consecutive_successes FROM worker_registry WHERE worker_id = ?", (worker_id,))
            row = cur.fetchone()
            if not row:
                return ""
            
            conn.execute("UPDATE worker_registry SET shards_completed = shards_completed + 1 WHERE worker_id = ?", (worker_id,))
            
            if row["status"] == "probation":
                new_streak = row["consecutive_successes"] + 1
                if new_streak >= 3:
                    conn.execute("UPDATE worker_registry SET status = 'trusted', consecutive_successes = 0 WHERE worker_id = ?", (worker_id,))
                    conn.commit()
                    return "graduated"
                else:
                    conn.execute("UPDATE worker_registry SET consecutive_successes = ? WHERE worker_id = ?", (new_streak, worker_id))
            conn.commit()
            return ""

    def record_failure(self, worker_id: str) -> str:
        with self._get_conn() as conn:
            cur = conn.execute("SELECT status, failed_validations FROM worker_registry WHERE worker_id = ?", (worker_id,))
            row = cur.fetchone()
            if not row:
                return ""
            
            failures = row["failed_validations"] + 1
            conn.execute("UPDATE worker_registry SET failed_validations = ?, consecutive_successes = 0 WHERE worker_id = ?", (failures, worker_id))
            
            status = row["status"]
            if (status == "probation" and failures >= 2) or (status == "trusted" and failures >= 5):
                self.ban_worker_conn(conn, worker_id, "Too many failed validations", "system")
                conn.commit()
                return "banned"
            
            conn.commit()
            return ""

    def ban_worker(self, worker_id: str, reason: str, banned_by: str = "system"):
        with self._get_conn() as conn:
            self.ban_worker_conn(conn, worker_id, reason, banned_by)
            conn.commit()

    def ban_worker_conn(self, conn, worker_id: str, reason: str, banned_by: str):
        conn.execute("UPDATE worker_registry SET status = 'banned', ban_reason = ? WHERE worker_id = ?", (reason, worker_id))
        conn.execute("INSERT INTO ban_log (worker_id, reason, banned_at, banned_by) VALUES (?, ?, ?, ?)", (worker_id, reason, time.time(), banned_by))

    def unban_worker(self, worker_id: str):
        with self._get_conn() as conn:
            conn.execute("UPDATE worker_registry SET status = 'probation', failed_validations = 0, consecutive_successes = 0 WHERE worker_id = ?", (worker_id,))
            conn.commit()

    def get_all_workers(self) -> List[Dict[str, Any]]:
        with self._get_conn() as conn:
            cur = conn.execute("SELECT * FROM worker_registry")
            return [dict(row) for row in cur.fetchall()]

    def get_trust_weight(self, worker_id: str) -> float:
        with self._get_conn() as conn:
            cur = conn.execute("SELECT status FROM worker_registry WHERE worker_id = ?", (worker_id,))
            row = cur.fetchone()
            if not row:
                return 0.0
            if row["status"] == "trusted":
                return 1.0
            elif row["status"] == "probation":
                return 0.5
            return 0.0
