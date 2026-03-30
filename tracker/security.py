from __future__ import annotations

import hashlib
import hmac
import os


MASTER_KEY = os.getenv("MASTER_KEY", "default-secret-key").encode("utf-8")
JOIN_PASSWORD = os.getenv("JOIN_PASSWORD", "Antigravity-2026")


def create_ticket(worker_id: str) -> str:
    """Generate a deterministic worker auth ticket from worker_id."""
    return hmac.new(MASTER_KEY, worker_id.encode("utf-8"), hashlib.sha256).hexdigest()


def is_valid_worker(worker_id: str, provided_ticket: str) -> bool:
    """Validate a worker ticket without any server-side session state."""
    expected = create_ticket(worker_id)
    return hmac.compare_digest(expected, provided_ticket)
