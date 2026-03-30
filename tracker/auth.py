"""
Worker authorization via pre-shared UUID tokens.

The tracker reads a flat-file of UUID tokens (one per line) and rejects any
worker that cannot present a valid token.

Environment variables
---------------------
AUTH_TOKENS_FILE : path to the token list (default: ``authorized_tokens.txt``
                   in the repo root, i.e. the cwd when the tracker is started).
AUTH_DISABLED    : set to ``1`` to bypass all auth checks (useful in local
                   development without a token file).
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path
from threading import RLock

_UUID4_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
    re.IGNORECASE,
)

_DEFAULT_TOKENS_FILE = "authorized_tokens.txt"

# Hot-reload: re-read the file at most every N seconds.
_RELOAD_INTERVAL_SEC = 5.0

_lock = RLock()
_tokens: set[str] = set()
_last_loaded: float = 0.0
_tokens_path: Path | None = None  # resolved once on first use


def _resolve_path() -> Path:
    global _tokens_path
    if _tokens_path is None:
        env_path = os.environ.get("AUTH_TOKENS_FILE", _DEFAULT_TOKENS_FILE)
        _tokens_path = Path(env_path)
    return _tokens_path


def _load_tokens(path: Path) -> set[str]:
    """Read lines from *path*, normalise to lower-case UUID strings."""
    tokens: set[str] = set()
    if not path.exists():
        return tokens
    try:
        for raw in path.read_text(encoding="utf-8").splitlines():
            tok = raw.strip()
            if tok and not tok.startswith("#") and _UUID4_RE.match(tok):
                tokens.add(tok.lower())
    except OSError:
        pass
    return tokens


def _maybe_reload() -> None:
    global _tokens, _last_loaded
    now = time.monotonic()
    with _lock:
        if now - _last_loaded < _RELOAD_INTERVAL_SEC:
            return
        path = _resolve_path()
        _tokens = _load_tokens(path)
        _last_loaded = now
        if _tokens:
            print(
                f"[auth] loaded {len(_tokens)} authorized token(s) from '{path}'",
                flush=True,
            )
        else:
            print(
                f"[auth] WARNING — no valid tokens in '{path}' "
                "(all workers will be rejected unless AUTH_DISABLED=1)",
                flush=True,
            )


def is_authorized(token: str | None) -> bool:
    """Return True if *token* is in the authorized set or auth is disabled."""
    if os.environ.get("AUTH_DISABLED", "0") == "1":
        return True
    if not token:
        return False
    _maybe_reload()
    with _lock:
        return token.strip().lower() in _tokens


def list_tokens() -> list[str]:
    """Return the current set of authorized tokens (sorted) — for admin/debug."""
    _maybe_reload()
    with _lock:
        return sorted(_tokens)


def force_reload() -> int:
    """Force an immediate reload and return the number of loaded tokens."""
    global _last_loaded
    with _lock:
        _last_loaded = 0.0
    _maybe_reload()
    with _lock:
        return len(_tokens)
