"""Minimal .env loader (no external deps).

Used so workers/trackers can share a local `.env` without committing secrets.
"""

from __future__ import annotations

import os
from pathlib import Path


def _iter_candidate_env_paths(dotenv_path: str | None) -> list[Path]:
    if dotenv_path:
        return [Path(dotenv_path)]

    # Try CWD and all parents, so running from subdirs still works.
    candidates: list[Path] = []
    try:
        cwd = Path.cwd().expanduser().resolve()
        for p in [cwd, *cwd.parents]:
            candidates.append(p / ".env")
    except OSError:
        candidates.append(Path(".env"))

    # Also try relative to this file location (works even if CWD is elsewhere).
    try:
        here = Path(__file__).expanduser().resolve()
        for p in [here.parent, *here.parents]:
            candidates.append(p / ".env")
    except OSError:
        pass

    # De-dupe while preserving order.
    seen: set[str] = set()
    out: list[Path] = []
    for c in candidates:
        key = str(c)
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out


def load_dotenv_if_present(dotenv_path: str | None = None) -> None:
    """
    Load KEY=VALUE pairs from `.env` into os.environ if missing.
    - Ignores blank lines and `#` comments.
    - Does not override existing env vars.
    """
    try:
        for candidate in _iter_candidate_env_paths(dotenv_path):
            try:
                p = candidate.expanduser().resolve()
                if not p.exists() or not p.is_file():
                    continue
                for raw in p.read_text(encoding="utf-8").splitlines():
                    line = raw.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip('"').strip("'")
                    if not k:
                        continue
                    if k not in os.environ:
                        os.environ[k] = v
                return
            except OSError:
                continue
    except OSError:
        return

