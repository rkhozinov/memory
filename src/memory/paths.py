"""Default data-directory + DB-path resolution.

Centralised so core/embeddings/rerank agree, and so the plugin works
regardless of where it was cloned. Resolution order for the data dir:

  1. ``MEMORY_DATA_DIR`` env var (explicit override)
  2. legacy ``~/repos/memory/data`` — only if a DB already exists there,
     so existing users keep their data without migration
  3. ``$XDG_DATA_HOME/memory``, else ``~/.local/share/memory``

DB path resolution order:

  1. ``MEMORY_DB`` env var (explicit file override)
  2. ``<resolved data dir>/sqlite_vec.db``
"""

from __future__ import annotations

import os
from pathlib import Path

_LEGACY = Path.home() / "repos" / "memory" / "data"


def data_dir() -> Path:
    """Resolve the data directory (DB + downloaded models + caches)."""
    env = os.environ.get("MEMORY_DATA_DIR")
    if env:
        return Path(env).expanduser()
    # Keep existing users on their legacy path if a DB is already there.
    if (_LEGACY / "sqlite_vec.db").exists():
        return _LEGACY
    xdg = os.environ.get("XDG_DATA_HOME")
    if xdg:
        return Path(xdg) / "memory"
    return Path.home() / ".local" / "share" / "memory"


def db_path() -> Path:
    """Resolve the SQLite DB file path."""
    env = os.environ.get("MEMORY_DB")
    if env:
        return Path(env).expanduser()
    return data_dir() / "sqlite_vec.db"
