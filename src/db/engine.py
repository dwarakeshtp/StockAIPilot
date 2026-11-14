"""SQLAlchemy engine factory for the StockPilot SQLite database."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from sqlalchemy import Engine, create_engine, event

SQLITE_PREFIX = "sqlite:///"


@lru_cache(maxsize=16)
def get_engine(url: str) -> Engine:
    """Return a process-wide engine for ``url``.

    File-based SQLite databases get their parent directory created, WAL journaling
    (concurrent readers while a job writes) and cross-thread connection use, which
    FastAPI's thread pool and the background job executor both need.
    """
    if not url.startswith("sqlite"):
        return create_engine(url)
    path = url.removeprefix(SQLITE_PREFIX)
    if url.startswith(SQLITE_PREFIX) and path and path != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(url, connect_args={"check_same_thread": False, "timeout": 30})

    @event.listens_for(engine, "connect")
    def _sqlite_pragmas(dbapi_connection, _record) -> None:  # pragma: no cover - driver hook
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.close()

    return engine
