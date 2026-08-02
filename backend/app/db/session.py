"""
Database engine and session management.

SQLite-specific `check_same_thread=False` is required because FastAPI may
serve a single request's dependency chain across different threads in the
sync-endpoint case; the session itself is still scoped to one request via
`get_db`, so this does not introduce cross-request sharing.

Milestone 15: WAL (Write-Ahead Logging) mode lets SQLite serve read queries
(e.g. a GET /analysis list request) concurrently with a write happening in
a background task (the analysis pipeline updating a record) instead of
blocking behind it -- relevant here since uploads kick off exactly that
kind of concurrent write while the user may be browsing history.
"""

import logging

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import get_settings

logger = logging.getLogger(__name__)

settings = get_settings()
_is_sqlite = settings.database_url.startswith("sqlite")

_connect_args = {"check_same_thread": False} if _is_sqlite else {}

engine = create_engine(settings.database_url, connect_args=_connect_args)

if _is_sqlite:

    @event.listens_for(Engine, "connect")
    def _enable_sqlite_wal(dbapi_connection, connection_record):  # noqa: ANN001, ARG001
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.close()


SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Session:
    """FastAPI dependency — yields a request-scoped DB session, always closed after use."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create any missing tables. Called once on app startup.

    NOT the schema-change mechanism -- `create_all` only ever creates MISSING
    tables and will never add a column to an existing one. Alembic owns schema
    evolution (R11):

        cd backend && alembic upgrade head

    This is kept because it makes a fresh database (and every test run) instant
    without applying the full migration chain. `tests/test_migrations.py`
    asserts the two produce identical schemas, so they cannot silently drift.
    """
    # Import models here (not at module top-level) so they register on
    # Base.metadata before create_all runs, without causing circular imports.
    from app.db.base import Base
    from app.models import analysis  # noqa: F401

    Base.metadata.create_all(bind=engine)
    logger.info("Database tables ensured at %s", settings.database_url)
