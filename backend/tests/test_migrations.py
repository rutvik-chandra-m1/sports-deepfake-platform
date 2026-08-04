"""
Migration tests (R11).

The point of these is the DRIFT GUARD: `init_db()` builds the schema from the
ORM models via `create_all`, while production databases are built by applying
migrations. Those two can silently diverge -- someone adds a column to the
model, forgets the migration, tests keep passing because they use create_all,
and the next real deployment is missing a column. `test_migrations_match_models`
fails in exactly that case.
"""

import subprocess
from pathlib import Path

import pytest
from sqlalchemy import create_engine, inspect

BACKEND_DIR = Path(__file__).resolve().parent.parent
ALEMBIC = BACKEND_DIR / ".venv" / "Scripts" / "alembic.exe"


def _run_alembic(*args: str, db_url: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [str(ALEMBIC), *args],
        cwd=str(BACKEND_DIR),
        env={"DATABASE_URL": db_url, "PATH": "", "SYSTEMROOT": "C:\\Windows"},
        capture_output=True,
        text=True,
    )


pytestmark = pytest.mark.skipif(
    not ALEMBIC.exists(), reason="alembic console script not present in this environment"
)


def _schema_of(engine) -> dict[str, set[str]]:
    inspector = inspect(engine)
    return {
        table: {col["name"] for col in inspector.get_columns(table)}
        for table in inspector.get_table_names()
        if table != "alembic_version"
    }


def test_migrations_build_the_same_schema_as_the_models(tmp_path):
    """THE drift guard -- see module docstring."""
    from app.db.base import Base
    from app.models import analysis  # noqa: F401 -- registers tables

    migrated_db = tmp_path / "migrated.db"
    result = _run_alembic("upgrade", "head", db_url=f"sqlite:///{migrated_db}")
    assert result.returncode == 0, f"alembic upgrade failed:\n{result.stderr}"

    migrated = _schema_of(create_engine(f"sqlite:///{migrated_db}"))

    models_db = tmp_path / "models.db"
    models_engine = create_engine(f"sqlite:///{models_db}")
    Base.metadata.create_all(models_engine)
    from_models = _schema_of(models_engine)

    assert migrated == from_models, (
        "Migrations and ORM models disagree. Add the missing migration with:\n"
        "  cd backend && alembic revision --autogenerate -m '<what changed>'\n"
        f"migrations: {migrated}\nmodels:     {from_models}"
    )


def test_migrations_round_trip(tmp_path):
    """Every migration must be reversible -- a downgrade path that was never
    executed is not a downgrade path."""
    db_url = f"sqlite:///{tmp_path / 'roundtrip.db'}"

    # Assert WITH stderr. This test flaked once in a full-suite run and the
    # bare `== 0` assertion carried no information about why -- three
    # subprocesses, no clue which failed or how. Alembic's own error is the
    # only thing that makes an intermittent failure actionable.
    for step in (("upgrade", "head"), ("downgrade", "base"), ("upgrade", "head")):
        result = _run_alembic(*step, db_url=db_url)
        assert result.returncode == 0, (
            f"alembic {' '.join(step)} failed (exit {result.returncode})"
            f"\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )


def test_existing_data_survives_migration(tmp_path):
    """The scenario that used to require deleting the database: an existing
    populated DB gaining new columns."""
    from sqlalchemy import text

    db_path = tmp_path / "populated.db"
    db_url = f"sqlite:///{db_path}"

    # Build only the baseline, then insert a row.
    assert _run_alembic("upgrade", "5106d5905b68", db_url=db_url).returncode == 0
    engine = create_engine(db_url)
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO analyses (filename, media_type, status, created_at) "
                "VALUES ('legacy.jpg', 'IMAGE', 'COMPLETED', '2026-01-01 00:00:00')"
            )
        )

    assert _run_alembic("upgrade", "head", db_url=db_url).returncode == 0

    with engine.connect() as conn:
        rows = conn.execute(text("SELECT filename, model_version FROM analyses")).fetchall()
    assert len(rows) == 1, "pre-existing row was lost by the migration"
    assert rows[0][0] == "legacy.jpg"
    assert rows[0][1] is None  # new column exists and is null for old rows
