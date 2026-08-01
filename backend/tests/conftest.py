"""
Test configuration.

Points DATABASE_URL at a throwaway temp file *before* anything under app/
is imported, so the engine created in app.db.session never touches the
real development database at database/app.db. Tables are created once
for the whole test session via init_db().
"""

import os
import tempfile

_tmp_dir = tempfile.mkdtemp(prefix="sports-deepfake-tests-")
os.environ["DATABASE_URL"] = f"sqlite:///{os.path.join(_tmp_dir, 'test.db')}"

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.db.session import init_db  # noqa: E402
from app.main import app  # noqa: E402

init_db()


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)
