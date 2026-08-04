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

# Raise the R9 rate limits far above anything the suite will hit.
#
# The limiter is in-process and its state is shared across the WHOLE session,
# so it does not reset between tests. With the production budget (10
# uploads/min) the suite silently began failing with 429s the moment enough
# upload-based tests existed -- and the failures landed on unrelated tests
# that merely happened to run later, which is a confusing way to learn about
# a limit.
#
# Rate-limiting behaviour is still covered: tests/test_security.py exercises
# SlidingWindowRateLimiter directly, where the budget can be set per test
# without a shared-state problem.
os.environ["RATE_LIMIT_REQUESTS"] = "100000"
os.environ["UPLOAD_RATE_LIMIT_REQUESTS"] = "100000"

# Run analyses inline (R10). TestClient used to execute BackgroundTasks
# synchronously, so "upload then assert completed" worked for free. The
# bounded pool is genuinely asynchronous, so without this every such test
# would race the worker.
os.environ["ANALYSIS_SYNCHRONOUS"] = "true"

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.db.session import init_db  # noqa: E402
from app.main import app  # noqa: E402

init_db()


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)
