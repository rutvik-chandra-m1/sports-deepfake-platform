"""
Background job and stale-record recovery tests (R10).

The recovery tests matter most: before this, a crash mid-analysis left a row
in PROCESSING forever, and the frontend polled it forever. Nothing detected
or reported it.
"""

from datetime import datetime, timedelta, timezone

import pytest

from app.core.config import Settings
from app.db.session import SessionLocal
from app.models.analysis import Analysis, AnalysisStatus, MediaType
from app.services import jobs


def _make_record(status: AnalysisStatus, age_seconds: int = 0) -> int:
    db = SessionLocal()
    try:
        record = Analysis(
            filename="stuck.jpg",
            media_type=MediaType.IMAGE,
            status=status,
            created_at=datetime.now(timezone.utc) - timedelta(seconds=age_seconds),
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        return record.id
    finally:
        db.close()


def _status_of(analysis_id: int) -> tuple[AnalysisStatus, str | None]:
    db = SessionLocal()
    try:
        record = db.get(Analysis, analysis_id)
        return record.status, record.explanation
    finally:
        db.close()


@pytest.mark.parametrize("stuck_status", [AnalysisStatus.PROCESSING, AnalysisStatus.PENDING])
def test_stale_records_are_failed_with_an_explanation(monkeypatch, stuck_status):
    """A row left mid-flight by a dead process must not stay that way."""
    analysis_id = _make_record(stuck_status, age_seconds=3600)

    monkeypatch.setattr(jobs, "get_settings", lambda: Settings(stale_analysis_seconds=300))
    assert jobs.recover_stale_records() >= 1

    status, explanation = _status_of(analysis_id)
    assert status == AnalysisStatus.FAILED
    # The user needs to know what happened and what to do, not a stack trace.
    assert "server stopped" in explanation
    assert "Re-run analysis" in explanation


def test_recent_in_flight_records_are_left_alone(monkeypatch):
    """A job that started seconds ago may be genuinely running. Failing it
    would destroy live work, so recovery only touches records older than the
    staleness window."""
    analysis_id = _make_record(AnalysisStatus.PROCESSING, age_seconds=0)

    monkeypatch.setattr(jobs, "get_settings", lambda: Settings(stale_analysis_seconds=300))
    jobs.recover_stale_records()

    status, _ = _status_of(analysis_id)
    assert status == AnalysisStatus.PROCESSING


def test_completed_records_are_never_touched(monkeypatch):
    analysis_id = _make_record(AnalysisStatus.COMPLETED, age_seconds=99999)

    monkeypatch.setattr(jobs, "get_settings", lambda: Settings(stale_analysis_seconds=300))
    jobs.recover_stale_records()

    status, _ = _status_of(analysis_id)
    assert status == AnalysisStatus.COMPLETED


def test_recovery_is_idempotent(monkeypatch):
    """Restarting twice must not keep rewriting rows."""
    _make_record(AnalysisStatus.PROCESSING, age_seconds=3600)
    monkeypatch.setattr(jobs, "get_settings", lambda: Settings(stale_analysis_seconds=300))

    first = jobs.recover_stale_records()
    second = jobs.recover_stale_records()
    assert first >= 1
    assert second == 0


def test_executor_is_bounded_and_reusable(monkeypatch):
    """Concurrency must be capped: analysis is CPU-bound, and an unbounded
    pool would starve the API it shares a machine with."""
    monkeypatch.setattr(jobs, "get_settings", lambda: Settings(max_concurrent_analyses=3))
    jobs.shutdown_executor(wait=True)
    try:
        executor = jobs.get_executor()
        assert executor._max_workers == 3
        assert jobs.get_executor() is executor  # cached, not rebuilt per call
    finally:
        jobs.shutdown_executor(wait=True)


def test_submit_runs_inline_in_synchronous_mode(monkeypatch):
    """The test-mode path must genuinely run the work, not silently skip it."""
    calls = []
    monkeypatch.setattr(jobs, "get_settings", lambda: Settings(analysis_synchronous=True))
    monkeypatch.setattr(
        "app.services.analysis_pipeline.run_analysis_pipeline", lambda i: calls.append(i)
    )

    jobs.submit_analysis(4242)
    assert calls == [4242]
