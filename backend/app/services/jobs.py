"""
Bounded background job execution and stale-record recovery (R10).

Two problems this fixes, both real and both previously unhandled:

1. **Records stuck in `processing` forever.** `run_analysis_pipeline` sets
   status to PROCESSING, then does several seconds of CPU-bound work. If the
   process dies in that window -- crash, Ctrl-C, deploy, OOM -- the row stays
   PROCESSING for eternity. Nothing ever reconciled it, and the frontend
   polled it forever. `recover_stale_records()` runs at startup and fails
   them with an honest explanation.

2. **Unbounded concurrency.** FastAPI's BackgroundTasks runs sync callables
   in Starlette's shared anyio threadpool (40 slots by default), which also
   serves every sync endpoint. Ten concurrent uploads -- each an 8-frame ViT
   pass plus MediaPipe plus optical flow -- would starve the API itself.
   A dedicated bounded pool keeps analysis work off the request threadpool
   entirely.

DELIBERATELY NOT a Celery/Redis queue. This is a single-process deployment;
a broker would add infrastructure to operate and secure without buying
correctness here. The honest limitation is recorded in docs/security.md and
docs/PROJECT_STATE.md: scaling to multiple workers needs a shared queue, and
`recover_stale_records()`'s "nothing can legitimately be running at startup"
assumption is only valid while there is exactly one process.
"""

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

from sqlalchemy import select

from app.core.config import get_settings
from app.db.session import SessionLocal
from app.models.analysis import Analysis, AnalysisStatus

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_executor: ThreadPoolExecutor | None = None


def get_executor() -> ThreadPoolExecutor:
    """Lazily creates the shared analysis pool."""
    global _executor
    with _lock:
        if _executor is None:
            workers = get_settings().max_concurrent_analyses
            _executor = ThreadPoolExecutor(
                max_workers=workers, thread_name_prefix="analysis"
            )
            logger.info("Analysis pool started with %d worker(s)", workers)
        return _executor


def shutdown_executor(wait: bool = False) -> None:
    """Stops the pool. `wait=False` at shutdown so a long analysis cannot
    hang the process; whatever was mid-flight is recovered on next startup."""
    global _executor
    with _lock:
        if _executor is not None:
            _executor.shutdown(wait=wait, cancel_futures=not wait)
            _executor = None


def submit_analysis(analysis_id: int) -> None:
    """Queues an analysis on the bounded pool.

    Replaces `BackgroundTasks.add_task`. Beyond bounding concurrency, this
    logs failures: a raw executor swallows exceptions into a Future nobody
    inspects, so a crashing job would vanish silently.
    """
    from app.services.analysis_pipeline import run_analysis_pipeline

    if get_settings().analysis_synchronous:
        # Deterministic path for tests: no pool, no waiting, no flakiness.
        run_analysis_pipeline(analysis_id)
        return

    def _run() -> None:
        try:
            run_analysis_pipeline(analysis_id)
        except Exception:  # noqa: BLE001 - last resort; the pipeline has its own handling
            logger.exception("Analysis job id=%s crashed outside the pipeline", analysis_id)

    get_executor().submit(_run)


def recover_stale_records() -> int:
    """
    Fails any record left mid-flight by a previous process. Returns the count.

    Safe because this runs at startup, before the pool accepts work, in a
    single-process deployment: nothing can legitimately be in progress yet.
    PENDING rows are included -- a record queued but never started is equally
    orphaned once the process that would have run it is gone.
    """
    settings = get_settings()
    cutoff = datetime.now(timezone.utc) - timedelta(seconds=settings.stale_analysis_seconds)

    db = SessionLocal()
    try:
        stale = db.execute(
            select(Analysis).where(
                Analysis.status.in_([AnalysisStatus.PENDING, AnalysisStatus.PROCESSING])
            )
        ).scalars().all()

        recovered = 0
        for record in stale:
            created = record.created_at
            # SQLite hands back naive datetimes; assume UTC to compare.
            if created is not None and created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
            if created is not None and created > cutoff:
                # Recent enough that it may belong to a still-running sibling
                # process. Left alone rather than risk failing live work.
                continue

            record.status = AnalysisStatus.FAILED
            record.explanation = (
                "Analysis did not finish — the server stopped while it was running. "
                "Use 'Re-run analysis' to try again."
            )
            record.completed_at = datetime.now(timezone.utc)
            recovered += 1

        if recovered:
            db.commit()
            logger.warning(
                "Recovered %d analysis record(s) left in-flight by a previous run", recovered
            )
        return recovered
    finally:
        db.close()
