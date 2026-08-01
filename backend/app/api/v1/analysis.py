"""
Analysis endpoints.

CRUD over analysis records backed by SQLite (Milestone 4), plus a manual
`POST /{id}/run` (Milestone 9) to (re-)trigger the analysis pipeline on an
existing record -- useful for reprocessing, or for records created directly
via `POST /analysis` rather than through a real upload.
"""

import logging

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.analysis import AnalysisStatus, Verdict
from app.schemas.analysis import AnalysisCreate, AnalysisList, AnalysisRead
from app.services import analysis_service
from app.services.analysis_pipeline import run_analysis_pipeline

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analysis", tags=["Analysis"])


@router.post("", response_model=AnalysisRead, status_code=status.HTTP_201_CREATED)
def create_analysis(payload: AnalysisCreate, db: Session = Depends(get_db)) -> AnalysisRead:
    record = analysis_service.create_analysis(db, payload)
    logger.info("Created analysis id=%s filename=%s", record.id, record.filename)
    return record


@router.get("", response_model=AnalysisList)
def list_analyses(
    offset: int = 0,
    limit: int = 50,
    search: str | None = None,
    verdict: Verdict | None = None,
    status: AnalysisStatus | None = None,
    db: Session = Depends(get_db),
) -> AnalysisList:
    items, total = analysis_service.list_analyses(
        db, offset=offset, limit=limit, search=search, verdict=verdict, status=status
    )
    return AnalysisList(total=total, items=items)


@router.get("/{analysis_id}", response_model=AnalysisRead)
def get_analysis(analysis_id: int, db: Session = Depends(get_db)) -> AnalysisRead:
    record = analysis_service.get_analysis(db, analysis_id)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis not found")
    return record


@router.post("/{analysis_id}/run", response_model=AnalysisRead, status_code=status.HTTP_202_ACCEPTED)
def run_analysis(
    analysis_id: int, background_tasks: BackgroundTasks, db: Session = Depends(get_db)
) -> AnalysisRead:
    record = analysis_service.get_analysis(db, analysis_id)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis not found")
    if not record.file_path:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="This analysis record has no file_path — nothing to analyze.",
        )

    record.status = AnalysisStatus.PENDING
    db.commit()
    db.refresh(record)

    background_tasks.add_task(run_analysis_pipeline, analysis_id)
    return record


@router.delete("/{analysis_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_analysis(analysis_id: int, db: Session = Depends(get_db)) -> None:
    deleted = analysis_service.delete_analysis(db, analysis_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis not found")
