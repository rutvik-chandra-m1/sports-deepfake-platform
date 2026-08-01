"""
Media upload endpoint.

Validates and stores the file, creates a `pending` Analysis record, then
schedules the real analysis pipeline (Milestone 9's fusion of the Milestone
7 DL detector + Milestone 8 forensic signals) as a background task -- the
response returns immediately with the pending record; poll
GET /analysis/{id} to watch it move to "processing" then "completed"/"failed".
"""

import logging

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.analysis import AnalysisStatus
from app.schemas.analysis import AnalysisCreate, AnalysisRead
from app.services import analysis_service, storage_service
from app.services.analysis_pipeline import run_analysis_pipeline
from app.utils.file_validation import FileTooLargeError, UnsupportedFileTypeError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/media", tags=["Media"])


@router.post("/upload", response_model=AnalysisRead, status_code=status.HTTP_201_CREATED)
async def upload_media(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> AnalysisRead:
    if not file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No file provided.")

    try:
        file_path, media_type, size_bytes = await storage_service.save_upload(file)
    except UnsupportedFileTypeError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except FileTooLargeError as exc:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=str(exc)
        ) from exc

    record = analysis_service.create_analysis(
        db,
        AnalysisCreate(
            filename=file.filename,
            media_type=media_type,
            file_path=file_path,
            status=AnalysisStatus.PENDING,
        ),
    )
    logger.info("Upload registered as analysis id=%s (%d bytes)", record.id, size_bytes)

    background_tasks.add_task(run_analysis_pipeline, record.id)

    return record
