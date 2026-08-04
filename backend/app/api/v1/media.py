"""
Media upload endpoint.

Validates and stores the file, creates a `pending` Analysis record, then
schedules the real analysis pipeline (Milestone 9's fusion of the Milestone
7 DL detector + Milestone 8 forensic signals) as a background task -- the
response returns immediately with the pending record; poll
GET /analysis/{id} to watch it move to "processing" then "completed"/"failed".
"""

import logging

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.core.security import UnsupportedFileContentError, require_api_key
from app.db.session import get_db
from app.models.analysis import AnalysisStatus
from app.schemas.analysis import AnalysisCreateInternal, AnalysisRead
from app.services import analysis_service, storage_service
from app.services.jobs import submit_analysis
from app.utils.file_validation import FileTooLargeError, UnsupportedFileTypeError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/media", tags=["Media"], dependencies=[Depends(require_api_key)])


@router.post("/upload", response_model=AnalysisRead, status_code=status.HTTP_201_CREATED)
async def upload_media(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> AnalysisRead:
    if not file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No file provided.")

    try:
        file_path, media_type, size_bytes = await storage_service.save_upload(file)
    except UnsupportedFileTypeError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except UnsupportedFileContentError as exc:
        # The bytes don't match the claimed extension -- a renamed payload.
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except FileTooLargeError as exc:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=str(exc)
        ) from exc

    record = analysis_service.create_analysis(
        db,
        # AnalysisCreateInternal, not AnalysisCreate: file_path is set here
        # from what storage_service actually wrote, never from the request.
        AnalysisCreateInternal(
            filename=file.filename,
            media_type=media_type,
            file_path=file_path,
            status=AnalysisStatus.PENDING,
        ),
    )
    logger.info("Upload registered as analysis id=%s (%d bytes)", record.id, size_bytes)

    # Bounded pool, not BackgroundTasks: analysis is CPU-bound and would
    # otherwise contend with every sync endpoint for the shared threadpool.
    submit_analysis(record.id)

    return record
