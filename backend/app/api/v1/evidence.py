"""
Visual evidence endpoints (R7): the uploaded media itself, and the
attention/forensic overlays explaining a verdict.

SECURITY NOTE -- read before changing these.
These are the only endpoints that serve file content, which makes them the
natural place for R9's arbitrary-file-read to creep back in. Two rules:

  1. The client supplies an **analysis id**, never a path. The path comes
     from the database record.
  2. That path is still re-validated with `assert_within_upload_dir()` before
     anything is opened -- the record is not trusted either.

Rule 2 is not redundant. A record could predate the schema change that
removed client-settable `file_path`, or be written by a future code path
that forgets. The check costs nothing and closes both.
"""

import logging
from pathlib import Path

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy.orm import Session

from app.core.security import PathNotAllowedError, assert_within_upload_dir, require_api_key
from app.db.session import get_db
from app.models.analysis import Analysis, MediaType
from app.services import analysis_service
from app.services.explainability.visual import (
    VisualExplanationError,
    build_attention_overlay,
    build_signal_visualisation,
)
from app.services.reporting import ReportError, build_report_json, build_report_pdf

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/analysis", tags=["Evidence"], dependencies=[Depends(require_api_key)]
)

_CONTENT_TYPES = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}


def _resolved_media_path(analysis_id: int, db: Session) -> tuple[Analysis, Path]:
    """Look the record up and return a validated on-disk path."""
    record = analysis_service.get_analysis(db, analysis_id)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis not found")
    if not record.file_path:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="This analysis has no stored media."
        )

    try:
        path = assert_within_upload_dir(record.file_path)
    except PathNotAllowedError:
        # Deliberately vague to the client; the real path is in the log only.
        logger.warning("Blocked media access for analysis id=%s outside upload dir", analysis_id)
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Stored media path is not permitted."
        ) from None

    if not path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Stored media file is missing."
        )
    return record, path


def _load_frame(record: Analysis, path: Path) -> tuple[np.ndarray, np.ndarray]:
    """First frame as RGB, for images and video alike."""
    import cv2

    from app.services.media_processing import process_media

    processed = process_media(str(path), record.media_type)
    if not processed.frames:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="No frames could be read."
        )
    return cv2.cvtColor(processed.frames[0].image, cv2.COLOR_BGR2RGB), processed.frames[0].image


@router.get("/{analysis_id}/media", summary="Original uploaded media")
def get_media(analysis_id: int, db: Session = Depends(get_db)) -> Response:
    """Serves the uploaded file, so the UI can show the user what was judged.

    Until this existed there was no way to see the evidence next to the
    verdict -- a real gap for a forensics tool.
    """
    record, path = _resolved_media_path(analysis_id, db)
    media_type = _CONTENT_TYPES.get(path.suffix.lower())
    if media_type is None:
        media_type = "video/mp4" if record.media_type is MediaType.VIDEO else "application/octet-stream"

    return Response(
        content=path.read_bytes(),
        media_type=media_type,
        headers={
            # Never render an upload as an inline document in the browser --
            # it is attacker-supplied content on our origin.
            "Content-Disposition": f'inline; filename="{path.name}"',
            "X-Content-Type-Options": "nosniff",
            "Cache-Control": "private, max-age=3600",
        },
    )


@router.get("/{analysis_id}/evidence/attention", summary="Attention-rollout heatmap")
def get_attention_overlay(analysis_id: int, db: Session = Depends(get_db)) -> Response:
    """Heatmap of where the model looked, blended over the image.

    Shows attention, NOT causation -- see the disclaimer header and
    `visual.py`.
    """
    record, path = _resolved_media_path(analysis_id, db)
    frame_rgb, _ = _load_frame(record, path)

    try:
        png, metadata = build_attention_overlay(frame_rgb)
    except VisualExplanationError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc

    return Response(
        content=png,
        media_type="image/png",
        headers={
            # Travels with the image so a caption can never drift from what
            # the maths actually supports.
            "X-Explanation-Disclaimer": metadata["disclaimer"],
            "X-Explanation-Method": metadata["method"],
            "X-Content-Type-Options": "nosniff",
        },
    )


@router.get("/{analysis_id}/evidence/{signal}", summary="Forensic artefact for one signal")
def get_signal_visualisation(
    analysis_id: int, signal: str, db: Session = Depends(get_db)
) -> Response:
    """Renders the raw artefact behind a classical signal (ELA map, FFT
    spectrum) -- the intermediate images the detectors used to discard."""
    record, path = _resolved_media_path(analysis_id, db)
    _, frame_bgr = _load_frame(record, path)

    try:
        png = build_signal_visualisation(frame_bgr, signal)
    except VisualExplanationError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    return Response(
        content=png,
        media_type="image/png",
        headers={"X-Content-Type-Options": "nosniff"},
    )


@router.get("/{analysis_id}/report.pdf", summary="Downloadable verification report")
def get_report_pdf(analysis_id: int, db: Session = Depends(get_db)) -> Response:
    """A PDF a user can keep or forward.

    Because it will travel beyond whoever ran the analysis, the measured
    error rate and the "not admissible as proof" statement are rendered on
    the first page, before the verdict is elaborated.
    """
    record = analysis_service.get_analysis(db, analysis_id)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis not found")

    try:
        pdf = build_report_pdf(record)
    except ReportError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc

    return Response(
        content=pdf,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="verification-report-{analysis_id}.pdf"',
            "X-Content-Type-Options": "nosniff",
        },
    )


@router.get("/{analysis_id}/report.json", summary="Machine-readable verification report")
def get_report_json(analysis_id: int, db: Session = Depends(get_db)) -> dict:
    """Same content as the PDF, including the limitations -- an integrator
    must not be able to consume a verdict without the caveats."""
    record = analysis_service.get_analysis(db, analysis_id)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis not found")
    return build_report_json(record)
