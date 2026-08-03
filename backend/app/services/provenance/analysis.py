"""
Runs all provenance checks over an uploaded file (R5).

Same resilience pattern as detection/forensic_analysis.py: one check failing
degrades to a non-applicable signal rather than aborting the batch.

Unlike every other detector in this project, these read the FILE, not decoded
frames -- metadata lives in the container and is destroyed the moment an
image is decoded to a pixel array. That is why the pipeline passes a path
here instead of frames.
"""

import logging
from collections.abc import Callable

from app.services.detection.types import ForensicSignal
from app.services.provenance.c2pa_verification import analyze_c2pa
from app.services.provenance.exif_analysis import analyze_ai_metadata, analyze_camera_metadata

logger = logging.getLogger(__name__)


def _safe(name: str, fn: Callable[..., ForensicSignal], *args) -> ForensicSignal:
    try:
        return fn(*args)
    except Exception as exc:  # noqa: BLE001 - any check failure becomes non-applicable
        logger.warning("Provenance check '%s' failed: %s", name, exc)
        return ForensicSignal(
            name=name,
            applicable=False,
            suspicion_score=None,
            summary=f"{name} failed: {type(exc).__name__}: {exc}",
            details={"error": str(exc)},
        )


def run_provenance_analysis(file_path: str) -> list[ForensicSignal]:
    """
    file_path: the stored upload. Callers MUST have already validated
    containment (app.core.security.assert_within_upload_dir) -- this opens
    whatever path it is given.

    Every signal here is non-applicable unless it finds positive evidence;
    see the module docstrings for why absence of metadata is deliberately
    treated as "no information" rather than as suspicion.
    """
    return [
        _safe("provenance_ai_metadata", analyze_ai_metadata, file_path),
        _safe("provenance_c2pa", analyze_c2pa, file_path),
        _safe("provenance_camera_metadata", analyze_camera_metadata, file_path),
    ]
