"""Metadata provenance verification (R5): EXIF/XMP inspection and C2PA
Content Credentials validation.

Closes the PPT's "integrate metadata provenance verification" objective.
See exif_analysis.py for the governing rule -- absence of metadata is never
treated as evidence of manipulation.
"""

from app.services.provenance.analysis import run_provenance_analysis
from app.services.provenance.c2pa_verification import C2PA_SIGNAL, analyze_c2pa
from app.services.provenance.exif_analysis import (
    AI_METADATA_SIGNAL,
    CAMERA_METADATA_SIGNAL,
    analyze_ai_metadata,
    analyze_camera_metadata,
)

PROVENANCE_SIGNAL_NAMES = (AI_METADATA_SIGNAL, C2PA_SIGNAL, CAMERA_METADATA_SIGNAL)

__all__ = [
    "run_provenance_analysis",
    "analyze_ai_metadata",
    "analyze_camera_metadata",
    "analyze_c2pa",
    "PROVENANCE_SIGNAL_NAMES",
    "AI_METADATA_SIGNAL",
    "CAMERA_METADATA_SIGNAL",
    "C2PA_SIGNAL",
]
