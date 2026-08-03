"""
Pydantic schemas for the Analysis resource. Kept deliberately separate from
the ORM model (app/models/analysis.py) so API contracts can evolve
independently of the database schema.
"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.models.analysis import AnalysisStatus, MediaType, RiskLevel, Verdict


class AnalysisCreate(BaseModel):
    """
    Payload to create an analysis record.

    SECURITY (R9): `file_path` is deliberately NOT a client-settable field.
    It used to be, and the analysis pipeline fed it straight to
    cv2.imread/VideoCapture -- so a client could name any path on the server
    and have its contents analysed, or probe the filesystem through the error
    text stored on the record. The path is now set server-side by
    `storage_service.save_upload()` only. Use `AnalysisCreateInternal` for
    that; this model is the public contract.
    """

    filename: str = Field(..., min_length=1, max_length=255)
    media_type: MediaType
    status: AnalysisStatus = AnalysisStatus.PENDING
    verdict: Verdict | None = None
    confidence_score: float | None = Field(default=None, ge=0, le=100)
    risk_level: RiskLevel | None = None
    explanation: str | None = None
    processing_duration_ms: int | None = Field(default=None, ge=0)


class AnalysisCreateInternal(AnalysisCreate):
    """Server-side creation, including the stored path.

    Separate from `AnalysisCreate` so `file_path` can never be populated from
    a request body: only `storage_service.save_upload()` -- which chooses the
    location itself -- constructs this.
    """

    file_path: str | None = None


class AnalysisRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    filename: str
    media_type: MediaType
    # SECURITY (R9): `file_path` is intentionally absent. Returning absolute
    # server paths disclosed the filesystem layout to every client for no
    # functional benefit -- the UI never displayed it.
    status: AnalysisStatus
    verdict: Verdict | None
    confidence_score: float | None
    risk_level: RiskLevel | None
    explanation: str | None
    detector_breakdown: str | None
    # Verdict provenance (R11) -- exposed because auditability is the point:
    # a stored verdict must say which model and combiner produced it.
    model_version: str | None
    pipeline_version: str | None
    fusion_method: str | None
    processing_duration_ms: int | None
    created_at: datetime
    completed_at: datetime | None


class AnalysisList(BaseModel):
    total: int
    items: list[AnalysisRead]
