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

    Milestone 4 scope: this lets a client register a record directly
    (useful for testing/demoing the history feature before real uploads
    exist). From Milestone 5 onward, the upload endpoint will create these
    records internally instead of clients posting them by hand.
    """

    filename: str = Field(..., min_length=1, max_length=255)
    media_type: MediaType
    file_path: str | None = None
    status: AnalysisStatus = AnalysisStatus.PENDING
    verdict: Verdict | None = None
    confidence_score: float | None = Field(default=None, ge=0, le=100)
    risk_level: RiskLevel | None = None
    explanation: str | None = None
    processing_duration_ms: int | None = Field(default=None, ge=0)


class AnalysisRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    filename: str
    media_type: MediaType
    file_path: str | None
    status: AnalysisStatus
    verdict: Verdict | None
    confidence_score: float | None
    risk_level: RiskLevel | None
    explanation: str | None
    detector_breakdown: str | None
    processing_duration_ms: int | None
    created_at: datetime
    completed_at: datetime | None


class AnalysisList(BaseModel):
    total: int
    items: list[AnalysisRead]
