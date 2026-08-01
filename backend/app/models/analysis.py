"""
Analysis ORM model.

This is the one row created per uploaded file. Its lifecycle:

    pending -> processing -> completed
                           -> failed

Fields for verdict/confidence/explanation stay nullable until a real
pipeline (Milestones 6-9) fills them in; for now they're populated directly
via the API so the History feature and frontend have real data to work with.
"""

import enum
from datetime import datetime, timezone

from sqlalchemy import DateTime, Enum, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class MediaType(str, enum.Enum):
    IMAGE = "image"
    VIDEO = "video"


class AnalysisStatus(str, enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class Verdict(str, enum.Enum):
    AUTHENTIC = "authentic"
    SUSPICIOUS = "suspicious"


class RiskLevel(str, enum.Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Analysis(Base):
    __tablename__ = "analyses"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)

    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    media_type: Mapped[MediaType] = mapped_column(Enum(MediaType), nullable=False)
    file_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)

    status: Mapped[AnalysisStatus] = mapped_column(
        Enum(AnalysisStatus), nullable=False, default=AnalysisStatus.PENDING, index=True
    )

    verdict: Mapped[Verdict | None] = mapped_column(Enum(Verdict), nullable=True, index=True)
    confidence_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    risk_level: Mapped[RiskLevel | None] = mapped_column(Enum(RiskLevel), nullable=True)
    explanation: Mapped[str | None] = mapped_column(Text, nullable=True)

    # JSON-serialized per-signal breakdown from the fusion engine (Milestone
    # 9): DL result + each forensic signal's applicability/score/summary and
    # the weight it was given. Populated by the pipeline, not the client.
    # Milestone 10's explainability layer reads this instead of re-running
    # detection just to produce human-readable reasons.
    detector_breakdown: Mapped[str | None] = mapped_column(Text, nullable=True)

    processing_duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, index=True
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Analysis id={self.id} filename={self.filename!r} status={self.status}>"
