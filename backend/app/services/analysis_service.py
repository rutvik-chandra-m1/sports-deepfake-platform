"""
Analysis service layer — all DB access for the Analysis resource lives here.
API routers (app/api/v1/analysis.py) should never touch the ORM/session
directly; they call these functions instead, so business logic stays
testable independent of HTTP.
"""

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models.analysis import Analysis, AnalysisStatus, Verdict
from app.schemas.analysis import AnalysisCreate


def create_analysis(db: Session, payload: AnalysisCreate) -> Analysis:
    record = Analysis(**payload.model_dump())
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def get_analysis(db: Session, analysis_id: int) -> Analysis | None:
    return db.get(Analysis, analysis_id)


def list_analyses(
    db: Session,
    offset: int = 0,
    limit: int = 50,
    search: str | None = None,
    verdict: Verdict | None = None,
    status: AnalysisStatus | None = None,
) -> tuple[list[Analysis], int]:
    """
    search matches filename (case-insensitive substring). verdict/status
    filter exactly. All filters are optional and combine with AND.
    """
    query = select(Analysis)
    count_query = select(func.count()).select_from(Analysis)

    if search:
        condition = Analysis.filename.ilike(f"%{search}%")
        query = query.where(condition)
        count_query = count_query.where(condition)
    if verdict is not None:
        query = query.where(Analysis.verdict == verdict)
        count_query = count_query.where(Analysis.verdict == verdict)
    if status is not None:
        query = query.where(Analysis.status == status)
        count_query = count_query.where(Analysis.status == status)

    total = db.scalar(count_query) or 0
    items = (
        db.execute(query.order_by(Analysis.created_at.desc()).offset(offset).limit(limit))
        .scalars()
        .all()
    )
    return list(items), total


def delete_analysis(db: Session, analysis_id: int) -> bool:
    record = get_analysis(db, analysis_id)
    if record is None:
        return False
    db.delete(record)
    db.commit()
    return True
