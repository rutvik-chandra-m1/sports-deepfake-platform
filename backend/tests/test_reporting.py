"""
Verification report tests (R13).

The limitations tests are the important ones. A formatted PDF saying
"SUSPICIOUS" carries authority a web page does not, and this system is wrong
about 3 times in 10 -- so the caveats must be impossible to remove by
accident.
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient

import app.core.security as security_module
import app.services.storage_service as storage_service_module
from app.core.config import Settings
from app.services.reporting import MEASURED_ACCURACY, ReportError, build_report_pdf


@pytest.fixture(autouse=True)
def _isolated_upload_dir(tmp_path, monkeypatch):
    settings = Settings(upload_dir=str(tmp_path))
    monkeypatch.setattr(storage_service_module, "get_settings", lambda: settings)
    monkeypatch.setattr(security_module, "get_settings", lambda: settings)


def _completed_analysis(client: TestClient) -> int:
    import cv2

    rng = np.random.default_rng(0)
    array = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", array)
    assert ok
    response = client.post(
        "/api/v1/media/upload", files={"file": ("r.jpg", encoded.tobytes(), "image/jpeg")}
    )
    assert response.status_code == 201
    return response.json()["id"]


def test_pdf_report_downloads(client: TestClient):
    analysis_id = _completed_analysis(client)
    response = client.get(f"/api/v1/analysis/{analysis_id}/report.pdf")

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert "attachment" in response.headers["content-disposition"]
    assert response.content.startswith(b"%PDF")


def test_pdf_contains_the_error_rate_on_the_first_page(client: TestClient):
    """A reader must meet the error rate BEFORE acting on the verdict."""
    pypdf = pytest.importorskip("pypdf")

    analysis_id = _completed_analysis(client)
    response = client.get(f"/api/v1/analysis/{analysis_id}/report.pdf")

    import io

    raw = pypdf.PdfReader(io.BytesIO(response.content)).pages[0].extract_text()
    # PDF extraction inserts newlines at line-wrap points, so a sentence can be
    # split mid-phrase. Collapse whitespace before matching on meaning.
    first_page = " ".join(raw.split())

    assert MEASURED_ACCURACY in first_page
    assert "3 in 10 verdicts are wrong" in first_page
    assert "not admissible as evidence" in first_page


def test_pdf_is_pure_ascii_so_nothing_renders_as_mojibake(client: TestClient):
    """REGRESSION: the explanation layer emits em-dashes, and ReportLab's
    built-in Helvetica is Latin-1 only, so they rendered as replacement
    characters in a document users are meant to forward."""
    pypdf = pytest.importorskip("pypdf")

    analysis_id = _completed_analysis(client)
    response = client.get(f"/api/v1/analysis/{analysis_id}/report.pdf")

    import io

    text = pypdf.PdfReader(io.BytesIO(response.content)).pages[0].extract_text()
    assert "�" not in text, "replacement character present -- encoding regression"


def test_json_report_carries_the_same_limitations(client: TestClient):
    """An integrator must not be able to consume a verdict without caveats."""
    analysis_id = _completed_analysis(client)
    payload = client.get(f"/api/v1/analysis/{analysis_id}/report.json").json()

    assert payload["limitations"], "limitations missing from the machine-readable report"
    assert any("3 in 10" in text for text in payload["limitations"])
    assert payload["system_performance"]["accuracy"] == MEASURED_ACCURACY
    # Verdict provenance travels with the report so it stays auditable.
    assert "model_version" in payload["provenance"]


def test_incomplete_analysis_cannot_be_exported(client: TestClient):
    """Exporting a pending record would produce a report with no verdict."""
    from app.db.session import SessionLocal
    from app.models.analysis import Analysis, AnalysisStatus, MediaType

    db = SessionLocal()
    try:
        record = Analysis(filename="p.jpg", media_type=MediaType.IMAGE, status=AnalysisStatus.PENDING)
        db.add(record)
        db.commit()
        db.refresh(record)
        pending_id = record.id
        with pytest.raises(ReportError):
            build_report_pdf(record)
    finally:
        db.close()

    assert client.get(f"/api/v1/analysis/{pending_id}/report.pdf").status_code == 409


def test_report_404s_for_unknown_analysis(client: TestClient):
    assert client.get("/api/v1/analysis/999999/report.pdf").status_code == 404
    assert client.get("/api/v1/analysis/999999/report.json").status_code == 404
