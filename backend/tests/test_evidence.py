"""
Visual evidence endpoint tests (R7).

These endpoints are the only ones that serve file content, which makes them
the most likely place for R9's arbitrary-file-read to reappear. The security
tests here matter more than the rendering ones.
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

import app.core.security as security_module
import app.services.storage_service as storage_service_module
from app.core.config import Settings


@pytest.fixture(autouse=True)
def _isolated_upload_dir(tmp_path, monkeypatch):
    settings = Settings(upload_dir=str(tmp_path))
    monkeypatch.setattr(storage_service_module, "get_settings", lambda: settings)
    monkeypatch.setattr(security_module, "get_settings", lambda: settings)
    return tmp_path


def _upload(client: TestClient, tmp_path) -> int:
    """Upload a real image through the API and return its analysis id."""
    import cv2

    rng = np.random.default_rng(0)
    array = rng.integers(0, 255, (96, 96, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", array)
    assert ok
    response = client.post(
        "/api/v1/media/upload", files={"file": ("shot.jpg", encoded.tobytes(), "image/jpeg")}
    )
    assert response.status_code == 201
    return response.json()["id"]


# --------------------------------------------------------------------------
# Security -- the reason these tests exist
# --------------------------------------------------------------------------

def test_media_endpoint_takes_an_id_not_a_path(client: TestClient, tmp_path):
    """The route is keyed by analysis id. There is no parameter through which
    a caller could name a file, which is the structural reason traversal is
    impossible here."""
    analysis_id = _upload(client, tmp_path)
    assert client.get(f"/api/v1/analysis/{analysis_id}/media").status_code == 200


def test_media_endpoint_refuses_a_record_pointing_outside_the_upload_dir(client: TestClient):
    """Defence in depth: even if a record somehow holds a hostile path -- an
    old row, or a future code path that forgets -- serving it must fail."""
    from app.db.session import SessionLocal
    from app.models.analysis import Analysis, AnalysisStatus, MediaType

    db = SessionLocal()
    try:
        record = Analysis(
            filename="evil.jpg",
            media_type=MediaType.IMAGE,
            file_path="C:/Windows/win.ini",
            status=AnalysisStatus.COMPLETED,
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        analysis_id = record.id
    finally:
        db.close()

    response = client.get(f"/api/v1/analysis/{analysis_id}/media")
    assert response.status_code == 403
    # The attempted path must not be echoed back to the caller.
    assert "win.ini" not in response.text
    assert "Windows" not in response.text


def test_media_endpoint_404s_for_unknown_analysis(client: TestClient):
    assert client.get("/api/v1/analysis/999999/media").status_code == 404


def test_media_response_sets_nosniff(client: TestClient, tmp_path):
    """Uploads are attacker-supplied bytes served from our origin, so the
    browser must not be allowed to sniff them into something executable."""
    analysis_id = _upload(client, tmp_path)
    response = client.get(f"/api/v1/analysis/{analysis_id}/media")
    assert response.headers["X-Content-Type-Options"] == "nosniff"


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------

def test_forensic_artefact_endpoints_return_images(client: TestClient, tmp_path):
    analysis_id = _upload(client, tmp_path)
    for signal in ("compression_analysis", "frequency_analysis"):
        response = client.get(f"/api/v1/analysis/{analysis_id}/evidence/{signal}")
        assert response.status_code == 200, signal
        assert response.headers["content-type"] == "image/png"
        # Decodable, and the right size for the source frame.
        image = Image.open(__import__("io").BytesIO(response.content))
        assert image.size == (96, 96)


def test_unknown_signal_is_404_not_500(client: TestClient, tmp_path):
    analysis_id = _upload(client, tmp_path)
    response = client.get(f"/api/v1/analysis/{analysis_id}/evidence/not_a_signal")
    assert response.status_code == 404


# --------------------------------------------------------------------------
# The honesty guard
# --------------------------------------------------------------------------

def test_attention_heatmap_stays_within_display_range():
    """REGRESSION: cubic upsampling overshot to -0.0125..1.0888. Since the
    renderer does `(heatmap * 255).astype(np.uint8)`, 1.0888 wrapped to 22 --
    the MOST attended regions would have rendered dark instead of hot."""
    pytest.importorskip("torch")
    from app.services.explainability.visual import compute_attention_rollout

    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, (128, 128, 3), dtype=np.uint8)
    try:
        heatmap = compute_attention_rollout(image)
    except Exception as exc:  # noqa: BLE001 - model may be unavailable offline
        pytest.skip(f"backbone unavailable: {exc}")

    assert heatmap.min() >= 0.0
    assert heatmap.max() <= 1.0
    assert heatmap.shape == (128, 128)


def test_heatmap_metadata_always_carries_the_disclaimer():
    """An attention map shows where the model looked, NOT why it decided.
    The disclaimer ships with the data so a caption cannot drift from what the
    maths supports."""
    from app.services.explainability.visual import _DISCLAIMER

    assert "not why it decided" in _DISCLAIMER
    assert "not, on their own, evidence" in _DISCLAIMER
