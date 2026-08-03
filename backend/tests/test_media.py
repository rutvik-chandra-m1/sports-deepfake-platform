
import pytest
from fastapi.testclient import TestClient

import app.core.security as security_module
import app.services.storage_service as storage_service_module
from app.core.config import Settings


@pytest.fixture(autouse=True)
def _isolated_upload_dir(tmp_path, monkeypatch):
    """Redirect every upload in this test file to a throwaway temp dir
    instead of the real uploads/ folder, so test runs don't leave files
    behind in the actual project directory."""
    test_settings = Settings(upload_dir=str(tmp_path))
    monkeypatch.setattr(storage_service_module, "get_settings", lambda: test_settings)
    # The R9 containment check reads settings through app.core.security. In
    # production both modules share one cached Settings, so they always agree;
    # in tests they must be patched together or storage writes to tmp_path
    # while containment still checks the real uploads/ root.
    monkeypatch.setattr(security_module, "get_settings", lambda: test_settings)


def test_upload_valid_image_creates_pending_analysis(client: TestClient):
    files = {"file": ("photo.jpg", b"\xff\xd8\xff\xe0fakejpegbytes", "image/jpeg")}
    response = client.post("/api/v1/media/upload", files=files)

    assert response.status_code == 201
    body = response.json()
    assert body["filename"] == "photo.jpg"
    assert body["media_type"] == "image"
    assert body["status"] == "pending"
    # R9: absolute server paths are no longer disclosed to clients.
    assert "file_path" not in body


def test_upload_valid_video_creates_pending_analysis(client: TestClient, tmp_path):
    # Real MP4 signature: "ftyp" at offset 4. Placeholder bytes are now
    # rejected by content validation (R9), which is the point of that check.
    payload = bytes([0, 0, 0, 24]) + b"ftypisom" + bytes(32)
    files = {"file": ("clip.mp4", payload, "video/mp4")}
    response = client.post("/api/v1/media/upload", files=files)

    assert response.status_code == 201
    body = response.json()
    assert body["media_type"] == "video"
    # file_path is no longer disclosed (R9); verify storage on disk instead.
    assert list((tmp_path / "videos").glob("*clip.mp4"))


def test_uploaded_file_is_stored_under_a_unique_name(client: TestClient, tmp_path):
    files = {"file": ("photo.jpg", b"\xff\xd8\xff\xe0content-one", "image/jpeg")}
    first = client.post("/api/v1/media/upload", files=files).json()
    second = client.post("/api/v1/media/upload", files=files).json()

    assert first["id"] != second["id"]
    # Same original filename, two distinct files on disk -- the UUID prefix
    # keeps concurrent uploads from colliding. Checked on disk because
    # file_path is deliberately no longer returned (R9).
    stored = list((tmp_path / "images").glob("*photo.jpg"))
    assert len(stored) == 2


def test_upload_rejects_unsupported_extension(client: TestClient):
    files = {"file": ("notes.txt", b"plain text content", "text/plain")}
    response = client.post("/api/v1/media/upload", files=files)

    assert response.status_code == 400
    assert "Unsupported file extension" in response.json()["detail"]


def test_upload_rejects_oversized_file(client: TestClient, tmp_path, monkeypatch):
    tiny_settings = Settings(upload_dir=str(tmp_path), max_upload_size_mb=0)
    monkeypatch.setattr(storage_service_module, "get_settings", lambda: tiny_settings)
    monkeypatch.setattr(security_module, "get_settings", lambda: tiny_settings)

    # Must carry a genuine JPEG signature: content validation runs on the first
    # chunk, so random bytes would be rejected as a content/extension mismatch
    # (400) before the size limit (413) could apply.
    files = {"file": ("photo.jpg", b"\xff\xd8\xff\xe0" + b"a" * 2048, "image/jpeg")}
    response = client.post("/api/v1/media/upload", files=files)

    assert response.status_code == 413
    # the partially-written file must be cleaned up, not left on disk
    assert list(tmp_path.rglob("*photo.jpg")) == []
