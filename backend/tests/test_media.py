from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import app.services.storage_service as storage_service_module
from app.core.config import Settings


@pytest.fixture(autouse=True)
def _isolated_upload_dir(tmp_path, monkeypatch):
    """Redirect every upload in this test file to a throwaway temp dir
    instead of the real uploads/ folder, so test runs don't leave files
    behind in the actual project directory."""
    test_settings = Settings(upload_dir=str(tmp_path))
    monkeypatch.setattr(storage_service_module, "get_settings", lambda: test_settings)


def test_upload_valid_image_creates_pending_analysis(client: TestClient):
    files = {"file": ("photo.jpg", b"\xff\xd8\xff\xe0fakejpegbytes", "image/jpeg")}
    response = client.post("/api/v1/media/upload", files=files)

    assert response.status_code == 201
    body = response.json()
    assert body["filename"] == "photo.jpg"
    assert body["media_type"] == "image"
    assert body["status"] == "pending"
    assert body["file_path"] is not None
    assert Path(body["file_path"]).exists()


def test_upload_valid_video_creates_pending_analysis(client: TestClient):
    files = {"file": ("clip.mp4", b"fakemp4bytescontent", "video/mp4")}
    response = client.post("/api/v1/media/upload", files=files)

    assert response.status_code == 201
    body = response.json()
    assert body["media_type"] == "video"
    assert Path(body["file_path"]).exists()


def test_uploaded_file_is_stored_under_a_unique_name(client: TestClient):
    files = {"file": ("photo.jpg", b"content-one", "image/jpeg")}
    first = client.post("/api/v1/media/upload", files=files).json()
    second = client.post("/api/v1/media/upload", files=files).json()

    assert first["file_path"] != second["file_path"]
    assert Path(first["file_path"]).exists()
    assert Path(second["file_path"]).exists()


def test_upload_rejects_unsupported_extension(client: TestClient):
    files = {"file": ("notes.txt", b"plain text content", "text/plain")}
    response = client.post("/api/v1/media/upload", files=files)

    assert response.status_code == 400
    assert "Unsupported file extension" in response.json()["detail"]


def test_upload_rejects_oversized_file(client: TestClient, tmp_path, monkeypatch):
    tiny_settings = Settings(upload_dir=str(tmp_path), max_upload_size_mb=0)
    monkeypatch.setattr(storage_service_module, "get_settings", lambda: tiny_settings)

    files = {"file": ("photo.jpg", b"a" * 2048, "image/jpeg")}
    response = client.post("/api/v1/media/upload", files=files)

    assert response.status_code == 413
    # the partially-written file must be cleaned up, not left on disk
    assert list(tmp_path.rglob("*photo.jpg")) == []
