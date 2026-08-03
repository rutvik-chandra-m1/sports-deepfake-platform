"""
Security regression tests (R9).

Each test here corresponds to a specific finding from the engineering review
and fails if that vulnerability is reintroduced. They assert on ATTACKER
outcomes ("the file was not read", "the path is not disclosed") rather than
on implementation details, so a future refactor that keeps the property
passes and one that loses it fails.
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient

import app.core.security as security_module
from app.core.config import Settings
from app.core.security import (
    INSECURE_SECRET_KEY,
    PathNotAllowedError,
    SlidingWindowRateLimiter,
    UnsupportedFileContentError,
    assert_content_matches_extension,
    assert_within_upload_dir,
    validate_production_settings,
)

# --------------------------------------------------------------------------
# Finding 1 (HIGH): arbitrary server-side file read
# --------------------------------------------------------------------------

@pytest.fixture()
def upload_root(tmp_path, monkeypatch):
    settings = Settings(upload_dir=str(tmp_path))
    monkeypatch.setattr(security_module, "get_settings", lambda: settings)
    return tmp_path


def test_path_inside_upload_dir_is_allowed(upload_root):
    target = upload_root / "images" / "photo.jpg"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"x")
    assert assert_within_upload_dir(str(target)) == target.resolve()


@pytest.mark.parametrize(
    "hostile",
    [
        "/etc/passwd",
        "C:/Windows/win.ini",
        "../../../../etc/shadow",
        "images/../../../../etc/passwd",
    ],
)
def test_paths_outside_upload_dir_are_rejected(upload_root, hostile):
    with pytest.raises(PathNotAllowedError):
        assert_within_upload_dir(hostile)


def test_traversal_out_and_back_is_rejected(upload_root):
    """`<upload>/../<sibling>/x.jpg` resolves outside the root even though it
    starts with the root as a prefix -- a naive startswith() check would pass
    it, which is why containment compares resolved paths."""
    sibling = upload_root.parent / "not-uploads" / "x.jpg"
    sibling.parent.mkdir(parents=True, exist_ok=True)
    sibling.write_bytes(b"x")
    with pytest.raises(PathNotAllowedError):
        assert_within_upload_dir(str(upload_root / ".." / "not-uploads" / "x.jpg"))


def test_rejection_message_does_not_disclose_the_attempted_path(upload_root):
    """The error is stored on the record and shown to the client, so it must
    not confirm filesystem layout to whoever probed it."""
    with pytest.raises(PathNotAllowedError) as exc:
        assert_within_upload_dir("/etc/passwd")
    assert "/etc/passwd" not in str(exc.value)


def test_client_cannot_set_file_path_via_api(client: TestClient):
    """The core exploit: POST a path, then ask the server to analyse it."""
    response = client.post(
        "/api/v1/analysis",
        json={
            "filename": "evil.jpg",
            "media_type": "image",
            "file_path": "C:/Windows/win.ini",
        },
    )
    assert response.status_code == 201

    # The unknown field is ignored, so the record has nothing to analyse and
    # /run refuses it rather than reading an arbitrary file.
    analysis_id = response.json()["id"]
    run = client.post(f"/api/v1/analysis/{analysis_id}/run")
    assert run.status_code == 400


# --------------------------------------------------------------------------
# Finding 5 (MEDIUM): internal path disclosure
# --------------------------------------------------------------------------

def test_api_never_discloses_server_file_paths(client: TestClient):
    import cv2

    image = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    upload = client.post(
        "/api/v1/media/upload", files={"file": ("p.jpg", encoded.tobytes(), "image/jpeg")}
    )
    assert upload.status_code == 201
    assert "file_path" not in upload.json()

    detail = client.get(f"/api/v1/analysis/{upload.json()['id']}")
    assert "file_path" not in detail.json()


# --------------------------------------------------------------------------
# Finding 3 (MEDIUM): extension-only upload validation
# --------------------------------------------------------------------------

def test_magic_bytes_accept_genuine_formats():
    assert_content_matches_extension(b"\xff\xd8\xff\xe0" + b"\x00" * 28, ".jpg")
    assert_content_matches_extension(b"\x89PNG\r\n\x1a\n" + b"\x00" * 24, ".png")
    assert_content_matches_extension(b"RIFF\x00\x00\x00\x00WEBP" + b"\x00" * 20, ".webp")


def test_magic_bytes_reject_renamed_payload():
    """A ZIP renamed to .jpg must not reach OpenCV's C++ decoders."""
    with pytest.raises(UnsupportedFileContentError):
        assert_content_matches_extension(b"PK\x03\x04" + b"\x00" * 28, ".jpg")


def test_upload_endpoint_rejects_content_extension_mismatch(client: TestClient):
    response = client.post(
        "/api/v1/media/upload",
        files={"file": ("payload.jpg", b"PK\x03\x04not-an-image", "image/jpeg")},
    )
    assert response.status_code == 400
    assert "does not match" in response.json()["detail"]


# --------------------------------------------------------------------------
# Finding 4 (MEDIUM): unbounded pagination
# --------------------------------------------------------------------------

def test_list_limit_is_clamped(client: TestClient):
    response = client.get("/api/v1/analysis?limit=1000000")
    assert response.status_code == 200
    assert len(response.json()["items"]) <= Settings().max_page_size


# --------------------------------------------------------------------------
# Finding 6 (MEDIUM): unsafe defaults reaching production
# --------------------------------------------------------------------------

def test_development_tolerates_default_settings():
    assert validate_production_settings(Settings(app_env="development")) == []


def test_production_rejects_placeholder_secret_and_missing_api_key():
    problems = validate_production_settings(
        Settings(app_env="production", secret_key=INSECURE_SECRET_KEY, api_key="", debug=True)
    )
    joined = " ".join(problems)
    assert "SECRET_KEY" in joined
    assert "API_KEY" in joined
    assert "DEBUG" in joined


def test_production_accepts_a_properly_configured_deployment():
    assert validate_production_settings(
        Settings(
            app_env="production",
            secret_key="a-real-secret",
            api_key="a-real-key",
            debug=False,
            cors_origins="https://example.org",
        )
    ) == []


# --------------------------------------------------------------------------
# Finding 4 (MEDIUM): rate limiting
# --------------------------------------------------------------------------

def test_rate_limiter_blocks_past_the_budget_and_isolates_clients():
    limiter = SlidingWindowRateLimiter(max_requests=3, window_seconds=60)
    assert all(limiter.allow("1.2.3.4") for _ in range(3))
    assert not limiter.allow("1.2.3.4")
    # One noisy client must not exhaust everyone else's budget.
    assert limiter.allow("5.6.7.8")


def test_rate_limiter_window_expires():
    limiter = SlidingWindowRateLimiter(max_requests=1, window_seconds=0)
    assert limiter.allow("1.2.3.4")
    assert limiter.allow("1.2.3.4")  # zero-length window -> immediately eligible again


# --------------------------------------------------------------------------
# Security headers
# --------------------------------------------------------------------------

def test_security_headers_present_on_responses(client: TestClient):
    response = client.get("/api/v1/health")
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"
    assert "frame-ancestors 'none'" in response.headers["Content-Security-Policy"]
