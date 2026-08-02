"""
Security primitives: path containment, upload content validation, API-key
auth, and production-readiness startup checks.

Each function here closes a specific finding from the engineering review in
docs/security.md. They are deliberately dependency-free (no `slowapi`, no
`python-magic`) -- the checks are small enough to implement and audit
directly, and every third-party package added to a security path is another
supply-chain surface.
"""

import hmac
import logging
import time
from collections import deque
from pathlib import Path
from threading import Lock

from fastapi import Header, HTTPException, Request, status

from app.core.config import Settings, get_settings

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# 1. Path containment -- fixes the arbitrary server-side file read
# --------------------------------------------------------------------------

class PathNotAllowedError(Exception):
    """Raised when a path resolves outside the permitted upload directory."""


def assert_within_upload_dir(candidate: str, settings: Settings | None = None) -> Path:
    """
    Resolve `candidate` and verify it sits inside the configured upload
    directory. Returns the resolved path; raises PathNotAllowedError otherwise.

    Why: `Analysis.file_path` used to be attacker-controllable, and the
    analysis pipeline fed it straight to cv2.imread/VideoCapture. That let a
    client name ANY path on the server and have its contents analysed, or
    probe the filesystem via the error text stored on the record.

    Uses `Path.resolve()` on both sides before comparing, so `..` traversal,
    symlinks, and mixed separators all collapse to a real location first --
    a string `startswith` check would not catch a symlink pointing out of the
    directory.
    """
    settings = settings or get_settings()
    upload_root = Path(settings.upload_dir).resolve()

    try:
        resolved = Path(candidate).resolve()
    except (OSError, ValueError) as exc:
        raise PathNotAllowedError(f"Path could not be resolved: {exc}") from exc

    if resolved == upload_root or upload_root in resolved.parents:
        return resolved

    # Deliberately does NOT echo the attempted path back to the caller -- that
    # would confirm filesystem layout to whoever probed it. The real path goes
    # to the server log only.
    logger.warning("Rejected path outside upload dir: %s (root=%s)", resolved, upload_root)
    raise PathNotAllowedError("Requested file is outside the permitted upload directory.")


# --------------------------------------------------------------------------
# 2. Upload content validation -- fixes extension-only trust
# --------------------------------------------------------------------------

# (magic bytes, offset) per accepted container. Extension alone proves
# nothing: anything can be renamed to .jpg.
_MAGIC_SIGNATURES: dict[str, list[tuple[bytes, int]]] = {
    ".jpg": [(b"\xff\xd8\xff", 0)],
    ".jpeg": [(b"\xff\xd8\xff", 0)],
    ".png": [(b"\x89PNG\r\n\x1a\n", 0)],
    ".webp": [(b"RIFF", 0), (b"WEBP", 8)],
    ".mp4": [(b"ftyp", 4)],
    ".mov": [(b"ftyp", 4)],
    ".avi": [(b"RIFF", 0), (b"AVI ", 8)],
    # Matroska/WebM share the EBML header.
    ".mkv": [(b"\x1a\x45\xdf\xa3", 0)],
}

# Enough to cover the deepest signature offset above.
MAGIC_HEADER_BYTES = 32


class UnsupportedFileContentError(Exception):
    """Raised when a file's actual bytes don't match its claimed extension."""


def assert_content_matches_extension(header: bytes, extension: str) -> None:
    """
    Verify the leading bytes match the claimed extension.

    Not a complete format parser and not a malware check -- it is a cheap
    guard against renamed payloads reaching OpenCV's decoders, which are C++
    and a historically rich source of memory-safety bugs.
    """
    signatures = _MAGIC_SIGNATURES.get(extension.lower())
    if signatures is None:
        raise UnsupportedFileContentError(f"No content signature known for '{extension}'.")

    for magic, offset in signatures:
        if header[offset : offset + len(magic)] != magic:
            raise UnsupportedFileContentError(
                f"File content does not match its '{extension}' extension."
            )


# --------------------------------------------------------------------------
# 3. API key authentication
# --------------------------------------------------------------------------

async def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """
    FastAPI dependency enforcing a shared API key.

    Enforced whenever `settings.api_key` is set. In development with no key
    configured it is a no-op, so local work and the test suite are unaffected;
    `validate_production_settings()` makes a missing key a startup failure
    outside development, so "unset" can never silently mean "open" in prod.
    """
    settings = get_settings()
    expected = settings.api_key
    if not expected:
        return

    # Constant-time compare: a normal `==` leaks key material through timing.
    if not x_api_key or not hmac.compare_digest(x_api_key, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid API key.",
            headers={"WWW-Authenticate": "ApiKey"},
        )


# --------------------------------------------------------------------------
# 4. Rate limiting
# --------------------------------------------------------------------------

class SlidingWindowRateLimiter:
    """
    Fixed-memory sliding-window limiter, keyed by client IP.

    In-process on purpose: this deployment is a single uvicorn process, so a
    shared Redis would add an external dependency without buying correctness.
    Documented in docs/security.md as the thing to replace when the service
    is ever scaled horizontally.
    """

    def __init__(self, max_requests: int, window_seconds: int) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._hits: dict[str, deque[float]] = {}
        self._lock = Lock()

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        cutoff = now - self.window_seconds
        with self._lock:
            bucket = self._hits.setdefault(key, deque())
            while bucket and bucket[0] < cutoff:
                bucket.popleft()
            if len(bucket) >= self.max_requests:
                return False
            bucket.append(now)
            # Stop unbounded growth from one-off IPs.
            if len(self._hits) > 10_000:
                for stale_key in [k for k, v in self._hits.items() if not v]:
                    del self._hits[stale_key]
            return True


def client_key(request: Request) -> str:
    """Best-effort client identity. X-Forwarded-For is honoured only when a
    trusted proxy is declared, since the header is client-settable."""
    settings = get_settings()
    if settings.trust_proxy_headers:
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


# --------------------------------------------------------------------------
# 5. Startup validation -- unsafe defaults must not reach production
# --------------------------------------------------------------------------

INSECURE_SECRET_KEY = "change-me-in-production"


def validate_production_settings(settings: Settings | None = None) -> list[str]:
    """
    Returns a list of configuration problems that are unacceptable outside
    development. `app.main` raises on a non-empty list at startup: failing
    loudly at boot beats running an internet-facing service with the
    shipped placeholder secret.
    """
    settings = settings or get_settings()
    if settings.app_env.lower() in {"development", "dev", "test", "testing"}:
        return []

    problems = []
    if settings.secret_key == INSECURE_SECRET_KEY:
        problems.append("SECRET_KEY is still the shipped placeholder.")
    if not settings.api_key:
        problems.append("API_KEY is empty, which would leave every endpoint unauthenticated.")
    if settings.debug:
        problems.append("DEBUG is true, which leaks stack traces to clients.")
    if "*" in settings.cors_origins:
        problems.append("CORS_ORIGINS contains '*' while credentials are allowed.")
    return problems
