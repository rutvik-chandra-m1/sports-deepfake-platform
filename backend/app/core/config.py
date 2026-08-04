"""
Application settings.

All configuration is sourced from environment variables (with sensible
defaults for local development), following the "config over code" principle
from docs/architecture.md. Never hardcode paths, thresholds, or secrets
elsewhere in the codebase — add a field here instead.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/app/core/config.py -> backend/
BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
PROJECT_ROOT = BACKEND_DIR.parent

_SQLITE_PREFIX = "sqlite:///"


def _resolve_relative_path(value: str) -> str:
    """Anchor a possibly-relative path to `backend/` rather than the current
    working directory.

    `.env` ships relative paths (`../models/pretrained`, `../uploads`, ...)
    which Python otherwise resolves against CWD -- so where uploads, reports,
    model weights and the database actually landed depended on which
    directory the process was launched from. Running a script from `ml/eval/`
    instead of `backend/` silently created a second `models/pretrained` tree
    and re-downloaded ~330MB of weights into it.

    BACKEND_DIR is the right anchor (not PROJECT_ROOT) because those `../`
    prefixes are written relative to the documented working directory --
    `cd backend && uvicorn app.main:app` -- so `../models/pretrained` is
    meant to mean PROJECT_ROOT/models/pretrained, matching the absolute
    defaults below.
    """
    path = Path(value)
    return str(path if path.is_absolute() else (BACKEND_DIR / path).resolve())


class Settings(BaseSettings):
    """Central application configuration."""

    model_config = SettingsConfigDict(
        env_file=str(BACKEND_DIR / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ----- General -----
    app_env: str = "development"
    app_name: str = "Sports Deepfake Detection & Verification Platform"
    app_version: str = "0.1.0"
    debug: bool = True

    # ----- Server -----
    backend_host: str = "0.0.0.0"
    backend_port: int = 8000

    # ----- CORS -----
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    # ----- Database -----
    database_url: str = f"sqlite:///{PROJECT_ROOT / 'database' / 'app.db'}"

    # ----- Storage -----
    upload_dir: str = str(PROJECT_ROOT / "uploads")
    reports_dir: str = str(PROJECT_ROOT / "reports")
    models_dir: str = str(PROJECT_ROOT / "models" / "pretrained")

    max_upload_size_mb: int = 200
    allowed_image_extensions: str = ".jpg,.jpeg,.png,.webp"
    allowed_video_extensions: str = ".mp4,.mov,.avi,.mkv"

    # ----- Logging -----
    log_level: str = "INFO"

    # ----- AI Models -----
    # Real, fine-tuned deepfake image classifier from Hugging Face Hub. See
    # docs/models.md for why this one, its reported metrics, and license.
    deepfake_image_model_id: str = "prithivMLmods/Deep-Fake-Detector-v2-Model"

    # Classical CV forensic assets (downloaded once, cached under models_dir).
    haarcascade_frontalface_url: str = (
        "https://raw.githubusercontent.com/opencv/opencv/4.x/data/haarcascades/"
        "haarcascade_frontalface_default.xml"
    )
    mediapipe_face_landmarker_url: str = (
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
        "face_landmarker/float16/latest/face_landmarker.task"
    )

    # ----- Fusion Engine (Milestone 9) -----
    # Nominal weight given to the deep-learning detector when it's available;
    # the remainder is split evenly across whichever forensic signals are
    # applicable, then everything is renormalized to sum to 1. Design
    # choices, not values fit/calibrated against a labeled dataset -- see
    # docs/models.md.
    # Fusion is organized into three weighted pools: the DL detector (a
    # single signal), classical forensic signals (Milestone 8/11), and
    # sports-specific signals (Milestone 12). Each pool's weight is split
    # evenly across whichever of its member signals are actually applicable
    # for a given analysis, then everything is renormalized to sum to 1 --
    # same renormalize-across-applicable-signals behavior introduced in
    # Milestone 9, just with three pools instead of one DL + one forensic
    # pool. Values are a documented design choice (DL still weighted
    # highest as the only signal with a published evaluation; sports
    # signals weighted lowest as the most experimental/heuristic), not
    # fit/calibrated against a labeled dataset -- see docs/models.md.
    # Reweighted after R3/R6 measured every pool on a held-out split
    # (docs/evaluation.md). The trained probe is this project's own
    # classifier, fitted to its labelled data and the only signal shown to
    # beat chance, so it carries most of the weight. The stock face-tuned DL
    # head measured 0.491 (chance) on non-face imagery and the classical
    # heuristics measured at or below chance with directions that were not
    # even stable across splits -- both are kept as corroboration at low
    # weight rather than removed, so their contribution stays visible in the
    # breakdown, but neither can now dominate a verdict.
    # Provenance (R5) is weighted meaningfully because when it fires it is
    # near-conclusive -- a signed manifest or an explicit generator tag beats
    # any pixel heuristic. It is simply absent most of the time, and the
    # renormalise-across-applicable-signals behaviour means an absent pool
    # costs nothing.
    fusion_provenance_weight: float = 0.35
    fusion_trained_weight: float = 0.60
    fusion_dl_weight: float = 0.15
    fusion_forensic_weight: float = 0.15
    fusion_sports_weight: float = 0.10
    fusion_verdict_threshold: float = 0.5  # fused score >= this -> "suspicious"
    fusion_risk_low_threshold: float = 0.3  # fused score below this -> "low" risk
    fusion_risk_high_threshold: float = 0.6  # fused score at/above this -> "high" risk

    # ----- Explainability (Milestone 10) -----
    # Per-signal thresholds for deciding whether a signal's score is
    # noteworthy enough to surface as a human-readable reason, vs.
    # "inconclusive" and left out. Independent from the fusion risk
    # thresholds above -- these judge one signal in isolation, not the
    # overall fused score.
    explanation_low_threshold: float = 0.35  # score <= this -> reassuring ("no X detected")
    explanation_high_threshold: float = 0.65  # score >= this -> concerning ("X detected")

    # ----- Background jobs (R10) -----
    # Analysis is CPU-bound (ViT forward pass + MediaPipe + optical flow).
    # More workers than cores just thrashes; this machine has 4.
    max_concurrent_analyses: int = 2
    # A PENDING/PROCESSING record older than this at startup is treated as
    # orphaned by a dead process and failed with an explanation.
    stale_analysis_seconds: int = 300
    # Run analyses inline instead of on the pool. Used by the test suite so
    # assertions are deterministic without sleeping -- previously guaranteed
    # for free by TestClient running BackgroundTasks synchronously, which a
    # real thread pool no longer does.
    analysis_synchronous: bool = False

    # ----- Security (R9) -----
    secret_key: str = "change-me-in-production"

    # Shared API key. Empty = auth disabled, which is convenient for local
    # development and the test suite but is rejected at startup whenever
    # app_env is not a development/test value (see
    # app.core.security.validate_production_settings), so "unset" can never
    # silently mean "open to the internet" in production.
    api_key: str = ""

    # Only honour X-Forwarded-For when genuinely behind a trusted proxy --
    # the header is client-settable, so trusting it by default would let
    # anyone spoof their identity and bypass rate limiting.
    trust_proxy_headers: bool = False

    rate_limit_requests: int = 60
    rate_limit_window_seconds: int = 60
    # Uploads are far more expensive than reads (full detection pipeline), so
    # they get their own, tighter budget.
    upload_rate_limit_requests: int = 10
    upload_rate_limit_window_seconds: int = 60

    # Cap on how many analyses a single list request may return; without it
    # `?limit=1000000` is a valid, trivially abusable request.
    max_page_size: int = 200

    @field_validator("upload_dir", "reports_dir", "models_dir")
    @classmethod
    def _anchor_storage_dirs(cls, value: str) -> str:
        return _resolve_relative_path(value)

    @field_validator("database_url")
    @classmethod
    def _anchor_sqlite_path(cls, value: str) -> str:
        """Same anchoring for the SQLite file. Non-SQLite URLs (Postgres etc.)
        and `:memory:` are passed through untouched."""
        if not value.startswith(_SQLITE_PREFIX):
            return value
        raw = value[len(_SQLITE_PREFIX):]
        if not raw or raw.startswith(":memory:"):
            return value
        return _SQLITE_PREFIX + _resolve_relative_path(raw)

    @property
    def cors_origins_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def allowed_image_extensions_list(self) -> list[str]:
        return [ext.strip().lower() for ext in self.allowed_image_extensions.split(",") if ext.strip()]

    @property
    def allowed_video_extensions_list(self) -> list[str]:
        return [ext.strip().lower() for ext in self.allowed_video_extensions.split(",") if ext.strip()]

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024

    def ensure_runtime_directories(self) -> None:
        """Create storage directories on startup if they don't already exist."""
        for path_str in (self.upload_dir, self.reports_dir, self.models_dir):
            path = Path(path_str)
            path.mkdir(parents=True, exist_ok=True)
        Path(self.database_url.replace("sqlite:///", "")).parent.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Cached settings accessor — settings are read from env once per process."""
    return Settings()
