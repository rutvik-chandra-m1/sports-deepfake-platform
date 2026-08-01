"""
Application settings.

All configuration is sourced from environment variables (with sensible
defaults for local development), following the "config over code" principle
from docs/architecture.md. Never hardcode paths, thresholds, or secrets
elsewhere in the codebase — add a field here instead.
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/app/core/config.py -> backend/
BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
PROJECT_ROOT = BACKEND_DIR.parent


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
    fusion_dl_weight: float = 0.4
    fusion_forensic_weight: float = 0.35
    fusion_sports_weight: float = 0.25
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

    # ----- Security -----
    secret_key: str = "change-me-in-production"

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
