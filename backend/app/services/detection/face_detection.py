"""
Lightweight face-region detection using OpenCV's Haar cascade classifier.

Used only as a face-region locator for lighting_analysis.py — deliberately
NOT MediaPipe here: modern MediaPipe (0.10+) requires the heavier Tasks API
and a downloaded model bundle even for simple bounding-box detection, and
current opencv-python releases no longer bundle cascade XML files either
(older versions did). So we lazily download the standard OpenCV cascade
file once and cache it under settings.models_dir — a much smaller, simpler
asset for a task that doesn't need MediaPipe's per-landmark precision.
"""

import logging
import urllib.request
from pathlib import Path

import cv2

from app.core.config import get_settings

logger = logging.getLogger(__name__)

_cascade: cv2.CascadeClassifier | None = None


class FaceDetectorLoadError(Exception):
    """Raised when the Haar cascade file can't be downloaded/loaded."""


def _cascade_path() -> Path:
    settings = get_settings()
    return Path(settings.models_dir) / "haarcascades" / "haarcascade_frontalface_default.xml"


def _ensure_cascade_downloaded() -> Path:
    path = _cascade_path()
    if path.exists():
        return path

    settings = get_settings()
    path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading Haar cascade from %s", settings.haarcascade_frontalface_url)
    try:
        urllib.request.urlretrieve(settings.haarcascade_frontalface_url, path)
    except Exception as exc:  # noqa: BLE001
        raise FaceDetectorLoadError(
            f"Could not download Haar cascade from {settings.haarcascade_frontalface_url}. "
            f"Requires internet access on first run. Original error: {type(exc).__name__}: {exc}"
        ) from exc
    return path


def _get_cascade() -> cv2.CascadeClassifier:
    global _cascade
    if _cascade is not None:
        return _cascade

    path = _ensure_cascade_downloaded()
    cascade = cv2.CascadeClassifier(str(path))
    if cascade.empty():
        raise FaceDetectorLoadError(f"Haar cascade file at {path} failed to load (empty classifier).")

    _cascade = cascade
    return cascade


def detect_largest_face(image_bgr) -> tuple[int, int, int, int] | None:
    """Returns (x, y, w, h) of the largest detected face, or None if no face found."""
    cascade = _get_cascade()
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        return None

    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    return int(x), int(y), int(w), int(h)
