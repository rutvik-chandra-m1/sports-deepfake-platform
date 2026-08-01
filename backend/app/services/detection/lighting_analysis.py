"""
Lighting-consistency analysis.

Compositing a face from a different source (or a different generation
pass) onto a background frequently leaves a subtle mismatch in brightness
and/or color balance between the face region and its surroundings, since
the two came from different lighting conditions. This module compares
face-region illumination statistics against a background sample when a
face is found (via face_detection.py's Haar cascade), and falls back to a
coarser whole-image quadrant-uniformity check when no face is detected.

suspicion_score is a heuristic, not a calibrated classifier — see
types.py::ForensicSignal.
"""

import cv2
import numpy as np

from app.services.detection.face_detection import FaceDetectorLoadError, detect_largest_face
from app.services.detection.types import ForensicSignal


def _region_stats(image_bgr: np.ndarray, mask: np.ndarray) -> tuple[float, np.ndarray]:
    """Mean brightness (LAB L-channel) and mean per-channel BGR, over the masked pixels."""
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0].astype(np.float32)

    brightness = float(l_channel[mask].mean()) if mask.any() else 0.0
    channel_means = image_bgr[mask].astype(np.float32).mean(axis=0) if mask.any() else np.zeros(3)
    return brightness, channel_means


def _face_vs_background(image_bgr: np.ndarray, bbox: tuple[int, int, int, int]) -> ForensicSignal:
    x, y, w, h = bbox
    height, width = image_bgr.shape[:2]

    face_mask = np.zeros((height, width), dtype=bool)
    face_mask[y : y + h, x : x + w] = True
    background_mask = ~face_mask

    face_brightness, face_channels = _region_stats(image_bgr, face_mask)
    bg_brightness, bg_channels = _region_stats(image_bgr, background_mask)

    brightness_diff = abs(face_brightness - bg_brightness)
    face_norm = face_channels / (face_channels.sum() + 1e-6)
    bg_norm = bg_channels / (bg_channels.sum() + 1e-6)
    color_balance_diff = float(np.abs(face_norm - bg_norm).sum())

    suspicion_score = float(
        np.clip(0.5 * (brightness_diff / 40.0) + 0.5 * (color_balance_diff / 0.3), 0.0, 1.0)
    )

    return ForensicSignal(
        name="lighting_analysis",
        applicable=True,
        suspicion_score=suspicion_score,
        summary=(
            f"Face-region lighting check: brightness difference={brightness_diff:.1f} "
            f"(LAB L, 0-255 scale), color-balance difference={color_balance_diff:.3f}."
        ),
        details={
            "mode": "face_vs_background",
            "face_brightness": face_brightness,
            "background_brightness": bg_brightness,
            "brightness_diff": brightness_diff,
            "color_balance_diff": color_balance_diff,
        },
    )


def _quadrant_fallback(image_bgr: np.ndarray) -> ForensicSignal:
    """Used when no face is detected: a coarser, lower-confidence signal
    comparing brightness across the four image quadrants."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = gray.shape
    quadrants = [
        gray[: h // 2, : w // 2],
        gray[: h // 2, w // 2 :],
        gray[h // 2 :, : w // 2],
        gray[h // 2 :, w // 2 :],
    ]
    means = [float(q.mean()) for q in quadrants]
    spread = float(max(means) - min(means))

    suspicion_score = float(np.clip(spread / 60.0, 0.0, 1.0))

    return ForensicSignal(
        name="lighting_analysis",
        applicable=True,
        suspicion_score=suspicion_score,
        summary=(
            f"No face detected — fell back to whole-image quadrant brightness check "
            f"(spread={spread:.1f}). Lower confidence than the face-aware check."
        ),
        details={"mode": "quadrant_fallback", "quadrant_means": means, "spread": spread},
    )


def analyze_lighting(image_bgr: np.ndarray) -> ForensicSignal:
    try:
        bbox = detect_largest_face(image_bgr)
    except FaceDetectorLoadError as exc:
        return ForensicSignal(
            name="lighting_analysis",
            applicable=False,
            suspicion_score=None,
            summary=f"Lighting analysis unavailable: {exc}",
            details={"error": str(exc)},
        )

    if bbox is None:
        return _quadrant_fallback(image_bgr)
    return _face_vs_background(image_bgr, bbox)
