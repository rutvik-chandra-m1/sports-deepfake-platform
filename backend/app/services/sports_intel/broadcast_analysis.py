"""
Broadcast overlay/graphics consistency (Milestone 12).

Sports broadcasts typically carry overlay graphics (scoreboards, logos,
tickers) in fixed border/corner regions, composited onto the live footage
during production. If an overlay graphic is edited or replaced after the
fact (e.g. to alter a score, team name, or sponsor logo), it's often
recompressed independently of the surrounding footage — the same Error
Level Analysis idea as detection/compression_analysis.py, but specifically
comparing the border/corner "overlay zone" against the center of the
frame, rather than treating the whole image uniformly.

suspicion_score is a heuristic, not a calibrated classifier — see
detection/types.py::ForensicSignal.
"""

import cv2
import numpy as np

from app.services.detection.types import ForensicSignal

DEFAULT_JPEG_QUALITY = 90
_BORDER_FRACTION = 0.15  # outer 15% of the frame on each side = "overlay zone"
# A border-vs-center compression difference ratio at/above this is treated
# as maximally suspicious. Heuristic, illustrative.
_MAX_EXPECTED_DIFF_RATIO = 2.0


def _error_level_map(image_bgr: np.ndarray, quality: int) -> np.ndarray:
    success, encoded = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not success:
        raise ValueError("Could not JPEG-encode image for overlay ELA")
    recompressed = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    diff = cv2.absdiff(image_bgr, recompressed).astype(np.float32)
    return diff.mean(axis=2)


def _border_and_center_means(ela_map: np.ndarray) -> tuple[float, float]:
    height, width = ela_map.shape
    border_h, border_w = int(height * _BORDER_FRACTION), int(width * _BORDER_FRACTION)

    border_mask = np.ones((height, width), dtype=bool)
    border_mask[border_h : height - border_h, border_w : width - border_w] = False
    center_mask = ~border_mask

    border_mean = float(ela_map[border_mask].mean()) if border_mask.any() else 0.0
    center_mean = float(ela_map[center_mask].mean()) if center_mask.any() else 0.0
    return border_mean, center_mean


def analyze_broadcast_overlay(image_bgr: np.ndarray, quality: int = DEFAULT_JPEG_QUALITY) -> ForensicSignal:
    ela_map = _error_level_map(image_bgr, quality=quality)
    border_mean, center_mean = _border_and_center_means(ela_map)

    diff_ratio = abs(border_mean - center_mean) / (center_mean + 1e-6)
    suspicion_score = float(np.clip(diff_ratio / _MAX_EXPECTED_DIFF_RATIO, 0.0, 1.0))

    return ForensicSignal(
        name="broadcast_overlay_analysis",
        applicable=True,
        suspicion_score=suspicion_score,
        summary=(
            f"Overlay-zone vs. center compression difference: border={border_mean:.2f}, "
            f"center={center_mean:.2f} (ratio={diff_ratio:.2f})."
        ),
        details={"border_mean": border_mean, "center_mean": center_mean, "diff_ratio": diff_ratio},
    )
