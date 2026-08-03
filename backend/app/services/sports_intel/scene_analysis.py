"""
Scene/stadium consistency (Milestone 12, video only).

Heuristic: compares the background region's color histogram across sampled
frames. A continuous shot of one stadium/broadcast scene should have a
fairly stable background color distribution even as the camera pans or the
subject moves; an abrupt shift (e.g. splicing in footage from a different
match, stadium, or broadcast) shows up as a large histogram distance
between consecutive frames.

Reuses face_detection.py's Haar cascade to exclude the foreground/face
region when present, so foreground motion doesn't dominate the "scene"
comparison; falls back to comparing whole-frame histograms if no face is
detected in a given frame.

suspicion_score is a heuristic, not a calibrated classifier — see
detection/types.py::ForensicSignal.
"""

import cv2
import numpy as np

from app.services.detection.face_detection import FaceDetectorLoadError, detect_largest_face
from app.services.detection.types import ForensicSignal

# Bhattacharyya distance (0=identical, 1=totally different) at/above this
# is treated as maximally suspicious. Heuristic, illustrative.
_MAX_EXPECTED_DISTANCE = 0.5


def _background_histogram(image_bgr: np.ndarray) -> np.ndarray:
    mask = None
    try:
        bbox = detect_largest_face(image_bgr)
    except FaceDetectorLoadError:
        bbox = None

    if bbox is not None:
        x, y, w, h = bbox
        mask = np.ones(image_bgr.shape[:2], dtype=np.uint8) * 255
        mask[y : y + h, x : x + w] = 0

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], mask, [30, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist


def analyze_scene_consistency(frames_bgr: list[np.ndarray]) -> ForensicSignal:
    if len(frames_bgr) < 2:
        return ForensicSignal(
            name="scene_consistency",
            applicable=False,
            suspicion_score=None,
            summary="Scene consistency needs 2+ frames (video); not applicable to a single image.",
            details={"frame_count": len(frames_bgr)},
        )

    histograms = [_background_histogram(f) for f in frames_bgr]
    distances = [
        float(cv2.compareHist(prev, curr, cv2.HISTCMP_BHATTACHARYYA))
        for prev, curr in zip(histograms, histograms[1:], strict=False)
    ]

    max_distance = max(distances)
    mean_distance = float(np.mean(distances))

    # Use max, not mean -- one abrupt splice point should be flagged even
    # if the rest of the clip is perfectly consistent.
    suspicion_score = float(np.clip(max_distance / _MAX_EXPECTED_DISTANCE, 0.0, 1.0))

    return ForensicSignal(
        name="scene_consistency",
        applicable=True,
        suspicion_score=suspicion_score,
        summary=(
            f"Background scene distance across {len(distances)} consecutive frame pair(s): "
            f"max={max_distance:.3f}, mean={mean_distance:.3f} (Bhattacharyya distance, 0=identical)."
        ),
        details={
            "max_distance": max_distance,
            "mean_distance": mean_distance,
            "frame_pairs_used": len(distances),
        },
    )
