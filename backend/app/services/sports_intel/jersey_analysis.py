"""
Jersey/team color consistency (Milestone 12, video only).

Heuristic: the dominant clothing color of the person in frame (approximated
as the region directly below the detected face — a proxy "torso" region,
since there's no dedicated person/jersey detector here) should stay roughly
consistent across a clip of one continuous shot. A sudden jersey-color
change mid-clip can indicate splicing between different source footage —
exactly the kind of "fake match highlights" the sports-specific scope
calls for.

Reuses face_detection.py's Haar cascade (no new external asset). This
checks internal self-consistency only — it does NOT identify or compare
against any real team's actual colors, which would need a maintained
team-color reference database, out of scope here.

suspicion_score is a heuristic, not a calibrated classifier — see
detection/types.py::ForensicSignal (reused here for a consistent shape
across both forensic and sports-intelligence signals).
"""

import cv2
import numpy as np

from app.services.detection.face_detection import FaceDetectorLoadError, detect_largest_face
from app.services.detection.types import ForensicSignal

# Heuristic scale for turning weighted HSV std into a 0-1 score.
# Illustrative, not calibrated against labeled sports footage.
_MAX_EXPECTED_VARIABILITY = 40.0


def _torso_region(image_bgr: np.ndarray, face_bbox: tuple[int, int, int, int]) -> np.ndarray | None:
    x, y, w, h = face_bbox
    height, width = image_bgr.shape[:2]

    top = y + h  # just below the face
    bottom = min(top + int(h * 1.5), height)
    left = max(x - int(w * 0.3), 0)
    right = min(x + w + int(w * 0.3), width)

    if bottom <= top or right <= left:
        return None
    return image_bgr[top:bottom, left:right]


def _dominant_color_hsv(region_bgr: np.ndarray) -> np.ndarray:
    """Mean color in HSV — more perceptually stable for 'is this the same
    clothing color' than raw BGR under lighting variation."""
    hsv = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    return hsv.reshape(-1, 3).mean(axis=0)


def analyze_jersey_consistency(frames_bgr: list[np.ndarray]) -> ForensicSignal:
    if len(frames_bgr) < 2:
        return ForensicSignal(
            name="jersey_color_consistency",
            applicable=False,
            suspicion_score=None,
            summary="Jersey consistency needs 2+ frames (video); not applicable to a single image.",
            details={"frame_count": len(frames_bgr)},
        )

    colors = []
    for frame in frames_bgr:
        try:
            bbox = detect_largest_face(frame)
        except FaceDetectorLoadError as exc:
            return ForensicSignal(
                name="jersey_color_consistency",
                applicable=False,
                suspicion_score=None,
                summary=f"Jersey consistency unavailable: {exc}",
                details={"error": str(exc)},
            )
        if bbox is None:
            continue
        region = _torso_region(frame, bbox)
        if region is None or region.size == 0:
            continue
        colors.append(_dominant_color_hsv(region))

    if len(colors) < 2:
        return ForensicSignal(
            name="jersey_color_consistency",
            applicable=False,
            suspicion_score=None,
            summary=(
                f"Could not locate a torso region in enough frames "
                f"({len(colors)}/{len(frames_bgr)}) to assess jersey consistency."
            ),
            details={"frames_with_torso_region": len(colors), "total_frames": len(frames_bgr)},
        )

    colors_arr = np.array(colors)
    std = colors_arr.std(axis=0)  # [hue_std, sat_std, val_std]
    # Hue differences dominate perceived color-identity change, so weight it highest.
    variability = float(std[0] * 1.0 + std[1] * 0.3 + std[2] * 0.2)

    suspicion_score = float(np.clip(variability / _MAX_EXPECTED_VARIABILITY, 0.0, 1.0))

    return ForensicSignal(
        name="jersey_color_consistency",
        applicable=True,
        suspicion_score=suspicion_score,
        summary=(
            f"Torso-region color variability across {len(colors)} frame(s): {variability:.1f} "
            "(weighted hue/saturation/value std). Higher values suggest inconsistent clothing color."
        ),
        details={
            "variability": variability,
            "hue_std": float(std[0]),
            "sat_std": float(std[1]),
            "val_std": float(std[2]),
            "frames_used": len(colors),
        },
    )
