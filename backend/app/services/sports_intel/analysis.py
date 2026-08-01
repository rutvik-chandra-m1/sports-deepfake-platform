"""
Runs all sports-specific detectors over a set of frames and returns their
signals together. Same resilience pattern as detection/forensic_analysis.py:
each detector's failure degrades to a non-applicable signal rather than
aborting the whole batch.
"""

import logging

import numpy as np

from app.services.detection.types import ForensicSignal
from app.services.sports_intel.broadcast_analysis import analyze_broadcast_overlay
from app.services.sports_intel.crowd_analysis import analyze_crowd_texture
from app.services.sports_intel.jersey_analysis import analyze_jersey_consistency
from app.services.sports_intel.scene_analysis import analyze_scene_consistency

logger = logging.getLogger(__name__)


def _safe(name: str, fn, *args) -> ForensicSignal:
    try:
        return fn(*args)
    except Exception as exc:  # noqa: BLE001 - any detector failure becomes a non-applicable signal
        logger.warning("Sports-intel detector '%s' failed: %s", name, exc)
        return ForensicSignal(
            name=name,
            applicable=False,
            suspicion_score=None,
            summary=f"{name} failed: {type(exc).__name__}: {exc}",
            details={"error": str(exc)},
        )


def run_sports_intelligence(frames_bgr: list[np.ndarray]) -> list[ForensicSignal]:
    """
    frames_bgr: sampled frames as returned by media_processing (BGR uint8).
    broadcast_overlay and crowd_texture run on the first frame (single-frame
    signals); jersey_color_consistency and scene_consistency run across all
    provided frames (video-only, need 2+ frames).
    """
    if not frames_bgr:
        raise ValueError("run_sports_intelligence requires at least one frame")

    first_frame = frames_bgr[0]

    return [
        _safe("broadcast_overlay_analysis", analyze_broadcast_overlay, first_frame),
        _safe("crowd_texture_analysis", analyze_crowd_texture, first_frame),
        _safe("jersey_color_consistency", analyze_jersey_consistency, frames_bgr),
        _safe("scene_consistency", analyze_scene_consistency, frames_bgr),
    ]
