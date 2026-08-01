"""
Runs all classical forensic detectors over a set of frames and returns
their signals together. Each detector's failure (e.g. no internet for the
landmark model on first run) is caught and represented as a non-applicable
signal rather than aborting the whole batch — the fusion engine (Milestone
9) should still get whatever signals *are* available.
"""

import logging

import cv2
import numpy as np

from app.services.detection.compression_analysis import analyze_compression
from app.services.detection.frequency_analysis import analyze_frequency
from app.services.detection.lighting_analysis import analyze_lighting
from app.services.detection.landmark_analysis import analyze_landmark_instability
from app.services.detection.optical_flow_analysis import analyze_optical_flow
from app.services.detection.types import ForensicSignal

logger = logging.getLogger(__name__)


def _safe(name: str, fn, *args) -> ForensicSignal:
    try:
        return fn(*args)
    except Exception as exc:  # noqa: BLE001 - any detector failure becomes a non-applicable signal
        logger.warning("Forensic detector '%s' failed: %s", name, exc)
        return ForensicSignal(
            name=name,
            applicable=False,
            suspicion_score=None,
            summary=f"{name} failed: {type(exc).__name__}: {exc}",
            details={"error": str(exc)},
        )


def run_forensic_analysis(frames_bgr: list[np.ndarray]) -> list[ForensicSignal]:
    """
    frames_bgr: sampled frames as returned by media_processing (BGR uint8).
    Runs frequency/compression/lighting on the first frame (representative
    for images and a fixed reference point for video); runs landmark
    instability and optical flow (Milestone 11) across all provided frames,
    since both are inherently multi-frame/video-only signals.
    """
    if not frames_bgr:
        raise ValueError("run_forensic_analysis requires at least one frame")

    first_frame = frames_bgr[0]
    frames_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_bgr]

    return [
        _safe("frequency_analysis", analyze_frequency, first_frame),
        _safe("compression_analysis", analyze_compression, first_frame),
        _safe("lighting_analysis", analyze_lighting, first_frame),
        _safe("landmark_instability", analyze_landmark_instability, frames_rgb),
        _safe("optical_flow_analysis", analyze_optical_flow, frames_bgr),
    ]
