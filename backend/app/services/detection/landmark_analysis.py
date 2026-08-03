"""
Facial landmark instability analysis (MediaPipe Face Landmarker).

Deepfake face-swap/reenactment often produces subtle frame-to-frame
landmark jitter beyond what natural head movement/facial expression would
cause. This module detects landmarks per frame, then — when 2+ frames with
a detected face are available — measures normalized frame-to-frame
landmark displacement as a jitter score.

Split into a network-dependent loader (_load_landmarker, downloads a model
bundle from Google's servers on first use — see docs/models.md) and a pure
math function (_jitter_from_landmark_sets) that Milestone 8's tests exercise
directly with synthetic coordinates, requiring no network or model download.

suspicion_score is a heuristic, not a calibrated classifier — see
types.py::ForensicSignal. Note natural head movement also causes
displacement; this is a coarse signal for the fusion engine (Milestone 9)
to weigh alongside the others, not a standalone verdict.
"""

import logging
import threading
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

from app.core.config import get_settings
from app.services.detection.types import ForensicSignal

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_landmarker: Any = None


class LandmarkerLoadError(Exception):
    """Raised when the Face Landmarker model bundle can't be downloaded/loaded."""


def _model_path() -> Path:
    settings = get_settings()
    return Path(settings.models_dir) / "mediapipe" / "face_landmarker.task"


def _ensure_model_downloaded() -> Path:
    path = _model_path()
    if path.exists():
        return path

    settings = get_settings()
    path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading MediaPipe Face Landmarker model from %s", settings.mediapipe_face_landmarker_url)
    try:
        urllib.request.urlretrieve(settings.mediapipe_face_landmarker_url, path)
    except Exception as exc:  # noqa: BLE001
        raise LandmarkerLoadError(
            f"Could not download Face Landmarker model from {settings.mediapipe_face_landmarker_url}. "
            f"Requires internet access on first run. Original error: {type(exc).__name__}: {exc}"
        ) from exc
    return path


def _load_landmarker() -> Any:
    global _landmarker
    with _lock:
        if _landmarker is not None:
            return _landmarker

        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode

        model_path = _ensure_model_downloaded()
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=RunningMode.IMAGE,
            num_faces=1,
        )
        _landmarker = FaceLandmarker.create_from_options(options)
        return _landmarker


def _detect_landmarks_one_frame(landmarker: Any, image_rgb: np.ndarray) -> np.ndarray | None:
    """Returns an (N, 2) array of normalized (x, y) landmark coordinates for
    the first detected face, or None if no face was found in this frame."""
    import mediapipe as mp

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    result = landmarker.detect(mp_image)

    if not result.face_landmarks:
        return None

    landmarks = result.face_landmarks[0]
    return np.array([[lm.x, lm.y] for lm in landmarks], dtype=np.float32)


def _inter_ocular_distance(landmarks: np.ndarray) -> float:
    """Rough scale normalizer: MediaPipe's canonical face mesh indices 33
    and 263 are the outer corners of the left/right eyes."""
    if landmarks.shape[0] <= 263:
        # Fallback for reduced landmark sets (e.g. in tests): use bounding-box diagonal.
        span = landmarks.max(axis=0) - landmarks.min(axis=0)
        return float(np.linalg.norm(span)) or 1.0
    return float(np.linalg.norm(landmarks[33] - landmarks[263])) or 1.0


def _jitter_from_landmark_sets(landmark_sets: list[np.ndarray]) -> tuple[float, int]:
    """
    Pure math, no MediaPipe/network involved — tested directly and
    independently of the model-loading path. Returns
    (mean_normalized_displacement, num_frame_pairs_used).
    """
    displacements = []
    for prev, curr in zip(landmark_sets, landmark_sets[1:], strict=False):
        if prev.shape != curr.shape:
            continue
        scale = _inter_ocular_distance(prev)
        per_point_displacement = np.linalg.norm(curr - prev, axis=1)
        displacements.append(float(per_point_displacement.mean()) / scale)

    if not displacements:
        return 0.0, 0

    return float(np.mean(displacements)), len(displacements)


def analyze_landmark_instability(frames_rgb: list[np.ndarray]) -> ForensicSignal:
    """
    frames_rgb: list of RGB uint8 frames (e.g. from media_processing,
    converted from BGR). For a single image, returns applicable=False —
    instability is inherently a multi-frame/video concept.
    """
    if len(frames_rgb) < 2:
        return ForensicSignal(
            name="landmark_instability",
            applicable=False,
            suspicion_score=None,
            summary="Landmark instability needs 2+ frames (video); not applicable to a single image.",
            details={"frame_count": len(frames_rgb)},
        )

    try:
        landmarker = _load_landmarker()
    except LandmarkerLoadError as exc:
        return ForensicSignal(
            name="landmark_instability",
            applicable=False,
            suspicion_score=None,
            summary=f"Landmark instability unavailable: {exc}",
            details={"error": str(exc)},
        )

    landmark_sets = []
    for frame in frames_rgb:
        landmarks = _detect_landmarks_one_frame(landmarker, frame)
        if landmarks is not None:
            landmark_sets.append(landmarks)

    if len(landmark_sets) < 2:
        return ForensicSignal(
            name="landmark_instability",
            applicable=False,
            suspicion_score=None,
            summary=(
                f"Face detected in only {len(landmark_sets)}/{len(frames_rgb)} sampled frames "
                "— not enough coverage to measure instability."
            ),
            details={"frames_with_face": len(landmark_sets), "total_frames": len(frames_rgb)},
        )

    jitter, pairs_used = _jitter_from_landmark_sets(landmark_sets)
    # Heuristic squash -- illustrative threshold, not calibrated.
    suspicion_score = float(np.clip(jitter / (jitter + 0.05), 0.0, 1.0))

    return ForensicSignal(
        name="landmark_instability",
        applicable=True,
        suspicion_score=suspicion_score,
        summary=(
            f"Landmark jitter (mean normalized frame-to-frame displacement)={jitter:.4f} "
            f"over {pairs_used} frame pair(s); face found in {len(landmark_sets)}/{len(frames_rgb)} frames."
        ),
        details={
            "jitter": jitter,
            "frame_pairs_used": pairs_used,
            "frames_with_face": len(landmark_sets),
            "total_frames": len(frames_rgb),
        },
    )
