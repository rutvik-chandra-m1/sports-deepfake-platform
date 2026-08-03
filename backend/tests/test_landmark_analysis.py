import numpy as np
import pytest

import app.services.detection.landmark_analysis as landmark_module
from app.services.detection.landmark_analysis import (
    LandmarkerLoadError,
    _jitter_from_landmark_sets,
    analyze_landmark_instability,
)


def test_identical_landmarks_across_frames_have_zero_jitter():
    rng = np.random.default_rng(0)
    base = rng.random((468, 2)).astype(np.float32)
    identical_sets = [base.copy() for _ in range(10)]

    jitter, pairs = _jitter_from_landmark_sets(identical_sets)

    assert jitter == pytest.approx(0.0, abs=1e-6)
    assert pairs == 9


def test_noisy_landmarks_have_higher_jitter_than_identical():
    rng = np.random.default_rng(0)
    base = rng.random((468, 2)).astype(np.float32)
    identical_sets = [base.copy() for _ in range(10)]
    jittery_sets = [base + rng.normal(0, 0.02, base.shape).astype(np.float32) for _ in range(10)]

    stable_jitter, _ = _jitter_from_landmark_sets(identical_sets)
    jittery_jitter, _ = _jitter_from_landmark_sets(jittery_sets)

    assert jittery_jitter > stable_jitter


def test_single_frame_is_not_applicable():
    frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    result = analyze_landmark_instability([frame])

    assert result.applicable is False
    assert "2+ frames" in result.summary


def test_returns_not_applicable_when_model_unavailable(monkeypatch):
    def _boom():
        raise LandmarkerLoadError("simulated: no internet to download model bundle")

    monkeypatch.setattr(landmark_module, "_load_landmarker", _boom)
    frames = [np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8) for _ in range(3)]

    result = analyze_landmark_instability(frames)

    assert result.applicable is False
    assert result.suspicion_score is None
    assert "unavailable" in result.summary
