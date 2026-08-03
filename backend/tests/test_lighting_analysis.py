import numpy as np
import pytest

import app.services.detection.lighting_analysis as lighting_module
from app.services.detection.face_detection import FaceDetectorLoadError
from app.services.detection.lighting_analysis import (
    _face_vs_background,
    _quadrant_fallback,
    analyze_lighting,
)


def test_face_vs_background_scores_zero_for_uniform_lighting():
    uniform = np.full((200, 200, 3), 120, dtype=np.uint8)
    result = _face_vs_background(uniform, bbox=(60, 60, 80, 80))

    assert result.applicable is True
    assert result.suspicion_score == pytest.approx(0.0, abs=1e-6)


def test_face_vs_background_scores_high_for_brightness_mismatch():
    mismatched = np.full((200, 200, 3), 60, dtype=np.uint8)
    mismatched[60:140, 60:140] = 220  # bright "face" patch on a dark background

    result = _face_vs_background(mismatched, bbox=(60, 60, 80, 80))

    assert result.suspicion_score > 0.8
    assert result.details["brightness_diff"] > 100


def test_quadrant_fallback_scores_low_for_uniform_image():
    uniform = np.full((200, 200, 3), 120, dtype=np.uint8)
    result = _quadrant_fallback(uniform)

    assert result.suspicion_score == pytest.approx(0.0, abs=1e-6)


def test_quadrant_fallback_scores_high_for_uneven_image():
    uneven = np.full((200, 200, 3), 120, dtype=np.uint8)
    uneven[:100, :100] = 10

    result = _quadrant_fallback(uneven)

    assert result.suspicion_score > 0.5


def test_analyze_lighting_falls_back_when_no_face_detected(monkeypatch):
    monkeypatch.setattr(lighting_module, "detect_largest_face", lambda image: None)
    image = np.full((100, 100, 3), 128, dtype=np.uint8)

    result = analyze_lighting(image)

    assert result.details["mode"] == "quadrant_fallback"


def test_analyze_lighting_uses_face_region_when_face_detected(monkeypatch):
    monkeypatch.setattr(lighting_module, "detect_largest_face", lambda image: (10, 10, 40, 40))
    image = np.full((100, 100, 3), 128, dtype=np.uint8)

    result = analyze_lighting(image)

    assert result.details["mode"] == "face_vs_background"


def test_analyze_lighting_returns_not_applicable_when_detector_unavailable(monkeypatch):
    def _boom(image):
        raise FaceDetectorLoadError("simulated: no internet to download cascade")

    monkeypatch.setattr(lighting_module, "detect_largest_face", _boom)
    image = np.full((100, 100, 3), 128, dtype=np.uint8)

    result = analyze_lighting(image)

    assert result.applicable is False
    assert result.suspicion_score is None
