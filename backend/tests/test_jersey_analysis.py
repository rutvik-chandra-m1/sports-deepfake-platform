import numpy as np
import pytest

import app.services.sports_intel.jersey_analysis as jersey_module
from app.services.detection.face_detection import FaceDetectorLoadError
from app.services.sports_intel.jersey_analysis import analyze_jersey_consistency


def _make_frame(torso_color: tuple[int, int, int]) -> np.ndarray:
    frame = np.full((200, 200, 3), 128, dtype=np.uint8)
    frame[90:170, 20:180] = torso_color  # below a face bbox of (50,50,40,40)
    return frame


def test_single_frame_is_not_applicable():
    result = analyze_jersey_consistency([_make_frame((0, 0, 200))])
    assert result.applicable is False
    assert "2+ frames" in result.summary


def test_consistent_jersey_color_scores_zero(monkeypatch):
    monkeypatch.setattr(jersey_module, "detect_largest_face", lambda frame: (50, 50, 40, 40))
    frames = [_make_frame((0, 0, 200)) for _ in range(5)]

    result = analyze_jersey_consistency(frames)

    assert result.applicable is True
    assert result.suspicion_score == pytest.approx(0.0, abs=1e-6)


def test_changing_jersey_color_scores_high(monkeypatch):
    monkeypatch.setattr(jersey_module, "detect_largest_face", lambda frame: (50, 50, 40, 40))
    frames = [
        _make_frame((0, 0, 200)),
        _make_frame((200, 0, 0)),
        _make_frame((0, 200, 0)),
        _make_frame((0, 0, 200)),
        _make_frame((200, 200, 0)),
    ]

    result = analyze_jersey_consistency(frames)

    assert result.suspicion_score > 0.5


def test_no_face_detected_is_not_applicable(monkeypatch):
    monkeypatch.setattr(jersey_module, "detect_largest_face", lambda frame: None)
    frames = [_make_frame((0, 0, 200)) for _ in range(3)]

    result = analyze_jersey_consistency(frames)

    assert result.applicable is False


def test_detector_unavailable_returns_not_applicable(monkeypatch):
    def _boom(frame):
        raise FaceDetectorLoadError("simulated: no internet")

    monkeypatch.setattr(jersey_module, "detect_largest_face", _boom)
    frames = [_make_frame((0, 0, 200)) for _ in range(3)]

    result = analyze_jersey_consistency(frames)

    assert result.applicable is False
    assert result.suspicion_score is None
