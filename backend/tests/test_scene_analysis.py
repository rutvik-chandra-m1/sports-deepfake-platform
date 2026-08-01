import numpy as np
import pytest

import app.services.sports_intel.scene_analysis as scene_module
from app.services.sports_intel.scene_analysis import analyze_scene_consistency


def _make_scene(color: tuple[int, int, int]) -> np.ndarray:
    return np.full((150, 150, 3), color, dtype=np.uint8)


def test_single_frame_is_not_applicable():
    result = analyze_scene_consistency([_make_scene((100, 150, 80))])
    assert result.applicable is False


def test_stable_background_scores_zero(monkeypatch):
    monkeypatch.setattr(scene_module, "detect_largest_face", lambda frame: None)
    frames = [_make_scene((100, 150, 80)) for _ in range(5)]

    result = analyze_scene_consistency(frames)

    assert result.applicable is True
    assert result.suspicion_score == pytest.approx(0.0, abs=1e-6)


def test_abrupt_background_change_scores_high(monkeypatch):
    monkeypatch.setattr(scene_module, "detect_largest_face", lambda frame: None)
    frames = [
        _make_scene((100, 150, 80)),
        _make_scene((100, 150, 80)),
        _make_scene((10, 10, 200)),  # abrupt splice point
        _make_scene((100, 150, 80)),
        _make_scene((100, 150, 80)),
    ]

    result = analyze_scene_consistency(frames)

    assert result.suspicion_score > 0.5


def test_returns_valid_signal_shape(monkeypatch):
    monkeypatch.setattr(scene_module, "detect_largest_face", lambda frame: None)
    frames = [_make_scene((100, 150, 80)) for _ in range(3)]

    result = analyze_scene_consistency(frames)

    assert result.name == "scene_consistency"
    assert set(result.details.keys()) >= {"max_distance", "mean_distance", "frame_pairs_used"}
