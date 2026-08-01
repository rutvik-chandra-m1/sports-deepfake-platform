import numpy as np
import pytest

from app.services.sports_intel import run_sports_intelligence


def test_runs_all_four_detectors_and_degrades_gracefully():
    frames = [np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8) for _ in range(3)]

    signals = run_sports_intelligence(frames)
    names = {s.name for s in signals}

    assert names == {
        "broadcast_overlay_analysis",
        "crowd_texture_analysis",
        "jersey_color_consistency",
        "scene_consistency",
    }
    for signal in signals:
        assert signal.applicable in (True, False)
        if signal.applicable:
            assert 0.0 <= signal.suspicion_score <= 1.0
        else:
            assert signal.suspicion_score is None


def test_raises_on_empty_frame_list():
    with pytest.raises(ValueError):
        run_sports_intelligence([])


def test_single_frame_still_runs_frame_level_detectors():
    frame = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    signals = run_sports_intelligence([frame])

    by_name = {s.name: s for s in signals}
    assert by_name["broadcast_overlay_analysis"].applicable is True
    assert by_name["crowd_texture_analysis"].applicable is True
    assert by_name["jersey_color_consistency"].applicable is False  # needs 2+ frames
    assert by_name["scene_consistency"].applicable is False  # needs 2+ frames
