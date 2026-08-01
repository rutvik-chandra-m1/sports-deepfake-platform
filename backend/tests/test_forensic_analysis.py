import numpy as np
import pytest

from app.services.detection.forensic_analysis import run_forensic_analysis


def test_runs_all_five_detectors_and_degrades_gracefully():
    frames = [np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8) for _ in range(3)]

    signals = run_forensic_analysis(frames)
    names = {s.name for s in signals}

    assert names == {
        "frequency_analysis",
        "compression_analysis",
        "lighting_analysis",
        "landmark_instability",
        "optical_flow_analysis",
    }
    # landmark_instability is expected to be unavailable in this sandbox
    # (no internet to Hugging Face/Google model storage) -- it must degrade
    # to a non-applicable signal, not raise.
    for signal in signals:
        assert signal.applicable in (True, False)
        if signal.applicable:
            assert 0.0 <= signal.suspicion_score <= 1.0
        else:
            assert signal.suspicion_score is None


def test_raises_on_empty_frame_list():
    with pytest.raises(ValueError):
        run_forensic_analysis([])


def test_single_frame_still_runs_image_level_detectors():
    frame = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    signals = run_forensic_analysis([frame])

    by_name = {s.name: s for s in signals}
    assert by_name["frequency_analysis"].applicable is True
    assert by_name["compression_analysis"].applicable is True
    assert by_name["lighting_analysis"].applicable is True
    assert by_name["landmark_instability"].applicable is False  # needs 2+ frames
    assert by_name["optical_flow_analysis"].applicable is False  # needs 2+ frames


def test_video_frames_enable_optical_flow_analysis():
    frames = [np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8) for _ in range(4)]
    signals = run_forensic_analysis(frames)

    by_name = {s.name: s for s in signals}
    assert by_name["optical_flow_analysis"].applicable is True
