import numpy as np

from app.services.detection.optical_flow_analysis import _flow_roughness, analyze_optical_flow


def test_identical_frames_have_near_zero_roughness():
    """Mathematical sanity check: zero pixel difference must mean zero
    computed flow, and therefore zero roughness — not a claim about
    detecting manipulation, just that the metric behaves correctly at this
    trivial, guaranteed-true edge case."""
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, (150, 150), dtype=np.uint8)

    roughness = _flow_roughness(frame, frame.copy())

    assert roughness < 1e-4


def test_uncorrelated_frames_have_higher_roughness_than_identical():
    """Any genuine difference between frames produces some non-zero
    roughness; two frames with no true correspondence at all (independent
    random noise) must score higher than two identical frames."""
    rng = np.random.default_rng(0)
    identical = rng.integers(0, 255, (150, 150), dtype=np.uint8)
    frame_a = rng.integers(0, 255, (150, 150), dtype=np.uint8)
    frame_b = rng.integers(0, 255, (150, 150), dtype=np.uint8)

    identical_roughness = _flow_roughness(identical, identical.copy())
    uncorrelated_roughness = _flow_roughness(frame_a, frame_b)

    assert uncorrelated_roughness > identical_roughness


def test_single_frame_is_not_applicable():
    frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    result = analyze_optical_flow([frame])

    assert result.applicable is False
    assert "2+ frames" in result.summary


def test_returns_valid_signal_shape_for_video():
    rng = np.random.default_rng(1)
    frames = [rng.integers(0, 255, (100, 100, 3), dtype=np.uint8) for _ in range(4)]

    result = analyze_optical_flow(frames)

    assert result.name == "optical_flow_analysis"
    assert result.applicable is True
    assert 0.0 <= result.suspicion_score <= 1.0
    assert result.details["frame_pairs_used"] == 3
