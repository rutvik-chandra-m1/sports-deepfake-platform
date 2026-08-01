import numpy as np
import pytest

import app.services.detection.image_detector as image_detector_module
from app.services.detection.image_detector import (
    ModelLoadError,
    predict_video,
    temporal_consistency_signal,
)
from app.services.detection.types import ImageDetectionResult, VideoDetectionResult


def _fake_predict_sequence(fake_probabilities: list[float]):
    """Returns a monkeypatch-able stand-in for `predict()` that yields the
    given fake_probability values in order, one per call."""
    calls = {"count": 0}

    def _predict(frame, model_id=None):
        value = fake_probabilities[calls["count"]]
        calls["count"] += 1
        return ImageDetectionResult(
            real_probability=1.0 - value, fake_probability=value, predicted_label="fake" if value >= 0.5 else "real", model_id="test/model"
        )

    _predict.calls = calls
    return _predict


def test_predict_video_aggregates_mean_and_std(monkeypatch):
    monkeypatch.setattr(image_detector_module, "predict", _fake_predict_sequence([0.1, 0.2, 0.3, 0.4]))
    frames = [np.zeros((10, 10, 3), dtype=np.uint8) for _ in range(4)]

    result = predict_video(frames, max_frames=4)

    assert result.num_frames_analyzed == 4
    assert result.mean_fake_probability == pytest.approx(0.25, abs=1e-6)
    assert result.std_fake_probability == pytest.approx(np.std([0.1, 0.2, 0.3, 0.4]), abs=1e-6)


def test_predict_video_respects_max_frames_via_even_sampling(monkeypatch):
    fake_predict = _fake_predict_sequence([0.5] * 4)  # only 4 values needed if sampling works
    monkeypatch.setattr(image_detector_module, "predict", fake_predict)
    frames = [np.zeros((10, 10, 3), dtype=np.uint8) for _ in range(20)]

    result = predict_video(frames, max_frames=4)

    assert result.num_frames_analyzed == 4
    assert fake_predict.calls["count"] == 4  # only sampled frames were analyzed, not all 20


def test_predict_video_fails_fast_on_first_frame_load_error(monkeypatch):
    call_count = {"n": 0}

    def _boom(frame, model_id=None):
        call_count["n"] += 1
        raise ModelLoadError("simulated: no internet")

    monkeypatch.setattr(image_detector_module, "predict", _boom)
    frames = [np.zeros((10, 10, 3), dtype=np.uint8) for _ in range(8)]

    with pytest.raises(ModelLoadError):
        predict_video(frames, max_frames=8)

    assert call_count["n"] == 1  # did not retry on every remaining frame


def test_temporal_consistency_signal_is_zero_for_perfectly_stable_predictions():
    video_result = VideoDetectionResult(
        mean_fake_probability=0.2,
        mean_real_probability=0.8,
        std_fake_probability=0.0,
        frame_results=[],
        num_frames_analyzed=5,
        model_id="test/model",
    )
    signal = temporal_consistency_signal(video_result)

    assert signal.applicable is True
    assert signal.suspicion_score == pytest.approx(0.0, abs=1e-6)


def test_temporal_consistency_signal_increases_with_higher_std():
    stable = VideoDetectionResult(
        mean_fake_probability=0.2, mean_real_probability=0.8, std_fake_probability=0.02,
        frame_results=[], num_frames_analyzed=5, model_id="test/model",
    )
    jittery = VideoDetectionResult(
        mean_fake_probability=0.2, mean_real_probability=0.8, std_fake_probability=0.3,
        frame_results=[], num_frames_analyzed=5, model_id="test/model",
    )

    assert temporal_consistency_signal(stable).suspicion_score < temporal_consistency_signal(jittery).suspicion_score
    assert temporal_consistency_signal(jittery).suspicion_score == pytest.approx(1.0, abs=1e-6)  # clipped at max
