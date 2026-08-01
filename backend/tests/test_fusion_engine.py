import pytest

from app.models.analysis import RiskLevel, Verdict
from app.services.detection.types import ForensicSignal, ImageDetectionResult
from app.services.fusion_engine import FusionError, fuse


def _dl(fake_probability: float) -> ImageDetectionResult:
    return ImageDetectionResult(
        real_probability=1.0 - fake_probability,
        fake_probability=fake_probability,
        predicted_label="fake" if fake_probability >= 0.5 else "real",
        model_id="test/model",
    )


def _signal(name: str, score: float | None, applicable: bool = True) -> ForensicSignal:
    return ForensicSignal(
        name=name,
        applicable=applicable,
        suspicion_score=score,
        summary=f"{name} test signal",
    )


ALL_FORENSIC_LOW = [
    _signal("frequency_analysis", 0.1),
    _signal("compression_analysis", 0.1),
    _signal("lighting_analysis", 0.1),
    _signal("landmark_instability", 0.1),
]

ALL_FORENSIC_HIGH = [
    _signal("frequency_analysis", 0.9),
    _signal("compression_analysis", 0.9),
    _signal("lighting_analysis", 0.9),
    _signal("landmark_instability", 0.9),
]


def test_all_signals_low_yields_authentic_verdict():
    result = fuse(_dl(0.05), ALL_FORENSIC_LOW)

    assert result.verdict == Verdict.AUTHENTIC
    assert result.risk_level == RiskLevel.LOW
    assert result.confidence_score > 50


def test_all_signals_high_yields_suspicious_verdict():
    result = fuse(_dl(0.95), ALL_FORENSIC_HIGH)

    assert result.verdict == Verdict.SUSPICIOUS
    assert result.risk_level == RiskLevel.HIGH
    assert result.confidence_score > 50


def test_dl_weighted_more_heavily_than_any_single_forensic_signal():
    # DL says very suspicious, forensic signals mildly disagree -- with DL
    # at nominal weight 0.5 vs 0.125 each, the fused score should still tilt
    # toward the DL result rather than the forensic majority.
    mostly_authentic_forensic = [
        _signal("frequency_analysis", 0.2),
        _signal("compression_analysis", 0.2),
        _signal("lighting_analysis", 0.2),
        _signal("landmark_instability", 0.2),
    ]
    result = fuse(_dl(0.9), mostly_authentic_forensic)

    assert result.fused_suspicion_score > 0.5  # DL's high score still wins out
    assert result.verdict == Verdict.SUSPICIOUS


def test_missing_dl_renormalizes_weight_across_forensic_signals():
    result_with_dl = fuse(_dl(0.5), ALL_FORENSIC_HIGH)
    result_without_dl = fuse(None, ALL_FORENSIC_HIGH)

    # Without DL, the four uniform forensic scores (all 0.9) alone determine
    # the fused score -- since they're all equal, the fused score should
    # equal that shared value, regardless of DL's presence/absence.
    assert result_without_dl.fused_suspicion_score == pytest.approx(0.9, abs=1e-6)
    assert "deep_learning" in dict(
        (k, v) for k, v in result_without_dl.detector_breakdown["unavailable"].items()
    )
    assert result_with_dl.fused_suspicion_score != result_without_dl.fused_suspicion_score


def test_missing_forensic_signals_still_produce_a_result():
    unavailable_signals = [
        _signal("frequency_analysis", None, applicable=False),
        _signal("compression_analysis", None, applicable=False),
        _signal("lighting_analysis", None, applicable=False),
        _signal("landmark_instability", None, applicable=False),
    ]
    result = fuse(_dl(0.8), unavailable_signals)

    # Only DL is applicable -> its renormalized weight is 1.0 -> fused score == DL's score
    assert result.fused_suspicion_score == pytest.approx(0.8, abs=1e-6)
    assert len(result.detector_breakdown["unavailable"]) == 4


def test_raises_fusion_error_when_nothing_is_applicable():
    unavailable_signals = [
        _signal("frequency_analysis", None, applicable=False),
        _signal("compression_analysis", None, applicable=False),
        _signal("lighting_analysis", None, applicable=False),
        _signal("landmark_instability", None, applicable=False),
    ]
    with pytest.raises(FusionError):
        fuse(None, unavailable_signals)


def test_confidence_score_reflects_distance_from_threshold():
    very_authentic = fuse(_dl(0.02), ALL_FORENSIC_LOW)
    borderline = fuse(_dl(0.48), ALL_FORENSIC_LOW)

    assert very_authentic.confidence_score > borderline.confidence_score


def test_detector_breakdown_is_json_serializable():
    result = fuse(_dl(0.7), ALL_FORENSIC_HIGH)
    serialized = result.detector_breakdown_json()

    assert "fused_suspicion_score" in serialized
    assert "deep_learning" in serialized
