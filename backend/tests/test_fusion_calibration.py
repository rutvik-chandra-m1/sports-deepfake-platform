"""
Contract tests between the trained calibration artifact and the serving code.

These exist because of a real, silent failure: `train_fusion.py` exported its
first feature as `"probe"` while the fusion engine emits `"trained_probe"`.
`apply_calibration()` therefore never found all its required signals and fell
back to the legacy weighted mean on EVERY request. The learned combiner --
the whole point of R4 -- was never actually used in production, and nothing
failed loudly. Only running the app and inspecting
`detector_breakdown["fusion"]["method"]` revealed it.

The lesson these tests encode: an artifact produced by one process and
consumed by another needs its contract asserted, because a mismatch degrades
silently rather than crashing.
"""

import json
from pathlib import Path

import pytest

from app.services import fusion_calibration
from app.services.detection.probe_detector import PROBE_SIGNAL_NAME
from app.services.fusion_engine import (
    FORENSIC_SIGNAL_NAMES,
    SPORTS_SIGNAL_NAMES,
    TRAINED_SIGNAL_NAMES,
)

CALIBRATION_PATH = (
    Path(__file__).resolve().parent.parent.parent / "models" / "configs" / "fusion_calibration.json"
)

pytestmark = pytest.mark.skipif(
    not CALIBRATION_PATH.exists(),
    reason="no trained calibration present (run ml/train/train_fusion.py)",
)


def _calibration() -> dict:
    return json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))


def test_every_calibration_feature_is_a_real_signal_name():
    """THE regression test. Every feature the calibration expects must be a
    name the fusion engine can actually produce, or the learned combiner
    silently never runs."""
    known = set(TRAINED_SIGNAL_NAMES) | set(FORENSIC_SIGNAL_NAMES) | set(SPORTS_SIGNAL_NAMES) | {
        "deep_learning"
    }
    unknown = [f for f in _calibration()["features"] if f not in known]
    assert not unknown, (
        f"Calibration expects signal(s) the engine never emits: {unknown}.\n"
        f"Engine emits: {sorted(known)}\n"
        "Fix the feature names in ml/train/train_fusion.py and regenerate."
    )


def test_probe_signal_name_matches_between_trainer_and_detector():
    assert PROBE_SIGNAL_NAME in _calibration()["features"]
    assert PROBE_SIGNAL_NAME in TRAINED_SIGNAL_NAMES


def test_weights_align_with_features():
    calibration = _calibration()
    assert len(calibration["features"]) == len(calibration["weights"])


def test_calibration_engages_when_all_signals_present():
    """Given every required signal, the learned path must be taken -- not the
    legacy fallback."""
    scores = {name: 0.5 for name in _calibration()["features"]}
    result = fusion_calibration.apply_calibration(scores)

    assert result is not None, "calibration did not engage despite all signals being present"
    probability, threshold, detail = result
    assert 0.0 <= probability <= 1.0
    assert detail["method"] == "learned_calibration"
    assert set(detail["contributions"]) == set(scores)


def test_calibration_declines_when_a_signal_is_missing():
    """Falling back is correct when evidence is incomplete -- the combiner was
    fitted on all features and cannot be evaluated with a hole in it."""
    features = _calibration()["features"]
    scores = {name: 0.5 for name in features[:-1]}  # drop one
    assert fusion_calibration.apply_calibration(scores) is None


def test_calibration_is_monotonic_in_the_probe():
    """The probe carries a large positive coefficient, so a higher probe score
    must not decrease the fused suspicion."""
    features = _calibration()["features"]
    low = {name: 0.5 for name in features}
    high = dict(low)
    low[PROBE_SIGNAL_NAME] = 0.1
    high[PROBE_SIGNAL_NAME] = 0.9

    low_p, _, _ = fusion_calibration.apply_calibration(low)
    high_p, _, _ = fusion_calibration.apply_calibration(high)
    assert high_p > low_p
