"""
Learned fusion: combines the per-signal scores with weights fitted to
labelled data, instead of the hand-picked constants.

Why this exists. R3 measured the hand-weighted engine at ROC-AUC 0.4331 --
below chance -- and the per-signal ablation showed why: most classical
signals are *inverted* (they score real images as more suspicious than
fakes). A fixed positive weight cannot express that; a fitted coefficient
can, and the fitted ones are negative exactly where the ablation says they
should be.

Measured on the held-out test split (n=275), see docs/evaluation.md:
    hand-weighted (R3)      0.4331
    trained probe alone     0.7534
    learned fusion          0.7715   (95% CI 0.716-0.824)

The calibration is fitted on the VAL split by ml/train/train_fusion.py --
never on train (where the probe's own scores are in-sample and optimistic)
and never on test.

Falls back to the legacy weighted mean whenever any required signal is
missing, so a video without a probe score, or a fresh clone with no
calibration file, still produces a verdict rather than an error.
"""

import json
import logging
import math
import threading
from pathlib import Path

from app.core.config import get_settings

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_calibration: dict | None = None
_load_failed_logged = False


def _calibration_path() -> Path:
    settings = get_settings()
    return Path(settings.models_dir).parent / "configs" / "fusion_calibration.json"


def load_calibration() -> dict | None:
    """Returns the fitted calibration, or None if unavailable (in which case
    callers should use the legacy weighted mean)."""
    global _calibration, _load_failed_logged

    with _lock:
        if _calibration is not None:
            return _calibration

        path = _calibration_path()
        if not path.exists():
            if not _load_failed_logged:
                logger.info(
                    "No fusion calibration at %s -- using legacy weighted mean. "
                    "Run ml/train/train_fusion.py to enable the learned combiner.", path,
                )
                _load_failed_logged = True
            return None

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            calibration = {
                "features": list(data["features"]),
                "weights": [float(w) for w in data["weights"]],
                "intercept": float(data["intercept"]),
                "operating_threshold": float(data.get("operating_threshold", 0.5)),
            }
            if len(calibration["features"]) != len(calibration["weights"]):
                raise ValueError("features/weights length mismatch")
        except (KeyError, ValueError, TypeError) as exc:
            if not _load_failed_logged:
                logger.warning("Fusion calibration at %s is malformed (%s) -- using legacy weights.", path, exc)
                _load_failed_logged = True
            return None

        _calibration = calibration
        logger.info(
            "Loaded learned fusion calibration over %d signals (threshold %.3f)",
            len(calibration["features"]), calibration["operating_threshold"],
        )
        return calibration


def apply_calibration(scores: dict[str, float]) -> tuple[float, float, dict] | None:
    """
    Applies the learned combiner to a name -> score mapping.

    Returns (probability, threshold, detail) or None when the calibration is
    unavailable or any required signal is missing -- the caller then falls
    back to the legacy weighted mean.
    """
    calibration = load_calibration()
    if calibration is None:
        return None

    missing = [name for name in calibration["features"] if name not in scores]
    if missing:
        logger.debug("Learned fusion skipped; missing signals: %s", missing)
        return None

    logit = calibration["intercept"]
    contributions = {}
    for name, weight in zip(calibration["features"], calibration["weights"]):
        contribution = weight * scores[name]
        contributions[name] = round(contribution, 6)
        logit += contribution

    probability = 1.0 / (1.0 + math.exp(-logit))
    detail = {
        "method": "learned_calibration",
        "logit": round(logit, 6),
        "contributions": contributions,
        "weights": dict(zip(calibration["features"], calibration["weights"])),
        "operating_threshold": calibration["operating_threshold"],
    }
    return probability, calibration["operating_threshold"], detail
