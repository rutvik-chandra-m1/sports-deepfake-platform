"""
Fusion engine — combines the deep learning image detector's output
(Milestone 7), classical forensic signals (Milestone 8, 11), and
sports-specific signals (Milestone 12) into a single verdict/confidence/
risk_level for the Analysis record.

TWO combining strategies, in preference order:

1. LEARNED CALIBRATION (default when available) -- coefficients fitted to
   the labelled val split by ml/train/train_fusion.py, applied by
   app/services/fusion_calibration.py. This is what makes the engine work:
   R3 measured the hand-weighted version at ROC-AUC 0.4331, *below chance*,
   because most classical signals are inverted (they rate real images as
   more suspicious than fakes) and a fixed positive weight cannot express
   that. The fitted coefficients are negative exactly where the ablation
   says they should be. Held-out test (n=275): 0.7715, 95% CI 0.716-0.824.

2. LEGACY WEIGHTED MEAN (fallback) -- three pools with nominal weights
   (settings.fusion_trained_weight / fusion_dl_weight /
   fusion_forensic_weight / fusion_sports_weight) split evenly across each
   pool's members, renormalized across whichever signals are applicable so
   an unavailable signal doesn't leave weight on the floor. Used when no
   calibration file is present, or when a signal the calibration requires is
   unavailable for this particular media. These pool weights are a
   documented design choice informed by the R3 ablation -- the trained probe
   carries most of the weight because it is the only signal measured above
   chance -- not values fitted to data.

`detector_breakdown["fusion"]["method"]` records which path produced any
given verdict, so a stored result is self-explanatory. See
docs/evaluation.md for the full measurement.

Milestone 10 turns the `detector_breakdown` this engine produces into a
natural-language explanation; see app/services/explainability/.
"""

import json
from dataclasses import asdict, dataclass, field

from app.core.config import Settings, get_settings
from app.models.analysis import RiskLevel, Verdict
from app.services import fusion_calibration
from app.services.detection.types import ForensicSignal, ImageDetectionResult

# This project's own trained classifier (R6). Weighted as its own pool, not
# lumped in with the classical heuristics: it is the only signal here fitted
# to this project's labelled data and measured on a held-out split, whereas
# R3 measured every classical signal at or below chance on still images.
TRAINED_SIGNAL_NAMES = ("trained_probe",)

FORENSIC_SIGNAL_NAMES = (
    "frequency_analysis",
    "compression_analysis",
    "lighting_analysis",
    "landmark_instability",
    "optical_flow_analysis",
    "temporal_consistency",
)

SPORTS_SIGNAL_NAMES = (
    "jersey_color_consistency",
    "scene_consistency",
    "broadcast_overlay_analysis",
    "crowd_texture_analysis",
)


class FusionError(Exception):
    """Raised when there are zero applicable signals to base a verdict on."""


@dataclass
class FusionResult:
    verdict: Verdict
    confidence_score: float  # 0-100, confidence in the *stated verdict direction*
    risk_level: RiskLevel
    fused_suspicion_score: float  # 0-1, raw weighted average (kept for transparency/debugging)
    explanation: str
    detector_breakdown: dict = field(default_factory=dict)  # JSON-serializable; stored as-is on the record

    def detector_breakdown_json(self) -> str:
        return json.dumps(self.detector_breakdown)


def _nominal_weights(settings: Settings) -> dict[str, float]:
    weights = {"deep_learning": settings.fusion_dl_weight}

    per_trained = settings.fusion_trained_weight / len(TRAINED_SIGNAL_NAMES)
    for name in TRAINED_SIGNAL_NAMES:
        weights[name] = per_trained

    per_forensic = settings.fusion_forensic_weight / len(FORENSIC_SIGNAL_NAMES)
    for name in FORENSIC_SIGNAL_NAMES:
        weights[name] = per_forensic

    per_sports = settings.fusion_sports_weight / len(SPORTS_SIGNAL_NAMES)
    for name in SPORTS_SIGNAL_NAMES:
        weights[name] = per_sports

    return weights


def _build_explanation(
    fused_score: float,
    verdict: Verdict,
    scores: dict[str, float],
    normalized_weights: dict[str, float],
    unavailable: list[tuple[str, str]],
) -> str:
    lines = [
        f"Verdict: {verdict.value.upper()} (fused suspicion score {fused_score:.2f} on a 0-1 scale).",
        "Contributing signals:",
    ]
    for name, score in sorted(scores.items(), key=lambda kv: -normalized_weights[kv[0]]):
        lines.append(f"  - {name}: score={score:.2f}, weight={normalized_weights[name] * 100:.0f}%")

    if unavailable:
        lines.append("Signals not available for this analysis:")
        for name, reason in unavailable:
            lines.append(f"  - {name}: {reason}")

    return "\n".join(lines)


def fuse(
    dl_result: ImageDetectionResult | None,
    forensic_signals: list[ForensicSignal],
    sports_signals: list[ForensicSignal] | None = None,
) -> FusionResult:
    settings = get_settings()
    nominal = _nominal_weights(settings)
    sports_signals = sports_signals or []

    scores: dict[str, float] = {}
    unavailable: list[tuple[str, str]] = []

    if dl_result is not None:
        scores["deep_learning"] = dl_result.fake_probability
    else:
        unavailable.append(("deep_learning", "model unavailable (see logs / summary on this run)"))

    for signal in [*forensic_signals, *sports_signals]:
        if signal.applicable and signal.suspicion_score is not None:
            scores[signal.name] = signal.suspicion_score
        else:
            unavailable.append((signal.name, signal.summary))

    if not scores:
        raise FusionError(
            "No applicable signals available (DL detector and all forensic/sports detectors "
            "failed/unavailable) -- cannot compute a verdict."
        )

    applicable_nominal = {name: nominal.get(name, 0.0) for name in scores}
    total_nominal = sum(applicable_nominal.values()) or 1.0
    normalized_weights = {name: w / total_nominal for name, w in applicable_nominal.items()}

    # Prefer the learned combiner (weights fitted to labelled data, which can
    # express the negative coefficients the R3 ablation showed are needed).
    # Falls back to the legacy weighted mean if the calibration is missing or
    # any signal it requires is unavailable for this particular media.
    calibrated = fusion_calibration.apply_calibration(scores)
    if calibrated is not None:
        fused_score, verdict_threshold, calibration_detail = calibrated
    else:
        fused_score = sum(scores[name] * normalized_weights[name] for name in scores)
        verdict_threshold = settings.fusion_verdict_threshold
        calibration_detail = {
            "method": "legacy_weighted_mean",
            "reason": "no calibration available, or a required signal was missing",
        }

    verdict = Verdict.SUSPICIOUS if fused_score >= verdict_threshold else Verdict.AUTHENTIC
    confidence_score = round(
        (fused_score if verdict == Verdict.SUSPICIOUS else (1.0 - fused_score)) * 100, 2
    )

    if fused_score < settings.fusion_risk_low_threshold:
        risk_level = RiskLevel.LOW
    elif fused_score < settings.fusion_risk_high_threshold:
        risk_level = RiskLevel.MEDIUM
    else:
        risk_level = RiskLevel.HIGH

    explanation = _build_explanation(fused_score, verdict, scores, normalized_weights, unavailable)

    breakdown = {
        "fused_suspicion_score": fused_score,
        "fusion": calibration_detail,
        "verdict_threshold": verdict_threshold,
        "signals": {
            name: {"score": scores[name], "weight": normalized_weights[name]} for name in scores
        },
        "unavailable": {name: reason for name, reason in unavailable},
        "dl_result": asdict(dl_result) if dl_result is not None else None,
        "forensic_signals": [asdict(s) for s in forensic_signals],
        "sports_signals": [asdict(s) for s in sports_signals],
    }

    return FusionResult(
        verdict=verdict,
        confidence_score=confidence_score,
        risk_level=risk_level,
        fused_suspicion_score=fused_score,
        explanation=explanation,
        detector_breakdown=breakdown,
    )
