"""
Fusion engine — combines the deep learning image detector's output
(Milestone 7), classical forensic signals (Milestone 8, 11), and
sports-specific signals (Milestone 12) into a single verdict/confidence/
risk_level for the Analysis record.

Weighting scheme: three pools, each with a nominal top-level weight
(settings.fusion_dl_weight / fusion_forensic_weight / fusion_sports_weight,
default 0.4 / 0.35 / 0.25) split evenly across that pool's member signals.
DL is weighted highest as the only signal here with a published, cited
evaluation (see docs/models.md); sports signals are weighted lowest as the
most experimental/heuristic pool. Whichever signals are actually applicable
(not None, no missing-model/network failure) have their nominal weights
renormalized to sum to 1 -- so an unavailable signal doesn't leave weight
on the floor; the rest simply account for 100% of the decision.

This weighting is a documented design choice, not a value fit/calibrated
against a labeled validation set -- see docs/models.md for the honesty
notes that apply to every heuristic score feeding into this engine.

Milestone 10 turns the `detector_breakdown` this engine produces into a
natural-language explanation; see app/services/explainability/.
"""

import json
from dataclasses import asdict, dataclass, field

from app.core.config import get_settings
from app.models.analysis import RiskLevel, Verdict
from app.services.detection.types import ForensicSignal, ImageDetectionResult

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


def _nominal_weights(settings) -> dict[str, float]:
    weights = {"deep_learning": settings.fusion_dl_weight}

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

    fused_score = sum(scores[name] * normalized_weights[name] for name in scores)

    verdict = Verdict.SUSPICIOUS if fused_score >= settings.fusion_verdict_threshold else Verdict.AUTHENTIC
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
