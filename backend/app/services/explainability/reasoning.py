"""
Explainability layer (Milestone 10): turns the fusion engine's raw
detector_breakdown (Milestone 9) into a natural-language verdict headline
plus a list of human-readable reasons, e.g.:

    Verdict: LIKELY AI GENERATED / MANIPULATED

    Reasons:
    - Frequency-domain irregularities detected, consistent with GAN/upsampling artifacts.
    - Facial landmarks show abnormal frame-to-frame instability, beyond natural movement.

This module adds no new detection logic — every phrase is templated from
scores the fusion engine already computed. Which signals are "noteworthy"
enough to surface (vs. inconclusive) is controlled by
settings.explanation_low_threshold / explanation_high_threshold, not
hardcoded.
"""

from dataclasses import dataclass, field

from app.core.config import get_settings
from app.models.analysis import RiskLevel, Verdict

# (signal_name -> {band -> phrase}). "high" = concerning/suspicious reading,
# "low" = reassuring reading supporting authenticity. Signals scoring in
# between ("medium") are treated as inconclusive and not surfaced.
_REASON_TEMPLATES: dict[str, dict[str, str]] = {
    "deep_learning": {
        "high": "The AI deepfake classifier flagged this as likely synthetic ({fake_pct:.0f}% fake probability).",
        "low": "The AI deepfake classifier found no signs of synthetic generation ({real_pct:.0f}% real probability).",
    },
    "frequency_analysis": {
        "high": "Frequency-domain irregularities detected, consistent with GAN/upsampling artifacts.",
        "low": "No frequency-domain irregularities detected — the spectral profile looks natural.",
    },
    "compression_analysis": {
        "high": "Inconsistent compression levels detected, suggesting possible splicing or re-editing.",
        "low": "Compression levels are consistent throughout, with no signs of splicing.",
    },
    "lighting_analysis": {
        "high": "Lighting and color balance on the face don't match the surrounding scene.",
        "low": "Lighting and color balance on the face match the surrounding scene.",
    },
    "landmark_instability": {
        "high": "Facial landmarks show abnormal frame-to-frame instability, beyond natural movement.",
        "low": "Facial landmarks remained stable across frames, consistent with natural movement.",
    },
    "optical_flow_analysis": {
        "high": "Motion between frames shows unusual local discontinuities, possibly from blended/composited content.",
        "low": "Motion between frames is spatially coherent, with no unusual local discontinuities.",
    },
    "temporal_consistency": {
        "high": "The AI classifier's confidence fluctuates unusually between frames, suggesting inconsistent content across the clip.",
        "low": "The AI classifier's confidence stayed consistent across frames.",
    },
    "jersey_color_consistency": {
        "high": "Clothing/jersey color varies unusually across the clip, possibly indicating spliced footage.",
        "low": "Clothing/jersey color stayed consistent throughout the clip.",
    },
    "scene_consistency": {
        "high": "The background scene changes abruptly at some point in the clip, possibly indicating spliced-in footage from a different source.",
        "low": "The background scene stayed consistent throughout the clip.",
    },
    "broadcast_overlay_analysis": {
        "high": "Broadcast overlay/graphics regions show different compression than the rest of the frame, possibly indicating an edited scoreboard, logo, or ticker.",
        "low": "Broadcast overlay/graphics regions show consistent compression with the rest of the frame.",
    },
    "crowd_texture_analysis": {
        "high": "A near-identical crowd texture patch was found repeated elsewhere in the frame, possibly indicating copy-pasted crowd padding.",
        "low": "No repeated crowd texture patches were found.",
    },
}

_UNAVAILABLE_NOTE_TEMPLATES: dict[str, str] = {
    "deep_learning": "AI deepfake classifier was unavailable for this run — verdict relies on forensic signals only.",
    "frequency_analysis": "Frequency-domain analysis was unavailable for this run.",
    "compression_analysis": "Compression analysis was unavailable for this run.",
    "lighting_analysis": "Lighting analysis was unavailable for this run.",
    "landmark_instability": (
        "Facial landmark instability could not be assessed (needs 2+ frames with a detected "
        "face, or the landmark model was unavailable)."
    ),
    "optical_flow_analysis": "Optical flow analysis needs 2+ frames (video only) — not applicable here.",
    "temporal_consistency": "Temporal consistency needs 2+ analyzed video frames — not applicable here.",
    "jersey_color_consistency": "Jersey consistency needs 2+ video frames with a detected face — not applicable here.",
    "scene_consistency": "Scene consistency needs 2+ video frames — not applicable here.",
    "broadcast_overlay_analysis": "Broadcast overlay analysis was unavailable for this run.",
    "crowd_texture_analysis": "Crowd texture analysis was unavailable for this run.",
}

_NO_CONCLUSIVE_SIGNAL_MESSAGE = (
    "No individual signal was strongly conclusive on its own; the verdict reflects their "
    "combined weighted signal."
)


@dataclass
class ExplanationReport:
    headline: str
    reasons: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def render(self) -> str:
        lines = [self.headline]
        if self.reasons:
            lines.append("")
            lines.append("Reasons:")
            lines.extend(f"- {r}" for r in self.reasons)
        if self.notes:
            lines.append("")
            lines.append("Notes:")
            lines.extend(f"- {n}" for n in self.notes)
        return "\n".join(lines)


def _headline(verdict: Verdict, risk_level: RiskLevel) -> str:
    if verdict == Verdict.SUSPICIOUS:
        if risk_level == RiskLevel.HIGH:
            return "Verdict: LIKELY AI GENERATED / MANIPULATED"
        return "Verdict: SUSPICIOUS — possible manipulation detected"

    if risk_level == RiskLevel.LOW:
        return "Verdict: AUTHENTIC — no significant signs of manipulation detected"
    return "Verdict: LIKELY AUTHENTIC — minor inconsistencies detected, likely not manipulation"


def _band(score: float) -> str:
    settings = get_settings()
    if score >= settings.explanation_high_threshold:
        return "high"
    if score <= settings.explanation_low_threshold:
        return "low"
    return "medium"


def _format_reason(name: str, score: float, band: str) -> str | None:
    template = _REASON_TEMPLATES.get(name, {}).get(band)
    if template is None:
        return None
    if name == "deep_learning":
        return template.format(fake_pct=score * 100, real_pct=(1 - score) * 100)
    return template


def generate_report(
    detector_breakdown: dict, verdict: Verdict, risk_level: RiskLevel
) -> ExplanationReport:
    signals: dict = detector_breakdown.get("signals", {})
    unavailable: dict = detector_breakdown.get("unavailable", {})

    scored: list[tuple[float, str]] = []
    for name, info in signals.items():
        score = info["score"]
        weight = info.get("weight", 0.0)
        reason = _format_reason(name, score, _band(score))
        if reason is not None:
            # How noteworthy: distance from the inconclusive midpoint,
            # scaled by how much this signal counted toward the verdict --
            # surfaces strong, heavily-weighted signals first.
            noteworthiness = abs(score - 0.5) * weight
            scored.append((noteworthiness, reason))

    scored.sort(key=lambda pair: -pair[0])
    reasons = [reason for _, reason in scored] or [_NO_CONCLUSIVE_SIGNAL_MESSAGE]

    notes = [
        _UNAVAILABLE_NOTE_TEMPLATES.get(name, f"{name} was unavailable for this run.")
        for name in unavailable
    ]

    return ExplanationReport(headline=_headline(verdict, risk_level), reasons=reasons, notes=notes)
