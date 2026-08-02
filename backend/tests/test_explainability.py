from app.models.analysis import RiskLevel, Verdict
from app.services.explainability.reasoning import _band, _headline, generate_report


def _breakdown(signals: dict[str, float], unavailable: dict[str, str] | None = None) -> dict:
    return {
        "signals": {name: {"score": score, "weight": 1.0 / len(signals)} for name, score in signals.items()},
        "unavailable": unavailable or {},
    }


# ----- Band thresholds -----


def test_band_classifies_high_low_medium():
    assert _band(0.9) == "high"
    assert _band(0.1) == "low"
    assert _band(0.5) == "medium"


# ----- Headline -----


def test_headline_suspicious_high_risk():
    assert "LIKELY AI GENERATED" in _headline(Verdict.SUSPICIOUS, RiskLevel.HIGH)


def test_headline_suspicious_medium_risk():
    headline = _headline(Verdict.SUSPICIOUS, RiskLevel.MEDIUM)
    assert "SUSPICIOUS" in headline


def test_headline_authentic_low_risk():
    headline = _headline(Verdict.AUTHENTIC, RiskLevel.LOW)
    assert "AUTHENTIC" in headline
    assert "no significant signs" in headline.lower()


def test_headline_authentic_medium_risk():
    headline = _headline(Verdict.AUTHENTIC, RiskLevel.MEDIUM)
    assert "LIKELY AUTHENTIC" in headline


# ----- Reason generation -----


def test_high_scores_produce_concerning_reasons_for_suspicious_verdict():
    breakdown = _breakdown(
        {
            "deep_learning": 0.9,
            "frequency_analysis": 0.8,
            "compression_analysis": 0.8,
            "lighting_analysis": 0.8,
            "landmark_instability": 0.8,
        }
    )
    report = generate_report(breakdown, Verdict.SUSPICIOUS, RiskLevel.HIGH)

    assert len(report.reasons) == 5
    assert any("deepfake classifier flagged" in r for r in report.reasons)
    assert any("Frequency-domain irregularities detected" in r for r in report.reasons)
    assert any("Inconsistent compression" in r for r in report.reasons)
    assert any("don't match the surrounding scene" in r for r in report.reasons)
    assert any("abnormal frame-to-frame instability" in r for r in report.reasons)


def test_low_scores_produce_reassuring_reasons_for_authentic_verdict():
    breakdown = _breakdown(
        {
            "deep_learning": 0.05,
            "frequency_analysis": 0.1,
            "compression_analysis": 0.1,
            "lighting_analysis": 0.1,
            "landmark_instability": 0.1,
        }
    )
    report = generate_report(breakdown, Verdict.AUTHENTIC, RiskLevel.LOW)

    assert any("found no signs of synthetic generation" in r for r in report.reasons)
    assert any("No frequency-domain irregularities" in r for r in report.reasons)
    assert any("consistent throughout" in r for r in report.reasons)
    assert any("match the surrounding scene" in r for r in report.reasons)
    assert any("remained stable" in r for r in report.reasons)


def test_medium_scores_are_excluded_from_reasons():
    breakdown = _breakdown({"frequency_analysis": 0.5})
    report = generate_report(breakdown, Verdict.AUTHENTIC, RiskLevel.MEDIUM)

    assert "Frequency-domain" not in "".join(report.reasons)


def test_falls_back_to_default_message_when_nothing_conclusive():
    breakdown = _breakdown({"frequency_analysis": 0.5, "compression_analysis": 0.45})
    report = generate_report(breakdown, Verdict.AUTHENTIC, RiskLevel.MEDIUM)

    assert len(report.reasons) == 1
    assert "combined weighted signal" in report.reasons[0]


def test_unavailable_signals_produce_notes():
    breakdown = _breakdown(
        {"frequency_analysis": 0.2},
        unavailable={
            "deep_learning": "model unavailable",
            "landmark_instability": "needs 2+ frames",
        },
    )
    report = generate_report(breakdown, Verdict.AUTHENTIC, RiskLevel.LOW)

    assert len(report.notes) == 2
    # "General deepfake classifier" -- the stock face-tuned model. Renamed from
    # "AI deepfake classifier" when R6 added `trained_probe`, so the two are
    # distinguishable to a reader of the report.
    assert any("General deepfake classifier was unavailable" in n for n in report.notes)
    assert any("2+ frames" in n for n in report.notes)


def test_trained_probe_has_its_own_reason_and_note_templates():
    """The trained probe carries most of the fusion weight, so it must never
    fall through to the generic "<name> was unavailable" fallback."""
    high = _breakdown({"trained_probe": 0.9})
    assert any("trained image classifier" in r for r in generate_report(high, Verdict.SUSPICIOUS, RiskLevel.HIGH).reasons)

    low = _breakdown({"trained_probe": 0.05})
    assert any("trained image classifier" in r for r in generate_report(low, Verdict.AUTHENTIC, RiskLevel.LOW).reasons)

    missing = _breakdown({"frequency_analysis": 0.2}, unavailable={"trained_probe": "head not found"})
    notes = generate_report(missing, Verdict.AUTHENTIC, RiskLevel.LOW).notes
    assert any("trained classifier was unavailable" in n and "low confidence" in n for n in notes)


def test_reasons_ordered_by_noteworthiness_times_weight():
    breakdown = {
        "signals": {
            "deep_learning": {"score": 0.95, "weight": 0.5},  # very extreme, heavily weighted -> first
            "frequency_analysis": {"score": 0.7, "weight": 0.125},  # moderately extreme, low weight
        },
        "unavailable": {},
    }
    report = generate_report(breakdown, Verdict.SUSPICIOUS, RiskLevel.HIGH)

    assert "deepfake classifier flagged" in report.reasons[0]


def test_render_omits_empty_sections():
    breakdown = _breakdown({"frequency_analysis": 0.9})
    report = generate_report(breakdown, Verdict.SUSPICIOUS, RiskLevel.HIGH)
    rendered = report.render()

    assert "Reasons:" in rendered
    assert "Notes:" not in rendered  # no unavailable signals -> no Notes section


def test_render_includes_notes_when_present():
    breakdown = _breakdown({"frequency_analysis": 0.9}, unavailable={"deep_learning": "unavailable"})
    report = generate_report(breakdown, Verdict.SUSPICIOUS, RiskLevel.HIGH)
    rendered = report.render()

    assert "Notes:" in rendered
