"""
Verification report generation (R13).

`reports/` has existed since Milestone 1 with nothing ever writing to it.
A downloadable report is the natural deliverable of a verification platform:
it is the artefact a user actually keeps, forwards, or files.

THE REASON THIS FILE IS CAREFUL
-------------------------------
A polished PDF carries authority that a web page does not. Someone handed a
formatted document saying "SUSPICIOUS" will treat it as a finding, and this
system is wrong roughly 3 times in 10. So the limitations are not an
appendix here -- accuracy, the error rate, and the "not admissible as proof"
statement are rendered on the FIRST page, before the verdict is elaborated.

That is a deliberate product decision, not decoration: the harm case for
this project is a false accusation against a real athlete, and the report is
the most likely thing to travel beyond the person who ran the analysis.
"""

import json
import logging
from datetime import datetime, timezone
from io import BytesIO

from app.models.analysis import Analysis

logger = logging.getLogger(__name__)

# Held-out test performance (docs/evaluation.md). Printed verbatim on every
# report so a reader can weigh the verdict against how often it is wrong.
MEASURED_ACCURACY = "71.3%"
MEASURED_AUC = "0.7402 (95% CI 0.680-0.796)"
MEASURED_N = 275

_LIMITATIONS = [
    f"This system is correct about {MEASURED_ACCURACY} of the time on a held-out test set of "
    f"{MEASURED_N} images. Roughly 3 in 10 verdicts are wrong.",
    "It is a research prototype, not a certified forensic tool. This report is NOT proof of "
    "manipulation and is not admissible as evidence.",
    "Sports-specific performance is effectively unmeasured (only 10 sports images in the test "
    "set). Treat any sports-related verdict with particular caution.",
    "Video analysis is unevaluated -- the measured results above cover still images only.",
    "Absence of metadata or Content Credentials is NOT evidence of manipulation; most genuine "
    "photographs carry neither.",
]


# ReportLab's built-in Helvetica is Latin-1 only, so the Unicode punctuation
# the explanation layer emits (em-dashes, curly quotes) rendered as literal
# replacement characters in the PDF. Mapping to ASCII equivalents is simpler
# and more portable than embedding a TTF, and the meaning is unchanged.
_ASCII_SUBSTITUTIONS = {
    "—": "--", "–": "-", "‘": "'", "’": "'",
    "“": '"', "”": '"', "…": "...", " ": " ",
    "•": "*", "×": "x",
}


def _pdf_safe(text: str) -> str:
    """Render-safe text for the built-in PDF fonts."""
    for source, replacement in _ASCII_SUBSTITUTIONS.items():
        text = text.replace(source, replacement)
    # Anything still outside Latin-1 would render as a black box; drop it
    # rather than ship mojibake in a document a user may forward.
    return text.encode("latin-1", "ignore").decode("latin-1")


class ReportError(Exception):
    """Raised when a report cannot be produced."""


def _verdict_colour(verdict: str | None):
    from reportlab.lib import colors

    if verdict == "suspicious":
        return colors.HexColor("#B3261E")
    if verdict == "authentic":
        return colors.HexColor("#1B6B3A")
    return colors.HexColor("#5F6368")


def build_report_pdf(record: Analysis) -> bytes:
    """Renders a one-file verification report for a completed analysis."""
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        HRFlowable,
        ListFlowable,
        ListItem,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    if record.status is None or record.status.value != "completed":
        raise ReportError("Only completed analyses can be exported.")

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "TitleX", parent=styles["Title"], fontSize=18, spaceAfter=2 * mm, alignment=TA_LEFT
    )
    body = ParagraphStyle("BodyX", parent=styles["BodyText"], fontSize=9.5, leading=14)
    small = ParagraphStyle("SmallX", parent=body, fontSize=8, textColor=colors.HexColor("#5F6368"))
    warn = ParagraphStyle(
        "WarnX", parent=body, fontSize=9, textColor=colors.HexColor("#7A2E1F"), leading=13
    )

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title=f"Verification Report #{record.id}",
        author="Sports Deepfake Detection & Verification Platform",
    )

    story = [
        Paragraph("Media Verification Report", title_style),
        Paragraph(
            "Sports Deepfake Detection &amp; Verification Platform -- research prototype",
            small,
        ),
        Spacer(1, 5 * mm),
    ]

    verdict = record.verdict.value if record.verdict else "unknown"
    risk = record.risk_level.value if record.risk_level else "unknown"
    summary_rows = [
        ["Verdict", verdict.upper()],
        ["Confidence", f"{record.confidence_score:.1f}%" if record.confidence_score else "n/a"],
        ["Risk level", risk],
        ["File", _pdf_safe(record.filename)],
        ["Analysed", record.completed_at.strftime("%Y-%m-%d %H:%M UTC") if record.completed_at else "n/a"],
        ["Record ID", f"#{record.id}"],
    ]
    table = Table(summary_rows, colWidths=[35 * mm, None])
    table.setStyle(
        TableStyle([
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9.5),
            ("TEXTCOLOR", (1, 0), (1, 0), _verdict_colour(record.verdict.value if record.verdict else None)),
            ("FONTNAME", (1, 0), (1, 0), "Helvetica-Bold"),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LINEBELOW", (0, 0), (-1, -2), 0.25, colors.HexColor("#E0E0E0")),
        ])
    )
    story += [table, Spacer(1, 6 * mm)]

    # Limitations FIRST -- see the module docstring. A reader must meet the
    # error rate before they act on the verdict, not after.
    story += [
        Paragraph("<b>Read this before acting on the verdict</b>", body),
        Spacer(1, 2 * mm),
        ListFlowable(
            [ListItem(Paragraph(_pdf_safe(text), warn), leftIndent=8) for text in _LIMITATIONS],
            bulletType="bullet",
            start="•",
        ),
        Spacer(1, 5 * mm),
        HRFlowable(width="100%", color=colors.HexColor("#E0E0E0")),
        Spacer(1, 4 * mm),
    ]

    breakdown = {}
    if record.detector_breakdown:
        try:
            breakdown = json.loads(record.detector_breakdown)
        except (ValueError, TypeError):
            logger.warning("Report %s: detector_breakdown is not valid JSON", record.id)

    reasons = breakdown.get("reasons") or []
    if reasons:
        story += [
            Paragraph("<b>Findings</b>", body),
            Spacer(1, 2 * mm),
            ListFlowable(
                [ListItem(Paragraph(_pdf_safe(r), body), leftIndent=8) for r in reasons],
                bulletType="bullet",
                start="•",
            ),
            Spacer(1, 5 * mm),
        ]

    signals = breakdown.get("signals") or {}
    if signals:
        rows = [["Signal", "Score", "Weight"]] + [
            [name.replace("_", " ").title(), f"{info.get('score', 0):.3f}", f"{info.get('weight', 0):.3f}"]
            for name, info in sorted(signals.items(), key=lambda kv: -kv[1].get("weight", 0))
        ]
        signal_table = Table(rows, colWidths=[80 * mm, 30 * mm, 30 * mm])
        signal_table.setStyle(
            TableStyle([
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F5F5F5")),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#E0E0E0")),
                ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
            ])
        )
        story += [Paragraph("<b>Signal breakdown</b>", body), Spacer(1, 2 * mm), signal_table,
                  Spacer(1, 5 * mm)]

    notes = breakdown.get("notes") or []
    if notes:
        story += [
            Paragraph("<b>Signals not available for this analysis</b>", body),
            Spacer(1, 2 * mm),
            ListFlowable(
                [ListItem(Paragraph(_pdf_safe(n), small), leftIndent=8) for n in notes],
                bulletType="bullet",
                start="•",
            ),
            Spacer(1, 5 * mm),
        ]

    # Provenance of the verdict itself (R11) -- which model and combiner
    # produced it, so the report stays auditable after a retrain.
    story += [
        HRFlowable(width="100%", color=colors.HexColor("#E0E0E0")),
        Spacer(1, 3 * mm),
        Paragraph(
            f"Model: {record.model_version or 'n/a'} &middot; "
            f"Pipeline: {record.pipeline_version or 'n/a'} &middot; "
            f"Fusion: {record.fusion_method or 'n/a'}<br/>"
            f"System performance at time of writing: ROC-AUC {MEASURED_AUC}, "
            f"accuracy {MEASURED_ACCURACY} on {MEASURED_N} held-out images.<br/>"
            f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}.",
            small,
        ),
    ]

    doc.build(story)
    return buffer.getvalue()


def build_report_json(record: Analysis) -> dict:
    """Machine-readable equivalent, for programmatic consumers.

    Carries the same limitations payload as the PDF -- a caller integrating
    this must not be able to receive a verdict without the caveats.
    """
    breakdown = {}
    if record.detector_breakdown:
        try:
            breakdown = json.loads(record.detector_breakdown)
        except (ValueError, TypeError):
            breakdown = {}

    return {
        "record_id": record.id,
        "filename": record.filename,
        "media_type": record.media_type.value if record.media_type else None,
        "verdict": record.verdict.value if record.verdict else None,
        "confidence_score": record.confidence_score,
        "risk_level": record.risk_level.value if record.risk_level else None,
        "analysed_at": record.completed_at.isoformat() if record.completed_at else None,
        "provenance": {
            "model_version": record.model_version,
            "pipeline_version": record.pipeline_version,
            "fusion_method": record.fusion_method,
        },
        "findings": breakdown.get("reasons", []),
        "unavailable_signals": breakdown.get("notes", []),
        "signals": breakdown.get("signals", {}),
        "system_performance": {
            "roc_auc": MEASURED_AUC,
            "accuracy": MEASURED_ACCURACY,
            "test_set_size": MEASURED_N,
        },
        "limitations": _LIMITATIONS,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
