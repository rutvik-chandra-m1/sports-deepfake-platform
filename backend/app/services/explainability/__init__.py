"""Explainability layer (Milestone 10): natural-language reasons generated
from the fusion engine's detector_breakdown (Milestone 9)."""

from app.services.explainability.reasoning import ExplanationReport, generate_report

__all__ = ["ExplanationReport", "generate_report"]
