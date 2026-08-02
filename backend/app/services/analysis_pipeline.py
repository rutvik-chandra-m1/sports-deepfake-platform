"""
Runs the full analysis pipeline for a single Analysis record: load media ->
extract frames -> DL detector + forensic analysis -> fuse -> persist.

Two layers, deliberately separated:

* `analyze_frames()` -- PURE detection. Takes frames, returns the fused
  verdict + explanation. Knows nothing about the database, HTTP, or the
  Analysis model. This is the layer the offline evaluation harness
  (ml/eval/) calls to score the pipeline over a labelled dataset; welding
  detection to persistence would have made measuring it impossible without
  standing up a database and inventing fake records.
* `run_analysis_pipeline()` -- the FastAPI BackgroundTask entrypoint. Loads
  the record, calls `analyze_frames()`, persists the outcome, and owns all
  the status/error bookkeeping.

The background task executes *after* the request/response cycle, so it opens
its own DB session rather than reusing the request-scoped one from
`get_db()`, which may already be closed.
"""

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone

import cv2

from app.core.security import PathNotAllowedError, assert_within_upload_dir
from app.db.session import SessionLocal
from app.models.analysis import Analysis, AnalysisStatus
from app.services import explainability, fusion_engine
from app.services.detection import (
    ModelLoadError,
    analyze_trained_probe,
    predict_image,
    predict_video,
    run_forensic_analysis,
    temporal_consistency_signal,
)
from app.services.detection.types import ForensicSignal, ImageDetectionResult
from app.services.media_processing import MediaReadError, process_media
from app.services.sports_intel import run_sports_intelligence

logger = logging.getLogger(__name__)


@dataclass
class AnalysisOutcome:
    """Everything `analyze_frames` produces, with no persistence concerns.

    `breakdown` is the JSON-serializable per-signal detail (raw scores,
    weights, availability, plus the rendered explanation) that the API
    stores on `Analysis.detector_breakdown` and the evaluation harness
    reads per-signal scores out of.
    """

    fusion: fusion_engine.FusionResult
    report: explainability.ExplanationReport
    breakdown: dict


def _run_dl_detection(
    frames_bgr: list, analysis_id: int | None = None
) -> tuple[ImageDetectionResult | None, ForensicSignal]:
    """
    Single image -> one predict() call, no temporal signal (not applicable).
    Video (2+ frames) -> predict_video() across sampled frames (Milestone 11);
    its mean becomes the "deep_learning" fusion input and its std becomes the
    temporal_consistency signal.
    """
    frames_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_bgr]

    if len(frames_bgr) == 1:
        not_applicable = ForensicSignal(
            name="temporal_consistency",
            applicable=False,
            suspicion_score=None,
            summary="Temporal consistency needs 2+ frames (video); not applicable to a single image.",
            details={"frame_count": 1},
        )
        try:
            return predict_image(frames_rgb[0]), not_applicable
        except ModelLoadError as exc:
            logger.warning("DL detector unavailable for analysis id=%s: %s", analysis_id, exc)
            return None, not_applicable

    try:
        video_result = predict_video(frames_rgb)
    except ModelLoadError as exc:
        logger.warning("DL detector unavailable for analysis id=%s: %s", analysis_id, exc)
        return None, ForensicSignal(
            name="temporal_consistency",
            applicable=False,
            suspicion_score=None,
            summary=f"Temporal consistency unavailable: {exc}",
            details={"error": str(exc)},
        )

    dl_result = ImageDetectionResult(
        real_probability=video_result.mean_real_probability,
        fake_probability=video_result.mean_fake_probability,
        predicted_label="fake" if video_result.mean_fake_probability >= 0.5 else "real",
        model_id=video_result.model_id,
    )
    return dl_result, temporal_consistency_signal(video_result)


def analyze_frames(frames_bgr: list, analysis_id: int | None = None) -> AnalysisOutcome:
    """
    The whole detection pipeline as a pure function: frames in, verdict out.

    No database, no HTTP, no Analysis model -- so it can be called directly
    by the offline evaluation harness over a labelled dataset, and by tests,
    without any persistence scaffolding.

    `analysis_id` is optional and used only to make log lines traceable back
    to a record when this is called from the background task.

    Raises fusion_engine.FusionError if no signal at all was applicable.
    """
    dl_result, temporal_signal = _run_dl_detection(frames_bgr, analysis_id)

    forensic_signals = run_forensic_analysis(frames_bgr)
    forensic_signals.append(temporal_signal)

    # This project's own trained classifier (R6). Runs on the representative
    # first frame, like the other single-frame signals. Self-degrades to a
    # non-applicable signal if no head has been trained yet.
    first_frame_rgb = cv2.cvtColor(frames_bgr[0], cv2.COLOR_BGR2RGB)
    try:
        forensic_signals.append(analyze_trained_probe(first_frame_rgb))
    except Exception as exc:  # noqa: BLE001 - never let one detector abort the batch
        logger.warning("Trained probe failed for analysis id=%s: %s", analysis_id, exc)

    sports_signals = run_sports_intelligence(frames_bgr)

    result = fusion_engine.fuse(dl_result, forensic_signals, sports_signals)
    report = explainability.generate_report(
        result.detector_breakdown, result.verdict, result.risk_level
    )

    breakdown = dict(result.detector_breakdown)
    breakdown["technical_summary"] = result.explanation
    breakdown["explanation_report"] = report.render()
    breakdown["headline"] = report.headline
    breakdown["reasons"] = report.reasons
    breakdown["notes"] = report.notes

    return AnalysisOutcome(fusion=result, report=report, breakdown=breakdown)


def run_analysis_pipeline(analysis_id: int) -> None:
    db = SessionLocal()
    try:
        record = db.get(Analysis, analysis_id)
        if record is None:
            logger.warning("run_analysis_pipeline: analysis id=%s not found", analysis_id)
            return

        record.status = AnalysisStatus.PROCESSING
        db.commit()

        start = time.monotonic()
        try:
            if not record.file_path:
                raise MediaReadError("Analysis record has no file_path to analyze.")

            # Defence in depth (R9): the schema no longer lets a client set
            # file_path, but this is the function that actually opens the file,
            # so it re-verifies containment rather than trusting that whatever
            # wrote the record did the right thing. A path outside the upload
            # directory fails the analysis instead of being read.
            safe_path = str(assert_within_upload_dir(record.file_path))

            processed = process_media(safe_path, record.media_type)
            frames_bgr = [f.image for f in processed.frames]

            outcome = analyze_frames(frames_bgr, analysis_id)
            result, report, breakdown = outcome.fusion, outcome.report, outcome.breakdown

            record.verdict = result.verdict
            record.confidence_score = result.confidence_score
            record.risk_level = result.risk_level
            record.explanation = report.render()
            record.detector_breakdown = json.dumps(breakdown)
            record.status = AnalysisStatus.COMPLETED

        except PathNotAllowedError as exc:
            logger.error("Analysis id=%s rejected: %s", analysis_id, exc)
            record.status = AnalysisStatus.FAILED
            record.explanation = "Analysis failed: the stored file path is not permitted."
        except (MediaReadError, fusion_engine.FusionError) as exc:
            logger.error("Analysis id=%s failed: %s", analysis_id, exc)
            record.status = AnalysisStatus.FAILED
            record.explanation = f"Analysis failed: {exc}"
        except Exception as exc:  # noqa: BLE001 - safety net: never leave a record stuck in "processing"
            logger.exception("Analysis id=%s failed unexpectedly", analysis_id)
            record.status = AnalysisStatus.FAILED
            record.explanation = f"Analysis failed unexpectedly: {type(exc).__name__}: {exc}"
        finally:
            record.processing_duration_ms = int((time.monotonic() - start) * 1000)
            record.completed_at = datetime.now(timezone.utc)
            db.commit()
    finally:
        db.close()
