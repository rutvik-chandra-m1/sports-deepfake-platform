import json
import pathlib

import numpy as np
import pytest
from fastapi.testclient import TestClient

import app.core.security as security_module
import app.services.analysis_pipeline as pipeline_module
import app.services.storage_service as storage_service_module
from app.core.config import Settings
from app.db.session import SessionLocal
from app.models.analysis import Analysis, AnalysisStatus, MediaType
from app.services.detection.image_detector import ModelLoadError
from app.services.detection.types import ForensicSignal, ImageDetectionResult
from app.services.media_processing import ExtractedFrame, MediaMetadata, MediaReadError, ProcessedMedia

_TEST_UPLOAD_DIR: str | None = None


@pytest.fixture(autouse=True)
def _isolated_upload_dir(tmp_path, monkeypatch):
    global _TEST_UPLOAD_DIR
    test_settings = Settings(upload_dir=str(tmp_path))
    monkeypatch.setattr(storage_service_module, "get_settings", lambda: test_settings)
    # The pipeline's path-containment check (R9) reads settings through
    # app.core.security, so it must see the same temp upload root -- otherwise
    # every test record's path is (correctly) rejected as being outside it.
    monkeypatch.setattr(security_module, "get_settings", lambda: test_settings)
    _TEST_UPLOAD_DIR = str(tmp_path)
    yield
    _TEST_UPLOAD_DIR = None


def _create_pending_record(file_path: str | None = None) -> int:
    """Records default to a path INSIDE the temp upload dir; anything outside
    is rejected by the containment check before analysis begins."""
    if file_path is None:
        file_path = str(pathlib.Path(_TEST_UPLOAD_DIR) / "does-not-matter.jpg")
    db = SessionLocal()
    try:
        record = Analysis(
            filename="test.jpg",
            media_type=MediaType.IMAGE,
            file_path=file_path,
            status=AnalysisStatus.PENDING,
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        return record.id
    finally:
        db.close()


def _get_record(analysis_id: int) -> Analysis:
    db = SessionLocal()
    try:
        record = db.get(Analysis, analysis_id)
        db.expunge(record)
        return record
    finally:
        db.close()


def _fake_processed_media() -> ProcessedMedia:
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    return ProcessedMedia(
        metadata=MediaMetadata(width=10, height=10),
        frames=[ExtractedFrame(index=0, timestamp_seconds=0.0, image=frame)],
    )


def _forensic_signals(score: float) -> list[ForensicSignal]:
    return [
        ForensicSignal(name="frequency_analysis", applicable=True, suspicion_score=score, summary="x"),
        ForensicSignal(name="compression_analysis", applicable=True, suspicion_score=score, summary="x"),
        ForensicSignal(name="lighting_analysis", applicable=True, suspicion_score=score, summary="x"),
        ForensicSignal(name="landmark_instability", applicable=False, suspicion_score=None, summary="n/a"),
    ]


def _mock_probe(monkeypatch, score: float | None) -> None:
    """Pin the trained probe's output.

    These control-flow tests mock every other detector, and the probe carries
    the largest fusion weight -- left unmocked it runs the real model against a
    10x10 synthetic frame and legitimately outvotes the mocked signals, so the
    test would be asserting on an unpinned input. `score=None` simulates an
    untrained/unavailable probe.
    """
    signal = (
        ForensicSignal(name="trained_probe", applicable=True, suspicion_score=score, summary="mocked probe")
        if score is not None
        else ForensicSignal(
            name="trained_probe", applicable=False, suspicion_score=None, summary="probe unavailable (mocked)"
        )
    )
    monkeypatch.setattr(pipeline_module, "analyze_trained_probe", lambda frame_rgb: signal)


# ----- Fast, fully mocked control-flow tests -----


def test_pipeline_marks_completed_with_verdict_on_success(monkeypatch):
    analysis_id = _create_pending_record()

    monkeypatch.setattr(pipeline_module, "process_media", lambda path, media_type: _fake_processed_media())
    _mock_probe(monkeypatch, 0.9)
    monkeypatch.setattr(
        pipeline_module,
        "predict_image",
        lambda frame: ImageDetectionResult(
            real_probability=0.1, fake_probability=0.9, predicted_label="fake", model_id="test/model"
        ),
    )
    monkeypatch.setattr(pipeline_module, "run_forensic_analysis", lambda frames: _forensic_signals(0.8))
    monkeypatch.setattr(pipeline_module, "run_sports_intelligence", lambda frames: [])

    pipeline_module.run_analysis_pipeline(analysis_id)
    record = _get_record(analysis_id)

    assert record.status == AnalysisStatus.COMPLETED
    assert record.verdict.value == "suspicious"
    assert record.confidence_score is not None
    assert record.risk_level is not None
    assert record.detector_breakdown is not None
    json.loads(record.detector_breakdown)  # must be valid JSON
    assert record.processing_duration_ms is not None
    assert record.completed_at is not None


def test_pipeline_marks_failed_on_media_read_error(monkeypatch):
    analysis_id = _create_pending_record()

    def _boom(path, media_type):
        raise MediaReadError("simulated: corrupt file")

    monkeypatch.setattr(pipeline_module, "process_media", _boom)

    pipeline_module.run_analysis_pipeline(analysis_id)
    record = _get_record(analysis_id)

    assert record.status == AnalysisStatus.FAILED
    assert "Analysis failed" in record.explanation
    assert record.completed_at is not None


def test_pipeline_marks_failed_on_unexpected_exception(monkeypatch):
    analysis_id = _create_pending_record()

    def _boom(path, media_type):
        raise RuntimeError("totally unexpected")

    monkeypatch.setattr(pipeline_module, "process_media", _boom)

    pipeline_module.run_analysis_pipeline(analysis_id)
    record = _get_record(analysis_id)

    assert record.status == AnalysisStatus.FAILED
    assert "unexpectedly" in record.explanation


def test_pipeline_handles_missing_record_gracefully():
    pipeline_module.run_analysis_pipeline(999_999)  # must not raise


def test_pipeline_degrades_gracefully_when_dl_detector_unavailable(monkeypatch):
    analysis_id = _create_pending_record()

    monkeypatch.setattr(pipeline_module, "process_media", lambda path, media_type: _fake_processed_media())

    def _dl_boom(frame):
        raise ModelLoadError("simulated: no internet")

    _mock_probe(monkeypatch, None)
    monkeypatch.setattr(pipeline_module, "predict_image", _dl_boom)
    monkeypatch.setattr(pipeline_module, "run_forensic_analysis", lambda frames: _forensic_signals(0.1))
    monkeypatch.setattr(pipeline_module, "run_sports_intelligence", lambda frames: [])

    pipeline_module.run_analysis_pipeline(analysis_id)
    record = _get_record(analysis_id)

    # Missing DL signal must NOT fail the whole analysis -- forensic-only result instead.
    assert record.status == AnalysisStatus.COMPLETED
    assert record.verdict.value == "authentic"
    breakdown = json.loads(record.detector_breakdown)
    assert "deep_learning" in breakdown["unavailable"]


def _fake_video_processed_media(num_frames: int = 4) -> ProcessedMedia:
    frames = [
        ExtractedFrame(index=i, timestamp_seconds=i * 0.5, image=np.zeros((10, 10, 3), dtype=np.uint8))
        for i in range(num_frames)
    ]
    return ProcessedMedia(metadata=MediaMetadata(width=10, height=10, frame_count=num_frames), frames=frames)


def test_pipeline_uses_predict_video_for_multi_frame_media(monkeypatch):
    from app.services.detection.types import VideoDetectionResult

    analysis_id = _create_pending_record()

    monkeypatch.setattr(
        pipeline_module, "process_media", lambda path, media_type: _fake_video_processed_media(4)
    )
    monkeypatch.setattr(pipeline_module, "run_forensic_analysis", lambda frames: _forensic_signals(0.1))
    monkeypatch.setattr(pipeline_module, "run_sports_intelligence", lambda frames: [])

    predict_video_calls = {"count": 0}

    def _fake_predict_video(frames, model_id=None):
        predict_video_calls["count"] += 1
        return VideoDetectionResult(
            mean_fake_probability=0.15,
            mean_real_probability=0.85,
            std_fake_probability=0.02,
            frame_results=[],
            num_frames_analyzed=len(frames),
            model_id="test/model",
        )

    monkeypatch.setattr(pipeline_module, "predict_video", _fake_predict_video)

    pipeline_module.run_analysis_pipeline(analysis_id)
    record = _get_record(analysis_id)

    assert predict_video_calls["count"] == 1
    assert record.status == AnalysisStatus.COMPLETED
    breakdown = json.loads(record.detector_breakdown)
    assert "temporal_consistency" in breakdown["signals"]
    assert breakdown["signals"]["deep_learning"]["score"] == pytest.approx(0.15, abs=1e-6)


def test_pipeline_marks_temporal_consistency_not_applicable_for_single_image(monkeypatch):
    analysis_id = _create_pending_record()

    monkeypatch.setattr(pipeline_module, "process_media", lambda path, media_type: _fake_processed_media())
    monkeypatch.setattr(
        pipeline_module,
        "predict_image",
        lambda frame: ImageDetectionResult(
            real_probability=0.9, fake_probability=0.1, predicted_label="real", model_id="test/model"
        ),
    )
    monkeypatch.setattr(pipeline_module, "run_forensic_analysis", lambda frames: _forensic_signals(0.1))
    monkeypatch.setattr(pipeline_module, "run_sports_intelligence", lambda frames: [])

    pipeline_module.run_analysis_pipeline(analysis_id)
    record = _get_record(analysis_id)

    breakdown = json.loads(record.detector_breakdown)
    assert "temporal_consistency" in breakdown["unavailable"]


# ----- One real, slow, end-to-end test (no mocking) via the actual API -----


def test_real_upload_completes_end_to_end_via_api(client: TestClient):
    import cv2

    image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    files = {"file": ("photo.jpg", encoded.tobytes(), "image/jpeg")}

    upload_response = client.post("/api/v1/media/upload", files=files)
    assert upload_response.status_code == 201
    analysis_id = upload_response.json()["id"]

    result = client.get(f"/api/v1/analysis/{analysis_id}")
    body = result.json()

    # TestClient runs BackgroundTasks synchronously before the response
    # returns, so by the time we GET it here, the pipeline has already run.
    assert body["status"] == "completed"
    assert body["verdict"] in ("authentic", "suspicious")
    assert body["confidence_score"] is not None
    assert body["risk_level"] is not None
    assert body["detector_breakdown"] is not None
    json.loads(body["detector_breakdown"])


def test_manual_run_endpoint_reprocesses_an_existing_record(client: TestClient):
    """Reprocessing goes through a real upload.

    Rewritten for R9: this used to POST a client-chosen `file_path`, which was
    the arbitrary-file-read vulnerability itself. `file_path` is now
    server-assigned only, so the legitimate route to a reprocessable record is
    to upload one.
    """
    import cv2

    image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    upload = client.post(
        "/api/v1/media/upload",
        files={"file": ("manual.jpg", encoded.tobytes(), "image/jpeg")},
    )
    assert upload.status_code == 201
    analysis_id = upload.json()["id"]

    run_response = client.post(f"/api/v1/analysis/{analysis_id}/run")
    assert run_response.status_code == 202

    final = client.get(f"/api/v1/analysis/{analysis_id}").json()
    assert final["status"] == "completed"
    assert final["verdict"] in ("authentic", "suspicious")


def test_manual_run_endpoint_404s_for_missing_record(client: TestClient):
    response = client.post("/api/v1/analysis/999999/run")
    assert response.status_code == 404


def test_manual_run_endpoint_400s_when_no_file_path(client: TestClient):
    create_response = client.post(
        "/api/v1/analysis", json={"filename": "no_file.jpg", "media_type": "image"}
    )
    analysis_id = create_response.json()["id"]

    response = client.post(f"/api/v1/analysis/{analysis_id}/run")
    assert response.status_code == 400
