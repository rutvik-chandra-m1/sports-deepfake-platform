"""Deepfake detection pipeline: DL image/video detector (M7, M11) + classical
forensic analysis (M8, M11). Fusion into a single verdict is Milestone 9."""

from app.services.detection.compression_analysis import analyze_compression
from app.services.detection.forensic_analysis import run_forensic_analysis
from app.services.detection.frequency_analysis import analyze_frequency
from app.services.detection.image_detector import (
    ModelLoadError,
    predict as predict_image,
    predict_video,
    temporal_consistency_signal,
)
from app.services.detection.landmark_analysis import analyze_landmark_instability
from app.services.detection.lighting_analysis import analyze_lighting
from app.services.detection.optical_flow_analysis import analyze_optical_flow
from app.services.detection.probe_detector import (
    PROBE_SIGNAL_NAME,
    ProbeUnavailableError,
    analyze_trained_probe,
)
from app.services.detection.types import ForensicSignal, ImageDetectionResult, VideoDetectionResult

__all__ = [
    "predict_image",
    "predict_video",
    "temporal_consistency_signal",
    "ModelLoadError",
    "analyze_trained_probe",
    "ProbeUnavailableError",
    "PROBE_SIGNAL_NAME",
    "analyze_frequency",
    "analyze_compression",
    "analyze_lighting",
    "analyze_landmark_instability",
    "analyze_optical_flow",
    "run_forensic_analysis",
    "ForensicSignal",
    "ImageDetectionResult",
    "VideoDetectionResult",
]
