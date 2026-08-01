from dataclasses import dataclass, field


@dataclass
class ImageDetectionResult:
    real_probability: float
    fake_probability: float
    predicted_label: str  # "real" | "fake"
    model_id: str
    raw_labels: dict[str, float] = field(default_factory=dict)


@dataclass
class VideoDetectionResult:
    """
    Aggregate of running the DL image detector across several sampled
    frames of a video (Milestone 11) — not a separate video-specific model,
    just the same per-frame classifier applied across time and summarized.
    """

    mean_fake_probability: float
    mean_real_probability: float
    std_fake_probability: float  # frame-to-frame variability -> temporal_consistency signal
    frame_results: list[ImageDetectionResult]
    num_frames_analyzed: int
    model_id: str


@dataclass
class ForensicSignal:
    """
    Common output shape for every classical (non-DL) forensic detector, so
    the fusion engine (Milestone 9) can consume them uniformly.

    suspicion_score: 0.0 (looks consistent/authentic by this signal) to 1.0
    (looks manipulated). This is a heuristic derived from classical signal
    processing, NOT a calibrated probability from a trained classifier —
    unlike image_detector.py's model output, there is no accuracy claim to
    cite here. It's one vote for the fusion engine, not a verdict on its own.
    applicable: False when there wasn't enough input to compute a meaningful
    score (e.g. landmark instability needs >=2 frames with a detected face).
    """

    name: str
    applicable: bool
    suspicion_score: float | None  # None when applicable=False
    summary: str
    details: dict[str, float | int | str] = field(default_factory=dict)
