"""
Deep learning image-level deepfake detector.

Model: prithivMLmods/Deep-Fake-Detector-v2-Model — a Vision Transformer
(base architecture google/vit-base-patch16-224-in21k) fine-tuned by its
author for binary Realism/Deepfake classification. Apache-2.0 licensed.

Its model card reports (on the author's own held-out test set, 56,001
images) — we cite this as *their* reported evaluation, not one we've
independently reproduced:

    accuracy      0.9212
    Realism       precision 0.9683  recall 0.8708  f1 0.9170
    Deepfake      precision 0.8826  recall 0.9715  f1 0.9249

See docs/models.md for the full citation, license, and limitations
(notably: trained on face imagery, so results on non-face sports content
like broadcast graphics or crowd shots should be treated as low-confidence
until the forensic + fusion layers (Milestones 8-9) add corroborating or
contradicting signal).

IMPORTANT — network requirement: weights (~330MB) download from Hugging
Face Hub on first use and are cached under settings.models_dir. This
requires outbound internet access at runtime; it is not vendored in the repo.
"""

import logging
import threading
from typing import Any

import numpy as np

from app.core.config import get_settings
from app.services.detection.types import ForensicSignal, ImageDetectionResult, VideoDetectionResult
from app.services.media_processing.frame_extractor import evenly_spaced_indices

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "prithivMLmods/Deep-Fake-Detector-v2-Model"
DEFAULT_MAX_VIDEO_FRAMES = 8

_lock = threading.Lock()
_loaded_model_id: str | None = None
_model: Any = None
_processor: Any = None


class ModelLoadError(Exception):
    """Raised when the pretrained model/processor can't be loaded (e.g. no
    internet on first run) or returns an unrecognized label set."""


def _load_model(model_id: str) -> tuple[Any, Any]:
    """Loads (and caches in-process) the model + processor for `model_id`.
    Safe to call repeatedly — only downloads/loads once per model_id per process."""
    global _model, _processor, _loaded_model_id

    with _lock:
        if _model is not None and _loaded_model_id == model_id:
            return _model, _processor

        # Imported lazily so importing this module doesn't require torch/transformers
        # to already be installed at import time (helps unit tests stay fast/optional).
        from transformers import AutoImageProcessor, AutoModelForImageClassification

        settings = get_settings()
        logger.info("Loading deepfake image detector '%s' (cache: %s)", model_id, settings.models_dir)

        processor = AutoImageProcessor.from_pretrained(model_id, cache_dir=settings.models_dir)
        model = AutoModelForImageClassification.from_pretrained(model_id, cache_dir=settings.models_dir)
        model.eval()

        _model, _processor, _loaded_model_id = model, processor, model_id
        return model, processor


def _run_inference(
    model: Any, processor: Any, image_rgb_uint8: np.ndarray, model_id: str
) -> ImageDetectionResult:
    """The actual forward pass + label normalization — separated from
    `predict()` so tests can exercise it against a small locally-constructed
    model without needing network access or real pretrained weights."""
    # torch imported lazily, matching the transformers import below: a
    # module-level `import torch` meant that importing ANY part of the app --
    # the API, the schemas, the migrations -- dragged in ~2GB of ML stack, so
    # nothing could be tested or started without it. Verified: `import
    # app.main` no longer loads torch.
    import torch

    inputs = processor(images=image_rgb_uint8, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]

    id2label = model.config.id2label
    label_probs = {id2label[i].lower(): probs[i].item() for i in range(len(probs))}

    # Normalize whatever the model calls its classes ("Realism"/"Deepfake",
    # "real"/"fake", etc. and in either index order) to a stable pair, so
    # callers never need to know this model's specific label wording.
    fake_probability = next(
        (v for k, v in label_probs.items() if "fake" in k), None
    )
    real_probability = next((v for k, v in label_probs.items() if "real" in k), None)

    if fake_probability is None or real_probability is None:
        raise ModelLoadError(
            f"Unrecognized label set from model '{model_id}': {list(id2label.values())}. "
            "Expected labels containing 'real' and 'fake'."
        )

    predicted_label = "fake" if fake_probability >= real_probability else "real"

    return ImageDetectionResult(
        real_probability=real_probability,
        fake_probability=fake_probability,
        predicted_label=predicted_label,
        model_id=model_id,
        raw_labels=label_probs,
    )


def predict(
    image_rgb_uint8: np.ndarray, model_id: str | None = None
) -> ImageDetectionResult:
    """
    Runs the deepfake image classifier on a single RGB frame.

    image_rgb_uint8: HxWx3 RGB uint8 array — e.g. an
    app.services.media_processing.ExtractedFrame.image converted from BGR
    to RGB. Pass the *raw* frame, not the generic [0,1]-normalized output of
    preprocess_frame(); the HF processor does its own resizing/normalization
    tuned to this specific model.
    """
    resolved_model_id = model_id or get_settings().deepfake_image_model_id

    try:
        model, processor = _load_model(resolved_model_id)
    except ModelLoadError:
        raise
    except Exception as exc:  # noqa: BLE001 - re-raised as a typed, catchable error
        raise ModelLoadError(
            f"Could not load deepfake detection model '{resolved_model_id}'. "
            "This requires internet access on first run to download weights from "
            f"Hugging Face Hub. Original error: {type(exc).__name__}: {exc}"
        ) from exc

    return _run_inference(model, processor, image_rgb_uint8, resolved_model_id)


def predict_video(
    frames_rgb: list[np.ndarray],
    model_id: str | None = None,
    max_frames: int = DEFAULT_MAX_VIDEO_FRAMES,
) -> VideoDetectionResult:
    """
    Runs the same per-frame classifier across several evenly-spaced sampled
    frames of a video (Milestone 11), rather than judging a whole video from
    frame 0 alone. Aggregates into a mean fake-probability (the video-level
    DL signal for the fusion engine) and a std-dev across frames (the
    temporal_consistency signal below).

    Fails fast: if the model can't be loaded, the first frame's predict()
    call raises immediately rather than retrying the (slow, doomed) load on
    every subsequent frame.
    """
    resolved_model_id = model_id or get_settings().deepfake_image_model_id

    indices = evenly_spaced_indices(len(frames_rgb), min(max_frames, len(frames_rgb)))
    sampled = [frames_rgb[i] for i in indices]

    frame_results = [predict(frame, model_id=resolved_model_id) for frame in sampled]

    fake_probs = np.array([r.fake_probability for r in frame_results])
    real_probs = np.array([r.real_probability for r in frame_results])

    return VideoDetectionResult(
        mean_fake_probability=float(fake_probs.mean()),
        mean_real_probability=float(real_probs.mean()),
        std_fake_probability=float(fake_probs.std()),
        frame_results=frame_results,
        num_frames_analyzed=len(frame_results),
        model_id=resolved_model_id,
    )


# A frame-to-frame fake-probability std of this magnitude or more is treated
# as maximally suspicious for the temporal_consistency signal below.
# Heuristic, illustrative -- not calibrated against labeled video data.
_MAX_EXPECTED_STD = 0.25


def temporal_consistency_signal(video_result: VideoDetectionResult) -> ForensicSignal:
    """
    A genuine, unedited clip of one identity should get fairly consistent
    fake-probability across frames; large frame-to-frame swings can indicate
    localized/frame-level tampering (e.g. splicing between sources, or
    reenactment artifacts that don't appear in every frame) rather than one
    consistent identity throughout.
    """
    std = video_result.std_fake_probability
    suspicion_score = float(np.clip(std / _MAX_EXPECTED_STD, 0.0, 1.0))

    return ForensicSignal(
        name="temporal_consistency",
        applicable=True,
        suspicion_score=suspicion_score,
        summary=(
            f"DL prediction variability across {video_result.num_frames_analyzed} frames: "
            f"std={std:.3f} (mean fake probability={video_result.mean_fake_probability:.2f})."
        ),
        details={
            "std_fake_probability": std,
            "mean_fake_probability": video_result.mean_fake_probability,
            "num_frames_analyzed": video_result.num_frames_analyzed,
        },
    )
