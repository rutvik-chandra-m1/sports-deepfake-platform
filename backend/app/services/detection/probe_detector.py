"""
Trained linear-probe detector -- this project's own transfer-learning
classifier, and the replacement for relying on the stock face-tuned head.

Background: `image_detector.py` runs a third-party ViT whose classification
head was fine-tuned on *faces*. R3 measured that head at ROC-AUC 0.491 --
chance -- on general and sports imagery, exactly the out-of-distribution
failure `docs/models.md` warned about. The ViT *trunk* underneath is a
general-purpose backbone (google/vit-base-patch16-224-in21k), so this module
keeps the trunk, discards the mismatched head, and applies a linear head
trained on this project's own labelled dataset instead.

The head is a plain JSON file (`models/configs/probe_head.json`) written by
`ml/train/train_probe.py`: standardisation statistics, a weight vector and a
bias. Applying it is one dot product, so the backend needs no scikit-learn
dependency and never unpickles a model file from disk -- deliberate, since
pickle deserialisation is arbitrary code execution.

Degrades exactly like every other detector: if the head file is missing or
the backbone cannot load, this returns a non-applicable ForensicSignal
rather than raising, and the fusion engine simply proceeds without it.
"""

import json
import logging
import threading
from pathlib import Path
from typing import Any

import numpy as np

from app.core.config import get_settings
from app.services.detection.types import ForensicSignal

logger = logging.getLogger(__name__)

PROBE_SIGNAL_NAME = "trained_probe"

_lock = threading.Lock()
_head: dict | None = None
_head_missing_logged = False


class ProbeUnavailableError(Exception):
    """Raised when the trained head is absent or malformed."""


def _head_path() -> Path:
    settings = get_settings()
    # Sits alongside the tracked model configs, not the gitignored weights
    # cache -- it is a small text artefact and belongs in version control.
    return Path(settings.models_dir).parent / "configs" / "probe_head.json"


def load_head() -> dict:
    """Loads and caches the exported probe head. Raises ProbeUnavailableError
    if it has not been trained yet."""
    global _head, _head_missing_logged

    with _lock:
        if _head is not None:
            return _head

        path = _head_path()
        if not path.exists():
            if not _head_missing_logged:
                logger.info(
                    "No trained probe head at %s -- run ml/train/train_probe.py to enable "
                    "the trained_probe signal. Pipeline continues without it.", path,
                )
                _head_missing_logged = True
            raise ProbeUnavailableError(f"probe head not found at {path}")

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            head = {
                "mean": np.asarray(data["mean"], dtype=np.float64),
                "std": np.asarray(data["std"], dtype=np.float64),
                "weights": np.asarray(data["weights"], dtype=np.float64),
                "bias": float(data["bias"]),
                "backbone_model_id": data.get("backbone_model_id", ""),
                "operating_threshold": float(data.get("operating_threshold", 0.5)),
                "feature_dim": int(data.get("feature_dim", len(data["weights"]))),
            }
        except (KeyError, ValueError, TypeError) as exc:
            raise ProbeUnavailableError(f"probe head at {path} is malformed: {exc}") from exc

        _head = head
        logger.info(
            "Loaded trained probe head (dim=%d, backbone=%s, threshold=%.3f)",
            head["feature_dim"], head["backbone_model_id"], head["operating_threshold"],
        )
        return head


def embed(image_rgb_uint8: np.ndarray, model: Any = None, processor: Any = None) -> np.ndarray:
    """CLS-token embedding from the ViT trunk. Same backbone (and same cached
    weights) the image detector already loads, so this adds no extra download."""
    import torch

    from app.services.detection.image_detector import _load_model

    if model is None or processor is None:
        model, processor = _load_model(get_settings().deepfake_image_model_id)

    inputs = processor(images=image_rgb_uint8, return_tensors="pt")
    with torch.no_grad():
        outputs = model.vit(**inputs)
        return outputs.last_hidden_state[:, 0].squeeze(0).numpy().astype(np.float64)


def predict_probability(image_rgb_uint8: np.ndarray) -> float:
    """Probability that the image is AI-generated, per the trained head."""
    head = load_head()
    embedding = embed(image_rgb_uint8)
    if embedding.shape[0] != head["weights"].shape[0]:
        raise ProbeUnavailableError(
            f"embedding dim {embedding.shape[0]} != head dim {head['weights'].shape[0]} "
            "-- the head was trained against a different backbone"
        )
    standardized = (embedding - head["mean"]) / head["std"]
    logit = float(standardized @ head["weights"] + head["bias"])
    return float(1.0 / (1.0 + np.exp(-logit)))


def analyze_trained_probe(image_rgb_uint8: np.ndarray) -> ForensicSignal:
    """ForensicSignal wrapper so the fusion engine consumes this uniformly
    with every other detector."""
    try:
        probability = predict_probability(image_rgb_uint8)
    except ProbeUnavailableError as exc:
        return ForensicSignal(
            name=PROBE_SIGNAL_NAME,
            applicable=False,
            suspicion_score=None,
            summary=f"Trained probe unavailable: {exc}",
            details={"error": str(exc)},
        )

    head = load_head()
    return ForensicSignal(
        name=PROBE_SIGNAL_NAME,
        applicable=True,
        suspicion_score=probability,
        summary=(
            f"Trained linear probe on this project's dataset: {probability * 100:.1f}% "
            f"probability AI-generated (decision threshold {head['operating_threshold']:.2f})."
        ),
        details={
            "probability_fake": probability,
            "operating_threshold": head["operating_threshold"],
            "backbone": head["backbone_model_id"],
        },
    )
