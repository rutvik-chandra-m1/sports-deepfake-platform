"""
Tests the detector's *integration logic* (label normalization, error
handling, probability math) against a tiny, locally-constructed,
randomly-initialized ViT — this requires no network access and makes no
claim about the real pretrained model's accuracy. Accuracy claims for the
real model (prithivMLmods/Deep-Fake-Detector-v2-Model) are documented in
docs/models.md, cited from its own model card, not tested here.
"""

import numpy as np
import pytest
from transformers import ViTConfig, ViTForImageClassification, ViTImageProcessor

import app.services.detection.image_detector as image_detector_module
from app.services.detection.image_detector import ModelLoadError, _run_inference, predict


def _tiny_vit(id2label: dict[int, str]):
    config = ViTConfig(
        image_size=32,
        patch_size=16,
        num_channels=3,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        num_labels=len(id2label),
    )
    config.id2label = id2label
    config.label2id = {label: idx for idx, label in id2label.items()}

    model = ViTForImageClassification(config)
    model.eval()
    processor = ViTImageProcessor(size={"height": 32, "width": 32})
    return model, processor


def _random_frame():
    return np.random.randint(0, 255, size=(48, 48, 3), dtype=np.uint8)


def test_run_inference_returns_complementary_probabilities():
    model, processor = _tiny_vit({0: "Realism", 1: "Deepfake"})
    result = _run_inference(model, processor, _random_frame(), model_id="test/tiny-vit")

    assert 0.0 <= result.real_probability <= 1.0
    assert 0.0 <= result.fake_probability <= 1.0
    assert result.real_probability == pytest.approx(1.0 - result.fake_probability, abs=1e-4)
    assert result.predicted_label in {"real", "fake"}
    assert result.model_id == "test/tiny-vit"


def test_run_inference_is_independent_of_label_index_order():
    # index 0 = fake this time (opposite of the previous test) — the
    # normalization logic must key off label text, not position.
    model, processor = _tiny_vit({0: "Deepfake", 1: "Realism"})
    result = _run_inference(model, processor, _random_frame(), model_id="test/tiny-vit")

    assert result.predicted_label in {"real", "fake"}
    assert result.real_probability == pytest.approx(1.0 - result.fake_probability, abs=1e-4)


def test_run_inference_raises_for_unrecognized_labels():
    model, processor = _tiny_vit({0: "cat", 1: "dog"})

    with pytest.raises(ModelLoadError, match="Unrecognized label set"):
        _run_inference(model, processor, _random_frame(), model_id="test/tiny-vit")


def test_predict_wraps_load_failures_as_model_load_error(monkeypatch):
    def _boom(model_id):
        raise OSError("simulated: no network / model not cached")

    monkeypatch.setattr(image_detector_module, "_load_model", _boom)

    with pytest.raises(ModelLoadError, match="Could not load deepfake detection model"):
        predict(_random_frame(), model_id="prithivMLmods/Deep-Fake-Detector-v2-Model")
