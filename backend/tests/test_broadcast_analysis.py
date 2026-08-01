import cv2
import numpy as np

from app.services.sports_intel.broadcast_analysis import analyze_broadcast_overlay


def test_uniformly_compressed_image_scores_low():
    rng = np.random.default_rng(0)
    base = rng.integers(100, 156, (256, 256, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", base, [cv2.IMWRITE_JPEG_QUALITY, 95])
    uniform = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

    result = analyze_broadcast_overlay(uniform)

    assert result.applicable is True
    assert result.suspicion_score < 0.1


def test_differently_compressed_border_scores_higher():
    rng = np.random.default_rng(0)
    base = rng.integers(100, 156, (256, 256, 3), dtype=np.uint8)
    ok, encoded_uniform = cv2.imencode(".jpg", base, [cv2.IMWRITE_JPEG_QUALITY, 95])
    uniform = cv2.imdecode(encoded_uniform, cv2.IMREAD_COLOR)

    edited = uniform.copy()
    border_patch = base[0:40, :]
    ok, encoded_patch = cv2.imencode(".jpg", border_patch, [cv2.IMWRITE_JPEG_QUALITY, 15])
    patch_recompressed = cv2.imdecode(encoded_patch, cv2.IMREAD_COLOR)
    edited[0:40, :] = patch_recompressed

    uniform_result = analyze_broadcast_overlay(uniform)
    edited_result = analyze_broadcast_overlay(edited)

    assert edited_result.suspicion_score > uniform_result.suspicion_score


def test_returns_valid_signal_shape():
    rng = np.random.default_rng(1)
    image = rng.integers(0, 255, (128, 128, 3), dtype=np.uint8)

    result = analyze_broadcast_overlay(image)

    assert result.name == "broadcast_overlay_analysis"
    assert result.applicable is True
    assert set(result.details.keys()) >= {"border_mean", "center_mean", "diff_ratio"}
