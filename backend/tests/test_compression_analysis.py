import cv2
import numpy as np

from app.services.detection.compression_analysis import analyze_compression


def test_uniformly_compressed_image_has_low_variation():
    rng = np.random.default_rng(0)
    base = rng.integers(100, 156, (256, 256, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", base, [cv2.IMWRITE_JPEG_QUALITY, 95])
    assert ok
    uniform_img = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

    result = analyze_compression(uniform_img)

    assert result.applicable is True
    assert 0.0 <= result.suspicion_score <= 1.0
    assert result.details["coefficient_of_variation"] < 0.15


def test_spliced_region_with_different_compression_has_higher_variation():
    rng = np.random.default_rng(0)
    base = rng.integers(100, 156, (256, 256, 3), dtype=np.uint8)
    ok, encoded_uniform = cv2.imencode(".jpg", base, [cv2.IMWRITE_JPEG_QUALITY, 95])
    uniform_img = cv2.imdecode(encoded_uniform, cv2.IMREAD_COLOR)

    spliced = uniform_img.copy()
    patch = base[64:160, 64:160]
    ok, encoded_patch = cv2.imencode(".jpg", patch, [cv2.IMWRITE_JPEG_QUALITY, 20])
    patch_recompressed = cv2.imdecode(encoded_patch, cv2.IMREAD_COLOR)
    spliced[64:160, 64:160] = patch_recompressed

    uniform_result = analyze_compression(uniform_img)
    spliced_result = analyze_compression(spliced)

    assert spliced_result.details["coefficient_of_variation"] > (
        uniform_result.details["coefficient_of_variation"] * 3
    )
    assert spliced_result.suspicion_score > uniform_result.suspicion_score


def test_returns_valid_signal_shape():
    rng = np.random.default_rng(1)
    image = rng.integers(0, 255, (128, 96, 3), dtype=np.uint8)

    result = analyze_compression(image)

    assert result.name == "compression_analysis"
    assert result.applicable is True
    assert set(result.details.keys()) >= {
        "mean_error_level",
        "block_mean",
        "block_std",
        "coefficient_of_variation",
    }
