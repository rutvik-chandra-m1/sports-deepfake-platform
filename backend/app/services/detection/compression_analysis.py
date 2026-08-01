"""
Compression-artifact analysis via Error Level Analysis (ELA).

Classical forensic technique (no pretrained weights): re-encode the image at
a fixed JPEG quality and measure the per-pixel difference from the original.
Regions that were compressed differently in the past (e.g. a face spliced in
from a different source, or re-saved at a different quality) tend to show a
different error level than the rest of an otherwise-uniform photo. We
measure this as the block-wise coefficient of variation of the ELA map --
authentic, uniformly-compressed photos tend to have fairly even ELA
response; spliced/manipulated regions stand out as outlier blocks.

suspicion_score is a heuristic derived from this variation, not a trained/
calibrated classifier -- see types.py::ForensicSignal.
"""

import cv2
import numpy as np

from app.services.detection.types import ForensicSignal

DEFAULT_JPEG_QUALITY = 90
DEFAULT_BLOCK_SIZE = 16


def _error_level_map(image_bgr: np.ndarray, quality: int = DEFAULT_JPEG_QUALITY) -> np.ndarray:
    success, encoded = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not success:
        raise ValueError("Could not JPEG-encode image for ELA")

    recompressed = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    diff = cv2.absdiff(image_bgr, recompressed).astype(np.float32)
    return diff.mean(axis=2)  # average over channels -> HxW error-level map


def _block_means(ela_map: np.ndarray, block_size: int = DEFAULT_BLOCK_SIZE) -> np.ndarray:
    height, width = ela_map.shape
    h_blocks, w_blocks = height // block_size, width // block_size
    if h_blocks == 0 or w_blocks == 0:
        return np.array([ela_map.mean()])

    trimmed = ela_map[: h_blocks * block_size, : w_blocks * block_size]
    reshaped = trimmed.reshape(h_blocks, block_size, w_blocks, block_size)
    return reshaped.mean(axis=(1, 3)).ravel()


def analyze_compression(
    image_bgr: np.ndarray, quality: int = DEFAULT_JPEG_QUALITY, block_size: int = DEFAULT_BLOCK_SIZE
) -> ForensicSignal:
    ela_map = _error_level_map(image_bgr, quality=quality)
    block_means = _block_means(ela_map, block_size=block_size)

    mean_error_level = float(ela_map.mean())
    block_mean = float(block_means.mean())
    block_std = float(block_means.std())
    coefficient_of_variation = block_std / (block_mean + 1e-6)

    suspicion_score = float(
        np.clip(coefficient_of_variation / (coefficient_of_variation + 1.0), 0.0, 1.0)
    )

    return ForensicSignal(
        name="compression_analysis",
        applicable=True,
        suspicion_score=suspicion_score,
        summary=(
            f"Error Level Analysis: mean error level={mean_error_level:.2f}, "
            f"block-wise coefficient of variation={coefficient_of_variation:.3f}."
        ),
        details={
            "mean_error_level": mean_error_level,
            "block_mean": block_mean,
            "block_std": block_std,
            "coefficient_of_variation": coefficient_of_variation,
        },
    )
