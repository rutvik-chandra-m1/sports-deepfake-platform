"""
Crowd-texture duplication check (Milestone 12).

A cheap way to fake a "packed stadium" is to copy-paste a patch of crowd
texture to fill empty seating areas. This looks for suspiciously repeated
patches by comparing small tiles across the upper portion of the frame (a
coarse proxy for "stands" in typical stadium framing) via cosine
similarity — genuine crowd texture looks locally repetitive but isn't
usually *identical* patch-for-patch; a strong near-exact match between two
well-separated tiles is a duplication tell.

Runs on a single representative frame — no temporal requirement, works for
images and video alike.

HONEST LIMITATION (found during testing): tiles are compared on a fixed,
non-overlapping grid, so this only catches duplicates that happen to land
on matching grid positions — a duplicate patch shifted by a few pixels from
any grid line will be missed even though a human (or a sliding-window/
keypoint-based copy-move detector) would catch it. A more robust version
would use all-offsets sliding-window or keypoint matching (e.g. ORB/SIFT),
which is meaningfully more expensive and out of scope here. Confirmed
empirically: a grid-aligned duplicate scores 1.0 (clean detection); the
same duplicate shifted off-grid was missed entirely in testing.

suspicion_score is a heuristic, not a calibrated classifier — see
detection/types.py::ForensicSignal.
"""

import cv2
import numpy as np

from app.services.detection.types import ForensicSignal

_TILE_SIZE = 24
_UPPER_FRACTION = 0.4  # scan only the upper 40% of the frame (crowd/stands proxy)
_MAX_REGION_WIDTH = 240  # downscale before tiling so cost stays bounded regardless of input resolution
_MIN_SEPARATION_PX = 40  # tiles must be at least this far apart to count as "different locations"
# Cosine similarity above this (in the downscaled-region tile comparison)
# is treated as a likely duplicate; the suspicion score ramps up over the
# narrow band just below "identical".
_DUPLICATE_SIMILARITY_FLOOR = 0.9


def _prepare_region(image_bgr: np.ndarray) -> np.ndarray:
    height, width = image_bgr.shape[:2]
    upper = image_bgr[: int(height * _UPPER_FRACTION), :]

    if upper.shape[1] > _MAX_REGION_WIDTH and upper.shape[1] > 0:
        scale = _MAX_REGION_WIDTH / upper.shape[1]
        new_size = (_MAX_REGION_WIDTH, max(1, int(upper.shape[0] * scale)))
        upper = cv2.resize(upper, new_size, interpolation=cv2.INTER_AREA)

    return cv2.cvtColor(upper, cv2.COLOR_BGR2GRAY)


def _extract_tiles(region_gray: np.ndarray, tile_size: int) -> tuple[np.ndarray, list[tuple[int, int]]]:
    height, width = region_gray.shape
    tiles, positions = [], []
    for y in range(0, height - tile_size, tile_size):
        for x in range(0, width - tile_size, tile_size):
            tiles.append(region_gray[y : y + tile_size, x : x + tile_size])
            positions.append((x, y))
    return np.array(tiles), positions


def analyze_crowd_texture(image_bgr: np.ndarray) -> ForensicSignal:
    gray = _prepare_region(image_bgr)
    tiles, positions = _extract_tiles(gray, _TILE_SIZE)

    if len(tiles) < 4:
        return ForensicSignal(
            name="crowd_texture_analysis",
            applicable=False,
            suspicion_score=None,
            summary="Frame too small to tile for crowd-texture analysis.",
            details={"num_tiles": len(tiles)},
        )

    flat = tiles.reshape(len(tiles), -1).astype(np.float32)
    flat = flat - flat.mean(axis=1, keepdims=True)  # compare texture pattern, not raw brightness
    norms = np.linalg.norm(flat, axis=1, keepdims=True)
    norms[norms == 0] = 1e-6
    normalized = flat / norms
    similarity = normalized @ normalized.T

    best_match = 0.0
    for i in range(len(tiles)):
        for j in range(i + 1, len(tiles)):
            dx = abs(positions[i][0] - positions[j][0])
            dy = abs(positions[i][1] - positions[j][1])
            if dx < _MIN_SEPARATION_PX and dy < _MIN_SEPARATION_PX:
                continue  # too close to be a meaningful "different location" duplicate
            best_match = max(best_match, float(similarity[i, j]))

    suspicion_score = float(
        np.clip((best_match - _DUPLICATE_SIMILARITY_FLOOR) / (1.0 - _DUPLICATE_SIMILARITY_FLOOR), 0.0, 1.0)
    )

    return ForensicSignal(
        name="crowd_texture_analysis",
        applicable=True,
        suspicion_score=suspicion_score,
        summary=(
            f"Highest well-separated tile similarity in the upper frame region: {best_match:.3f} "
            "(1.0 = identical patch found elsewhere)."
        ),
        details={"best_tile_similarity": best_match, "num_tiles": len(tiles)},
    )
