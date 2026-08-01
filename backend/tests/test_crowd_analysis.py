import numpy as np

from app.services.sports_intel.crowd_analysis import analyze_crowd_texture


def test_unique_texture_scores_zero():
    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, (150, 220, 3), dtype=np.uint8)

    result = analyze_crowd_texture(image)

    assert result.applicable is True
    assert result.suspicion_score == 0.0


def test_grid_aligned_duplicate_patch_scores_high():
    """The detector compares tiles on a fixed grid (documented limitation:
    only catches grid-aligned duplicates) -- this test uses a duplicate
    that lands on the grid to verify the core mechanism works correctly."""
    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, (150, 220, 3), dtype=np.uint8)

    duplicated = image.copy()
    duplicated[0:48, 144:192] = image[0:48, 0:48]  # tile_size=24 -> both offsets are grid-aligned

    result = analyze_crowd_texture(duplicated)

    assert result.suspicion_score == 1.0
    assert result.details["best_tile_similarity"] > 0.99


def test_too_small_frame_is_not_applicable():
    tiny = np.zeros((10, 10, 3), dtype=np.uint8)
    result = analyze_crowd_texture(tiny)

    assert result.applicable is False


def test_returns_valid_signal_shape():
    rng = np.random.default_rng(1)
    image = rng.integers(0, 255, (150, 200, 3), dtype=np.uint8)

    result = analyze_crowd_texture(image)

    assert result.name == "crowd_texture_analysis"
    assert set(result.details.keys()) >= {"best_tile_similarity", "num_tiles"}
