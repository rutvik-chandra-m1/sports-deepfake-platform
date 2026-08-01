import numpy as np

from app.services.detection.frequency_analysis import analyze_frequency


def test_white_noise_has_low_bumpiness_and_score():
    rng = np.random.default_rng(0)
    noise = rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)

    result = analyze_frequency(noise)

    assert result.applicable is True
    assert 0.0 <= result.suspicion_score <= 1.0
    assert result.details["bumpiness"] < 0.05


def test_strong_periodic_grating_has_much_higher_bumpiness_than_noise():
    """A strong single-frequency sinusoidal grating concentrates energy at
    one exact radius in the power spectrum -- a stand-in for the kind of
    periodic upsampling/checkerboard artifact this detector targets."""
    rng = np.random.default_rng(0)
    noise = rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)

    size = 256
    freq = 20
    xx, _ = np.meshgrid(np.arange(size), np.arange(size))
    sinusoid = (np.sin(2 * np.pi * freq * xx / size) * 100 + 128).astype(np.uint8)
    grating = np.repeat(sinusoid[:, :, None], 3, axis=2)

    noise_result = analyze_frequency(noise)
    grating_result = analyze_frequency(grating)

    assert grating_result.details["bumpiness"] > noise_result.details["bumpiness"] * 5
    assert grating_result.suspicion_score > noise_result.suspicion_score


def test_returns_valid_signal_shape():
    rng = np.random.default_rng(1)
    image = rng.integers(0, 255, (128, 96, 3), dtype=np.uint8)

    result = analyze_frequency(image)

    assert result.name == "frequency_analysis"
    assert result.applicable is True
    assert set(result.details.keys()) >= {
        "bumpiness",
        "high_freq_ratio",
        "low_freq_energy",
        "mid_freq_energy",
        "high_freq_energy",
    }
