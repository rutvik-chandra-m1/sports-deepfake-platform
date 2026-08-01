"""
Frequency-domain (FFT) forensic analysis.

Classical signal-processing technique (no pretrained weights): natural
photographs tend to have 2D power spectra that decay smoothly and roughly
monotonically with radial frequency. GAN/upsampling-based image synthesis
often leaves periodic or plateaued artifacts in the frequency domain (see
e.g. Durall et al., "Watch your Up-Convolution: CNN Based Generative Deep
Neural Networks are Failing to Reproduce Spectral Distributions", CVPR 2020,
and related "unmasking deepfakes with simple features" work).

suspicion_score here is a heuristic derived from the *shape* of the radial
power spectrum, not a trained/calibrated classifier — see
app/services/detection/types.py::ForensicSignal for what that does and
doesn't mean.
"""

import cv2
import numpy as np

from app.services.detection.types import ForensicSignal

DEFAULT_NUM_BINS = 50


def _radial_profile(magnitude: np.ndarray, num_bins: int = DEFAULT_NUM_BINS) -> np.ndarray:
    """Azimuthally averages a 2D magnitude spectrum into a 1D profile vs radius."""
    height, width = magnitude.shape
    cy, cx = height / 2, width / 2
    y_idx, x_idx = np.indices((height, width))
    radius = np.sqrt((x_idx - cx) ** 2 + (y_idx - cy) ** 2)

    max_radius = radius.max() or 1.0
    bin_edges = np.linspace(0, max_radius, num_bins + 1)
    bin_indices = np.clip(np.digitize(radius.ravel(), bin_edges) - 1, 0, num_bins - 1)

    sums = np.bincount(bin_indices, weights=magnitude.ravel(), minlength=num_bins)
    counts = np.bincount(bin_indices, minlength=num_bins)
    counts[counts == 0] = 1
    return sums / counts


def analyze_frequency(image_bgr: np.ndarray, num_bins: int = DEFAULT_NUM_BINS) -> ForensicSignal:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

    spectrum = np.fft.fftshift(np.fft.fft2(gray))
    magnitude = np.log1p(np.abs(spectrum))

    profile = _radial_profile(magnitude, num_bins=num_bins)

    quarter = max(num_bins // 4, 1)
    low = profile[:quarter]
    mid = profile[quarter : 2 * quarter]
    high = profile[2 * quarter :]
    total_energy = float(profile.sum()) or 1e-8
    high_freq_ratio = float(high.sum()) / total_energy

    # "Bumpiness": natural spectra decay smoothly, so their discrete second
    # derivative w.r.t. radius stays small and fairly flat. Periodic/grid
    # artifacts show up as sharp local peaks -> large second-derivative
    # variance relative to the profile's own scale.
    second_derivative = np.diff(profile, n=2)
    bumpiness = float(np.std(second_derivative) / (np.mean(np.abs(profile)) + 1e-8))

    # Squash bumpiness (unbounded, image-dependent scale) into [0, 1] with a
    # simple saturating curve. Not fit against labeled data -- a coarse
    # heuristic indicator, documented as such in docs/models.md.
    suspicion_score = float(np.clip(bumpiness / (bumpiness + 1.0), 0.0, 1.0))

    return ForensicSignal(
        name="frequency_analysis",
        applicable=True,
        suspicion_score=suspicion_score,
        summary=(
            f"Frequency-domain analysis: spectral bumpiness={bumpiness:.3f}, "
            f"high-frequency energy ratio={high_freq_ratio:.3f}."
        ),
        details={
            "bumpiness": bumpiness,
            "high_freq_ratio": high_freq_ratio,
            "low_freq_energy": float(low.sum()),
            "mid_freq_energy": float(mid.sum()),
            "high_freq_energy": float(high.sum()),
        },
    )
