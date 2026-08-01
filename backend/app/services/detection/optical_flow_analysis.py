"""
Optical-flow-based motion-anomaly detection (Milestone 11, video only).

Classical technique (no pretrained weights): dense optical flow between
consecutive frames should vary smoothly under natural motion. Face-swap
compositing/warping can introduce a locally rough or discontinuous flow
field right around a manipulated region, since blended content moves
slightly differently from its surroundings frame to frame.

We measure this as the mean spatial "roughness" (Laplacian variance) of the
flow-magnitude field across consecutive frame pairs — the same idea as
frequency_analysis.py's spectral "bumpiness", applied to motion instead of
pixel intensity.

HONEST LIMITATION (found during testing, not glossed over): this metric
responds to *any* spatial complexity in the flow field, including entirely
natural causes — a moving subject's silhouette against a static background
produces real flow discontinuities at its edges via ordinary occlusion, with
no manipulation involved. In testing, a synthetic scene with structured
foreground objects under simple translation scored *higher* roughness than
independent random noise. We could not construct a synthetic scenario that
cleanly isolates "manipulation-like" discontinuity from ordinary scene
structure; that would need labeled real-vs-manipulated video data, which is
out of scope here. Treat this as a weak, noisy signal — it carries the
smallest fusion weight of any signal for that reason — not a standalone
detector. See docs/models.md.

suspicion_score is a heuristic, not a calibrated classifier — see
types.py::ForensicSignal.
"""

import cv2
import numpy as np

from app.services.detection.types import ForensicSignal

# Roughness values at/above this are treated as maximally suspicious.
# Heuristic, illustrative, NOT calibrated against labeled video data --
# see the limitation note above.
_MAX_EXPECTED_ROUGHNESS = 1.0


def _flow_roughness(prev_gray: np.ndarray, curr_gray: np.ndarray) -> float:
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray,
        curr_gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )
    magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    laplacian = cv2.Laplacian(magnitude, cv2.CV_32F)
    return float(laplacian.var())


def analyze_optical_flow(frames_bgr: list[np.ndarray]) -> ForensicSignal:
    if len(frames_bgr) < 2:
        return ForensicSignal(
            name="optical_flow_analysis",
            applicable=False,
            suspicion_score=None,
            summary="Optical flow analysis needs 2+ frames (video); not applicable to a single image.",
            details={"frame_count": len(frames_bgr)},
        )

    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames_bgr]
    roughness_values = [_flow_roughness(grays[i], grays[i + 1]) for i in range(len(grays) - 1)]

    mean_roughness = float(np.mean(roughness_values))
    suspicion_score = float(np.clip(mean_roughness / _MAX_EXPECTED_ROUGHNESS, 0.0, 1.0))

    return ForensicSignal(
        name="optical_flow_analysis",
        applicable=True,
        suspicion_score=suspicion_score,
        summary=(
            f"Optical flow roughness (motion-field discontinuity) across {len(roughness_values)} "
            f"frame pair(s): mean={mean_roughness:.2f}."
        ),
        details={"mean_roughness": mean_roughness, "frame_pairs_used": len(roughness_values)},
    )
