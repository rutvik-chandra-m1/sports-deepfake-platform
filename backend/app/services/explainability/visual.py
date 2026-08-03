"""
Visual explanation: attention-rollout heatmaps over the ViT backbone (R7).

Closes the gap `docs/architecture.md` had claimed as done for months --
"highlighted-frame evidence" that did not exist. The explainability layer was
text-only.

WHAT THIS SHOWS, AND WHAT IT DOES NOT
-------------------------------------
Attention rollout shows **where the model looked**. It does NOT show **why it
decided**. Those are different claims and conflating them is the standard way
attention visualisations mislead people: a bright region means "this patch
influenced the representation", not "this patch is fake". The UI must label it
accordingly, and `_DISCLAIMER` below is returned with every heatmap so the
caption cannot drift away from the maths.

METHOD
------
Attention rollout (Abnar & Zuidema, *Quantifying Attention Flow in
Transformers*, ACL 2020):

1. take the per-layer attention matrices, averaged over heads
2. add the identity to account for residual connections, and renormalise --
   without this, rollout attributes everything to the attention path and
   ignores the residual stream that carries most of the signal
3. multiply the layers together to get end-to-end token-to-token flow
4. read the CLS row: how much each image patch contributed to the summary
   vector the classifier head actually consumes

Step 4 matters for correctness here: our trained probe reads the CLS
embedding, so the CLS row is genuinely the right thing to attribute -- not an
arbitrary choice.
"""

import logging
import threading

import numpy as np

logger = logging.getLogger(__name__)

_DISCLAIMER = (
    "Shows where the model looked, not why it decided. Bright regions influenced the "
    "model's summary of the image; they are not, on their own, evidence of manipulation."
)


class VisualExplanationError(Exception):
    """Raised when a heatmap cannot be produced."""


_lock = threading.Lock()
_eager_model = None
_eager_processor = None


def _load_eager_model():
    """Loads a SEPARATE model instance using eager attention.

    Why a second instance instead of reusing the cached one: transformers
    defaults to SDPA (fused scaled-dot-product attention), which never
    materialises the attention matrices -- `output_attentions=True` silently
    returns nothing, which is exactly how this first failed. Only the eager
    implementation exposes them.

    The alternatives were worse. Flipping the shared model's
    `_attn_implementation` mutates state used by every concurrent inference
    request and would slow the hot path for everyone. Loading the whole
    backbone in eager mode would tax every prediction to benefit the rare
    visualisation request.

    Cost: ~330MB of additional resident memory, and only after someone
    actually requests a heatmap. Weights come from the same local cache, so
    there is no extra download.
    """
    global _eager_model, _eager_processor

    with _lock:
        if _eager_model is not None:
            return _eager_model, _eager_processor

        from transformers import AutoImageProcessor, AutoModelForImageClassification

        from app.core.config import get_settings

        settings = get_settings()
        model_id = settings.deepfake_image_model_id
        logger.info("Loading eager-attention backbone for visual explanations (%s)", model_id)

        processor = AutoImageProcessor.from_pretrained(model_id, cache_dir=settings.models_dir)
        model = AutoModelForImageClassification.from_pretrained(
            model_id, cache_dir=settings.models_dir, attn_implementation="eager"
        )
        model.eval()

        _eager_model, _eager_processor = model, processor
        return model, processor


def compute_attention_rollout(image_rgb_uint8: np.ndarray) -> np.ndarray:
    """
    Returns a (H, W) float array in [0, 1], the same size as the input,
    scoring each pixel by how much its patch fed the CLS summary.
    """
    import torch

    try:
        model, processor = _load_eager_model()
    except Exception as exc:  # noqa: BLE001 - surfaced as a typed error
        raise VisualExplanationError(f"Backbone unavailable: {exc}") from exc

    inputs = processor(images=image_rgb_uint8, return_tensors="pt")
    with torch.no_grad():
        outputs = model.vit(**inputs, output_attentions=True)

    attentions = outputs.attentions
    if not attentions:
        raise VisualExplanationError("Backbone returned no attention maps")

    # (layers, tokens, tokens) after averaging heads.
    rollout = None
    for layer_attention in attentions:
        attention = layer_attention[0].mean(dim=0)  # average heads -> (tokens, tokens)

        # Residual connections carry information around attention; ignoring
        # them makes rollout over-attribute to whatever attention happened to
        # focus on. I + A, renormalised, is the standard correction.
        identity = torch.eye(attention.size(0), dtype=attention.dtype)
        attention = attention + identity
        attention = attention / attention.sum(dim=-1, keepdim=True)

        rollout = attention if rollout is None else attention @ rollout

    # Row 0 is the CLS token: its attention over the patch tokens.
    cls_to_patches = rollout[0, 1:].numpy()

    grid = int(np.sqrt(cls_to_patches.size))
    if grid * grid != cls_to_patches.size:
        raise VisualExplanationError(
            f"Patch count {cls_to_patches.size} is not square; cannot form a heatmap grid"
        )
    heatmap = cls_to_patches.reshape(grid, grid)

    # Normalise to [0, 1] for display. A flat map (no variation) would divide
    # by zero, so guard it.
    spread = heatmap.max() - heatmap.min()
    heatmap = (heatmap - heatmap.min()) / spread if spread > 1e-12 else np.zeros_like(heatmap)

    import cv2

    height, width = image_rgb_uint8.shape[:2]
    upsampled = cv2.resize(
        heatmap.astype(np.float32), (width, height), interpolation=cv2.INTER_CUBIC
    )

    # Clip AFTER resizing. Cubic interpolation overshoots at sharp edges, and
    # measured output ran -0.0125..1.0888 -- outside the [0, 1] this function
    # promises. That is not cosmetic: render_overlay does
    # `(heatmap * 255).astype(np.uint8)`, and 1.0888 * 255 = 277 wraps to 22,
    # so the MOST attended regions would have rendered as dark instead of hot.
    return np.clip(upsampled, 0.0, 1.0)


def render_overlay(image_rgb_uint8: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5) -> bytes:
    """Blends the heatmap over the image and encodes it as PNG bytes."""
    import cv2

    coloured = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    coloured = cv2.cvtColor(coloured, cv2.COLOR_BGR2RGB)

    overlay = (alpha * coloured + (1 - alpha) * image_rgb_uint8).clip(0, 255).astype(np.uint8)

    ok, encoded = cv2.imencode(".png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    if not ok:
        raise VisualExplanationError("Could not encode the overlay image")
    return encoded.tobytes()


def build_attention_overlay(image_rgb_uint8: np.ndarray) -> tuple[bytes, dict]:
    """Returns (png_bytes, metadata). Metadata always carries the disclaimer."""
    heatmap = compute_attention_rollout(image_rgb_uint8)
    png = render_overlay(image_rgb_uint8, heatmap)
    return png, {
        "method": "attention_rollout",
        "reference": "Abnar & Zuidema, Quantifying Attention Flow in Transformers, ACL 2020",
        "disclaimer": _DISCLAIMER,
        "peak_attention_fraction": round(float((heatmap > 0.7).mean()), 4),
    }


def build_signal_visualisation(image_bgr: np.ndarray, signal: str) -> bytes:
    """
    Renders the raw forensic artefact behind a classical signal, so a reader
    can see the evidence rather than only a number.

    These are the intermediate images the detectors compute and previously
    discarded.
    """
    import cv2

    if signal == "compression_analysis":
        # Error Level Analysis: re-encode and amplify the difference.
        from app.services.detection.compression_analysis import _error_level_map

        ela = _error_level_map(image_bgr)
        normalised = cv2.normalize(ela, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        rendered = cv2.applyColorMap(normalised, cv2.COLORMAP_INFERNO)

    elif signal == "frequency_analysis":
        # Log magnitude spectrum, centred: periodic generator artefacts show
        # up as off-centre peaks.
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        spectrum = np.fft.fftshift(np.fft.fft2(gray))
        magnitude = np.log1p(np.abs(spectrum))
        normalised = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        rendered = cv2.applyColorMap(normalised, cv2.COLORMAP_VIRIDIS)

    else:
        raise VisualExplanationError(f"No visualisation available for signal '{signal}'")

    ok, encoded = cv2.imencode(".png", rendered)
    if not ok:
        raise VisualExplanationError("Could not encode the visualisation")
    return encoded.tobytes()
