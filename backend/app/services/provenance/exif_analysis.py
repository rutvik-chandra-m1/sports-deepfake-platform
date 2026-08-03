"""
EXIF / XMP / PNG-text provenance analysis (R5).

Closes the PPT's "integrate metadata provenance verification" objective,
which had no implementation at all before this.

THE GOVERNING RULE, and the easiest thing to get catastrophically wrong:

    ABSENCE OF METADATA IS NOT EVIDENCE OF MANIPULATION.

Measured on this project's own dataset: 0 of 6 sampled *genuine* OpenFake
photographs carry any EXIF, because social platforms, CDNs and dataset
pipelines routinely strip it -- this project's own `normalize_dataset.py`
strips it too. Scoring "no EXIF" as suspicious would therefore flag ordinary
real photographs, which for a tool that publicly accuses people of faking
images is the worst possible failure mode.

So every signal here is `applicable=False` unless it finds *positive*
evidence one way or the other. A missing signal contributes nothing to the
verdict rather than contributing a made-up 0.5.

What DOES count as positive evidence:
  * an explicit generator marker (Stable Diffusion, Midjourney, DALL-E, ...)
    written by the tool that produced the image
  * IPTC `digitalSourceType = trainedAlgorithmicMedia`, the standard
    machine-readable "this is AI-generated" declaration
  * plausible camera EXIF (Make/Model/exposure), weak evidence of capture
"""

import logging
import re

from PIL import ExifTags, Image

from app.services.detection.types import ForensicSignal

logger = logging.getLogger(__name__)

AI_METADATA_SIGNAL = "provenance_ai_metadata"
CAMERA_METADATA_SIGNAL = "provenance_camera_metadata"

# Substrings that identify a generator when they appear in a Software tag,
# PNG text chunk, or XMP payload. Matched case-insensitively.
_GENERATOR_MARKERS = (
    "stable diffusion", "stablediffusion", "sd-webui", "automatic1111", "comfyui",
    "midjourney", "dall-e", "dalle", "openai", "firefly", "adobe firefly",
    "novelai", "imagen", "flux", "ideogram", "leonardo.ai", "playground ai",
    "craiyon", "dreamstudio", "invokeai", "fooocus", "diffusers",
)

# The IPTC/CAI standard declaration that content is synthetic. Its presence
# is an explicit, machine-readable statement by the producing tool.
_IPTC_SYNTHETIC_MARKERS = (
    "trainedalgorithmicmedia",
    "compositesynthetic",
    "algorithmicmedia",
)

# PNG text keys that generators commonly write.
_GENERATOR_TEXT_KEYS = ("parameters", "prompt", "workflow", "sd-metadata", "software")


# XMP properties that name the producing software. Searching the WHOLE XMP
# payload instead was a real defect: a genuine Nikon D810 photograph was
# flagged as AI-generated because its 12KB of camera settings contained
# `aux:imagenumber="177034"`, and "imagen" is a generator marker. Camera
# settings are not software identifiers; only these fields are.
_SOFTWARE_XMP_PROPERTIES = (
    "xmp:creatortool",
    "tiff:software",
    "dc:creator",
    "photoshop:credit",
    "exif:software",
    "digitalsourcetype",
)


def _extract_software_fields(image: Image.Image, tags: dict) -> list[str]:
    """Only fields that plausibly identify the producing tool."""
    fields: list[str] = []

    for name in ("Software", "ProcessingSoftware", "HostComputer", "Artist"):
        if name in tags:
            fields.append(str(tags[name]))

    info = image.info or {}
    for key, value in info.items():
        key_lower = key.lower()
        if not isinstance(value, (str, bytes)):
            continue
        text = value.decode("utf-8", "ignore") if isinstance(value, bytes) else value

        # PNG text chunks written by generation tools carry the whole prompt
        # and settings; the key alone identifies the tool, so the value is
        # fair game.
        if key_lower in _GENERATOR_TEXT_KEYS:
            fields.append(f"{key} {text}")
            continue

        # XMP: pull out only software-identifying properties, not the whole
        # document.
        if key_lower in ("xmp", "xml:com.adobe.xmp"):
            for prop in _SOFTWARE_XMP_PROPERTIES:
                for match in re.finditer(
                    rf'{re.escape(prop)}\s*=\s*"([^"]*)"', text, flags=re.IGNORECASE
                ):
                    fields.append(match.group(1))
                for match in re.finditer(
                    rf"<{re.escape(prop)}[^>]*>(.*?)</{re.escape(prop)}>",
                    text,
                    flags=re.IGNORECASE | re.DOTALL,
                ):
                    fields.append(match.group(1))

    return fields


def _collect_metadata_text(image: Image.Image) -> tuple[dict, str]:
    """Returns (exif_tags_by_name, lowercase text of SOFTWARE-identifying fields only)."""
    tags: dict = {}
    try:
        exif = image.getexif()
        tags = {ExifTags.TAGS.get(key, str(key)): value for key, value in exif.items()}
    except Exception as exc:  # noqa: BLE001 - malformed EXIF must not break analysis
        logger.debug("EXIF read failed: %s", exc)

    return tags, " ".join(_extract_software_fields(image, tags)).lower()


def _matches_as_token(marker: str, text: str) -> bool:
    """Marker must stand alone, not sit inside a longer word.

    Plain substring matching produced a false positive on a real photograph
    ("imagen" inside "imagenumber"), so a marker must be bounded by something
    that is not a letter or digit. Markers legitimately contain spaces, dots
    and hyphens ("stable diffusion", "dall-e", "leonardo.ai"), so \\b cannot
    be used directly.
    """
    pattern = rf"(?<![a-z0-9]){re.escape(marker)}(?![a-z0-9])"
    return re.search(pattern, text) is not None


def _find_generator(blob: str) -> str | None:
    for marker in _GENERATOR_MARKERS:
        if _matches_as_token(marker, blob):
            return marker
    return None


def _find_iptc_synthetic_declaration(blob: str) -> str | None:
    normalized = re.sub(r"[\s_-]", "", blob)
    for marker in _IPTC_SYNTHETIC_MARKERS:
        if marker in normalized:
            return marker
    return None


def analyze_ai_metadata(image_path: str) -> ForensicSignal:
    """Looks for an explicit declaration that the image was machine-generated.

    Only ever fires POSITIVE. Finding nothing means the tool that made this
    image did not label it -- which is true of most AI images in the wild
    (metadata is trivially stripped) and of every real photograph. That is
    not evidence of anything, so the signal reports not-applicable.
    """
    try:
        with Image.open(image_path) as image:
            _, blob = _collect_metadata_text(image)
    except Exception as exc:  # noqa: BLE001
        return ForensicSignal(
            name=AI_METADATA_SIGNAL,
            applicable=False,
            suspicion_score=None,
            summary=f"Could not read metadata: {type(exc).__name__}",
            details={"error": str(exc)},
        )

    iptc = _find_iptc_synthetic_declaration(blob)
    if iptc:
        return ForensicSignal(
            name=AI_METADATA_SIGNAL,
            applicable=True,
            suspicion_score=1.0,
            summary=(
                "The file's own metadata declares it as AI-generated "
                f"(IPTC digitalSourceType: {iptc})."
            ),
            details={"evidence": "iptc_digital_source_type", "marker": iptc},
        )

    generator = _find_generator(blob)
    if generator:
        return ForensicSignal(
            name=AI_METADATA_SIGNAL,
            applicable=True,
            suspicion_score=0.95,
            summary=f"Metadata names an AI image generator ('{generator}').",
            details={"evidence": "generator_software_tag", "marker": generator},
        )

    return ForensicSignal(
        name=AI_METADATA_SIGNAL,
        applicable=False,
        suspicion_score=None,
        summary=(
            "No AI-generation marker in the metadata. This is NOT evidence the image is "
            "genuine -- such markers are absent from most images and are trivially removed."
        ),
        details={"evidence": "none"},
    )


def analyze_camera_metadata(image_path: str) -> ForensicSignal:
    """Looks for camera-capture EXIF: weak positive evidence of a photograph.

    Weak on purpose, and low-confidence by design: EXIF is trivially forged,
    so its presence raises no bar for a motivated adversary. It is worth a
    small nudge toward "authentic", never a verdict.
    """
    try:
        with Image.open(image_path) as image:
            tags, _ = _collect_metadata_text(image)
    except Exception as exc:  # noqa: BLE001
        return ForensicSignal(
            name=CAMERA_METADATA_SIGNAL,
            applicable=False,
            suspicion_score=None,
            summary=f"Could not read metadata: {type(exc).__name__}",
            details={"error": str(exc)},
        )

    make, model = tags.get("Make"), tags.get("Model")
    capture_fields = [
        field for field in ("ExposureTime", "FNumber", "ISOSpeedRatings", "FocalLength", "DateTimeOriginal")
        if field in tags
    ]

    if not (make or model):
        return ForensicSignal(
            name=CAMERA_METADATA_SIGNAL,
            applicable=False,
            suspicion_score=None,
            summary=(
                "No camera metadata present. Expected for most web images -- platforms strip "
                "EXIF on upload -- so this says nothing about authenticity either way."
            ),
            details={"evidence": "none"},
        )

    # More capture fields -> a more complete, harder-to-fabricate record.
    # Floor at 0.35 rather than 0: EXIF is forgeable, so even a full record
    # is only a nudge.
    score = 0.35 if len(capture_fields) >= 3 else 0.45

    return ForensicSignal(
        name=CAMERA_METADATA_SIGNAL,
        applicable=True,
        suspicion_score=score,
        summary=(
            f"Camera metadata present ({str(make).strip()} {str(model).strip()}".rstrip()
            + f", {len(capture_fields)} capture field(s)). Consistent with a real photograph, "
            "though EXIF can be forged."
        ),
        details={
            "evidence": "camera_exif",
            "make": str(make) if make else "",
            "model": str(model) if model else "",
            "capture_fields": ", ".join(capture_fields),
        },
    )
