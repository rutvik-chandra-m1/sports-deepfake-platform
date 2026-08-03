"""
C2PA (Content Credentials) manifest verification (R5).

C2PA is the Content Authenticity Initiative's standard for cryptographically
signed provenance: a manifest embedded in the file records who created it,
with what tool, and what was done to it, signed so tampering is detectable.
It is the mechanism the PPT's literature survey cites for provenance
verification.

THE SAME GOVERNING RULE AS EXIF, and it matters even more here:

    NO MANIFEST IS NOT EVIDENCE OF MANIPULATION.

The overwhelming majority of images in circulation carry no Content
Credentials at all -- adoption is early, and any re-encode or screenshot
destroys the manifest. "No C2PA" therefore describes almost every genuine
photograph ever taken, and treating it as suspicious would flag nearly
everything. This module reports not-applicable in that case.

What it CAN establish, when a manifest is present:
  * valid signature   -> a verifiable provenance chain (evidence FOR
                         authenticity, and the claim generator may itself
                         declare the content AI-generated)
  * failed validation -> the manifest does not match the pixels: the file was
                         altered after signing. Strong evidence of tampering.
"""

import logging

from app.services.detection.types import ForensicSignal

logger = logging.getLogger(__name__)

C2PA_SIGNAL = "provenance_c2pa"

# Claim-generator / assertion substrings that indicate the signer itself
# declared the content synthetic.
_SYNTHETIC_CLAIM_MARKERS = (
    "trainedalgorithmicmedia",
    "compositesynthetic",
    "algorithmicmedia",
)


class C2paUnavailableError(Exception):
    """Raised when the c2pa library is not installed."""


def _load_reader():
    """Imported lazily so the ~83MB native library is not required just to
    import the app -- consistent with how torch is handled."""
    try:
        from c2pa import Reader
    except ImportError as exc:  # pragma: no cover - depends on install
        raise C2paUnavailableError(
            "c2pa is not installed; Content Credentials cannot be verified"
        ) from exc
    return Reader


def read_manifest(image_path: str) -> dict | None:
    """Returns the parsed manifest store, or None when the file carries no
    Content Credentials.

    The library raises for "no manifest found" as well as for genuinely
    malformed data, so the two are distinguished by message rather than by
    exception type.
    """
    import json

    Reader = _load_reader()
    try:
        with Reader(image_path) as reader:
            return json.loads(reader.json())
    except Exception as exc:  # noqa: BLE001 - library raises broadly
        # "This file has no Content Credentials" is by far the common case and
        # is NOT an error condition. The library signals it by raising, so it
        # must be told apart from a genuinely corrupt manifest. Matched on the
        # exception CLASS name as well as the message, because the observed
        # message ("ManifestNotFound: no JUMBF data found") does not contain
        # the phrasings a message-only check would naturally look for.
        name = type(exc).__name__.lower()
        message = str(exc).lower()
        not_found_markers = ("manifestnotfound", "no claim", "jumbf", "manifest not found")
        if any(marker in name for marker in not_found_markers) or any(
            marker in message for marker in not_found_markers
        ):
            return None
        raise


def analyze_c2pa(image_path: str) -> ForensicSignal:
    try:
        manifest_store = read_manifest(image_path)
    except C2paUnavailableError as exc:
        return ForensicSignal(
            name=C2PA_SIGNAL,
            applicable=False,
            suspicion_score=None,
            summary=f"Content Credentials could not be checked: {exc}",
            details={"error": str(exc)},
        )
    except Exception as exc:  # noqa: BLE001
        # A manifest that exists but cannot be parsed is itself noteworthy,
        # but not something to convert into a suspicion score -- report it.
        logger.warning("C2PA read failed for %s: %s", image_path, exc)
        return ForensicSignal(
            name=C2PA_SIGNAL,
            applicable=False,
            suspicion_score=None,
            summary=f"Content Credentials present but unreadable: {type(exc).__name__}",
            details={"error": str(exc)},
        )

    if manifest_store is None:
        return ForensicSignal(
            name=C2PA_SIGNAL,
            applicable=False,
            suspicion_score=None,
            summary=(
                "No Content Credentials (C2PA) found. This is the norm -- most images carry "
                "none, and any re-encode strips them -- so it is NOT evidence of manipulation."
            ),
            details={"evidence": "no_manifest"},
        )

    validation = manifest_store.get("validation_status") or []
    active_id = manifest_store.get("active_manifest")
    manifests = manifest_store.get("manifests", {})
    active = manifests.get(active_id, {}) if active_id else {}
    claim_generator = str(active.get("claim_generator", "")) or "unknown"

    if validation:
        # Non-empty validation_status means at least one check failed: the
        # signed manifest does not match the file as it now stands.
        codes = [str(item.get("code", "unknown")) for item in validation]
        return ForensicSignal(
            name=C2PA_SIGNAL,
            applicable=True,
            suspicion_score=0.95,
            summary=(
                "Content Credentials FAILED validation -- the file was altered after it was "
                f"signed ({', '.join(codes[:3])})."
            ),
            details={
                "evidence": "invalid_manifest",
                "claim_generator": claim_generator,
                "validation_codes": ", ".join(codes[:5]),
            },
        )

    # Valid manifest. Check whether the signer declared the content synthetic.
    blob = str(active).lower().replace("_", "").replace("-", "").replace(" ", "")
    declared_synthetic = next((m for m in _SYNTHETIC_CLAIM_MARKERS if m in blob), None)

    if declared_synthetic:
        return ForensicSignal(
            name=C2PA_SIGNAL,
            applicable=True,
            suspicion_score=1.0,
            summary=(
                "Valid Content Credentials, and the signed manifest itself declares the "
                f"content AI-generated (claim generator: {claim_generator})."
            ),
            details={
                "evidence": "valid_manifest_declares_synthetic",
                "claim_generator": claim_generator,
                "marker": declared_synthetic,
            },
        )

    return ForensicSignal(
        name=C2PA_SIGNAL,
        applicable=True,
        suspicion_score=0.15,
        summary=(
            "Valid Content Credentials with an intact provenance chain "
            f"(claim generator: {claim_generator}). Strong evidence the file is unaltered "
            "since signing."
        ),
        details={"evidence": "valid_manifest", "claim_generator": claim_generator},
    )
