"""
Provenance verification tests (R5).

The most important tests here are the NEGATIVE ones. The dangerous failure
mode for this feature is not missing a marker -- it is treating *absence* of
metadata as evidence of manipulation, which would flag ordinary photographs
and, for a tool that publicly accuses people of faking images, cause real
harm. Measured on this project's own dataset, 0 of 6 sampled genuine
photographs carried any EXIF at all.
"""

import numpy as np
import pytest
from PIL import Image, PngImagePlugin

from app.services.provenance import (
    analyze_ai_metadata,
    analyze_c2pa,
    analyze_camera_metadata,
    run_provenance_analysis,
)


@pytest.fixture()
def image_array():
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)


def _png_with_text(path, array, key: str, value: str):
    meta = PngImagePlugin.PngInfo()
    meta.add_text(key, value)
    Image.fromarray(array).save(path, pnginfo=meta)
    return str(path)


def _jpeg_with_exif(path, array, **tags):
    image = Image.fromarray(array)
    exif = image.getexif()
    # 305 Software, 271 Make, 272 Model
    tag_ids = {"software": 305, "make": 271, "model": 272, "artist": 315}
    for name, value in tags.items():
        exif[tag_ids[name]] = value
    image.save(path, exif=exif)
    return str(path)


# --------------------------------------------------------------------------
# POSITIVE detection -- an explicit declaration IS found
# --------------------------------------------------------------------------

def test_detects_iptc_synthetic_declaration(tmp_path, image_array):
    """IPTC digitalSourceType=trainedAlgorithmicMedia is THE standard
    machine-readable "this is AI" statement."""
    path = _png_with_text(
        tmp_path / "iptc.png", image_array, "XML:com.adobe.xmp",
        "<x:xmpmeta><digitalSourceType>"
        "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia"
        "</digitalSourceType></x:xmpmeta>",
    )
    signal = analyze_ai_metadata(path)
    assert signal.applicable
    assert signal.suspicion_score == 1.0


def test_detects_generation_tool_png_chunk(tmp_path, image_array):
    path = _png_with_text(
        tmp_path / "a1111.png", image_array, "parameters",
        "a photo of a cat, Steps: 20, Model: stable diffusion v1-5",
    )
    signal = analyze_ai_metadata(path)
    assert signal.applicable
    assert signal.suspicion_score >= 0.9


@pytest.mark.parametrize("software", ["Midjourney v6", "DALL-E 3", "Adobe Firefly", "ComfyUI"])
def test_detects_generator_in_exif_software_tag(tmp_path, image_array, software):
    path = _jpeg_with_exif(tmp_path / f"{software[:4]}.jpg", image_array, software=software)
    assert analyze_ai_metadata(path).applicable


# --------------------------------------------------------------------------
# NEGATIVE -- the failure mode that would harm real people
# --------------------------------------------------------------------------

def test_absent_metadata_is_never_treated_as_suspicious(tmp_path, image_array):
    """A bare image with no metadata must yield NO signal at all -- not a
    score of 0.5, not a nudge toward "fake"."""
    path = str(tmp_path / "bare.png")
    Image.fromarray(image_array).save(path)

    for signal in run_provenance_analysis(path):
        assert not signal.applicable, f"{signal.name} fired on an image with no metadata"
        assert signal.suspicion_score is None


def test_absence_notes_explicitly_say_it_is_not_evidence(tmp_path, image_array):
    """The wording matters: a reader must not infer guilt from silence.

    Checks that each absence message actively disclaims evidentiary value,
    rather than merely stating the metadata is missing. Several phrasings are
    accepted because the right wording differs per signal -- "not evidence of
    manipulation" reads oddly for camera EXIF, where "says nothing about
    authenticity either way" is the accurate statement.
    """
    path = str(tmp_path / "bare2.png")
    Image.fromarray(image_array).save(path)

    disclaimers = (
        "not evidence",
        "does not indicate",
        "says nothing",
        "either way",
        "is the norm",
        "expected",
    )
    for signal in run_provenance_analysis(path):
        summary = signal.summary.lower()
        assert any(phrase in summary for phrase in disclaimers), (
            f"{signal.name} reports absence without disclaiming evidentiary value: {signal.summary!r}"
        )


def test_ordinary_editing_software_is_not_flagged_as_ai(tmp_path, image_array):
    """Photoshop/Lightroom appear in millions of genuine photographs."""
    for software in ["Adobe Photoshop Lightroom Classic 13.2", "GIMP 2.10", "Capture One 23"]:
        path = _jpeg_with_exif(tmp_path / "edit.jpg", image_array, software=software)
        assert not analyze_ai_metadata(path).applicable, f"{software} wrongly flagged as AI"


def test_camera_metadata_containing_marker_substring_is_not_flagged(tmp_path, image_array):
    """REGRESSION: a real Nikon D810 photograph was flagged as AI-generated
    because its XMP contained `aux:imagenumber="177034"` and "imagen" is a
    generator marker. Markers must match as whole tokens, and only in
    software-identifying fields -- not anywhere in 12KB of camera settings."""
    path = _png_with_text(
        tmp_path / "nikon.png", image_array, "XML:com.adobe.xmp",
        '<x:xmpmeta><rdf:Description aux:imagenumber="177034" '
        'aux:lens="150.0-600.0 mm" xmp:CreatorTool="Adobe Lightroom"/></x:xmpmeta>',
    )
    assert not analyze_ai_metadata(path).applicable


# --------------------------------------------------------------------------
# Camera metadata -- weak positive evidence only
# --------------------------------------------------------------------------

def test_camera_metadata_is_recognised_but_only_weakly(tmp_path, image_array):
    path = _jpeg_with_exif(
        tmp_path / "cam.jpg", image_array, make="NIKON CORPORATION", model="NIKON D810"
    )
    signal = analyze_camera_metadata(path)
    assert signal.applicable
    # Below 0.5 (leans authentic) but well above 0 -- EXIF is forgeable, so it
    # must never be treated as proof.
    assert 0.3 <= signal.suspicion_score < 0.5
    assert "forged" in signal.summary


# --------------------------------------------------------------------------
# C2PA
# --------------------------------------------------------------------------

def test_missing_c2pa_manifest_is_not_applicable_not_an_error(tmp_path, image_array):
    """"No Content Credentials" is the norm and must be reported as absence of
    information, not as a read failure or as suspicion."""
    path = str(tmp_path / "noc2pa.jpg")
    Image.fromarray(image_array).save(path)

    signal = analyze_c2pa(path)
    assert not signal.applicable
    assert signal.suspicion_score is None
    assert "not evidence" in signal.summary.lower()


def test_provenance_runs_all_checks_and_survives_a_bad_path():
    """One failing check must not abort the batch."""
    signals = run_provenance_analysis("/nonexistent/definitely-not-here.jpg")
    assert len(signals) == 3
    assert all(not s.applicable for s in signals)
