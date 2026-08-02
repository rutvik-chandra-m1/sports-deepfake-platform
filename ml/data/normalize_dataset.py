"""
Normalizes every image through an identical transform chain, to remove the
container-level confounds that audit_dataset.py found (metadata-only
ROC-AUC was 1.0000 before this step -- real vs fake was perfectly
separable from width/squareness/format alone, with no pixel content).

Transform chain, applied uniformly to EVERY image regardless of source:
  1. convert to RGB
  2. centre-crop to square
  3. resample to TARGET_SIZE
  4. re-encode as JPEG at JPEG_QUALITY

Why a target SMALLER than every source (384 < 512 native synthetic < real
photos up to 7360px): if the target equalled the synthetic size, synthetics
would pass through unresampled while real photos got downsampled, and the
resampling artefact itself would become a new leak in the opposite
direction. Everything must actually be resampled for the playing field to
be level.

Also filters out images that are not colour photographs at all -- the
Wikimedia sports pull brought in 1910s line drawings and B&W newsprint
scans, which are neither real photographs nor AI-generated images, so
labelling them "real" teaches a detector nonsense.

KNOWN COST, documented not hidden: uniform downsampling + re-encoding
attenuates exactly the high-frequency and compression-history evidence that
frequency_analysis.py and compression_analysis.py look for. That is a real
loss of signal. It is accepted because the alternative -- leaving a
perfectly-separable confound in place -- makes every downstream number
meaningless rather than merely weaker. Uniform preprocessing is also
standard practice for detection benchmarks, and images that circulate in
the real world have generally been resized and re-encoded anyway, so the
normalized set is arguably closer to the deployment condition than raw
generator output.

Usage:
    python normalize_dataset.py --datasets-dir ../../datasets
"""

import argparse
import csv
import logging
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TARGET_SIZE = 384
JPEG_QUALITY = 90

# Below this mean per-pixel channel spread the image is effectively
# greyscale (B&W newsprint scans, monochrome prints).
GRAYSCALE_SPREAD_THRESHOLD = 8.0
# Shannon entropy (bits) of a 4-bit-per-channel colour histogram. Natural
# photographs are well above this; flat-shaded line art and diagrams are not.
MIN_COLOR_ENTROPY = 4.0
# Anything smaller than the target in either dimension would be upscaled,
# which introduces its own distinctive artefacts.
MIN_SOURCE_DIMENSION = TARGET_SIZE


def color_entropy(arr: np.ndarray) -> float:
    """Shannon entropy of the quantized colour histogram -- a cheap
    'is this a photograph or flat-shaded artwork' proxy."""
    quantized = (arr // 16).astype(np.uint8)  # 4 bits per channel -> 4096 bins
    flat = quantized[:, :, 0].astype(np.uint32) * 256 + quantized[:, :, 1] * 16 + quantized[:, :, 2]
    counts = np.bincount(flat.ravel(), minlength=4096).astype(np.float64)
    probs = counts[counts > 0] / counts.sum()
    return float(-(probs * np.log2(probs)).sum())


def center_crop_square(img: Image.Image) -> Image.Image:
    width, height = img.size
    side = min(width, height)
    left, top = (width - side) // 2, (height - side) // 2
    return img.crop((left, top, left + side, top + side))


def assess(path: Path) -> tuple[Image.Image | None, str | None]:
    """Returns (rgb_image, rejection_reason). Exactly one is non-None."""
    try:
        with Image.open(path) as img:
            rgb = img.convert("RGB")
            rgb.load()
    except Exception as exc:  # noqa: BLE001 - unreadable file is a valid rejection
        return None, f"unreadable ({type(exc).__name__})"

    width, height = rgb.size
    if min(width, height) < MIN_SOURCE_DIMENSION:
        return None, f"too small ({width}x{height}, would upscale)"

    arr = np.asarray(rgb, dtype=np.float32)
    spread = float(np.mean(arr.max(axis=2) - arr.min(axis=2)))
    if spread < GRAYSCALE_SPREAD_THRESHOLD:
        return None, f"effectively greyscale (spread={spread:.1f})"

    entropy = color_entropy(np.asarray(rgb))
    if entropy < MIN_COLOR_ENTROPY:
        return None, f"not a photograph -- flat colour distribution (entropy={entropy:.2f})"

    return rgb, None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets-dir", type=str, default="../../datasets")
    parser.add_argument("--out-subdir", type=str, default="normalized")
    args = parser.parse_args()

    datasets_dir = Path(args.datasets_dir).resolve()
    out_root = datasets_dir / args.out_subdir
    manifest_path = datasets_dir / "manifest.csv"
    if not manifest_path.exists():
        raise SystemExit(f"No manifest at {manifest_path} -- run build_manifest.py first.")

    with manifest_path.open(encoding="utf-8") as f:
        manifest = list(csv.DictReader(f))

    kept_rows, rejections = [], []
    for entry in manifest:
        src = datasets_dir / entry["path"]
        rgb, reason = assess(src)
        if rgb is None:
            rejections.append((entry["path"], entry["domain"], entry["class"], reason))
            continue

        processed = center_crop_square(rgb).resize((TARGET_SIZE, TARGET_SIZE), Image.LANCZOS)

        # Flatten source subdirectories into <class>/ so the output layout is
        # uniform and carries no path-based hint of origin.
        dest_dir = out_root / entry["class"]
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_name = Path(entry["path"]).stem + ".jpg"
        processed.save(dest_dir / dest_name, "JPEG", quality=JPEG_QUALITY)

        row = dict(entry)
        row["path"] = f"{args.out_subdir}/{entry['class']}/{dest_name}"
        row["original_path"] = entry["path"]
        kept_rows.append(row)

    fieldnames = list(manifest[0].keys()) + ["original_path"]
    out_manifest = datasets_dir / "manifest_normalized.csv"
    with out_manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept_rows)

    print(f"\nNormalized {len(kept_rows)}/{len(manifest)} images -> {out_root}")
    print(f"Manifest -> {out_manifest}")
    print(f"Transform: RGB -> centre-crop square -> {TARGET_SIZE}x{TARGET_SIZE} LANCZOS -> JPEG q{JPEG_QUALITY}")

    if rejections:
        print(f"\nRejected {len(rejections)}:")
        by_reason = Counter(r[3].split(" (")[0].split(" --")[0] for r in rejections)
        for reason, count in by_reason.most_common():
            print(f"  {count:3d}  {reason}")
        print("\n  Detail (domain/class -> file -> reason):")
        for path, domain, cls, reason in rejections:
            print(f"    {domain}/{cls:4s}  {Path(path).name:<34} {reason}")

    kept_counts = Counter((r["domain"], r["class"]) for r in kept_rows)
    print("\nSurviving composition:")
    for (domain, cls), count in sorted(kept_counts.items()):
        print(f"  {domain}/{cls}: {count}")

    split_counts = Counter((r["split"], r["class"]) for r in kept_rows)
    print("\nSurviving splits:")
    for split in ("train", "val", "test"):
        real = split_counts.get((split, "real"), 0)
        fake = split_counts.get((split, "fake"), 0)
        print(f"  {split:5s}: real={real:3d} fake={fake:3d} total={real + fake:3d}")


if __name__ == "__main__":
    main()
