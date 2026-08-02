"""
Pulls a pilot-sized, balanced subset of a public real-vs-AI-generated image
dataset as the R2 dataset's general backbone -- giving R3 a real evaluation
baseline without hours of scraping/generation, before the smaller
sports-specific supplement (fetch_wikimedia_sports.py +
generate_synthetic.py) adds domain-specific coverage on top.

Uses HF `datasets` streaming mode so only the requested number of examples
is transferred, not the whole dataset.

DEFAULT SOURCE: ComplexDataLab/OpenFake (a published 2025 research
benchmark). Chosen after the original pick, Parveshiiii/AI-vs-Real, was
withdrawn: measured directly, every one of its "real" images was exactly
178x218 and every "fake" exactly 1024x1024, so class and source were
perfectly correlated and a width threshold alone scored ROC-AUC 1.0000. See
docs/dataset.md. OpenFake's classes also differ in resolution (cameras
shoot bigger than generators emit) but every image is above the 384px
normalization target, so normalize_dataset.py only ever downsamples and no
single class picks up upscaling artefacts.

ALWAYS run probe_hf_dataset.py on a candidate before adopting it here, and
audit_dataset.py after downloading.

Schema (OpenFake): {"image": PIL.Image, "label": "real"|"fake",
"model": str, "prompt": str, ...}. The generator name is captured into the
manifest so generator diversity can be reported rather than assumed.

Usage:
    python fetch_hf_backbone.py --per-class 250 --out ../../datasets/backbone
"""

import argparse
import csv
import logging
from collections import Counter
from pathlib import Path

from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_DATASET_ID = "ComplexDataLab/OpenFake"
DEFAULT_CONFIG = "core"
DEFAULT_LABEL_KEY = "label"

# Maps whatever a dataset calls its classes onto this project's vocabulary.
# Covers integer-coded and string-coded label columns alike.
LABEL_TO_CLASS = {
    "real": "real", "fake": "fake",
    "Real": "real", "Fake": "fake",
    0: "fake", 1: "real",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-class", type=int, default=250)
    parser.add_argument("--out", type=str, default="../../datasets/backbone")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET_ID)
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG)
    parser.add_argument("--label-key", type=str, default=DEFAULT_LABEL_KEY)
    parser.add_argument("--seed", type=int, default=42, help="shuffle seed (reproducible sampling)")
    parser.add_argument(
        "--shuffle-buffer", type=int, default=2000,
        help="reservoir size for streaming shuffle; larger = more diverse, more startup latency",
    )
    parser.add_argument(
        "--min-dimension", type=int, default=384,
        help="skip images whose smaller side is below this; they would need upscaling at "
             "normalization time, which adds interpolation artefacts to one class only",
    )
    parser.add_argument(
        "--max-side", type=int, default=1600,
        help="downscale anything larger than this on the long side before saving. OpenFake's "
             "real images are camera-resolution (up to 6720x4480); holding those in a shuffle "
             "buffer exhausted RAM mid-run on a 16GB machine. Still far above --min-dimension, "
             "so normalization is unaffected.",
    )
    args = parser.parse_args()

    from datasets import load_dataset

    out_dir = Path(args.out).resolve()
    (out_dir / "real").mkdir(parents=True, exist_ok=True)
    (out_dir / "fake").mkdir(parents=True, exist_ok=True)

    logger.info("Opening %s (config=%s) in streaming mode (no full download)...", args.dataset, args.config)
    stream = load_dataset(args.dataset, args.config, split="train", streaming=True)

    # Shuffle the stream. Without this the sample is badly unrepresentative:
    # the underlying parquet is class-sorted (an unshuffled run drew every
    # "real" from indices 0-249 and every "fake" from a single contiguous
    # block at 4000-4249), so a whole class could come from one generator or
    # one source batch. shuffle() fills a reservoir buffer and draws from it,
    # which costs some startup latency but makes the sample actually diverse.
    stream = stream.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)

    counts = {"real": 0, "fake": 0}
    skipped_too_small = 0
    generators = Counter()

    # Written incrementally rather than buffered until the end. A previous run
    # died on a MemoryError partway through and took the whole manifest with
    # it, orphaning ~500 already-downloaded images that no downstream script
    # could see. Streaming rows means a crash costs only the current image.
    manifest_path = out_dir / "manifest.csv"
    manifest_fields = ["filename", "class", "source", "source_index", "generator", "width", "height"]
    manifest_file = manifest_path.open("w", newline="", encoding="utf-8")
    manifest_writer = csv.DictWriter(manifest_file, fieldnames=manifest_fields)
    manifest_writer.writeheader()

    for i, example in enumerate(stream):
        if counts["real"] >= args.per_class and counts["fake"] >= args.per_class:
            break

        label = example.get(args.label_key)
        cls = LABEL_TO_CLASS.get(label)
        if cls is None or counts[cls] >= args.per_class:
            continue

        image = example["image"]
        if min(image.size) < args.min_dimension:
            skipped_too_small += 1
            continue
        if image.mode != "RGB":
            image = image.convert("RGB")
        if max(image.size) > args.max_side:
            scale = args.max_side / max(image.size)
            image = image.resize(
                (max(1, int(image.width * scale)), max(1, int(image.height * scale))),
                Image.LANCZOS,
            )

        filename = f"backbone_{i}.jpg"
        dest = out_dir / cls / filename
        image.save(dest, "JPEG", quality=92)

        # Which generator produced a fake (where the dataset records it) --
        # kept so generator diversity can be reported rather than assumed.
        generator = example.get("model") or ""
        if cls == "fake" and generator:
            generators[generator] += 1

        counts[cls] += 1
        manifest_writer.writerow(
            {
                "filename": filename,
                "class": cls,
                "source": f"hf:{args.dataset}",
                "source_index": i,
                "generator": generator,
                "width": image.width,
                "height": image.height,
            }
        )
        manifest_file.flush()
        if sum(counts.values()) % 25 == 0:
            logger.info("Progress: real=%d fake=%d (target %d each)", counts["real"], counts["fake"], args.per_class)

    manifest_file.close()

    logger.info(
        "Done: real=%d fake=%d written under %s (manifest: %s)",
        counts["real"], counts["fake"], out_dir, manifest_path,
    )
    if skipped_too_small:
        logger.info(
            "Skipped %d image(s) with a side below %dpx (would need upscaling at normalization).",
            skipped_too_small, args.min_dimension,
        )
    if generators:
        logger.info("Generator mix across fakes (%d distinct):", len(generators))
        for name, count in generators.most_common():
            logger.info("    %-40s %d", name, count)
    # Dropping the reference (rather than letting it go out of scope at
    # process exit) avoids a benign-but-noisy crash trace on Windows: the
    # streaming iterator's background prefetch thread otherwise sometimes
    # loses a race with interpreter shutdown after main() returns, logging
    # a "Fatal Python error: _enter_buffered_busy" that looks alarming but
    # occurs strictly after the code above has already finished and the
    # manifest is already written.
    del stream
    if counts["real"] < args.per_class or counts["fake"] < args.per_class:
        logger.warning(
            "Stream exhausted before reaching target (%d requested per class) -- "
            "dataset may be smaller or less balanced within the streamed order than expected.",
            args.per_class,
        )


if __name__ == "__main__":
    main()
