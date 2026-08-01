"""
Pulls a pilot-sized, balanced subset of Parveshiiii/AI-vs-Real (MIT
licensed, not gated, 14k real-vs-AI-generated images) as the R2 dataset's
general backbone -- gets a real, working evaluation baseline (R3) quickly
without hours of scraping/generation, before the smaller sports-specific
supplement (fetch_wikimedia_sports.py + generate_synthetic.py) adds
domain-specific coverage on top.

Uses HF `datasets` streaming mode so we only transfer as many examples as
requested, not the full ~2GB dataset.

Schema: {"image": PIL.Image, "binary_label": int} where 0 = AI-generated,
1 = real (per the dataset card).

Usage:
    python fetch_hf_backbone.py --per-class 250 --out ../../datasets/backbone
"""

import argparse
import csv
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATASET_ID = "Parveshiiii/AI-vs-Real"
LABEL_TO_CLASS = {0: "fake", 1: "real"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-class", type=int, default=250)
    parser.add_argument("--out", type=str, default="../../datasets/backbone")
    args = parser.parse_args()

    from datasets import load_dataset

    out_dir = Path(args.out).resolve()
    (out_dir / "real").mkdir(parents=True, exist_ok=True)
    (out_dir / "fake").mkdir(parents=True, exist_ok=True)

    logger.info("Opening %s in streaming mode (no full download)...", DATASET_ID)
    stream = load_dataset(DATASET_ID, split="train", streaming=True)

    counts = {"real": 0, "fake": 0}
    manifest_rows = []
    target_total = args.per_class * 2

    for i, example in enumerate(stream):
        if counts["real"] >= args.per_class and counts["fake"] >= args.per_class:
            break

        label = example.get("binary_label")
        cls = LABEL_TO_CLASS.get(label)
        if cls is None or counts[cls] >= args.per_class:
            continue

        image = example["image"]
        if image.mode != "RGB":
            image = image.convert("RGB")

        filename = f"backbone_{i}.jpg"
        dest = out_dir / cls / filename
        image.save(dest, "JPEG", quality=92)

        counts[cls] += 1
        manifest_rows.append(
            {
                "filename": filename,
                "class": cls,
                "source": "hf:" + DATASET_ID,
                "source_index": i,
                "width": image.width,
                "height": image.height,
            }
        )
        if sum(counts.values()) % 25 == 0:
            logger.info("Progress: real=%d fake=%d (target %d each)", counts["real"], counts["fake"], args.per_class)

    manifest_path = out_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "class", "source", "source_index", "width", "height"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    logger.info(
        "Done: real=%d fake=%d written under %s (manifest: %s)",
        counts["real"], counts["fake"], out_dir, manifest_path,
    )
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
