"""
Merges the three R2 sources (HF backbone, Wikimedia real sports photos,
locally-generated synthetic sports photos) into one manifest with a
deterministic train/val/test split, then writes docs/dataset.md's data
tables.

Split-by-source, not by file, per the roadmap's leakage-avoidance rule:
  - Backbone images: split using a hash of the filename (stable, but these
    are already independent stock/generated images from a pre-shuffled
    dataset, so per-file splitting doesn't leak the way a photoshoot burst
    would).
  - Wikimedia images: split by `source_title`'s category/event, approximated
    here by grouping on the `category` column recorded during download, so
    an entire sport category lands in one split rather than being scattered
    (avoids the same photographer/event appearing in both train and test).
  - Synthetic images: split by `prompt` (same reasoning -- all images from
    one prompt template stay together, since they share the same synthetic
    "scene").

Usage:
    python build_manifest.py --datasets-dir ../../datasets --out ../../datasets/manifest.csv
"""

import argparse
import csv
import hashlib
from pathlib import Path

SPLIT_RATIOS = {"train": 0.7, "val": 0.15, "test": 0.15}


def _assign_split(group_key: str) -> str:
    """Deterministic hash-based split so re-running produces the same split
    without needing to persist state -- same group_key always lands in the
    same split."""
    digest = hashlib.sha256(group_key.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF  # -> [0, 1)
    if bucket < SPLIT_RATIOS["train"]:
        return "train"
    if bucket < SPLIT_RATIOS["train"] + SPLIT_RATIOS["val"]:
        return "val"
    return "test"


def load_backbone(datasets_dir: Path) -> list[dict]:
    manifest_path = datasets_dir / "backbone" / "manifest.csv"
    if not manifest_path.exists():
        return []
    rows = []
    with manifest_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "path": f"backbone/{row['class']}/{row['filename']}",
                    "class": row["class"],
                    "domain": "general",
                    "source": row["source"],
                    "split_group": row["filename"],  # per-file split OK, see docstring
                    "sport": "",
                    "framing": "",
                    "license": "MIT",
                    "attribution": "",
                }
            )
    return rows


def load_wikimedia(datasets_dir: Path) -> list[dict]:
    manifest_path = datasets_dir / "sports_real" / "attribution.csv"
    if not manifest_path.exists():
        return []
    rows = []
    with manifest_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            attribution = row["artist"] if row["license"] != "cc0" else "None required (CC0)"
            rows.append(
                {
                    "path": f"sports_real/{row['filename']}",
                    "class": "real",
                    "domain": "sports",
                    "source": "wikimedia_commons",
                    "split_group": row["category"],  # whole category -> one split
                    "sport": row["category"].replace("Category:", ""),
                    "framing": "",
                    "license": row["license"],
                    "attribution": f"{attribution} (source: {row['source_url']})",
                }
            )
    return rows


def load_synthetic(datasets_dir: Path) -> list[dict]:
    manifest_path = datasets_dir / "sports_fake" / "generation_manifest.csv"
    if not manifest_path.exists():
        return []
    rows = []
    with manifest_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            prompt = row["prompt"]
            framing = "close-up" if "close-up" in prompt else ("wide" if "wide shot" in prompt else "mid")
            rows.append(
                {
                    "path": f"sports_fake/{row['filename']}",
                    "class": "fake",
                    "domain": "sports",
                    "source": row.get("source", "local-diffusion"),
                    "split_group": prompt,  # same prompt/scene -> one split
                    "sport": "",
                    "framing": framing,
                    "license": "N/A (generated, no real subject)",
                    "attribution": "",
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets-dir", type=str, default="../../datasets")
    parser.add_argument("--out", type=str, default="../../datasets/manifest.csv")
    args = parser.parse_args()

    datasets_dir = Path(args.datasets_dir).resolve()

    all_rows = load_backbone(datasets_dir) + load_wikimedia(datasets_dir) + load_synthetic(datasets_dir)
    if not all_rows:
        raise SystemExit(
            "No source manifests found under "
            f"{datasets_dir} -- run fetch_hf_backbone.py, fetch_wikimedia_sports.py, "
            "and generate_synthetic.py first."
        )

    for row in all_rows:
        row["split"] = _assign_split(row["split_group"])

    out_path = Path(args.out).resolve()
    fieldnames = ["path", "class", "domain", "source", "sport", "framing", "license", "attribution", "split"]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow({k: row[k] for k in fieldnames})

    # Summary
    from collections import Counter

    by_split_class = Counter((r["split"], r["class"]) for r in all_rows)
    by_domain = Counter(r["domain"] for r in all_rows)
    print(f"Wrote {len(all_rows)} rows to {out_path}\n")
    print("By domain:", dict(by_domain))
    print("By split x class:")
    for split in ("train", "val", "test"):
        for cls in ("real", "fake"):
            print(f"  {split:5s} / {cls:4s}: {by_split_class.get((split, cls), 0)}")


if __name__ == "__main__":
    main()
