"""
Cheaply vet a candidate Hugging Face image dataset BEFORE downloading it in
bulk. Streams a small sample and reports, per class, the distribution of
image size and format.

Why this exists: the first backbone chosen for this project
(Parveshiiii/AI-vs-Real) turned out to be completely source-confounded --
every "real" image was exactly 178x218 and every "fake" exactly 1024x1024,
with zero overlap between the classes. A detector trained on it would learn
"thumbnail vs high-resolution render", not "photographed vs generated". That
was only discovered after downloading 700 images and inspecting them.

The check: if the class-conditional size distributions are disjoint, the
dataset is source-confounded and unusable as a detection benchmark, no matter
how balanced its class counts look.

Usage:
    python probe_hf_dataset.py --dataset ComplexDataLab/OpenFake --n 60
    python probe_hf_dataset.py --dataset foo/bar --n 60 --label-key label --image-key image
"""

import argparse
import logging
from collections import Counter, defaultdict

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def guess_keys(example: dict) -> tuple[str | None, str | None]:
    """Find the image column and the label column without knowing the schema."""
    image_key = next(
        (k for k, v in example.items() if hasattr(v, "size") and hasattr(v, "mode")), None
    )
    label_key = next(
        (
            k
            for k, v in example.items()
            if k != image_key and isinstance(v, (int, bool, str)) and k.lower() not in {"id", "path", "file_name"}
        ),
        None,
    )
    return image_key, label_key


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--config", type=str, default=None, help="dataset config name, if it has several")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--n", type=int, default=60, help="examples to sample")
    parser.add_argument("--image-key", type=str, default=None)
    parser.add_argument("--label-key", type=str, default=None)
    parser.add_argument("--shuffle-buffer", type=int, default=500)
    parser.add_argument(
        "--norm-target", type=int, default=384,
        help="normalize_dataset.py TARGET_SIZE; decides whether a class would need upscaling",
    )
    args = parser.parse_args()

    from datasets import load_dataset

    print(f"Streaming {args.n} examples from {args.dataset} [{args.split}]...\n")
    stream = load_dataset(args.dataset, args.config, split=args.split, streaming=True)
    stream = stream.shuffle(seed=0, buffer_size=args.shuffle_buffer)

    sizes_by_label: dict[object, Counter] = defaultdict(Counter)
    formats_by_label: dict[object, Counter] = defaultdict(Counter)
    modes_by_label: dict[object, Counter] = defaultdict(Counter)
    image_key, label_key = args.image_key, args.label_key

    seen = 0
    for example in stream:
        if seen >= args.n:
            break
        if image_key is None or label_key is None:
            guessed_image, guessed_label = guess_keys(example)
            image_key = image_key or guessed_image
            label_key = label_key or guessed_label
            if image_key is None:
                raise SystemExit(f"Could not find an image column. Columns: {list(example.keys())}")
            print(f"Using image_key='{image_key}', label_key='{label_key}'")
            print(f"All columns: {list(example.keys())}\n")

        image = example[image_key]
        label = example.get(label_key, "<none>")
        sizes_by_label[label][image.size] += 1
        formats_by_label[label][getattr(image, "format", None) or "?"] += 1
        modes_by_label[label][image.mode] += 1
        seen += 1

    del stream

    print("=" * 74)
    print(f"CLASS-CONDITIONAL CONTAINER DISTRIBUTIONS  (n={seen})")
    print("=" * 74)
    for label in sorted(sizes_by_label, key=str):
        sizes = sizes_by_label[label]
        total = sum(sizes.values())
        print(f"\nlabel={label!r}  (n={total})")
        print(f"  distinct sizes : {len(sizes)}")
        for size, count in sizes.most_common(5):
            print(f"      {size}: {count}")
        print(f"  formats        : {dict(formats_by_label[label])}")
        print(f"  modes          : {dict(modes_by_label[label])}")

    labels = list(sizes_by_label)
    print("\n" + "=" * 74)
    print("VERDICT")
    print("=" * 74)
    if len(labels) < 2:
        print("  Only one label seen -- increase --n or --shuffle-buffer.")
        return

    size_sets = {label: set(sizes_by_label[label]) for label in labels}
    all_overlap = set.intersection(*size_sets.values())
    union = set.union(*size_sets.values())
    overlap_fraction = len(all_overlap) / len(union) if union else 0.0

    # The question that actually decides usability is not "do the raw sizes
    # overlap" -- real photos legitimately come off cameras at far higher
    # resolution than generators emit, so disjointness is expected and is not
    # by itself disqualifying. What matters is whether normalizing to a common
    # size can fix it, and that turns on DOWNSAMPLING vs UPSCALING: if one
    # class sits below the target it must be upscaled, which stamps
    # interpolation artefacts onto exactly that class and simply swaps one
    # confound for another. If every class is above the target, all of them
    # get downsampled and the confound genuinely disappears.
    min_dim_by_label = {
        label: min(min(size) for size in sizes_by_label[label]) for label in labels
    }
    variety_by_label = {label: len(size_sets[label]) for label in labels}

    print(f"  distinct sizes per class   : { {str(l): v for l, v in variety_by_label.items()} }")
    print(f"  smallest dimension per class: { {str(l): v for l, v in min_dim_by_label.items()} }")
    print(f"  sizes shared by ALL classes : {len(all_overlap)} of {len(union)} ({overlap_fraction:.1%})")

    worst_min_dim = min(min_dim_by_label.values())
    single_size_classes = [str(l) for l, v in variety_by_label.items() if v == 1]

    print(f"\n  normalization target assumed: {args.norm_target}px (normalize_dataset.py TARGET_SIZE)")

    if single_size_classes and not all_overlap:
        print(f"\n  UNUSABLE - class(es) {single_size_classes} have exactly ONE image size, disjoint")
        print("  from the others. Class and source are perfectly correlated; a single")
        print("  resolution threshold separates them and no preprocessing can undo that.")
    elif worst_min_dim < args.norm_target:
        below = [str(l) for l, v in min_dim_by_label.items() if v < args.norm_target]
        print(f"\n  RISKY - class(es) {below} contain images below the {args.norm_target}px target.")
        print("  Normalizing would UPSCALE those, adding interpolation artefacts to one class")
        print("  only -- trading a resolution confound for an upsampling confound. Either")
        print("  lower the target, or filter sub-target images out (losing that data).")
    elif not all_overlap:
        print("\n  USABLE AFTER NORMALIZATION - sizes are disjoint, but every class is above")
        print(f"  the {args.norm_target}px target, so normalization only ever downsamples and no")
        print("  class picks up upscaling artefacts. This size gap is the expected, realistic")
        print("  camera-vs-generator difference, not careless dataset construction.")
        print("  REQUIRED: run normalize_dataset.py, then audit_dataset.py to confirm the")
        print("  container-tier AUC actually falls to ~0.50.")
    else:
        print("\n  PROMISING - classes share a size distribution; no obvious source confound.")
        print("  (Still run audit_dataset.py after downloading -- this only checks size/format.)")


if __name__ == "__main__":
    main()
