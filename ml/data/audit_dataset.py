"""
Dataset confound audit -- run this BEFORE trusting any evaluation number.

The question this answers: **can you tell real from fake without looking at
the picture?** If yes, the dataset is broken, and any accuracy number from
R3 measures the confound rather than detection ability.

Method: extract only "container" properties that carry no manipulation
evidence -- pixel dimensions, aspect ratio, file format, file size, whether
the image is effectively grayscale, and mean saturation. Fit a small
gradient-boosted classifier on those alone, cross-validated. A well-built
dataset should score near chance (~0.5 AUC). Anything much above that is a
shortcut the real detector will silently learn instead of learning
detection.

This is a standard "shortcut learning" / spurious-correlation check
(cf. Geirhos et al., "Shortcut Learning in Deep Neural Networks", 2020).
It is deliberately run as its own step, not folded into R3, so a broken
dataset is caught before it contaminates a headline metric.

Usage:
    python audit_dataset.py --datasets-dir ../../datasets
    python audit_dataset.py --datasets-dir ../../datasets --json-out ../../reports/dataset_audit.json
"""

import argparse
import csv
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# An image whose mean per-pixel channel spread is below this is treated as
# effectively grayscale (covers true 'L' mode and RGB scans of B&W prints).
GRAYSCALE_SATURATION_THRESHOLD = 8.0


def dhash(path: Path, hash_size: int = 8) -> int | None:
    """Difference hash -- a 64-bit perceptual fingerprint. Robust to
    rescaling and re-encoding, so it catches the same image appearing twice
    at different sizes/formats (which exact checksums would miss entirely)."""
    try:
        with Image.open(path) as img:
            small = img.convert("L").resize((hash_size + 1, hash_size), Image.LANCZOS)
            arr = np.asarray(small, dtype=np.int16)
    except Exception:  # noqa: BLE001 - unreadable handled by probe_image
        return None
    bits = (arr[:, 1:] > arr[:, :-1]).flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def find_duplicate_leakage(rows: list[dict], max_hamming: int = 4) -> dict:
    """
    Near-duplicate detection across splits. If the same (or a barely-altered)
    image sits in both train and test, every evaluation number is inflated --
    the model is being tested on something it memorized. This is the single
    most common way a benchmark silently lies.
    """
    hashed = [r for r in rows if r.get("dhash") is not None]
    exact_groups: dict[int, list[dict]] = defaultdict(list)
    for row in hashed:
        exact_groups[row["dhash"]].append(row)

    exact_dupes = {h: g for h, g in exact_groups.items() if len(g) > 1}

    cross_split, cross_class = [], []
    for group in exact_dupes.values():
        splits = {g["split"] for g in group}
        classes = {g["class"] for g in group}
        if len(splits) > 1:
            cross_split.append([g["path"] for g in group])
        if len(classes) > 1:
            cross_class.append([g["path"] for g in group])

    # Near-duplicates (small Hamming distance) across split boundaries.
    near_cross_split = []
    unique_hashes = sorted(exact_groups.keys())
    for i, h1 in enumerate(unique_hashes):
        for h2 in unique_hashes[i + 1:]:
            if bin(h1 ^ h2).count("1") <= max_hamming:
                s1 = {g["split"] for g in exact_groups[h1]}
                s2 = {g["split"] for g in exact_groups[h2]}
                if s1 != s2 or len(s1 | s2) > 1:
                    near_cross_split.append(
                        (exact_groups[h1][0]["path"], exact_groups[h2][0]["path"],
                         bin(h1 ^ h2).count("1"))
                    )

    return {
        "n_hashed": len(hashed),
        "exact_duplicate_groups": len(exact_dupes),
        "exact_duplicate_images": sum(len(g) for g in exact_dupes.values()),
        "cross_split_exact_groups": len(cross_split),
        "cross_class_exact_groups": len(cross_class),
        "near_duplicate_cross_split_pairs": len(near_cross_split),
        "examples_cross_split": cross_split[:5],
        "examples_cross_class": cross_class[:5],
        "examples_near_cross_split": near_cross_split[:5],
    }


def probe_image(path: Path) -> dict | None:
    """Container-level properties only. Deliberately no manipulation-relevant
    signal here -- that is the entire point of the audit."""
    try:
        with Image.open(path) as img:
            fmt = img.format or "UNKNOWN"
            mode = img.mode
            width, height = img.size
            rgb = img.convert("RGB")
            arr = np.asarray(rgb, dtype=np.float32)
    except Exception as exc:  # noqa: BLE001 - a corrupt file is itself a finding
        logger.warning("Could not read %s: %s", path, exc)
        return None

    # Mean spread between channels: ~0 for grayscale, larger for colour.
    channel_spread = float(np.mean(arr.max(axis=2) - arr.min(axis=2)))

    return {
        "width": width,
        "height": height,
        "megapixels": (width * height) / 1e6,
        "aspect_ratio": width / height if height else 0.0,
        "is_square": 1.0 if abs(width - height) <= 2 else 0.0,
        "format": fmt,
        "mode": mode,
        "file_size_kb": path.stat().st_size / 1024,
        "channel_spread": channel_spread,
        "is_grayscale": 1.0 if channel_spread < GRAYSCALE_SATURATION_THRESHOLD else 0.0,
        "mean_brightness": float(arr.mean()),
    }


def load_manifest(datasets_dir: Path, manifest_name: str) -> list[dict]:
    manifest_path = datasets_dir / manifest_name
    if not manifest_path.exists():
        raise SystemExit(f"No manifest at {manifest_path} -- run build_manifest.py first.")
    with manifest_path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def summarize(rows: list[dict], group_keys: tuple[str, ...]) -> dict:
    """Per-group distribution summary of the probed properties."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in group_keys)].append(row)

    summary = {}
    for key, items in sorted(groups.items()):
        widths = [i["width"] for i in items]
        heights = [i["height"] for i in items]
        summary["/".join(key)] = {
            "count": len(items),
            "formats": dict(Counter(i["format"] for i in items)),
            "modes": dict(Counter(i["mode"] for i in items)),
            "square_fraction": round(float(np.mean([i["is_square"] for i in items])), 3),
            "grayscale_fraction": round(float(np.mean([i["is_grayscale"] for i in items])), 3),
            "width_min_med_max": [int(min(widths)), int(np.median(widths)), int(max(widths))],
            "height_min_med_max": [int(min(heights)), int(np.median(heights)), int(max(heights))],
            "megapixels_median": round(float(np.median([i["megapixels"] for i in items])), 3),
            "file_size_kb_median": round(float(np.median([i["file_size_kb"] for i in items])), 1),
            "channel_spread_median": round(float(np.median([i["channel_spread"] for i in items])), 1),
        }
    return summary


# Two categorically different tiers, deliberately scored separately.
#
# CONTAINER: how the file was stored. Carries no information about whether
# the depicted scene was photographed or generated -- if these separate the
# classes, the dataset is broken and must be fixed.
#
# CONTENT_STATS: crude global image statistics. These *can* legitimately
# differ between real photographs and generative output (generated images
# do skew more saturated, more evenly exposed, and smoother/more
# compressible). A model using them is not cheating -- but it is using the
# shallowest possible evidence, so this score is the trivial baseline any
# real detector must beat to have demonstrated anything.
CONTAINER_FEATURES = ["width", "height", "megapixels", "aspect_ratio", "is_square"]
CONTENT_STAT_FEATURES = ["file_size_kb", "channel_spread", "is_grayscale", "mean_brightness"]


def metadata_only_separability(
    rows: list[dict], feature_set: str, label_key: str = "class"
) -> dict:
    """
    Fit a classifier on non-content features only and report cross-validated
    ROC-AUC. For feature_set='container', ~0.5 is required for the dataset to
    be sound. For 'content_stats', the score is the trivial baseline.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    if feature_set == "container":
        numeric_features, use_format = CONTAINER_FEATURES, True
    elif feature_set == "content_stats":
        numeric_features, use_format = CONTENT_STAT_FEATURES, False
    else:
        numeric_features, use_format = CONTAINER_FEATURES + CONTENT_STAT_FEATURES, True

    formats = sorted({r["format"] for r in rows}) if use_format else []

    X, y = [], []
    for row in rows:
        features = [row[f] for f in numeric_features]
        features += [1.0 if row["format"] == fmt else 0.0 for fmt in formats]
        X.append(features)
        y.append(1 if row[label_key] == "fake" else 0)

    X_arr, y_arr = np.array(X, dtype=np.float32), np.array(y)
    if len(set(y_arr.tolist())) < 2:
        return {"error": "only one class present", "n": len(y_arr)}

    n_splits = min(5, int(np.bincount(y_arr).min()))
    if n_splits < 2:
        return {"error": "too few samples in minority class for CV", "n": len(y_arr)}

    model = HistGradientBoostingClassifier(max_iter=200, random_state=0)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    scores = cross_val_score(model, X_arr, y_arr, cv=cv, scoring="roc_auc")

    # Which single feature is most responsible? Fit once and read permutation
    # importance so the report names the specific leak, not just its size.
    from sklearn.inspection import permutation_importance

    model.fit(X_arr, y_arr)
    perm = permutation_importance(model, X_arr, y_arr, n_repeats=5, random_state=0, scoring="roc_auc")
    feature_names = numeric_features + [f"format={f}" for f in formats]
    ranked = sorted(zip(feature_names, perm.importances_mean), key=lambda kv: -kv[1])

    return {
        "n": int(len(y_arr)),
        "n_fake": int(y_arr.sum()),
        "n_real": int(len(y_arr) - y_arr.sum()),
        "cv_folds": n_splits,
        "roc_auc_mean": round(float(scores.mean()), 4),
        "roc_auc_std": round(float(scores.std()), 4),
        "top_leaking_features": [(name, round(float(imp), 4)) for name, imp in ranked[:5]],
    }


def container_verdict(auc: float) -> str:
    """Container features must not separate the classes. Anything meaningfully
    above chance is a dataset defect, not a finding about detection."""
    if auc >= 0.9:
        return "BROKEN - storage format alone nearly solves it; fix the dataset before evaluating"
    if auc >= 0.75:
        return "SEVERE confound - strong container shortcut present"
    if auc >= 0.6:
        return "MODERATE confound - some container shortcut remains"
    return "OK - near chance; no container shortcut"


def content_stats_verdict(auc: float) -> str:
    """Content statistics legitimately differ between real and generated
    imagery, so this is a baseline to beat, not a defect."""
    if auc >= 0.85:
        return f"HIGH trivial baseline ({auc:.2f}) - a real detector must clear this bar convincingly"
    if auc >= 0.65:
        return f"MODERATE trivial baseline ({auc:.2f}) - report detector scores against this, not against 0.5"
    return f"LOW trivial baseline ({auc:.2f}) - global statistics alone explain little"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets-dir", type=str, default="../../datasets")
    parser.add_argument("--manifest", type=str, default="manifest.csv")
    parser.add_argument("--json-out", type=str, default=None)
    args = parser.parse_args()

    datasets_dir = Path(args.datasets_dir).resolve()
    manifest = load_manifest(datasets_dir, args.manifest)

    logger.info("Probing %d images...", len(manifest))
    rows = []
    for entry in manifest:
        probed = probe_image(datasets_dir / entry["path"])
        if probed is None:
            continue
        probed.update(
            {"path": entry["path"], "class": entry["class"], "domain": entry["domain"], "split": entry["split"]}
        )
        probed["dhash"] = dhash(datasets_dir / entry["path"])
        rows.append(probed)

    report: dict = {"n_probed": len(rows), "n_manifest": len(manifest)}

    print("\n" + "=" * 78)
    print("PER-GROUP CONTAINER PROPERTIES (domain/class)")
    print("=" * 78)
    report["by_domain_class"] = summarize(rows, ("domain", "class"))
    for group, stats in report["by_domain_class"].items():
        print(f"\n{group}  (n={stats['count']})")
        print(f"  formats            : {stats['formats']}")
        print(f"  modes              : {stats['modes']}")
        print(f"  square fraction    : {stats['square_fraction']}")
        print(f"  grayscale fraction : {stats['grayscale_fraction']}")
        print(f"  width  (min/med/max): {stats['width_min_med_max']}")
        print(f"  height (min/med/max): {stats['height_min_med_max']}")
        print(f"  median megapixels  : {stats['megapixels_median']}")
        print(f"  median file size KB: {stats['file_size_kb_median']}")
        print(f"  median chan spread : {stats['channel_spread_median']}")

    scopes = [
        ("ALL", rows),
        ("general", [r for r in rows if r["domain"] == "general"]),
        ("sports", [r for r in rows if r["domain"] == "sports"]),
    ]

    print("\n" + "=" * 78)
    print("TIER 1 - CONTAINER SEPARABILITY   (dataset soundness check)")
    print("  Features: width, height, megapixels, aspect ratio, squareness, file format.")
    print("  These say nothing about whether a scene was photographed or generated.")
    print("  REQUIREMENT: ~0.50. Anything higher is a dataset defect to fix.")
    print("=" * 78)

    report["container_separability"] = {}
    for scope, subset in scopes:
        result = metadata_only_separability(subset, "container")
        report["container_separability"][scope] = result
        print(f"\n[{scope}]  n={result.get('n')} (real={result.get('n_real')}, fake={result.get('n_fake')})")
        if "error" in result:
            print(f"  skipped: {result['error']}")
            continue
        result["verdict"] = container_verdict(result["roc_auc_mean"])
        print(f"  container-only ROC-AUC : {result['roc_auc_mean']:.4f} (+/- {result['roc_auc_std']:.4f})")
        print(f"  verdict                : {result['verdict']}")
        print("  top features           :")
        for name, imp in result["top_leaking_features"]:
            print(f"      {name:<22} {imp:+.4f}")

    print("\n" + "=" * 78)
    print("TIER 2 - CONTENT-STATISTICS BASELINE   (the bar R3 must beat)")
    print("  Features: saturation, brightness, compressibility (JPEG size at fixed")
    print("  dimensions/quality). Generated imagery genuinely does skew more")
    print("  saturated, more evenly exposed, and smoother -- so this is real but")
    print("  very shallow signal. NOT a defect; it is the trivial baseline.")
    print("=" * 78)

    report["content_stats_baseline"] = {}
    for scope, subset in scopes:
        result = metadata_only_separability(subset, "content_stats")
        report["content_stats_baseline"][scope] = result
        print(f"\n[{scope}]  n={result.get('n')} (real={result.get('n_real')}, fake={result.get('n_fake')})")
        if "error" in result:
            print(f"  skipped: {result['error']}")
            continue
        result["verdict"] = content_stats_verdict(result["roc_auc_mean"])
        print(f"  content-stats ROC-AUC  : {result['roc_auc_mean']:.4f} (+/- {result['roc_auc_std']:.4f})")
        print(f"  interpretation         : {result['verdict']}")
        print("  top features           :")
        for name, imp in result["top_leaking_features"]:
            print(f"      {name:<22} {imp:+.4f}")

    print("\n" + "=" * 78)
    print("TIER 3 - NEAR-DUPLICATE LEAKAGE   (is the test set actually held out?)")
    print("  Perceptual (difference-hash) match, robust to resize/re-encode.")
    print("  REQUIREMENT: zero cross-split matches. Any are memorization, not skill.")
    print("=" * 78)

    dupes = find_duplicate_leakage(rows)
    report["duplicate_leakage"] = dupes
    print(f"\n  images hashed                  : {dupes['n_hashed']}")
    print(f"  exact-duplicate groups         : {dupes['exact_duplicate_groups']}"
          f" ({dupes['exact_duplicate_images']} images)")
    print(f"  duplicate groups spanning splits: {dupes['cross_split_exact_groups']}")
    print(f"  duplicate groups spanning CLASSES: {dupes['cross_class_exact_groups']}"
          f"   <-- any >0 means contradictory labels")
    print(f"  near-duplicate cross-split pairs : {dupes['near_duplicate_cross_split_pairs']}")

    leak_total = (
        dupes["cross_split_exact_groups"]
        + dupes["cross_class_exact_groups"]
        + dupes["near_duplicate_cross_split_pairs"]
    )
    print(f"\n  verdict: {'OK - no cross-split leakage detected' if leak_total == 0 else f'LEAKAGE - {leak_total} problem group(s)/pair(s), fix before evaluating'}")
    for label, key in [
        ("cross-split", "examples_cross_split"),
        ("cross-class", "examples_cross_class"),
        ("near cross-split", "examples_near_cross_split"),
    ]:
        if dupes[key]:
            print(f"    {label} examples:")
            for example in dupes[key]:
                print(f"      {example}")

    if args.json_out:
        out_path = Path(args.json_out).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nJSON report -> {out_path}")


if __name__ == "__main__":
    main()
