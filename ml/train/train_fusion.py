"""
Learns the fusion weights from data, replacing the hand-picked constants
that R3 measured at ROC-AUC 0.4331 (below chance).

RUN WITH THE **ML** VENV:

    cd ml/train
    ../.venv/Scripts/python.exe train_fusion.py

What this fixes, concretely. R3's per-signal ablation found five of six
applicable signals were INVERTED -- they scored real images as more
suspicious than fakes. `lighting_analysis` sat at AUC 0.281, which means
that flipped it carries 0.719 of genuine signal. Fixed equal weights cannot
express a negative coefficient, so the old engine was actively summing
evidence in the wrong direction. A logistic regression can, and does.

DATA HYGIENE -- fitted on VAL, not train:
the linear probe was trained on the train split, so its scores there are
in-sample and optimistic. Fitting fusion on train would teach it to
over-trust the probe. Val is out-of-sample for both the probe and the
classical signals, so it is the only honest place to fit the combiner.
Test is touched once, at the end.

Exports plain JSON so the backend applies it with numpy alone -- no
scikit-learn dependency and no pickle loading in the serving path.
"""

import argparse
import csv
import json
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Signals that actually fire on still images (R3 found the other five are
# video-only and never populate for an image dataset).
CLASSICAL_SIGNALS = [
    "deep_learning",
    "frequency_analysis",
    "compression_analysis",
    "lighting_analysis",
    "broadcast_overlay_analysis",
    "crowd_texture_analysis",
]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def bootstrap_auc_ci(y_true, y_score, n_boot=2000, seed=0):
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(seed)
    y_true, y_score = np.asarray(y_true), np.asarray(y_score)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        if len(set(y_true[idx].tolist())) < 2:
            continue
        stats.append(roc_auc_score(y_true[idx], y_score[idx]))
    if not stats:
        return None, None
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=str, default="../../reports/evaluation/predictions.csv")
    parser.add_argument("--embeddings", type=str, default="../../datasets/embeddings.npz")
    parser.add_argument("--probe-head", type=str, default="../../models/configs/probe_head.json")
    parser.add_argument("--out", type=str, default="../../models/configs/fusion_calibration.json")
    parser.add_argument("--report", type=str, default="../../reports/evaluation/fusion_training.json")
    args = parser.parse_args()

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

    # --- probe scores from the frozen embeddings + exported head ---
    head = json.loads(Path(args.probe_head).resolve().read_text(encoding="utf-8"))
    emb = np.load(Path(args.embeddings).resolve(), allow_pickle=False)
    mean, std = np.array(head["mean"]), np.array(head["std"])
    weights, bias = np.array(head["weights"]), head["bias"]
    probe_scores = sigmoid(((emb["embeddings"] - mean) / std) @ weights + bias)
    probe_by_path = dict(zip(emb["paths"].astype(str), probe_scores))

    # --- classical signal scores from the R3 inference cache ---
    with Path(args.predictions).resolve().open(encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["status"] == "ok"]

    features, labels, splits, domains, kept_paths = [], [], [], [], []
    for row in rows:
        path = row["path"]
        if path not in probe_by_path:
            continue
        vector = [probe_by_path[path]]
        usable = True
        for name in CLASSICAL_SIGNALS:
            raw = row.get(f"signal_{name}", "")
            if row.get(f"applicable_{name}") != "1" or raw == "":
                usable = False
                break
            vector.append(float(raw))
        if not usable:
            continue
        features.append(vector)
        labels.append(1 if row["true_class"] == "fake" else 0)
        splits.append(row["split"])
        domains.append(row["domain"])
        kept_paths.append(path)

    X = np.array(features, dtype=np.float64)
    y = np.array(labels)
    splits = np.array(splits)
    domains = np.array(domains)
    feature_names = ["probe"] + CLASSICAL_SIGNALS

    val_mask, test_mask = splits == "val", splits == "test"
    logger.info(
        "usable rows: %d (val=%d, test=%d) with %d features",
        len(X), val_mask.sum(), test_mask.sum(), X.shape[1],
    )

    # Fitted on val only -- see the data-hygiene note in the module docstring.
    # C=1.0 with 7 features against ~113 rows is a reasonable ratio; stronger
    # regularisation would just shrink everything toward the broken uniform
    # weighting this is meant to replace.
    model = LogisticRegression(max_iter=5000, C=1.0, random_state=0)
    model.fit(X[val_mask], y[val_mask])

    coefficients = dict(zip(feature_names, model.coef_[0].round(4).tolist()))
    logger.info("Learned coefficients: %s", coefficients)

    def fused(mask):
        return model.predict_proba(X[mask])[:, 1]

    # Operating threshold from val, not test.
    val_scores = fused(val_mask)
    fpr, tpr, thresholds = roc_curve(y[val_mask], val_scores)
    threshold = float(thresholds[np.argmax(tpr - fpr)])

    report = {
        "fitted_on": "val",
        "features": feature_names,
        "coefficients": coefficients,
        "intercept": round(float(model.intercept_[0]), 4),
        "operating_threshold_from_val": round(threshold, 4),
        "splits": {},
        "per_signal_direction": {},
    }

    for name, mask in (("val", val_mask), ("test", test_mask)):
        if mask.sum() == 0:
            continue
        s = fused(mask)
        auc = float(roc_auc_score(y[mask], s))
        lo, hi = bootstrap_auc_ci(y[mask], s)
        report["splits"][name] = {
            "n": int(mask.sum()),
            "n_fake": int(y[mask].sum()),
            "roc_auc": round(auc, 4),
            "roc_auc_ci95": [round(lo, 4), round(hi, 4)] if lo else None,
            "accuracy": round(float(accuracy_score(y[mask], (s >= threshold).astype(int))), 4),
            "predicted_fake_rate": round(float((s >= threshold).mean()), 4),
            "actual_fake_rate": round(float(y[mask].mean()), 4),
        }

    # Standalone AUC per input feature on test, to show which the combiner
    # was right to trust and which it correctly down-weighted or flipped.
    for i, name in enumerate(feature_names):
        raw_auc = float(roc_auc_score(y[test_mask], X[test_mask, i]))
        report["per_signal_direction"][name] = {
            "standalone_test_auc": round(raw_auc, 4),
            "auc_if_flipped": round(1 - raw_auc, 4),
            "learned_coefficient": coefficients[name],
            "interpretation": (
                "learned NEGATIVE weight -- signal is inverted, engine now flips it"
                if coefficients[name] < 0
                else "learned positive weight"
            ),
        }

    report["test_by_domain"] = {}
    for domain in sorted(set(domains[test_mask])):
        dmask = test_mask & (domains == domain)
        if len(set(y[dmask].tolist())) < 2:
            report["test_by_domain"][domain] = {"n": int(dmask.sum()), "note": "single class"}
            continue
        report["test_by_domain"][domain] = {
            "n": int(dmask.sum()),
            "roc_auc": round(float(roc_auc_score(y[dmask], fused(dmask))), 4),
        }

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "_comment": (
                    "Learned fusion. Apply as: p = sigmoid(dot(weights, [probe, "
                    + ", ".join(CLASSICAL_SIGNALS)
                    + "]) + intercept). Requires ALL listed signals to be applicable; "
                    "if any is missing the backend falls back to the legacy weighted mean."
                ),
                "features": feature_names,
                "weights": model.coef_[0].tolist(),
                "intercept": float(model.intercept_[0]),
                "operating_threshold": round(threshold, 4),
                "fitted_on": {"split": "val", "n": int(val_mask.sum())},
                "metrics": report["splits"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    report_path = Path(args.report).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n" + "=" * 74)
    print("LEARNED FUSION RESULTS")
    print("=" * 74)
    print(f"\n  fitted on val (n={int(val_mask.sum())}), threshold {threshold:.4f}\n")
    print(f"  {'feature':<28} {'standalone':>11} {'flipped':>9} {'coef':>9}")
    print("  " + "-" * 60)
    for name, info in report["per_signal_direction"].items():
        print(f"  {name:<28} {info['standalone_test_auc']:>11.4f} {info['auc_if_flipped']:>9.4f}"
              f" {info['learned_coefficient']:>9.3f}")
    print()
    for name, m in report["splits"].items():
        print(f"  {name:<5} n={m['n']:<4} ROC-AUC {m['roc_auc']:.4f} CI95 {m['roc_auc_ci95']}"
              f"  acc {m['accuracy']:.4f}  flag-rate {m['predicted_fake_rate']:.3f}"
              f" (actual {m['actual_fake_rate']:.3f})")
    print("\n  test by domain:")
    for domain, m in report["test_by_domain"].items():
        print(f"    {domain:<10} n={m['n']:<4} {m.get('roc_auc', m.get('note'))}")
    print(f"\n  calibration -> {out_path}")


if __name__ == "__main__":
    main()
