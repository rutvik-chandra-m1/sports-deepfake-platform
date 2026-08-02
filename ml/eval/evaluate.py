"""
Turns run_inference.py's prediction cache into the evaluation report:
headline metrics, per-signal standalone performance, baselines, per-domain
and per-generator breakdowns, and plots.

RUN THIS WITH THE **ML** VENV (needs scikit-learn/matplotlib):

    cd ml/eval
    ../.venv/Scripts/python.exe evaluate.py

Reporting principles, deliberately encoded here rather than left to whoever
reads the numbers:

* Every metric is reported on the TEST split only, unless explicitly
  labelled otherwise. Val exists for R4 calibration fitting.
* Every AUC is reported alongside a bootstrap 95% confidence interval. At
  pilot scale (test n<100) the interval is wide enough that ignoring it
  would be actively misleading.
* Accuracy is reported against the majority-class baseline, and AUC against
  the content-statistics baseline from the dataset audit -- NOT against
  0.50. A detector that beats chance but not "how saturated is this image"
  has demonstrated nothing.
* Per-signal AUC is computed on that signal's own applicable subset, and
  the subset size is always printed: a signal that only fires on 12 images
  has not been measured in any meaningful sense.
"""

import argparse
import csv
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SIGNALS = [
    "trained_probe",
    "deep_learning",
    "frequency_analysis",
    "compression_analysis",
    "lighting_analysis",
    "landmark_instability",
    "optical_flow_analysis",
    "temporal_consistency",
    "jersey_color_consistency",
    "scene_consistency",
    "broadcast_overlay_analysis",
    "crowd_texture_analysis",
]

# From reports/dataset_audit_normalized.json -- what a classifier achieves
# using only global image statistics (saturation/brightness/compressibility).
# The real bar for the detector, since beating 0.50 proves nothing.
# Re-read these from the audit JSON whenever the dataset is rebuilt; they
# moved from {0.5797, 0.6071} to the values below when the backbone grew
# from 500 to ~1790 images.
CONTENT_STATS_BASELINE = {"ALL": 0.5458, "general": 0.5323, "sports": 0.8350}


def bootstrap_auc_ci(y_true, y_score, n_boot=2000, seed=0):
    """Percentile bootstrap CI for ROC-AUC. Essential at pilot scale, where
    a point estimate alone invites over-reading a few dozen samples."""
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(seed)
    y_true, y_score = np.asarray(y_true), np.asarray(y_score)
    n = len(y_true)
    if n == 0 or len(set(y_true.tolist())) < 2:
        return None, None
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(set(y_true[idx].tolist())) < 2:
            continue
        stats.append(roc_auc_score(y_true[idx], y_score[idx]))
    if not stats:
        return None, None
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def load_predictions(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    ok = [r for r in rows if r["status"] == "ok"]
    if len(ok) != len(rows):
        logger.warning(
            "%d/%d rows had status != ok and are excluded from metrics",
            len(rows) - len(ok), len(rows),
        )
    return ok


def headline_metrics(rows: list[dict], threshold: float) -> dict:
    from sklearn.metrics import (
        accuracy_score, average_precision_score, confusion_matrix,
        f1_score, precision_score, recall_score, roc_auc_score,
    )

    y_true = np.array([1 if r["true_class"] == "fake" else 0 for r in rows])
    y_score = np.array([float(r["fused_suspicion_score"]) for r in rows])
    y_pred = (y_score >= threshold).astype(int)

    if len(set(y_true.tolist())) < 2:
        return {"error": "only one class present", "n": len(rows)}

    auc = roc_auc_score(y_true, y_score)
    lo, hi = bootstrap_auc_ci(y_true, y_score)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    majority = max(y_true.mean(), 1 - y_true.mean())

    return {
        "n": int(len(rows)),
        "n_real": int((y_true == 0).sum()),
        "n_fake": int((y_true == 1).sum()),
        "threshold": threshold,
        "roc_auc": round(float(auc), 4),
        "roc_auc_ci95": [round(lo, 4), round(hi, 4)] if lo is not None else None,
        "pr_auc": round(float(average_precision_score(y_true, y_score)), 4),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "majority_class_baseline": round(float(majority), 4),
        "precision_fake": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall_fake": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1_fake": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "predicted_fake_rate": round(float(y_pred.mean()), 4),
        "actual_fake_rate": round(float(y_true.mean()), 4),
        "score_mean_real": round(float(y_score[y_true == 0].mean()), 4),
        "score_mean_fake": round(float(y_score[y_true == 1].mean()), 4),
        "score_min": round(float(y_score.min()), 4),
        "score_max": round(float(y_score.max()), 4),
    }


def per_signal_metrics(rows: list[dict]) -> dict:
    """Each signal's standalone discriminative power, on its own applicable
    subset. This is the ablation that says which of the 11 signals are
    actually earning their place in the fusion."""
    from sklearn.metrics import roc_auc_score

    results = {}
    for signal in SIGNALS:
        subset = [r for r in rows if r.get(f"applicable_{signal}") == "1" and r.get(f"signal_{signal}")]
        y_true = np.array([1 if r["true_class"] == "fake" else 0 for r in subset])
        y_score = np.array([float(r[f"signal_{signal}"]) for r in subset])

        entry = {
            "n_applicable": len(subset),
            "applicable_rate": round(len(subset) / len(rows), 4) if rows else 0.0,
        }
        if len(subset) < 20 or len(set(y_true.tolist())) < 2:
            entry["auc"] = None
            entry["note"] = (
                "not applicable to any image in this subset"
                if not subset
                else f"too few applicable samples (n={len(subset)}) to measure"
            )
        else:
            auc = float(roc_auc_score(y_true, y_score))
            lo, hi = bootstrap_auc_ci(y_true, y_score)
            entry.update(
                {
                    "auc": round(auc, 4),
                    "auc_ci95": [round(lo, 4), round(hi, 4)] if lo is not None else None,
                    # AUC below 0.5 means the signal is anti-correlated: it
                    # scores real images as MORE suspicious than fakes.
                    "direction": "correct" if auc >= 0.5 else "INVERTED",
                    "mean_score_real": round(float(y_score[y_true == 0].mean()), 4),
                    "mean_score_fake": round(float(y_score[y_true == 1].mean()), 4),
                }
            )
        results[signal] = entry
    return results


def per_generator_metrics(rows: list[dict], threshold: float) -> dict:
    """Recall per generator -- which generators does this pipeline actually
    catch? A detector strong on one family and blind to another looks fine
    in aggregate and fails in deployment."""
    by_gen = defaultdict(list)
    for r in rows:
        if r["true_class"] == "fake" and r.get("generator"):
            by_gen[r["generator"]].append(r)

    results = {}
    for gen, items in sorted(by_gen.items(), key=lambda kv: -len(kv[1])):
        scores = np.array([float(r["fused_suspicion_score"]) for r in items])
        results[gen] = {
            "n": len(items),
            "recall": round(float((scores >= threshold).mean()), 4),
            "mean_score": round(float(scores.mean()), 4),
        }
    return results


def make_plots(rows: list[dict], out_dir: Path, threshold: float) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, roc_curve

    y_true = np.array([1 if r["true_class"] == "fake" else 0 for r in rows])
    y_score = np.array([float(r["fused_suspicion_score"]) for r in rows])
    written = []

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    fpr, tpr, _ = roc_curve(y_true, y_score)
    axes[0].plot(fpr, tpr, label="fused score")
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.4, label="chance")
    axes[0].set_xlabel("false positive rate")
    axes[0].set_ylabel("true positive rate")
    axes[0].set_title("ROC (test split)")
    axes[0].legend(loc="lower right", fontsize=8)

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    axes[1].plot(recall, precision)
    axes[1].axhline(y_true.mean(), ls="--", c="k", alpha=0.4, label="prevalence")
    axes[1].set_xlabel("recall")
    axes[1].set_ylabel("precision")
    axes[1].set_title("Precision-Recall (test split)")
    axes[1].legend(loc="lower left", fontsize=8)

    bins = np.linspace(0, 1, 26)
    axes[2].hist(y_score[y_true == 0], bins=bins, alpha=0.6, label="real", density=True)
    axes[2].hist(y_score[y_true == 1], bins=bins, alpha=0.6, label="fake", density=True)
    axes[2].axvline(threshold, c="r", ls="--", label=f"threshold={threshold}")
    axes[2].set_xlabel("fused suspicion score")
    axes[2].set_ylabel("density")
    axes[2].set_title("Score distribution by true class")
    axes[2].legend(fontsize=8)

    fig.tight_layout()
    path = out_dir / "evaluation_overview.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    written.append(str(path))
    return written


def print_section(title: str) -> None:
    print("\n" + "=" * 76)
    print(title)
    print("=" * 76)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=str, default="../../reports/evaluation/predictions.csv")
    parser.add_argument("--out-dir", type=str, default="../../reports/evaluation")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="fusion verdict threshold (settings.fusion_verdict_threshold)")
    args = parser.parse_args()

    predictions_path = Path(args.predictions).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = load_predictions(predictions_path)
    rows = [r for r in all_rows if r["split"] == args.split]
    if not rows:
        raise SystemExit(f"No rows for split={args.split} in {predictions_path}")

    report = {
        "split": args.split,
        "threshold": args.threshold,
        "n_scored_total": len(all_rows),
        "content_stats_baseline": CONTENT_STATS_BASELINE,
    }

    print_section(f"HEADLINE METRICS  (split={args.split}, threshold={args.threshold})")
    overall = headline_metrics(rows, args.threshold)
    report["overall"] = overall
    for key in ("n", "n_real", "n_fake", "roc_auc", "roc_auc_ci95", "pr_auc",
                "accuracy", "majority_class_baseline", "precision_fake",
                "recall_fake", "f1_fake", "predicted_fake_rate", "actual_fake_rate"):
        print(f"  {key:<26}: {overall.get(key)}")
    cm = overall["confusion_matrix"]
    print(f"  confusion matrix          : TN={cm['tn']} FP={cm['fp']} FN={cm['fn']} TP={cm['tp']}")
    print(f"  score range               : [{overall['score_min']}, {overall['score_max']}]")
    print(f"  mean score real / fake    : {overall['score_mean_real']} / {overall['score_mean_fake']}")

    baseline = CONTENT_STATS_BASELINE.get("ALL")
    print(f"\n  BARS TO BEAT:")
    print(f"    majority-class accuracy : {overall['majority_class_baseline']}"
          f"   -> detector accuracy {overall['accuracy']}")
    print(f"    content-stats AUC       : {baseline}"
          f"   -> detector AUC {overall['roc_auc']}")
    verdict = "BEATS" if overall["roc_auc"] > baseline else "DOES NOT BEAT"
    print(f"    => detector {verdict} the trivial content-statistics baseline")

    print_section("PER-DOMAIN")
    report["by_domain"] = {}
    for domain in sorted({r["domain"] for r in rows}):
        subset = [r for r in rows if r["domain"] == domain]
        metrics = headline_metrics(subset, args.threshold)
        report["by_domain"][domain] = metrics
        if "error" in metrics:
            print(f"\n  [{domain}] n={metrics['n']} -- {metrics['error']}")
            continue
        bl = CONTENT_STATS_BASELINE.get(domain)
        print(f"\n  [{domain}] n={metrics['n']} (real={metrics['n_real']}, fake={metrics['n_fake']})")
        print(f"    ROC-AUC  : {metrics['roc_auc']}  CI95={metrics['roc_auc_ci95']}"
              f"   (content-stats baseline {bl})")
        print(f"    accuracy : {metrics['accuracy']}  (majority {metrics['majority_class_baseline']})")
        print(f"    recall   : {metrics['recall_fake']}   precision: {metrics['precision_fake']}")

    print_section("PER-SIGNAL STANDALONE ABLATION  (which signals actually earn their weight?)")
    print("  AUC on each signal's own applicable subset. 'INVERTED' means the signal")
    print("  scores REAL images as more suspicious than fakes -- worse than useless.")
    signals = per_signal_metrics(rows)
    report["per_signal"] = signals
    print(f"\n  {'signal':<28} {'n':>5} {'appl%':>7} {'AUC':>7}  {'CI95':<18} {'dir':<9}")
    print("  " + "-" * 74)
    for name, info in signals.items():
        if info["auc"] is None:
            print(f"  {name:<28} {info['n_applicable']:>5} {info['applicable_rate']*100:>6.1f}%"
                  f" {'--':>7}  {'':<18} {info.get('note','')}")
        else:
            ci = f"[{info['auc_ci95'][0]:.3f}, {info['auc_ci95'][1]:.3f}]" if info.get("auc_ci95") else ""
            print(f"  {name:<28} {info['n_applicable']:>5} {info['applicable_rate']*100:>6.1f}%"
                  f" {info['auc']:>7.4f}  {ci:<18} {info['direction']}")

    print_section("PER-GENERATOR RECALL  (which generators does it catch?)")
    generators = per_generator_metrics(rows, args.threshold)
    report["per_generator"] = generators
    if generators:
        print(f"\n  {'generator':<40} {'n':>4} {'recall':>8} {'mean score':>11}")
        print("  " + "-" * 66)
        for gen, info in generators.items():
            print(f"  {gen:<40} {info['n']:>4} {info['recall']:>8.3f} {info['mean_score']:>11.4f}")
    else:
        print("\n  No generator metadata on fakes in this split.")

    plots = make_plots(rows, out_dir, args.threshold)
    report["plots"] = plots

    json_path = out_dir / f"evaluation_{args.split}.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nJSON report -> {json_path}")
    for plot in plots:
        print(f"Plot        -> {plot}")


if __name__ == "__main__":
    main()
