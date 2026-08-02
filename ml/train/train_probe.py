"""
Trains a linear probe on frozen ViT embeddings: the project's actual
transfer-learning step, and the fix for R3's finding that the stock
face-tuned classification head scores ROC-AUC 0.491 (chance) on this data.

RUN WITH THE **ML** VENV (needs scikit-learn):

    cd ml/train
    ../.venv/Scripts/python.exe train_probe.py

Method: L2-regularised logistic regression on 768-d CLS features, with the
regularisation strength chosen by cross-validation *within the training
split only*. Val is used to pick the operating threshold; test is touched
exactly once, for the final report.

Exports plain JSON (weights, bias, feature standardisation) rather than a
pickled sklearn object, so the backend can apply the head with numpy alone
and needs no scikit-learn dependency, no pickle deserialisation of a file
on disk, and no version-coupling between training and serving.
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


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
    parser.add_argument("--embeddings", type=str, default="../../datasets/embeddings.npz")
    parser.add_argument("--out", type=str, default="../../models/configs/probe_head.json")
    parser.add_argument("--report", type=str, default="../../reports/evaluation/probe_training.json")
    args = parser.parse_args()

    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import Pipeline

    data = np.load(Path(args.embeddings).resolve(), allow_pickle=False)
    X, y = data["embeddings"], data["labels"]
    splits, domains = data["splits"].astype(str), data["domains"].astype(str)
    model_id = str(data["model_id"][0])

    train_mask, val_mask, test_mask = splits == "train", splits == "val", splits == "test"
    logger.info(
        "train=%d (fake %d) | val=%d (fake %d) | test=%d (fake %d) | dim=%d",
        train_mask.sum(), y[train_mask].sum(),
        val_mask.sum(), y[val_mask].sum(),
        test_mask.sum(), y[test_mask].sum(), X.shape[1],
    )

    # Standardise using TRAIN statistics only -- computing them over the whole
    # dataset would leak val/test distribution into the fitted model.
    mean, std = X[train_mask].mean(axis=0), X[train_mask].std(axis=0)
    std[std < 1e-6] = 1.0

    def z(a):
        return (a - mean) / std

    # PCA before the logistic head. Without it, 768 features against ~318
    # training rows memorises the training set (measured: train AUC 0.974 vs
    # val 0.612). Both the component count and the regularisation strength are
    # chosen by cross-validation *inside train* -- never on val, which is
    # reserved for picking the operating threshold.
    grid = GridSearchCV(
        Pipeline([
            ("pca", PCA(random_state=0)),
            ("clf", LogisticRegression(max_iter=5000, random_state=0)),
        ]),
        param_grid={
            "pca__n_components": [8, 16, 32, 64, 128],
            "clf__C": np.logspace(-3, 2, 10),
        },
        cv=5,
        scoring="roc_auc",
        n_jobs=-1,
    )
    grid.fit(z(X[train_mask]), y[train_mask])
    best = grid.best_estimator_
    n_components = best.named_steps["pca"].n_components
    C = best.named_steps["clf"].C
    logger.info(
        "Chosen n_components=%d, C=%.5f (train CV AUC %.4f)",
        n_components, C, grid.best_score_,
    )

    def scores(mask):
        return best.predict_proba(z(X[mask]))[:, 1]

    # Collapse standardisation + PCA + linear head into ONE linear map in the
    # original 768-d space, so serving needs only a dot product:
    #   logit = w_pca . (components @ (z - pca_mean)) + b
    #         = (w_pca @ components) . z  -  (w_pca @ components) . pca_mean + b
    pca = best.named_steps["pca"]
    clf = best.named_steps["clf"]
    effective_w = clf.coef_[0] @ pca.components_          # (768,)
    effective_b = float(clf.intercept_[0] - effective_w @ pca.mean_)

    # Verify the collapse is exact before exporting it -- a silent algebra
    # error here would ship a head that behaves differently in production
    # than it did in training.
    check_logits_pipeline = best.decision_function(z(X[test_mask]))
    check_logits_linear = z(X[test_mask]) @ effective_w + effective_b
    max_drift = float(np.abs(check_logits_pipeline - check_logits_linear).max())
    if max_drift > 1e-6:
        raise SystemExit(f"Collapsed linear head disagrees with pipeline (max drift {max_drift:.2e})")
    logger.info("Collapsed head verified against pipeline (max logit drift %.2e)", max_drift)

    report = {
        "backbone": model_id,
        "method": "linear probe (logistic regression) on frozen ViT CLS embeddings",
        "feature_dim": int(X.shape[1]),
        "chosen_n_components": int(n_components),
        "chosen_C": float(C),
        "train_cv_auc": round(float(grid.best_score_), 4),
        "n_train": int(train_mask.sum()),
        "splits": {},
    }

    for name, mask in (("train", train_mask), ("val", val_mask), ("test", test_mask)):
        if mask.sum() == 0 or len(set(y[mask].tolist())) < 2:
            continue
        s = scores(mask)
        auc = float(roc_auc_score(y[mask], s))
        lo, hi = bootstrap_auc_ci(y[mask], s)
        report["splits"][name] = {
            "n": int(mask.sum()),
            "roc_auc": round(auc, 4),
            "roc_auc_ci95": [round(lo, 4), round(hi, 4)] if lo else None,
            "accuracy_at_0.5": round(float(accuracy_score(y[mask], (s >= 0.5).astype(int))), 4),
        }

    # Operating threshold picked on VAL (never test) by maximising Youden's J.
    val_scores = scores(val_mask)
    from sklearn.metrics import roc_curve

    fpr, tpr, thresholds = roc_curve(y[val_mask], val_scores)
    best_threshold = float(thresholds[np.argmax(tpr - fpr)])
    report["operating_threshold_from_val"] = round(best_threshold, 4)

    test_scores = scores(test_mask)
    report["test_accuracy_at_val_threshold"] = round(
        float(accuracy_score(y[test_mask], (test_scores >= best_threshold).astype(int))), 4
    )

    # Per-domain on test, so the sports subset is never silently folded in.
    report["test_by_domain"] = {}
    for domain in sorted(set(domains[test_mask])):
        dmask = test_mask & (domains == domain)
        if len(set(y[dmask].tolist())) < 2:
            report["test_by_domain"][domain] = {"n": int(dmask.sum()), "note": "single class, AUC undefined"}
            continue
        s = scores(dmask)
        report["test_by_domain"][domain] = {
            "n": int(dmask.sum()),
            "roc_auc": round(float(roc_auc_score(y[dmask], s)), 4),
        }

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "_comment": (
                    "Linear probe head over frozen ViT CLS embeddings. Apply as: "
                    "p = sigmoid(dot(w, (embedding - mean) / std) + b). Plain JSON on purpose "
                    "-- the backend applies this with numpy and needs no scikit-learn."
                ),
                "backbone_model_id": model_id,
                "feature_dim": int(X.shape[1]),
                "mean": mean.tolist(),
                "std": std.tolist(),
                "weights": effective_w.tolist(),
                "bias": effective_b,
                "pca_n_components": int(n_components),
                "operating_threshold": round(best_threshold, 4),
                "trained_on": {
                    "n_train": int(train_mask.sum()),
                    "dataset_manifest": "datasets/manifest_normalized.csv",
                },
                "metrics": report["splits"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    report_path = Path(args.report).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n" + "=" * 70)
    print("LINEAR PROBE RESULTS")
    print("=" * 70)
    for name, m in report["splits"].items():
        ci = f" CI95 {m['roc_auc_ci95']}" if m.get("roc_auc_ci95") else ""
        print(f"  {name:<6} n={m['n']:<4} ROC-AUC {m['roc_auc']:.4f}{ci}   acc@0.5 {m['accuracy_at_0.5']:.4f}")
    print(f"\n  operating threshold (from val): {report['operating_threshold_from_val']}")
    print(f"  TEST accuracy at that threshold: {report['test_accuracy_at_val_threshold']}")
    print("\n  test by domain:")
    for domain, m in report["test_by_domain"].items():
        print(f"    {domain:<10} n={m['n']:<4} {m.get('roc_auc', m.get('note'))}")
    print(f"\n  head -> {out_path}")
    print(f"  report -> {report_path}")


if __name__ == "__main__":
    main()
