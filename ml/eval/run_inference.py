"""
Runs the real detection pipeline over the labelled dataset and records every
signal's raw score per image. Produces the prediction cache that
`evaluate.py` turns into metrics.

RUN THIS WITH THE **BACKEND** VENV -- it imports the actual production
pipeline (`app.services.analysis_pipeline.analyze_frames`), so it needs
torch/opencv/mediapipe, not scikit-learn:

    cd ml/eval
    ../../backend/.venv/Scripts/python.exe run_inference.py

Deliberately calls the same `analyze_frames()` the API calls -- not a
reimplementation. If this scores well but the API behaves differently, that
is a bug worth knowing about; a parallel evaluation-only copy of the
pipeline would hide exactly that.

Resumable: re-running skips images already present in the output CSV, so a
long run can be interrupted and continued.
"""

import argparse
import csv
import logging
import os
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BACKEND_DIR = REPO_ROOT / "backend"

# Point at a throwaway DB before importing anything under app/: the pipeline
# module imports app.db.session, which builds an Engine at import time. Nothing
# here touches the database, but this guarantees the developer's real
# database/app.db is never even opened by an evaluation run.
os.environ.setdefault("DATABASE_URL", "sqlite:///./_eval_scratch.db")
sys.path.insert(0, str(BACKEND_DIR))

# Every signal the fusion engine knows about, in a stable column order.
ALL_SIGNALS = [
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

BASE_COLUMNS = [
    "path", "true_class", "domain", "split", "generator",
    "status", "fused_suspicion_score", "verdict", "confidence_score", "risk_level",
    "elapsed_s", "error",
]


def load_manifest(datasets_dir: Path, manifest_name: str, splits: set[str] | None) -> list[dict]:
    with (datasets_dir / manifest_name).open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if splits:
        rows = [r for r in rows if r["split"] in splits]
    return rows


def load_done(out_path: Path) -> set[str]:
    if not out_path.exists():
        return set()
    with out_path.open(encoding="utf-8") as f:
        return {row["path"] for row in csv.DictReader(f)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets-dir", type=str, default="../../datasets")
    parser.add_argument("--manifest", type=str, default="manifest_normalized.csv")
    parser.add_argument("--out", type=str, default="../../reports/evaluation/predictions.csv")
    parser.add_argument(
        "--splits", type=str, default="train,val,test",
        help="comma-separated splits to score; scoring all of them lets evaluate.py "
             "fit calibration on val and report on test without a second inference pass",
    )
    parser.add_argument("--limit", type=int, default=None, help="stop after N images (smoke test)")
    args = parser.parse_args()

    import cv2  # noqa: E402 - after sys.path setup

    from app.services.analysis_pipeline import analyze_frames  # noqa: E402
    from app.services.fusion_engine import FusionError  # noqa: E402

    datasets_dir = Path(args.datasets_dir).resolve()
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    splits = {s.strip() for s in args.splits.split(",") if s.strip()}
    rows = load_manifest(datasets_dir, args.manifest, splits)
    done = load_done(out_path)
    todo = [r for r in rows if r["path"] not in done]
    if args.limit:
        todo = todo[: args.limit]

    logger.info(
        "Manifest %s: %d rows in splits %s; %d already scored; %d to do",
        args.manifest, len(rows), sorted(splits), len(done), len(todo),
    )
    if not todo:
        logger.info("Nothing to do. Delete %s to force a full re-run.", out_path)
        return

    fieldnames = BASE_COLUMNS + [f"signal_{name}" for name in ALL_SIGNALS] + [
        f"applicable_{name}" for name in ALL_SIGNALS
    ]
    write_header = not out_path.exists()
    started = time.monotonic()

    with out_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for i, entry in enumerate(todo, start=1):
            image_path = datasets_dir / entry["path"]
            record = {
                "path": entry["path"],
                "true_class": entry["class"],
                "domain": entry["domain"],
                "split": entry["split"],
                "generator": entry.get("generator", ""),
                "status": "ok",
                "error": "",
            }

            t0 = time.monotonic()
            try:
                image = cv2.imread(str(image_path))
                if image is None:
                    raise ValueError(f"cv2 could not read {image_path}")

                outcome = analyze_frames([image])
                fusion = outcome.fusion
                signals = outcome.breakdown.get("signals", {})

                record.update(
                    {
                        "fused_suspicion_score": f"{fusion.fused_suspicion_score:.6f}",
                        "verdict": fusion.verdict.value,
                        "confidence_score": f"{fusion.confidence_score:.2f}",
                        "risk_level": fusion.risk_level.value,
                    }
                )
                for name in ALL_SIGNALS:
                    info = signals.get(name)
                    record[f"signal_{name}"] = f"{info['score']:.6f}" if info else ""
                    record[f"applicable_{name}"] = "1" if info else "0"

            except FusionError as exc:
                record.update({"status": "fusion_error", "error": str(exc)})
            except Exception as exc:  # noqa: BLE001 - one bad image must not kill a long run
                logger.warning("Failed on %s: %s", entry["path"], exc)
                record.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"})

            record["elapsed_s"] = f"{time.monotonic() - t0:.2f}"
            writer.writerow(record)
            f.flush()  # keep the cache durable so an interrupted run loses nothing

            if i % 10 == 0 or i == len(todo):
                per_image = (time.monotonic() - started) / i
                remaining = (len(todo) - i) * per_image
                logger.info(
                    "[%d/%d] %.2fs/image, ~%.0fs remaining", i, len(todo), per_image, remaining
                )

    logger.info("Wrote predictions -> %s", out_path)
    logger.info("Next: run evaluate.py with the ML venv (needs scikit-learn/matplotlib).")


if __name__ == "__main__":
    main()
