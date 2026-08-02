"""
Extracts ViT embeddings for every image in the dataset, once, so a
classifier head can be trained on them cheaply.

RUN WITH THE **BACKEND** VENV (needs torch/transformers):

    cd ml/train
    ../../backend/.venv/Scripts/python.exe extract_embeddings.py

Why a linear probe rather than full fine-tuning: this machine has no GPU
(Intel i7-8665U). Backpropagating through ViT-B/16 on CPU is impractical,
but a *frozen* backbone needs only one forward pass per image (~0.3s), and
the resulting 768-d features train a logistic head in milliseconds. Linear
probing is the standard way to measure what a frozen representation already
knows, and it is exactly the transfer-learning step the PPT calls for.

Reuses the ALREADY-CACHED deepfake ViT rather than downloading a second
backbone: its trunk is google/vit-base-patch16-224-in21k, so the features
are general-purpose even though its classification head was tuned for faces.
R3 measured that head at ROC-AUC 0.491 (chance) on this data precisely
because the head is face-specific -- the trunk beneath it is untested and is
what this script harvests.
"""

import argparse
import csv
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BACKEND_DIR = REPO_ROOT / "backend"
os.environ.setdefault("DATABASE_URL", "sqlite:///./_train_scratch.db")
sys.path.insert(0, str(BACKEND_DIR))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets-dir", type=str, default="../../datasets")
    parser.add_argument("--manifest", type=str, default="manifest_normalized.csv")
    parser.add_argument("--out", type=str, default="../../datasets/embeddings.npz")
    args = parser.parse_args()

    import cv2  # noqa: E402
    import torch  # noqa: E402

    from app.core.config import get_settings  # noqa: E402
    from app.services.detection.image_detector import _load_model  # noqa: E402

    settings = get_settings()
    model_id = settings.deepfake_image_model_id
    logger.info("Loading backbone %s ...", model_id)
    model, processor = _load_model(model_id)

    datasets_dir = Path(args.datasets_dir).resolve()
    with (datasets_dir / args.manifest).open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    logger.info("Extracting embeddings for %d images", len(rows))

    embeddings, labels, splits, domains, paths, generators = [], [], [], [], [], []
    started = time.monotonic()

    for i, entry in enumerate(rows, start=1):
        image_bgr = cv2.imread(str(datasets_dir / entry["path"]))
        if image_bgr is None:
            logger.warning("Unreadable, skipping: %s", entry["path"])
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        inputs = processor(images=image_rgb, return_tensors="pt")
        with torch.no_grad():
            # The ViT trunk, not the classification head. CLS token of the
            # final layer is the standard sentence-level/image-level summary.
            outputs = model.vit(**inputs)
            cls_token = outputs.last_hidden_state[:, 0]  # (1, 768)

        embeddings.append(cls_token.squeeze(0).numpy().astype(np.float32))
        labels.append(1 if entry["class"] == "fake" else 0)
        splits.append(entry["split"])
        domains.append(entry["domain"])
        paths.append(entry["path"])
        generators.append(entry.get("generator", ""))

        if i % 50 == 0 or i == len(rows):
            per = (time.monotonic() - started) / i
            logger.info("[%d/%d] %.2fs/image, ~%.0fs remaining", i, len(rows), per, (len(rows) - i) * per)

    out_path = Path(args.out).resolve()
    np.savez_compressed(
        out_path,
        embeddings=np.stack(embeddings),
        labels=np.array(labels),
        splits=np.array(splits),
        domains=np.array(domains),
        paths=np.array(paths),
        generators=np.array(generators),
        model_id=np.array([model_id]),
    )
    logger.info("Wrote %s embeddings of dim %d -> %s", len(embeddings), embeddings[0].shape[0], out_path)


if __name__ == "__main__":
    main()
