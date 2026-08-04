"""
Pre-download every model asset into the local cache.

Run at Docker BUILD time so the resulting image is self-contained. Without
this, the first upload after a deploy blocks for as long as it takes to pull
~330MB from the Hugging Face Hub -- and a container that needs the public
internet to answer its first request is not something you want to deploy
behind a firewall or scale horizontally.

Also useful on a laptop before demoing offline:

    cd backend && .venv/Scripts/python.exe ../scripts/warm_model_cache.py
"""

import logging
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("warm_model_cache")


def main() -> int:
    from app.core.config import get_settings

    settings = get_settings()
    Path(settings.models_dir).mkdir(parents=True, exist_ok=True)
    logger.info("Cache directory: %s", settings.models_dir)

    failures: list[str] = []

    # 1. Vision Transformer classifier (~330MB) -- the big one.
    try:
        from app.services.detection.image_detector import _load_model

        _load_model(settings.deepfake_image_model_id)
        logger.info("OK  transformer classifier (%s)", settings.deepfake_image_model_id)
    except Exception as exc:  # noqa: BLE001 -- report every failure, not the first
        failures.append(f"transformer classifier: {exc}")

    # 2. Haar cascade (~900KB).
    try:
        from app.services.detection.face_detection import _ensure_cascade_downloaded

        logger.info("OK  haar cascade (%s)", _ensure_cascade_downloaded().name)
    except Exception as exc:  # noqa: BLE001
        failures.append(f"haar cascade: {exc}")

    # 3. MediaPipe face landmarker bundle (~3MB).
    try:
        from app.services.detection.landmark_analysis import _ensure_model_downloaded

        logger.info("OK  mediapipe landmarker (%s)", _ensure_model_downloaded().name)
    except ImportError:
        logger.warning("SKIP mediapipe landmarker: module not importable in this environment")
    except Exception as exc:  # noqa: BLE001
        failures.append(f"mediapipe landmarker: {exc}")

    if failures:
        for failure in failures:
            logger.error("FAILED %s", failure)
        # Hard-fail the build. A silently half-warmed image looks fine until
        # it hits production and starts downloading weights per container.
        return 1

    logger.info("All model assets cached.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
