"""
Centralized logging configuration.

Call `configure_logging()` once at application startup. Every module should
then obtain its logger via `logging.getLogger(__name__)` rather than
configuring handlers itself.
"""

import logging
import sys

from app.core.config import get_settings

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def configure_logging() -> None:
    settings = get_settings()

    root_logger = logging.getLogger()
    root_logger.setLevel(settings.log_level.upper())

    # Avoid duplicate handlers on reload
    root_logger.handlers.clear()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
    root_logger.addHandler(console_handler)

    # Quiet noisy third-party loggers unless in debug mode
    if not settings.debug:
        logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

    logging.getLogger(__name__).info(
        "Logging configured (level=%s, env=%s)", settings.log_level.upper(), settings.app_env
    )
