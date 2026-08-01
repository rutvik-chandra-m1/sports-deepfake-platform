"""
Application entrypoint.

Run locally with:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

Or via the helper script:
    ../scripts/run_backend.sh
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from app.api.v1.router import api_router
from app.core.config import get_settings
from app.core.logging_config import configure_logging
from app.db.session import init_db

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ----- Startup -----
    configure_logging()
    settings = get_settings()
    settings.ensure_runtime_directories()
    init_db()
    logger.info("Starting %s v%s (%s)", settings.app_name, settings.app_version, settings.app_env)
    yield
    # ----- Shutdown -----
    logger.info("Shutting down %s", settings.app_name)


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="REST API for AI-powered sports deepfake detection and verification.",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    # Milestone 15: detector_breakdown JSON (full per-signal scores/weights/
    # reasons) can run several KB per record; compress responses over 1KB.
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    app.include_router(api_router, prefix="/api/v1")

    @app.get("/", tags=["Root"], summary="API root")
    def root() -> dict:
        return {
            "message": f"{settings.app_name} API",
            "version": settings.app_version,
            "docs": "/docs",
        }

    return app


app = create_app()
