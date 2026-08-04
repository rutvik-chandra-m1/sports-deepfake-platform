"""
Application entrypoint.

Run locally with:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

Or via the helper script:
    ../scripts/run_backend.sh
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import RequestResponseEndpoint
from starlette.responses import Response

from app.api.v1.router import api_router
from app.core.config import get_settings
from app.core.logging_config import configure_logging
from app.core.security import (
    SlidingWindowRateLimiter,
    client_key,
    validate_production_settings,
)
from app.db.session import init_db
from app.services.jobs import recover_stale_records, shutdown_executor

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ----- Startup -----
    configure_logging()
    settings = get_settings()

    # Refuse to boot with development defaults outside development (R9).
    # Failing loudly here beats serving traffic with the shipped placeholder
    # secret and no authentication.
    problems = validate_production_settings(settings)
    if problems:
        for problem in problems:
            logger.critical("Unsafe configuration for APP_ENV=%s: %s", settings.app_env, problem)
        raise RuntimeError(
            f"Refusing to start with {len(problems)} unsafe setting(s) for APP_ENV="
            f"{settings.app_env}. See the log above and docs/security.md."
        )
    if not settings.api_key:
        logger.warning(
            "API_KEY is not set -- all endpoints are UNAUTHENTICATED. Acceptable for local "
            "development only; set API_KEY before exposing this service."
        )

    settings.ensure_runtime_directories()
    init_db()

    # Reconcile anything a previous process left mid-flight (R10). Must run
    # before the pool accepts work, while "nothing is legitimately running"
    # still holds.
    recover_stale_records()

    logger.info("Starting %s v%s (%s)", settings.app_name, settings.app_version, settings.app_env)
    yield
    # ----- Shutdown -----
    # Don't wait: a long analysis must not hang shutdown. Whatever is
    # mid-flight gets recovered on the next startup.
    shutdown_executor(wait=False)
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

    # ----- R9 security middleware -----
    # Two budgets: uploads run the full detection pipeline and are far more
    # expensive than reads, so they get a tighter one.
    general_limiter = SlidingWindowRateLimiter(
        settings.rate_limit_requests, settings.rate_limit_window_seconds
    )
    upload_limiter = SlidingWindowRateLimiter(
        settings.upload_rate_limit_requests, settings.upload_rate_limit_window_seconds
    )

    @app.middleware("http")
    async def rate_limit_and_harden(request: Request, call_next: RequestResponseEndpoint) -> Response:
        key = client_key(request)
        is_upload = request.url.path.endswith("/media/upload")
        limiter = upload_limiter if is_upload else general_limiter

        if not limiter.allow(key):
            retry_after = (
                settings.upload_rate_limit_window_seconds
                if is_upload
                else settings.rate_limit_window_seconds
            )
            logger.warning("Rate limit exceeded for %s on %s", key, request.url.path)
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={"detail": "Rate limit exceeded. Please retry shortly."},
                headers={"Retry-After": str(retry_after)},
            )

        response = await call_next(request)

        # Defensive headers. This API serves JSON to a separate SPA origin, so
        # a restrictive CSP costs nothing here and blocks the API being framed
        # or its responses being MIME-sniffed into something executable.
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = "default-src 'none'; frame-ancestors 'none'"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
        return response

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
