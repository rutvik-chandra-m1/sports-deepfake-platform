"""
Aggregates all v1 routers.

Future milestones append their routers here, e.g.:
    from app.api.v1 import media, analysis, history
    api_router.include_router(media.router, prefix="/media", tags=["Media"])
    api_router.include_router(analysis.router, prefix="/analysis", tags=["Analysis"])
    api_router.include_router(history.router, prefix="/history", tags=["History"])
"""

from fastapi import APIRouter

from app.api.v1 import analysis, evidence, health, media

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(media.router)
api_router.include_router(analysis.router)
# Registered after `analysis` so its more specific paths
# (/{id}/media, /{id}/evidence/...) are matched by their own handlers.
api_router.include_router(evidence.router)
