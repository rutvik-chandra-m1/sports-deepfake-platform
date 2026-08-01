"""Response schemas for the health-check endpoint."""

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(..., examples=["ok"])
    app_name: str
    app_version: str
    environment: str
