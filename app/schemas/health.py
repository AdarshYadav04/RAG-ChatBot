"""Health check schemas."""

from typing import Dict, Optional
from pydantic import BaseModel


class ComponentStatus(BaseModel):
    status: str
    latency_ms: Optional[float] = None
    details: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str
    components: Dict[str, ComponentStatus]
    uptime_seconds: float
