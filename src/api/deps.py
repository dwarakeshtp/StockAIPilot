"""FastAPI dependency getters."""

from __future__ import annotations

from fastapi import HTTPException, Request

from src.services import Services


def get_services(request: Request) -> Services:
    """Services built by the lifespan (503 while the backend is not ready)."""
    services = getattr(request.app.state, "services", None)
    if services is None:
        raise HTTPException(status_code=503, detail="Backend is starting or the database "
                                                    "is not initialised.")
    return services
