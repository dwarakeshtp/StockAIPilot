"""Map domain exceptions to HTTP responses."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.db.repository import DatabaseNotInitialisedError
from src.services.pipeline import UnknownItemError, UnknownRunError


def _message(exc: Exception) -> str:
    return str(exc.args[0]) if exc.args else exc.__class__.__name__


def register_error_handlers(app: FastAPI) -> None:
    """404 for unknown items/runs, 503 for an empty database."""

    @app.exception_handler(UnknownItemError)
    async def _unknown_item(_: Request, exc: UnknownItemError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": f"Unknown item: {_message(exc)}"})

    @app.exception_handler(UnknownRunError)
    async def _unknown_run(_: Request, exc: UnknownRunError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": f"Run not found: {_message(exc)}"})

    @app.exception_handler(DatabaseNotInitialisedError)
    async def _not_initialised(_: Request, exc: DatabaseNotInitialisedError) -> JSONResponse:
        return JSONResponse(status_code=503, content={"detail": _message(exc)})
