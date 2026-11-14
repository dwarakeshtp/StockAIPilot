"""FastAPI application: data, forecasting, recommendation and agent-workflow API.

Run with::

    uvicorn src.api.main:app --reload     # docs at http://localhost:8000/docs
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from starlette.concurrency import run_in_threadpool

from src.api.errors import register_error_handlers
from src.api.routers import decisions, evaluation, items, jobs, runs, system
from src.config.settings import Settings, load_settings
from src.db.loader import init_db
from src.logging.logger import configure_logging, get_logger
from src.services import Services

logger = get_logger(__name__)


def create_app(settings: Settings | None = None, services: Services | None = None) -> FastAPI:
    """Build the API.

    Args:
        settings: Settings to use (loaded from ``config/settings.yaml`` by default).
        services: Pre-built services (tests); skips database initialisation.
    """
    settings = settings or (services.settings if services else load_settings())

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        owned = services is None
        if owned:
            configure_logging(settings.outputs.logs_dir, level="INFO")
            settings.ensure_output_dirs()
            if settings.database.init_on_startup:
                report = await run_in_threadpool(init_db, settings,
                                                 settings.database.init_source)
                logger.info("database_ready", source=report.source, items=report.items,
                            skipped=report.skipped)
            app.state.services = Services.build(settings)
        else:
            app.state.services = services
        try:
            yield
        finally:
            if owned:
                app.state.services.close()
            app.state.services = None

    app = FastAPI(
        title="StockPilot AI API",
        version="1.0.0",
        description="Agentic demand forecasting and inventory reorder recommendations "
                    "(M5 CA_1 FOODS). Four LangGraph agents - Forecast, Risk, Inventory, "
                    "Critic - over a SQLite data layer.",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.services = None
    register_error_handlers(app)
    for module in (system, items, runs, jobs, decisions, evaluation):
        app.include_router(module.router)

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse("/docs")

    return app


app = create_app()
