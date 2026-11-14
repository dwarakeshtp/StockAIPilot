"""System endpoints: health, backend info and the pipeline graph."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Request

from src.api.deps import get_services
from src.api.schemas import Health, SystemInfo
from src.db.loader import read_meta
from src.orchestration.graph import pipeline_mermaid
from src.services import Services

router = APIRouter(tags=["system"])
Svc = Annotated[Services, Depends(get_services)]


@router.get("/health", response_model=Health)
def health(request: Request) -> Health:
    """Liveness probe; reports whether the database holds data."""
    services: Services | None = getattr(request.app.state, "services", None)
    if services is None:
        return Health(status="starting", database="unavailable")
    source = read_meta(services.engine).get("source")
    return Health(database="ready" if source else "empty", source=source)


@router.get("/system/info", response_model=SystemInfo)
def system_info(services: Svc) -> SystemInfo:
    """Data source, row counts, history window and LLM provider chain."""
    repo = services.repository
    meta = repo.meta
    start, end = repo.history_window
    return SystemInfo(
        data_source=repo.source, store_id=meta.get("store_id"), category=meta.get("category"),
        loaded_at=meta.get("loaded_at"), items=repo.counts()["items"],
        demo_items=repo.item_ids, history_start=start, history_end=end, forecast_origin=end,
        horizon_days=services.settings.forecast.horizon, counts=repo.counts(),
        llm=services.pipeline.llm_description, database_url=services.settings.database.url,
    )


@router.get("/system/graph")
def system_graph(services: Svc) -> dict[str, str]:
    """Mermaid source of the compiled LangGraph pipeline."""
    return {"mermaid": pipeline_mermaid(services.pipeline.runner.deps)}
