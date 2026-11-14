"""Agent workflow endpoints: forecasts, pipeline runs, traces and what-if re-plans."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query

from src.agents.schemas import AgentTraceEntry, PipelineResult
from src.api.deps import get_services
from src.api.schemas import ForecastResponse, RunRequest, RunSummary, WhatIfRequest
from src.db.stores import RUN_KIND_PIPELINE
from src.services import Services

router = APIRouter(tags=["agents"])
Svc = Annotated[Services, Depends(get_services)]


@router.post("/forecasts/{item_id}", response_model=ForecastResponse, tags=["forecasting"])
def forecast(item_id: str, services: Svc) -> ForecastResponse:
    """Run the data pipeline plus the Forecast and Risk agents (nothing is stored)."""
    return ForecastResponse(**services.pipeline.forecast(item_id))


@router.post("/runs", response_model=PipelineResult)
def run_pipeline(body: RunRequest, services: Svc) -> PipelineResult:
    """Run all four agents for one item synchronously and store the result."""
    try:
        return services.pipeline.run_item(body.item_id, body.overrides or None)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None


@router.get("/runs", response_model=list[RunSummary])
def list_runs(
    services: Svc,
    item_id: str | None = None,
    kind: str | None = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 50,
) -> list[RunSummary]:
    """Run log, newest first."""
    return [RunSummary(**r) for r in services.pipeline.run_store.list(item_id, kind, limit)]


@router.get("/runs/latest", response_model=list[PipelineResult])
def latest_runs(
    services: Svc,
    item_ids: Annotated[list[str] | None, Query()] = None,
    kind: str = RUN_KIND_PIPELINE,
) -> list[PipelineResult]:
    """Latest run per item (optionally restricted to ``item_ids``)."""
    return list(services.pipeline.latest(item_ids, kind=kind).values())


@router.get("/runs/{run_id}", response_model=PipelineResult)
def get_run(run_id: str, services: Svc) -> PipelineResult:
    """One stored run with every agent output."""
    return services.pipeline.get_run(run_id)


@router.get("/runs/{run_id}/trace", response_model=list[AgentTraceEntry])
def get_trace(run_id: str, services: Svc) -> list[AgentTraceEntry]:
    """Step-by-step agent trace of a run."""
    return services.pipeline.get_run(run_id).trace


@router.post("/runs/{run_id}/what-if", response_model=PipelineResult)
def what_if(run_id: str, body: WhatIfRequest, services: Svc) -> PipelineResult:
    """Re-plan inventory and re-run the Critic with overrides, reusing the run's forecast.

    Supported overrides: ``lead_time_days``, ``lead_time_std``, ``on_hand``,
    ``service_level`` and ``demand_multiplier``.
    """
    try:
        return services.pipeline.what_if(body.overrides, run_id=run_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
