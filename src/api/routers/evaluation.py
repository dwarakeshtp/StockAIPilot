"""Evaluation endpoints."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query

from src.api.deps import get_services
from src.services import Services
from src.services.evaluation import TABLES

router = APIRouter(prefix="/evaluation", tags=["evaluation"])
Svc = Annotated[Services, Depends(get_services)]


@router.get("/summary")
def evaluation_summary(services: Svc) -> dict[str, Any]:
    """Output of ``scripts/run_evaluation.py`` (hold-out, simulation, scenarios)."""
    summary = services.evaluation.summary()
    if summary is None:
        raise HTTPException(status_code=404, detail="No evaluation yet. Run "
                                                    "`python scripts/run_evaluation.py`.")
    return summary


@router.get("/tables/{name}")
def evaluation_table(name: str, services: Svc) -> list[dict[str, Any]]:
    """One evaluation table: ``holdout_by_model``, ``inventory_simulation`` or
    ``scenario_evaluation``."""
    if name not in TABLES:
        raise HTTPException(status_code=404, detail=f"Unknown table. Choose from {list(TABLES)}")
    try:
        return services.evaluation.table(name)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"{name} not generated yet.") from None


@router.get("/holdout")
def holdout(services: Svc,
            item_ids: Annotated[list[str] | None, Query()] = None) -> dict[str, Any]:
    """Live hold-out accuracy of the latest stored runs."""
    return services.evaluation.holdout(item_ids)
