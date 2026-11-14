"""Recommendation, purchase-order and planner decision endpoints."""

from __future__ import annotations

import io
from typing import Annotated, Literal

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response

from src.api.deps import get_services
from src.api.schemas import DecisionRequest, Recommendation
from src.orchestration.decisions import Decision
from src.services import Services

router = APIRouter(tags=["decisions"])
Svc = Annotated[Services, Depends(get_services)]
PO_COLUMNS = ["item_id", "product", "department", "quantity", "unit_cost", "line_value",
              "decision", "reviewer", "reason", "run_id", "decided_at"]


@router.get("/recommendations", response_model=list[Recommendation],
            tags=["recommendations"])
def recommendations(
    services: Svc, item_ids: Annotated[list[str] | None, Query()] = None,
) -> list[Recommendation]:
    """Latest recommendation per item with its decision status."""
    return [Recommendation(**r) for r in services.pipeline.recommendations(item_ids)]


@router.get("/purchase-orders/draft", tags=["recommendations"])
def purchase_order_draft(services: Svc, format: Literal["csv", "json"] = "csv"):  # noqa: A002
    """Purchase-order lines from approved or modified decisions (CSV by default)."""
    lines = services.pipeline.po_draft()
    if format == "json":
        return lines
    buffer = io.StringIO()
    pd.DataFrame(lines, columns=PO_COLUMNS).to_csv(buffer, index=False)
    return Response(content=buffer.getvalue(), media_type="text/csv",
                    headers={"Content-Disposition": 'attachment; filename="po_draft.csv"'})


@router.post("/decisions", response_model=Decision, status_code=201)
def record_decision(body: DecisionRequest, services: Svc) -> Decision:
    """Record APPROVED / MODIFIED / REJECTED for a stored run.

    Returns 400 with a plain-language message when the decision breaks a rule.
    """
    try:
        return services.decisions.record(body.run_id, body.decision, body.reason,
                                         final_qty=body.final_qty, reviewer=body.reviewer)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None


@router.get("/decisions", response_model=list[Decision])
def list_decisions(services: Svc, item_id: str | None = None) -> list[Decision]:
    """Decision log, oldest first."""
    return services.decisions.list(item_id)


@router.get("/decisions/latest", response_model=list[Decision])
def latest_decisions(services: Svc) -> list[Decision]:
    """Most recent decision per item."""
    return list(services.decisions.latest_by_item().values())
