"""Data and inventory endpoints: items, sales, history, actuals, calendar, stock."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query

from src.api.deps import get_services
from src.api.schemas import (
    InventoryPosition,
    InventoryUpdate,
    Item,
    ItemDetail,
    ItemPage,
    Rows,
    frame_records,
)
from src.services import Services
from src.services.pipeline import UnknownItemError

router = APIRouter(tags=["data"])
Svc = Annotated[Services, Depends(get_services)]


def _require(services: Services, item_id: str) -> None:
    if not services.repository.has_item(item_id):
        raise UnknownItemError(item_id)


def _inventory(services: Services, item_id: str) -> InventoryPosition:
    params = services.repository.item_inventory_params(item_id)
    return InventoryPosition(**params,
                             updated_at=services.repository.inventory_updated_at(item_id))


@router.get("/items", response_model=ItemPage)
def list_items(
    services: Svc,
    search: Annotated[str | None, Query(description="Matches item code or product name")] = None,
    department: str | None = None,
    demo_only: bool = False,
    limit: Annotated[int, Query(ge=1, le=500)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> ItemPage:
    """Page through the item master (demo items first)."""
    rows, total = services.repository.search_items(search, department, demo_only, limit, offset)
    return ItemPage(items=[Item(**r) for r in rows], total=total, limit=limit, offset=offset)


@router.get("/items/{item_id}", response_model=ItemDetail)
def get_item(item_id: str, services: Svc) -> ItemDetail:
    """Item metadata, inventory position, sales statistics and latest run id."""
    row = services.repository.item(item_id)
    if row is None:
        raise UnknownItemError(item_id)
    latest = services.pipeline.latest([item_id]).get(item_id)
    return ItemDetail(item=Item(**row), inventory=_inventory(services, item_id),
                      stats=services.repository.item_stats(item_id),
                      latest_run_id=latest.run_id if latest else None)


@router.get("/items/{item_id}/sales", response_model=Rows)
def item_sales(item_id: str, services: Svc) -> Rows:
    """Raw daily sales joined with price and calendar (the pipeline's input)."""
    try:
        df = services.repository.item_sales(item_id)
    except KeyError:
        raise UnknownItemError(item_id) from None
    return Rows(item_id=item_id, rows=frame_records(df))


@router.get("/items/{item_id}/history", response_model=Rows)
def item_history(item_id: str, services: Svc) -> Rows:
    """Cleaned daily history with imputation and suspected-stockout flags."""
    return Rows(item_id=item_id, rows=frame_records(services.pipeline.history(item_id)))


@router.get("/items/{item_id}/actuals", response_model=Rows)
def item_actuals(item_id: str, services: Svc) -> Rows:
    """Observed sales after the forecast origin (evaluation hold-out)."""
    _require(services, item_id)
    return Rows(item_id=item_id, rows=frame_records(services.repository.future_actuals(item_id)))


@router.get("/calendar/future", response_model=Rows)
def future_calendar(services: Svc) -> Rows:
    """Calendar of the forecast horizon (events and SNAP days)."""
    return Rows(rows=frame_records(services.repository.future_calendar))


@router.get("/items/{item_id}/inventory", response_model=InventoryPosition, tags=["inventory"])
def get_inventory(item_id: str, services: Svc) -> InventoryPosition:
    """Current stock position and supplier terms."""
    _require(services, item_id)
    return _inventory(services, item_id)


@router.put("/items/{item_id}/inventory", response_model=InventoryPosition, tags=["inventory"])
def update_inventory(item_id: str, body: InventoryUpdate, services: Svc) -> InventoryPosition:
    """Update stock fields (simulates a stock-feed update); marks the row as not simulated."""
    _require(services, item_id)
    changes = body.model_dump(exclude_none=True)
    if not changes:
        raise HTTPException(status_code=422, detail="No fields to update.")
    services.repository.update_inventory(item_id, changes)
    return _inventory(services, item_id)
