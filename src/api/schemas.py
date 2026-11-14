"""Request and response models of the HTTP API (agent models are reused as-is)."""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from src.agents.schemas import AgentTraceEntry, ForecastOutput, RiskOutput
from src.data.quality import DataQualityReport
from src.db.stores import Job, JobStatus
from src.orchestration.decisions import DecisionType


def frame_records(df: pd.DataFrame | None) -> list[dict[str, Any]]:
    """DataFrame -> JSON-safe records (ISO dates, NaN/NaT as ``null``)."""
    if df is None or df.empty:
        return []
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = out[col].dt.strftime("%Y-%m-%d")
    records = out.astype(object).where(out.notna(), None).to_dict(orient="records")
    return [{k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in r.items()}
            for r in records]


class Health(BaseModel):
    """Liveness and database readiness."""

    status: str = "ok"
    database: str
    source: str | None = None


class SystemInfo(BaseModel):
    """What the backend is serving."""

    data_source: str
    store_id: str | None
    category: str | None
    loaded_at: str | None
    items: int
    demo_items: list[str]
    history_start: str
    history_end: str
    forecast_origin: str
    horizon_days: int
    counts: dict[str, int]
    llm: str
    database_url: str


class Item(BaseModel):
    """Item master row with display metadata."""

    item_id: str
    name: str
    label: str
    department: str
    unit: str
    perishable: bool
    profile_note: str = ""
    dept_id: str | None = None
    store_id: str | None = None
    is_demo: bool = False


class ItemPage(BaseModel):
    """One page of items."""

    items: list[Item]
    total: int
    limit: int
    offset: int


class InventoryPosition(BaseModel):
    """Stock and supplier terms for one item."""

    item_id: str
    on_hand: int
    on_order: int
    on_order_eta_days: int
    lead_time_days: int
    lead_time_std: float
    moq: int
    pack_size: int
    unit_cost: float
    holding_cost_rate: float
    is_simulated: bool
    updated_at: str | None = None


class InventoryUpdate(BaseModel):
    """Partial update from a stock feed or a manual count."""

    model_config = ConfigDict(extra="forbid")

    on_hand: int | None = Field(default=None, ge=0)
    on_order: int | None = Field(default=None, ge=0)
    on_order_eta_days: int | None = Field(default=None, ge=0)
    lead_time_days: int | None = Field(default=None, ge=1)
    lead_time_std: float | None = Field(default=None, ge=0)
    moq: int | None = Field(default=None, ge=0)
    pack_size: int | None = Field(default=None, ge=1)
    unit_cost: float | None = Field(default=None, ge=0)
    holding_cost_rate: float | None = Field(default=None, ge=0)


class ItemDetail(BaseModel):
    """Item, its inventory position and headline sales statistics."""

    item: Item
    inventory: InventoryPosition
    stats: dict[str, Any]
    latest_run_id: str | None = None


class Rows(BaseModel):
    """Tabular payload (dates as ISO strings)."""

    item_id: str | None = None
    rows: list[dict[str, Any]]


class ForecastResponse(BaseModel):
    """Forecast and Risk agent outputs for one item."""

    run_id: str
    item_id: str
    product_name: str | None = None
    data_source: str
    data_quality: DataQualityReport
    forecast: ForecastOutput
    risk: RiskOutput
    trace: list[AgentTraceEntry]


class RunRequest(BaseModel):
    """Run the full agent pipeline for one item."""

    item_id: str
    overrides: dict[str, float] = Field(default_factory=dict)


class WhatIfRequest(BaseModel):
    """Re-plan a stored run under new assumptions."""

    overrides: dict[str, float] = Field(
        default_factory=dict,
        examples=[{"lead_time_days": 9, "service_level": 0.98, "demand_multiplier": 1.2}])


class RunSummary(BaseModel):
    """Row of the run log."""

    run_id: str
    kind: str
    item_id: str
    created_at: str
    action: str | None
    qty: int | None
    verdict: str | None
    revision_count: int | None


class JobRequest(BaseModel):
    """Queue pipeline runs; ``item_ids`` empty or ``all_demo`` = all demo items."""

    item_ids: list[str] = Field(default_factory=list)
    all_demo: bool = False


class JobResponse(BaseModel):
    """Background job state."""

    job_id: str
    status: JobStatus
    total: int
    done: int
    progress: float
    item_ids: list[str]
    run_ids: list[str]
    errors: list[str]
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None

    @classmethod
    def of(cls, job: Job) -> JobResponse:
        """Build from the stored job."""
        return cls(**job.model_dump(), progress=round(job.done / job.total, 3) if job.total else 1)


class Recommendation(BaseModel):
    """Latest recommendation per item (overview table row)."""

    run_id: str
    item_id: str
    product_name: str
    department: str
    created_at: str
    action: str
    quantity: int
    order_value: float
    risk_level: str
    confidence: float
    stockout_probability: float
    days_of_cover: float | None
    model: str
    wape: float | None
    verdict: str
    revision_count: int
    headline: str
    decision: str
    final_qty: int | None


class DecisionRequest(BaseModel):
    """Planner decision on a stored run."""

    run_id: str
    decision: DecisionType
    final_qty: int | None = Field(default=None, ge=0)
    reason: str = ""
    reviewer: str = "planner"
