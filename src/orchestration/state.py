"""Shared LangGraph state for the pipeline."""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

import pandas as pd

from src.agents.schemas import (
    AgentTraceEntry,
    CriticOutput,
    ForecastOutput,
    InventoryOutput,
    RevisionRequest,
    RiskOutput,
)
from src.data.quality import DataQualityReport


class PipelineState(TypedDict, total=False):
    """State passed between graph nodes.

    ``trace`` uses an additive reducer so every node appends its own entry.
    """

    run_id: str
    item_id: str
    product_name: str | None
    department: str | None
    overrides: dict[str, float]
    data_source: str
    raw_sales: pd.DataFrame
    clean: pd.DataFrame
    data_quality: DataQualityReport
    features: pd.DataFrame
    future: pd.DataFrame
    holidays: pd.DataFrame
    params: dict[str, Any]
    history_max: float
    forecast: ForecastOutput
    risk: RiskOutput
    inventory: InventoryOutput
    critic: CriticOutput
    revision_count: int
    revision_request: RevisionRequest | None
    trace: Annotated[list[AgentTraceEntry], operator.add]
