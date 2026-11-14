"""Pydantic contracts exchanged between agents, orchestration and the dashboard.

Every agent returns one of these models, which makes agent outputs typed,
validated, serialisable to JSON traces and easy to render in the UI.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field

from src.data.quality import DataQualityReport
from src.evaluation.metrics import ForecastMetrics


class RiskLevel(StrEnum):
    """Qualitative risk classification."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class InventoryAction(StrEnum):
    """Inventory action proposed by the planning agent."""

    REORDER = "REORDER"
    EXPEDITE = "EXPEDITE"
    HOLD = "HOLD"


class Verdict(StrEnum):
    """Critic verdict on the combined recommendation."""

    APPROVED = "APPROVED"
    APPROVED_WITH_WARNINGS = "APPROVED_WITH_WARNINGS"
    NEEDS_REVISION = "NEEDS_REVISION"
    NEEDS_HUMAN_REVIEW = "NEEDS_HUMAN_REVIEW"


class RevisionHint(StrEnum):
    """Instruction from the Critic to the Forecast Agent."""

    USE_ENSEMBLE = "USE_ENSEMBLE"
    USE_ALTERNATIVE_MODEL = "USE_ALTERNATIVE_MODEL"


Severity = Literal["INFO", "WARNING", "CRITICAL"]


class RevisionRequest(BaseModel):
    """A Critic request for the Forecast Agent to revise its output."""

    hint: RevisionHint
    reason: str
    previous_model: str


class ForecastPoint(BaseModel):
    """One forecasted day with its prediction interval."""

    date: date
    yhat: float
    lower: float
    upper: float


class BacktestPoint(BaseModel):
    """One backtest day: actual vs predicted."""

    date: date
    fold: int
    actual: float | None
    yhat: float
    lower: float
    upper: float


class ForecastOutput(BaseModel):
    """Output of the Forecast Agent."""

    item_id: str
    model_used: str
    baseline_model: str = "seasonal_naive"
    horizon: int
    forecast: list[ForecastPoint]
    model_forecasts: dict[str, list[ForecastPoint]] = Field(default_factory=dict)
    metrics_by_model: dict[str, ForecastMetrics]
    selected_metrics: ForecastMetrics
    skill_vs_naive: float | None = None
    model_agreement_pct: float = Field(ge=0.0, le=1.0)
    arima_order: tuple[int, int, int] | None = None
    backtest: list[BacktestPoint] = Field(default_factory=list)
    revision_applied: RevisionHint | None = None
    reasoning: list[str] = Field(default_factory=list)
    narrative: str | None = None

    @property
    def total_forecast(self) -> float:
        """Sum of point forecasts over the horizon."""
        return float(sum(p.yhat for p in self.forecast))

    @property
    def mean_daily_demand(self) -> float:
        """Average forecasted daily demand."""
        return self.total_forecast / max(len(self.forecast), 1)


class Anomaly(BaseModel):
    """A historical demand anomaly."""

    date: date
    units: float
    expected: float
    score: float
    method: Literal["zscore", "iqr", "zscore+iqr"]
    kind: Literal["spike", "drop"]


class RiskOutput(BaseModel):
    """Output of the Risk & Uncertainty Agent."""

    item_id: str
    confidence_score: float = Field(ge=0.0, le=100.0)
    risk_level: RiskLevel
    confidence_components: dict[str, float]
    anomalies: list[Anomaly] = Field(default_factory=list)
    recent_anomaly_count: int = 0
    spike_detected: bool = False
    drop_detected: bool = False
    forecast_spike: bool = False
    recent_to_baseline_ratio: float
    relative_interval_width: float
    uncertainty_multiplier: float
    reasoning: list[str] = Field(default_factory=list)
    narrative: str | None = None


class InventoryParams(BaseModel):
    """Inventory and supplier parameters for one item (simulated in this POC)."""

    item_id: str
    on_hand: float = Field(ge=0)
    on_order: float = Field(default=0, ge=0)
    on_order_eta_days: int = Field(default=0, ge=0)
    lead_time_days: float = Field(gt=0)
    lead_time_std: float = Field(default=0, ge=0)
    moq: int = Field(default=1, ge=1)
    pack_size: int = Field(default=1, ge=1)
    unit_cost: float = Field(default=1.0, ge=0)
    holding_cost_rate: float = Field(default=0.25, ge=0)
    is_simulated: bool = True


class ProjectedStockPoint(BaseModel):
    """Projected end-of-day stock for one future day."""

    date: date
    demand: float
    arrivals: float
    stock: float


class InventoryOutput(BaseModel):
    """Output of the Inventory Planning Agent."""

    item_id: str
    params: InventoryParams
    service_level: float
    z_value: float
    review_period_days: int
    mean_daily_demand: float
    demand_std: float
    adjusted_demand_std: float
    lead_time_demand: float
    safety_stock: float
    reorder_point: float
    inventory_position: float
    reorder_qty: int
    eoq: float
    order_value: float
    days_of_cover: float
    stockout_probability: float = Field(ge=0.0, le=1.0)
    projected_stockout_date: date | None = None
    projected_stock: list[ProjectedStockPoint] = Field(default_factory=list)
    action: InventoryAction
    overrides_applied: dict[str, float] = Field(default_factory=dict)
    reasoning: list[str] = Field(default_factory=list)
    narrative: str | None = None


class ValidationIssue(BaseModel):
    """A finding raised by one Critic validation rule."""

    rule: str
    severity: Severity
    agent: str
    message: str
    revision_hint: RevisionHint | None = None


class RecommendationSummary(BaseModel):
    """Final business recommendation shown to the planner."""

    item_id: str
    action: InventoryAction
    quantity: int
    order_value: float
    risk_level: RiskLevel
    confidence_score: float
    headline: str
    rationale: str
    key_risks: list[str] = Field(default_factory=list)
    requires_human_review: bool = True


class CriticOutput(BaseModel):
    """Output of the Critic Agent."""

    item_id: str
    verdict: Verdict
    issues: list[ValidationIssue] = Field(default_factory=list)
    revision_request: RevisionRequest | None = None
    explanation: str
    explanation_source: str
    grounding_passed: bool = True
    ungrounded_numbers: list[str] = Field(default_factory=list)
    recommendation: RecommendationSummary
    reasoning: list[str] = Field(default_factory=list)

    @property
    def issue_counts(self) -> dict[str, int]:
        """Number of issues per severity."""
        counts = {"INFO": 0, "WARNING": 0, "CRITICAL": 0}
        for issue in self.issues:
            counts[issue.severity] += 1
        return counts


class AgentTraceEntry(BaseModel):
    """One step of the pipeline as recorded in the JSON trace."""

    step: int
    node: str
    agent: str
    started_at: datetime
    finished_at: datetime
    duration_ms: float
    status: Literal["success", "error"]
    input_summary: dict[str, Any] = Field(default_factory=dict)
    output_summary: dict[str, Any] = Field(default_factory=dict)
    reasoning: list[str] = Field(default_factory=list)
    llm_provider: str | None = None
    error: str | None = None


class PipelineResult(BaseModel):
    """Everything produced by one end-to-end run for one item."""

    run_id: str
    item_id: str
    created_at: datetime
    data_source: str
    data_quality: DataQualityReport
    forecast: ForecastOutput
    risk: RiskOutput
    inventory: InventoryOutput
    critic: CriticOutput
    revision_count: int = 0
    overrides: dict[str, float] = Field(default_factory=dict)
    trace: list[AgentTraceEntry] = Field(default_factory=list)
    trace_path: str | None = None
