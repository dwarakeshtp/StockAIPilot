"""Builders for agent outputs used across tests."""

from __future__ import annotations

from datetime import date, timedelta

from src.agents.schemas import (
    ForecastOutput,
    ForecastPoint,
    InventoryParams,
    RiskLevel,
    RiskOutput,
)
from src.data.quality import DataQualityReport
from src.evaluation.metrics import ForecastMetrics

START = date(2016, 5, 23)


def make_forecast(
    mean: float = 10.0,
    width: float = 4.0,
    rmse: float = 2.0,
    wape: float = 0.2,
    agreement: float = 0.9,
    model: str = "arima",
    yhat_override: list[float] | None = None,
) -> ForecastOutput:
    """ForecastOutput with a flat 7-day forecast."""
    values = yhat_override or [mean] * 7
    points = [ForecastPoint(date=START + timedelta(days=i), yhat=v, lower=max(v - width / 2, 0),
                            upper=v + width / 2) for i, v in enumerate(values)]
    metrics = ForecastMetrics(mae=rmse * 0.8, rmse=rmse, mape=wape, wape=wape, n=28)
    naive = ForecastMetrics(mae=rmse, rmse=rmse * 1.2, mape=wape, wape=wape * 1.1, n=28)
    return ForecastOutput(
        item_id="X", model_used=model, horizon=7, forecast=points,
        metrics_by_model={"seasonal_naive": naive, "arima": metrics, "prophet": metrics,
                          "ensemble": metrics},
        selected_metrics=metrics, skill_vs_naive=0.2, model_agreement_pct=agreement,
    )


def make_risk(level: RiskLevel = RiskLevel.LOW, score: float = 80.0,
              multiplier: float = 1.0) -> RiskOutput:
    """RiskOutput with the given level and multiplier."""
    return RiskOutput(
        item_id="X", confidence_score=score, risk_level=level,
        confidence_components={"accuracy": 0.8, "precision": 0.7, "agreement": 0.9,
                               "data_quality": 1.0},
        recent_to_baseline_ratio=1.0, relative_interval_width=0.4,
        uncertainty_multiplier=multiplier,
    )


def make_params(**overrides) -> InventoryParams:
    """InventoryParams with sensible defaults."""
    base = {"item_id": "X", "on_hand": 50, "on_order": 0, "on_order_eta_days": 0,
            "lead_time_days": 4, "lead_time_std": 0, "moq": 12, "pack_size": 6,
            "unit_cost": 2.0, "holding_cost_rate": 0.25, "is_simulated": True}
    base.update(overrides)
    return InventoryParams(**base)


def make_quality(score: float = 1.0) -> DataQualityReport:
    """DataQualityReport with the given score."""
    return DataQualityReport(item_id="X", n_rows=730, start_date=date(2014, 5, 24),
                             end_date=date(2016, 5, 22), score=score)
