"""Tests for anomaly detection, confidence scoring and the Risk Agent."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from src.agents.anomaly import detect_anomalies, detect_level_shift
from src.agents.confidence import (
    classify_risk,
    confidence_components,
    confidence_score,
    relative_interval_width,
)
from src.agents.risk_agent import RiskAgent
from src.agents.schemas import ForecastOutput, ForecastPoint, RiskLevel
from src.config.settings import RiskWeights
from src.data.quality import DataQualityReport
from src.evaluation.metrics import ForecastMetrics


def _features(values: np.ndarray) -> pd.DataFrame:
    dates = pd.date_range("2016-01-01", periods=len(values), freq="D")
    return pd.DataFrame({"date": dates, "units": values.astype(float), "y": values.astype(float),
                         "is_suspected_stockout": False})


def _forecast(mean: float, width: float = 4.0, wape: float = 0.2,
              agreement: float = 0.9) -> ForecastOutput:
    start = date(2016, 5, 1)
    points = [ForecastPoint(date=start + timedelta(days=i), yhat=mean, lower=mean - width / 2,
                            upper=mean + width / 2) for i in range(7)]
    metrics = ForecastMetrics(mae=1.0, rmse=1.5, mape=0.2, wape=wape, n=28)
    return ForecastOutput(item_id="X", model_used="arima", horizon=7, forecast=points,
                          metrics_by_model={"arima": metrics}, selected_metrics=metrics,
                          model_agreement_pct=agreement)


def _dq(score: float = 1.0) -> DataQualityReport:
    return DataQualityReport(item_id="X", n_rows=200, start_date=date(2016, 1, 1),
                             end_date=date(2016, 7, 1), score=score)


def test_detect_anomalies_finds_injected_spike() -> None:
    rng = np.random.default_rng(0)
    values = rng.poisson(20, 200).astype(float)
    values[180] = 120
    found = detect_anomalies(_features(values), lookback_days=60)
    assert any(a.kind == "spike" and a.units == 120 for a in found)


def test_detect_anomalies_ignores_stockout_days() -> None:
    values = np.full(120, 20.0)
    values[100:110] = 0
    feats = _features(values)
    feats.loc[100:109, "is_suspected_stockout"] = True
    assert not any(a.kind == "drop" for a in detect_anomalies(feats, lookback_days=60))


def test_level_shift_spike_and_drop() -> None:
    up = np.concatenate([np.full(100, 10.0), np.full(7, 20.0)])
    down = np.concatenate([np.full(100, 10.0), np.full(7, 4.0)])
    assert detect_level_shift(_features(up)).spike
    assert detect_level_shift(_features(down)).drop
    flat = detect_level_shift(_features(np.full(107, 10.0)))
    assert not flat.spike and not flat.drop and flat.ratio == pytest.approx(1.0)


def test_confidence_components_and_score() -> None:
    comps = confidence_components(wape=0.2, rel_width=1.0, agreement=0.9, data_quality=1.0)
    assert comps == {"accuracy": 0.8, "precision": 0.5, "agreement": 0.9, "data_quality": 1.0}
    score = confidence_score(comps, RiskWeights())
    assert score == pytest.approx(100 * (0.32 + 0.15 + 0.18 + 0.1))
    assert confidence_score(comps, RiskWeights(), recent_anomalies=10) == pytest.approx(score - 15)


def test_relative_interval_width_zero_forecast() -> None:
    assert relative_interval_width([0, 0], [0, 0], [1, 1]) == 0.0
    assert relative_interval_width([10, 10], [5, 5], [15, 15]) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("score", "escalate", "expected"),
    [(80, False, RiskLevel.LOW), (60, False, RiskLevel.MEDIUM), (30, False, RiskLevel.HIGH),
     (80, True, RiskLevel.MEDIUM), (30, True, RiskLevel.HIGH)],
)
def test_classify_risk(score: float, escalate: bool, expected: RiskLevel) -> None:
    assert classify_risk(score, 45, 70, escalate) == expected


def test_risk_agent_low_vs_high(settings) -> None:
    rng = np.random.default_rng(1)
    feats = _features(rng.poisson(20, 200))
    agent = RiskAgent(settings)
    good = agent.run("X", feats, _forecast(20, width=6, wape=0.1, agreement=0.95), _dq(1.0))
    bad = agent.run("X", feats, _forecast(20, width=60, wape=0.9, agreement=0.2), _dq(0.6))
    assert good.confidence_score > bad.confidence_score
    assert good.risk_level == RiskLevel.LOW
    assert bad.risk_level == RiskLevel.HIGH
    assert bad.uncertainty_multiplier > good.uncertainty_multiplier
    assert good.reasoning and good.narrative is None


def test_risk_agent_escalates_on_forecast_spike(settings) -> None:
    feats = _features(np.full(200, 10.0))
    out = RiskAgent(settings).run("X", feats, _forecast(25, width=4, wape=0.1), _dq())
    assert out.forecast_spike
    assert out.risk_level != RiskLevel.LOW
