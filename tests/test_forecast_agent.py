"""Tests for the Forecast Agent."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.agents.forecast_agent import ForecastAgent, is_plausible
from src.agents.schemas import ForecastOutput, RevisionHint, RevisionRequest
from src.data.cleaning import clean_item_sales
from src.data.features import build_features, build_future_frame, prophet_holidays


@pytest.fixture()
def inputs(repository):
    item = repository.item_ids[0]
    clean, _ = clean_item_sales(repository.item_sales(item))
    features = build_features(clean)
    future = build_future_frame(features, repository.future_calendar, 7)
    return item, features, future, prophet_holidays(features, future)


def test_forecast_agent_output_contract(settings, inputs) -> None:
    agent = ForecastAgent(settings)
    out = agent.run(*inputs)
    assert isinstance(out, ForecastOutput)
    assert out.model_used in {"arima", "prophet"}
    assert len(out.forecast) == 7
    assert set(out.metrics_by_model) == {"seasonal_naive", "arima", "prophet", "ensemble"}
    for metrics in out.metrics_by_model.values():
        assert metrics.mae >= 0 and metrics.rmse >= metrics.mae - 1e-9
    assert 0 <= out.model_agreement_pct <= 1
    assert len(out.backtest) == settings.forecast.backtest_folds * 7
    assert out.narrative is None  # LLM disabled in tests
    assert out.reasoning


def test_is_plausible_rejects_exploding_forecast() -> None:
    dates = pd.date_range("2016-01-01", periods=3)
    good = pd.DataFrame({"date": dates, "yhat": [5.0, 6, 7], "lower": [1.0, 2, 3],
                         "upper": [9.0, 10, 11]})
    history = pd.Series([0, 10, 20])
    assert is_plausible(good, history)
    exploding = good.assign(yhat=[0, 1e38, 0])
    assert not is_plausible(exploding, history)
    assert not is_plausible(good.assign(lower=[np.nan, 1, 1]), history)


def test_forecast_agent_revision_uses_cache(settings, inputs) -> None:
    agent = ForecastAgent(settings)
    first = agent.run(*inputs)
    revision = RevisionRequest(hint=RevisionHint.USE_ENSEMBLE, reason="test",
                               previous_model=first.model_used)
    second = agent.run(*inputs, revision=revision)
    assert second.model_used == "ensemble"
    assert second.revision_applied == RevisionHint.USE_ENSEMBLE
    assert len(agent._cache) == 1

    alt = agent.run(*inputs, revision=RevisionRequest(
        hint=RevisionHint.USE_ALTERNATIVE_MODEL, reason="test", previous_model=first.model_used))
    assert alt.model_used != first.model_used
