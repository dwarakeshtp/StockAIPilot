"""Tests for forecasting models, metrics and backtesting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.cleaning import clean_item_sales
from src.data.features import PROPHET_REGRESSORS, build_features, build_future_frame
from src.evaluation.backtest import ensemble_backtest, rolling_origin_backtest
from src.evaluation.metrics import compute_metrics, mae, mape, rmse, wape
from src.models import (
    ARIMAForecaster,
    EnsembleForecaster,
    NotFittedError,
    ProphetForecaster,
    SeasonalNaiveForecaster,
)


@pytest.fixture(scope="module")
def features(synthetic_dataset) -> pd.DataFrame:
    item = synthetic_dataset.item_ids[1]
    clean, _ = clean_item_sales(synthetic_dataset.sales[synthetic_dataset.sales.item_id == item])
    return build_features(clean)


@pytest.fixture(scope="module")
def future(features, synthetic_dataset) -> pd.DataFrame:
    return build_future_frame(features, synthetic_dataset.future_calendar, 7)


def test_metrics_known_values() -> None:
    actual, pred = [10, 0, 20], [12, 1, 18]
    assert mae(actual, pred) == pytest.approx(5 / 3)
    assert rmse(actual, pred) == pytest.approx(np.sqrt(9 / 3))
    assert mape(actual, pred) == pytest.approx((0.2 + 0.1) / 2)
    assert wape(actual, pred) == pytest.approx(5 / 30)


def test_metrics_handle_nan_and_zero() -> None:
    m = compute_metrics([np.nan, 0, 0], [1, 2, 3])
    assert m.n == 2
    assert m.mape is None and m.wape is None
    with pytest.raises(ValueError):
        compute_metrics([1, 2], [1])


def _check_forecast(frame: pd.DataFrame, horizon: int = 7) -> None:
    assert len(frame) == horizon
    assert (frame["yhat"] >= 0).all()
    assert (frame["lower"] <= frame["yhat"]).all()
    assert (frame["yhat"] <= frame["upper"]).all()


def test_seasonal_naive_repeats_last_week(features, future) -> None:
    model = SeasonalNaiveForecaster().fit(features)
    pred = model.predict(future)
    _check_forecast(pred)
    last_week = features["y"].tail(7).fillna(features["y"].tail(28).mean()).to_numpy()
    assert np.allclose(pred["yhat"].to_numpy(), np.clip(last_week, 0, None))


def test_predict_before_fit_raises(future) -> None:
    with pytest.raises(NotFittedError):
        SeasonalNaiveForecaster().predict(future)


def test_arima_fits_and_forecasts(features, future) -> None:
    model = ARIMAForecaster(order_grid=[(1, 0, 1), (0, 1, 1)], fit_window=200)
    pred = model.fit(features).predict(future)
    _check_forecast(pred)
    assert model.selected_order in [(1, 0, 1), (0, 1, 1)]
    assert model.aic is not None


def test_arima_handles_missing_values(features, future) -> None:
    gappy = features.copy()
    gappy.loc[gappy.index[-20:-10], "y"] = np.nan
    pred = ARIMAForecaster(order=(1, 0, 1), fit_window=200).fit(gappy).predict(future)
    _check_forecast(pred)


def test_prophet_fits_and_forecasts(features, future) -> None:
    model = ProphetForecaster(regressors=PROPHET_REGRESSORS, uncertainty_samples=100)
    pred = model.fit(features).predict(future)
    _check_forecast(pred)


def test_ensemble_averages_members(features, future) -> None:
    members = [SeasonalNaiveForecaster(), ARIMAForecaster(order=(1, 0, 1), fit_window=200)]
    ens = EnsembleForecaster(members).fit(features).predict(future)
    _check_forecast(ens)
    a, b = (m.predict(future)["yhat"] for m in members)
    assert np.allclose(ens["yhat"], (a + b) / 2)


def test_rolling_backtest_and_ensemble(features) -> None:
    res_naive = rolling_origin_backtest(SeasonalNaiveForecaster, features, 7, 3)
    res_arima = rolling_origin_backtest(
        lambda: ARIMAForecaster(order=(1, 0, 1), fit_window=200), features, 7, 3
    )
    assert len(res_naive.predictions) == 21
    assert res_naive.predictions["fold"].nunique() == 3
    assert res_naive.metrics.n > 0
    ens = ensemble_backtest([res_naive, res_arima])
    assert len(ens.predictions) == 21
    assert ens.metrics.mae >= 0


def test_backtest_rejects_short_history(features) -> None:
    with pytest.raises(ValueError):
        rolling_origin_backtest(SeasonalNaiveForecaster, features.head(40), 7, 4)
