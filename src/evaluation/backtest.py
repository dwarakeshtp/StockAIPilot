"""Rolling-origin backtesting for any :class:`BaseForecaster`.

For ``folds`` = k and ``horizon`` = h, the model is trained on data up to
``n - k*h`` and evaluated on the next ``h`` days, then the origin moves forward
by ``h`` days. This mimics weekly re-forecasting and never uses future data.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import pandas as pd

from src.evaluation.metrics import ForecastMetrics, compute_metrics
from src.models.base import BaseForecaster
from src.models.ensemble import average_forecasts

ForecasterFactory = Callable[[], BaseForecaster]


@dataclass
class BacktestResult:
    """Fold-level predictions and aggregate metrics of one model."""

    model_name: str
    predictions: pd.DataFrame  # fold, date, actual, yhat, lower, upper
    metrics: ForecastMetrics


def rolling_origin_backtest(
    factory: ForecasterFactory,
    features: pd.DataFrame,
    horizon: int,
    folds: int,
    regressors: Sequence[str] = (),
    model_name: str | None = None,
) -> BacktestResult:
    """Run a rolling-origin backtest.

    Args:
        factory: Callable returning a fresh, unfitted forecaster.
        features: Item feature frame with ``date``, ``y`` and regressors.
        horizon: Days per fold.
        folds: Number of folds.
        regressors: Regressor columns passed to ``predict``.
        model_name: Optional display name (defaults to the forecaster's name).

    Returns:
        A :class:`BacktestResult`.

    Raises:
        ValueError: if the history is too short for the requested folds.
    """
    data = features.sort_values("date").reset_index(drop=True)
    n = len(data)
    if n <= folds * horizon + 30:
        raise ValueError("History too short for the requested backtest")
    frames = []
    name = model_name
    for fold, k in enumerate(range(folds, 0, -1), start=1):
        cutoff = n - k * horizon
        train, test = data.iloc[:cutoff], data.iloc[cutoff : cutoff + horizon]
        model = factory()
        name = name or model.name
        pred = model.fit(train).predict(test[["date", *regressors]])
        pred["actual"] = test["y"].to_numpy()
        pred["fold"] = fold
        frames.append(pred)
    predictions = pd.concat(frames, ignore_index=True)
    return BacktestResult(
        model_name=name or "model",
        predictions=predictions,
        metrics=compute_metrics(predictions["actual"], predictions["yhat"]),
    )


def ensemble_backtest(results: Sequence[BacktestResult], name: str = "ensemble") -> BacktestResult:
    """Combine member backtests into an equal-weight ensemble without refitting."""
    if not results:
        raise ValueError("No backtest results to combine")
    base = results[0].predictions[["date", "fold", "actual"]]
    avg = average_forecasts([r.predictions for r in results])
    merged = base.merge(avg, on="date", how="left")
    return BacktestResult(
        model_name=name,
        predictions=merged,
        metrics=compute_metrics(merged["actual"], merged["yhat"]),
    )
