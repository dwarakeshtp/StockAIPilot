"""Seasonal ARIMA baseline built on statsmodels ``SARIMAX``.

The model is univariate (a classic statistical baseline). The state-space
implementation handles missing (censored) observations natively. When no order
is given, a small grid is searched by AIC.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Self

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.models.base import BaseForecaster

Order = tuple[int, int, int]


class ARIMAForecaster(BaseForecaster):
    """SARIMA(p,d,q)(P,D,Q,s) with optional AIC-based order selection."""

    name = "arima"

    def __init__(
        self,
        order: Order | None = None,
        seasonal_order: Order = (1, 0, 1),
        seasonal_period: int = 7,
        order_grid: Sequence[Order] = ((1, 0, 1), (2, 0, 1), (1, 1, 1)),
        fit_window: int = 365,
        interval_width: float = 0.8,
    ) -> None:
        super().__init__(interval_width)
        self.order = order
        self.seasonal_order = seasonal_order
        self.seasonal_period = seasonal_period
        self.order_grid = list(order_grid)
        self.fit_window = fit_window
        self.selected_order: Order | None = None
        self.aic: float | None = None
        self.stable: bool = True
        self._result = None

    def _fit_order(self, y: pd.Series, order: Order, enforce: bool = False):
        trend = "c" if order[1] == 0 else None
        model = SARIMAX(
            y.to_numpy(dtype=float),
            order=order,
            seasonal_order=(*self.seasonal_order, self.seasonal_period),
            trend=trend,
            enforce_stationarity=enforce,
            enforce_invertibility=enforce,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return model.fit(disp=False, maxiter=60)

    def _is_sane(self, result, y: pd.Series, steps: int = 28) -> bool:
        """Reject fits whose forecasts are non-finite or wildly above history."""
        forecast = result.get_forecast(steps=steps)
        mean = np.asarray(forecast.predicted_mean, dtype=float)
        ci = np.asarray(forecast.conf_int(alpha=1 - self.interval_width), dtype=float)
        cap = 10 * max(float(np.nanmax(y)), 1.0)
        return bool(
            np.all(np.isfinite(mean)) and np.all(np.isfinite(ci))
            and np.max(np.abs(mean)) <= cap and np.max(np.abs(ci)) <= 5 * cap
        )

    def select_order(self, history: pd.DataFrame) -> Order:
        """Return the grid order with the lowest AIC among numerically sane fits."""
        y = self._target(history)
        best_order, best_aic = self.order_grid[0], np.inf
        for order in self.order_grid:
            try:
                result = self._fit_order(y, order)
                aic = float(result.aic)
                if not self._is_sane(result, y):
                    continue
            except (ValueError, np.linalg.LinAlgError):
                continue
            if np.isfinite(aic) and aic < best_aic:
                best_order, best_aic = order, aic
        return best_order

    def _target(self, history: pd.DataFrame) -> pd.Series:
        self._require_columns(history, ("date", "y"))
        y = history.sort_values("date")["y"].astype(float).tail(self.fit_window)
        if y.notna().sum() < 3 * self.seasonal_period:
            raise ValueError("Not enough observations to fit ARIMA")
        return y.reset_index(drop=True)

    def fit(self, history: pd.DataFrame) -> Self:
        """Fit the model (selecting the order first when ``order`` is ``None``).

        If the fit is numerically unstable, it is retried with stationarity and
        invertibility enforced, then with simpler fallback orders.
        """
        y = self._target(history)
        order = self.order or self.select_order(history)
        attempts = [(order, False), (order, True), ((1, 0, 1), True), ((0, 1, 1), True)]
        result, used, stable = None, order, False
        for candidate, enforce in attempts:
            try:
                fitted = self._fit_order(y, candidate, enforce)
            except (ValueError, np.linalg.LinAlgError):
                continue
            result, used = fitted, candidate
            if self._is_sane(fitted, y):
                stable = True
                break
        if result is None:
            raise ValueError("ARIMA could not be fitted")
        self.stable = stable
        self._result = result
        self.selected_order = used
        self.aic = float(result.aic)
        self._fitted = True
        return self

    def _predict(self, future: pd.DataFrame) -> pd.DataFrame:
        forecast = self._result.get_forecast(steps=len(future))
        mean = np.asarray(forecast.predicted_mean, dtype=float)
        ci = np.asarray(forecast.conf_int(alpha=1 - self.interval_width), dtype=float)
        return pd.DataFrame(
            {"date": future["date"].to_numpy(), "yhat": mean, "lower": ci[:, 0],
             "upper": ci[:, 1]}
        )
