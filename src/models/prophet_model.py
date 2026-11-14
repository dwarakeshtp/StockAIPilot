"""Prophet model with weekly/yearly seasonality, M5 events and regressors."""

from __future__ import annotations

import logging as std_logging
from collections.abc import Sequence
from typing import Self

import pandas as pd

from src.models.base import BaseForecaster

std_logging.getLogger("cmdstanpy").disabled = True
std_logging.getLogger("prophet").setLevel(std_logging.ERROR)


class ProphetForecaster(BaseForecaster):
    """Wrapper around ``prophet.Prophet`` that follows :class:`BaseForecaster`."""

    name = "prophet"

    def __init__(
        self,
        holidays: pd.DataFrame | None = None,
        regressors: Sequence[str] = ("snap", "price_ratio"),
        interval_width: float = 0.8,
        uncertainty_samples: int = 300,
        yearly_seasonality: bool | str = "auto",
    ) -> None:
        super().__init__(interval_width)
        self.holidays = holidays if holidays is not None and not holidays.empty else None
        self.regressors = list(regressors)
        self.uncertainty_samples = uncertainty_samples
        self.yearly_seasonality = yearly_seasonality
        self._model = None
        self._active_regressors: list[str] = []

    def fit(self, history: pd.DataFrame) -> Self:
        """Fit Prophet on ``history`` (rows with missing ``y`` are ignored)."""
        from prophet import Prophet

        # prophet.forecaster resets its logger to INFO on import.
        std_logging.getLogger("prophet").setLevel(std_logging.ERROR)
        self._require_columns(history, ("date", "y"))
        df = history.rename(columns={"date": "ds"})
        self._active_regressors = [
            r for r in self.regressors if r in df.columns and df[r].nunique() > 1
        ]
        df = df[["ds", "y", *self._active_regressors]].dropna(subset=["y"])
        if len(df) < 30:
            raise ValueError("Not enough observations to fit Prophet")
        yearly = self.yearly_seasonality
        if yearly == "auto":
            yearly = (df["ds"].max() - df["ds"].min()).days >= 365
        model = Prophet(
            interval_width=self.interval_width,
            weekly_seasonality=True,
            yearly_seasonality=yearly,
            daily_seasonality=False,
            holidays=self.holidays,
            uncertainty_samples=self.uncertainty_samples,
        )
        for reg in self._active_regressors:
            model.add_regressor(reg)
        model.fit(df)
        self._model = model
        self._fitted = True
        return self

    def _predict(self, future: pd.DataFrame) -> pd.DataFrame:
        frame = future.rename(columns={"date": "ds"}).copy()
        for reg in self._active_regressors:
            if reg not in frame.columns:
                raise ValueError(f"Future frame is missing regressor '{reg}'")
        pred = self._model.predict(frame[["ds", *self._active_regressors]])
        return pd.DataFrame(
            {"date": pred["ds"], "yhat": pred["yhat"], "lower": pred["yhat_lower"],
             "upper": pred["yhat_upper"]}
        )
