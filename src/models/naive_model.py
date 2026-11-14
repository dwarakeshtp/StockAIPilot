"""Seasonal-naive reference model: tomorrow equals the same weekday last week."""

from __future__ import annotations

from typing import Self

import numpy as np
import pandas as pd

from src.models.base import BaseForecaster


class SeasonalNaiveForecaster(BaseForecaster):
    """Repeat the last observed season; intervals from seasonal-difference residuals."""

    name = "seasonal_naive"

    def __init__(self, season_length: int = 7, interval_width: float = 0.8) -> None:
        super().__init__(interval_width)
        self.season_length = season_length
        self._last_season: np.ndarray | None = None
        self._resid_std: float = 0.0

    def fit(self, history: pd.DataFrame) -> Self:
        """Store the last season of the target (censored days filled with recent mean)."""
        self._require_columns(history, ("date", "y"))
        y = history.sort_values("date")["y"].astype(float).reset_index(drop=True)
        if len(y) < self.season_length:
            raise ValueError("History shorter than one season")
        fill = float(y.tail(28).mean()) if y.tail(28).notna().any() else 0.0
        y_filled = y.fillna(fill)
        self._last_season = y_filled.tail(self.season_length).to_numpy()
        resid = (y - y.shift(self.season_length)).dropna()
        self._resid_std = float(resid.std()) if len(resid) > 1 else float(y_filled.std() or 0.0)
        self._fitted = True
        return self

    def _predict(self, future: pd.DataFrame) -> pd.DataFrame:
        assert self._last_season is not None
        steps = len(future)
        reps = int(np.ceil(steps / self.season_length))
        yhat = np.tile(self._last_season, reps)[:steps]
        horizon_scale = np.sqrt(np.floor(np.arange(steps) / self.season_length) + 1)
        half = self.z * self._resid_std * horizon_scale
        return pd.DataFrame(
            {"date": future["date"].to_numpy(), "yhat": yhat, "lower": yhat - half,
             "upper": yhat + half}
        )
