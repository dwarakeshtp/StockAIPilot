"""Equal-weight ensemble of forecasters (used when the Critic requests it)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Self

import pandas as pd

from src.models.base import FORECAST_COLUMNS, BaseForecaster


def average_forecasts(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """Average ``yhat``, ``lower`` and ``upper`` of aligned forecast frames."""
    if not frames:
        raise ValueError("At least one forecast frame is required")
    stacked = pd.concat([f[FORECAST_COLUMNS] for f in frames])
    out = stacked.groupby("date", as_index=False)[["yhat", "lower", "upper"]].mean()
    return BaseForecaster.finalize(out)


class EnsembleForecaster(BaseForecaster):
    """Fits every member and averages their forecasts."""

    name = "ensemble"

    def __init__(self, members: Sequence[BaseForecaster], interval_width: float = 0.8) -> None:
        super().__init__(interval_width)
        if not members:
            raise ValueError("Ensemble requires at least one member")
        self.members = list(members)

    def fit(self, history: pd.DataFrame) -> Self:
        """Fit every member on ``history``."""
        for member in self.members:
            member.fit(history)
        self._fitted = True
        return self

    def _predict(self, future: pd.DataFrame) -> pd.DataFrame:
        return average_forecasts([m.predict(future) for m in self.members])
