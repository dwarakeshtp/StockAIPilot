"""Forecaster abstraction shared by every forecasting model."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar, Self

import numpy as np
import pandas as pd
from scipy.stats import norm

FORECAST_COLUMNS: list[str] = ["date", "yhat", "lower", "upper"]


class NotFittedError(RuntimeError):
    """Raised when ``predict`` is called before ``fit``."""


class BaseForecaster(ABC):
    """Common interface: ``fit(history)`` then ``predict(future)``.

    ``history`` must contain ``date`` and the target ``y`` (``NaN`` allowed for
    censored days) plus any regressors; ``future`` contains ``date`` and the
    same regressors for each day to forecast.
    """

    name: ClassVar[str] = "base"

    def __init__(self, interval_width: float = 0.8) -> None:
        if not 0.5 < interval_width < 1.0:
            raise ValueError("interval_width must be in (0.5, 1.0)")
        self.interval_width = interval_width
        self._fitted = False

    @property
    def z(self) -> float:
        """Two-sided normal quantile for the configured interval width."""
        return float(norm.ppf(0.5 + self.interval_width / 2))

    @abstractmethod
    def fit(self, history: pd.DataFrame) -> Self:
        """Fit the model on ``history`` and return ``self``."""

    @abstractmethod
    def _predict(self, future: pd.DataFrame) -> pd.DataFrame:
        """Model-specific prediction returning :data:`FORECAST_COLUMNS`."""

    def predict(self, future: pd.DataFrame) -> pd.DataFrame:
        """Forecast each row of ``future``; results are non-negative and ordered.

        Raises:
            NotFittedError: if called before :meth:`fit`.
        """
        if not self._fitted:
            raise NotFittedError(f"{self.name} must be fitted before predict")
        return self.finalize(self._predict(future))

    @staticmethod
    def finalize(frame: pd.DataFrame) -> pd.DataFrame:
        """Clip to zero, map non-finite values to NaN and enforce ``lower <= yhat <= upper``."""
        out = frame[FORECAST_COLUMNS].copy()
        out["date"] = pd.to_datetime(out["date"])
        for col in ("yhat", "lower", "upper"):
            values = out[col].astype(float).to_numpy()
            values = np.where(np.isfinite(values), values, np.nan)
            out[col] = np.clip(values, 0, None)
        out["lower"] = np.minimum(out["lower"], out["yhat"])
        out["upper"] = np.maximum(out["upper"], out["yhat"])
        return out.reset_index(drop=True)

    @staticmethod
    def _require_columns(df: pd.DataFrame, columns: tuple[str, ...]) -> None:
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
