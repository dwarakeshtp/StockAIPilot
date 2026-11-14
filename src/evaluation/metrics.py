"""Point-forecast accuracy metrics.

MAPE is computed on non-zero actuals only (it is undefined at zero). Because
retail demand is often intermittent, WAPE (sum |e| / sum actual) is reported as
a robust companion metric.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from pydantic import BaseModel


class ForecastMetrics(BaseModel):
    """Accuracy metrics for a set of forecasts."""

    mae: float
    rmse: float
    mape: float | None = None
    wape: float | None = None
    n: int = 0


def _clean_pairs(actual: Sequence[float], predicted: Sequence[float]) -> tuple[np.ndarray, ...]:
    a = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    if a.shape != p.shape:
        raise ValueError(f"Shape mismatch: actual {a.shape} vs predicted {p.shape}")
    mask = ~(np.isnan(a) | np.isnan(p))
    return a[mask], p[mask]


def mae(actual: Sequence[float], predicted: Sequence[float]) -> float:
    """Mean absolute error."""
    a, p = _clean_pairs(actual, predicted)
    return float(np.mean(np.abs(a - p))) if a.size else float("nan")


def rmse(actual: Sequence[float], predicted: Sequence[float]) -> float:
    """Root mean squared error."""
    a, p = _clean_pairs(actual, predicted)
    return float(np.sqrt(np.mean((a - p) ** 2))) if a.size else float("nan")


def mape(actual: Sequence[float], predicted: Sequence[float]) -> float | None:
    """Mean absolute percentage error (fraction) over non-zero actuals; ``None`` if none."""
    a, p = _clean_pairs(actual, predicted)
    nz = a != 0
    if not nz.any():
        return None
    return float(np.mean(np.abs((a[nz] - p[nz]) / a[nz])))


def wape(actual: Sequence[float], predicted: Sequence[float]) -> float | None:
    """Weighted absolute percentage error (fraction); ``None`` if actuals sum to zero."""
    a, p = _clean_pairs(actual, predicted)
    total = np.sum(np.abs(a))
    if total == 0:
        return None
    return float(np.sum(np.abs(a - p)) / total)


def compute_metrics(actual: Sequence[float], predicted: Sequence[float]) -> ForecastMetrics:
    """Compute MAE, RMSE, MAPE and WAPE, ignoring pairs with missing values."""
    a, p = _clean_pairs(actual, predicted)
    m, w = mape(a, p), wape(a, p)
    return ForecastMetrics(
        mae=round(mae(a, p), 4),
        rmse=round(rmse(a, p), 4),
        mape=None if m is None else round(m, 4),
        wape=None if w is None else round(w, 4),
        n=int(a.size),
    )
