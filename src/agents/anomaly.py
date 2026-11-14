"""Demand anomaly and level-shift detection (pure functions)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.agents.schemas import Anomaly


def detect_anomalies(
    features: pd.DataFrame,
    lookback_days: int = 90,
    window: int = 28,
    z_threshold: float = 3.0,
    iqr_multiplier: float = 3.0,
) -> list[Anomaly]:
    """Flag unusual days in the last ``lookback_days`` with z-score and IQR rules.

    The expected value of each day is the trailing ``window``-day mean (shifted
    one day, so the day itself is excluded). A day is anomalous if its z-score
    exceeds ``z_threshold`` or its residual lies outside the IQR fence. Days
    flagged as suspected stockouts are ignored (those are supply, not demand,
    events).

    Args:
        features: Feature frame with ``date``, ``units`` and ``is_suspected_stockout``.
        lookback_days: Number of recent days to scan.
        window: Trailing window for the expected value.
        z_threshold: Absolute z-score threshold.
        iqr_multiplier: Fence multiplier applied to the residual IQR.

    Returns:
        Anomalies sorted by date.
    """
    df = features.sort_values("date").reset_index(drop=True)
    units = df["units"].astype(float)
    expected = units.shift(1).rolling(window, min_periods=7).mean()
    std = units.shift(1).rolling(window, min_periods=7).std().replace(0, np.nan)
    z = (units - expected) / std
    resid = units - expected

    recent = df.index >= len(df) - lookback_days
    stockout = df.get("is_suspected_stockout", pd.Series(False, index=df.index)).astype(bool)
    q1, q3 = resid[recent].quantile([0.25, 0.75])
    iqr = q3 - q1
    lo_fence, hi_fence = q1 - iqr_multiplier * iqr, q3 + iqr_multiplier * iqr

    anomalies: list[Anomaly] = []
    for i in np.flatnonzero(recent):
        if stockout.iloc[i] or pd.isna(expected.iloc[i]):
            continue
        z_hit = pd.notna(z.iloc[i]) and abs(z.iloc[i]) >= z_threshold
        iqr_hit = iqr > 0 and (resid.iloc[i] > hi_fence or resid.iloc[i] < lo_fence)
        if not (z_hit or iqr_hit):
            continue
        method = "zscore+iqr" if z_hit and iqr_hit else ("zscore" if z_hit else "iqr")
        anomalies.append(
            Anomaly(
                date=df.loc[i, "date"].date(),
                units=float(units.iloc[i]),
                expected=round(float(expected.iloc[i]), 2),
                score=round(float(z.iloc[i]) if pd.notna(z.iloc[i]) else 0.0, 2),
                method=method,
                kind="spike" if resid.iloc[i] > 0 else "drop",
            )
        )
    return anomalies


@dataclass(frozen=True)
class LevelShift:
    """Recent demand level relative to the preceding baseline."""

    ratio: float
    spike: bool
    drop: bool


def detect_level_shift(
    features: pd.DataFrame, recent_days: int = 7, baseline_days: int = 28, ratio: float = 1.5
) -> LevelShift:
    """Compare the mean of the last ``recent_days`` with the prior ``baseline_days``.

    Censored (``NaN``) target values are ignored.
    """
    y = features.sort_values("date")["y"].astype(float)
    recent = y.tail(recent_days).mean()
    baseline = y.iloc[-(recent_days + baseline_days) : -recent_days].mean()
    if pd.isna(recent) or pd.isna(baseline) or baseline <= 0:
        return LevelShift(ratio=1.0, spike=False, drop=False)
    r = float(recent / baseline)
    return LevelShift(ratio=round(r, 3), spike=r >= ratio, drop=r <= 1 / ratio)
