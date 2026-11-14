"""Feature engineering and chronological splitting.

Rolling statistics are shifted by one day so a feature at day *t* only uses
information available at the end of day *t-1* (no target leakage).
"""

from __future__ import annotations

import pandas as pd

PROPHET_REGRESSORS: tuple[str, ...] = ("snap", "price_ratio")


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar, event, price and rolling-demand features to a cleaned item frame.

    Args:
        df: Output of :func:`src.data.cleaning.clean_item_sales`.

    Returns:
        A copy of ``df`` with additional feature columns.
    """
    out = df.sort_values("date").reset_index(drop=True).copy()
    out["dow"] = out["date"].dt.dayofweek
    out["month"] = out["date"].dt.month
    out["is_weekend"] = (out["dow"] >= 5).astype(int)
    out["is_event"] = (out["event_name"] != "").astype(int)

    price_ref = out["sell_price"].rolling(28, min_periods=1).mean()
    out["price_ratio"] = (out["sell_price"] / price_ref).fillna(1.0)
    out["price_change_pct"] = (out["price_ratio"] - 1.0).round(4)

    lagged = out["units"].shift(1)
    out["rolling_mean_7"] = lagged.rolling(7, min_periods=1).mean()
    out["rolling_mean_28"] = lagged.rolling(28, min_periods=1).mean()
    out["rolling_std_28"] = lagged.rolling(28, min_periods=2).std()
    out["lag_7"] = out["units"].shift(7)
    out["lag_28"] = out["units"].shift(28)
    return out


def build_future_frame(
    features: pd.DataFrame, future_calendar: pd.DataFrame, horizon: int
) -> pd.DataFrame:
    """Create the regressor frame for the next ``horizon`` days.

    Prices are unknown in the future, so the last observed price is carried
    forward (``price_ratio`` computed against the trailing 28-day average).
    Calendar rows missing from ``future_calendar`` default to no event / no SNAP.

    Args:
        features: Feature frame of one item (history).
        future_calendar: Calendar rows after the last history date.
        horizon: Number of days to forecast.

    Returns:
        DataFrame with ``date, event_name, event_type, is_event, snap, dow,
        sell_price, price_ratio`` for each future day.
    """
    last_date = features["date"].max()
    dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="D")
    cal = future_calendar.copy()
    cal["date"] = pd.to_datetime(cal["date"])
    frame = pd.DataFrame({"date": dates}).merge(cal, on="date", how="left")
    frame["event_name"] = frame["event_name"].fillna("")
    frame["event_type"] = frame["event_type"].fillna("")
    frame["snap"] = frame["snap"].fillna(0).astype(int)
    frame["is_event"] = (frame["event_name"] != "").astype(int)
    frame["dow"] = frame["date"].dt.dayofweek
    last_price = float(features["sell_price"].iloc[-1])
    ref_price = float(features["sell_price"].tail(28).mean())
    frame["sell_price"] = last_price
    frame["price_ratio"] = last_price / ref_price if ref_price else 1.0
    return frame


def chronological_split(df: pd.DataFrame, test_days: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a time-ordered frame into train and the final ``test_days`` rows.

    Raises:
        ValueError: if ``test_days`` is not smaller than the number of rows.
    """
    if test_days <= 0 or test_days >= len(df):
        raise ValueError("test_days must be between 1 and len(df) - 1")
    ordered = df.sort_values("date").reset_index(drop=True)
    return ordered.iloc[:-test_days].copy(), ordered.iloc[-test_days:].copy()


def prophet_holidays(features: pd.DataFrame, future: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return a Prophet ``holidays`` frame (``holiday, ds``) from event columns."""
    frames = [features[["date", "event_name"]]]
    if future is not None:
        frames.append(future[["date", "event_name"]])
    events = pd.concat(frames, ignore_index=True)
    events = events[events["event_name"] != ""].drop_duplicates()
    return events.rename(columns={"date": "ds", "event_name": "holiday"})[["holiday", "ds"]]
