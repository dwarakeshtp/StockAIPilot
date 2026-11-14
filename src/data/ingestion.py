"""Data ingestion: build, persist and load the long-format sales dataset.

Three interchangeable sources implement :class:`SalesDataSource`:

* :class:`M5RawSource` - builds the single-store, single-category subset from the
  raw M5 CSV files (used once by ``scripts/build_subset.py``).
* :class:`ProcessedParquetSource` - loads the committed processed subset (default).
* :class:`SyntheticSource` - generates an M5-shaped dataset (tests / offline fallback).

All sources return a :class:`SalesDataset` whose ``sales`` frame follows
:data:`SALES_COLUMNS`, so downstream layers never depend on the origin of the data.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.config.settings import Settings
from src.logging.logger import get_logger

logger = get_logger(__name__)

SALES_COLUMNS: list[str] = [
    "date",
    "item_id",
    "dept_id",
    "cat_id",
    "store_id",
    "units",
    "sell_price",
    "event_name",
    "event_type",
    "snap",
]
FUTURE_CALENDAR_COLUMNS: list[str] = ["date", "event_name", "event_type", "snap"]


@dataclass(frozen=True)
class SalesDataset:
    """Container for the sales history and the calendar of the forecast window."""

    sales: pd.DataFrame
    future_calendar: pd.DataFrame
    source: str
    future_actuals: pd.DataFrame | None = field(default=None)

    @property
    def item_ids(self) -> list[str]:
        """Sorted list of item identifiers in the dataset."""
        return sorted(self.sales["item_id"].unique().tolist())

    @property
    def last_date(self) -> pd.Timestamp:
        """Last date with observed sales (the forecast origin)."""
        return pd.Timestamp(self.sales["date"].max())


class SalesDataSource(ABC):
    """Abstract source of a :class:`SalesDataset`."""

    name: str = "abstract"

    @abstractmethod
    def load(self) -> SalesDataset:
        """Load and return the dataset."""


def validate_sales_schema(df: pd.DataFrame) -> None:
    """Raise ``ValueError`` if ``df`` misses any of :data:`SALES_COLUMNS`."""
    missing = [c for c in SALES_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Sales data is missing required columns: {missing}")


class ProcessedParquetSource(SalesDataSource):
    """Load the processed subset written by :func:`save_dataset`."""

    name = "m5_processed"

    def __init__(self, sales_path: Path, future_calendar_path: Path) -> None:
        self.sales_path = sales_path
        self.future_calendar_path = future_calendar_path

    def available(self) -> bool:
        """Whether the processed files exist on disk."""
        return self.sales_path.exists() and self.future_calendar_path.exists()

    def load(self) -> SalesDataset:
        """Read the parquet files and normalise dtypes."""
        sales = pd.read_parquet(self.sales_path)
        sales["date"] = pd.to_datetime(sales["date"])
        validate_sales_schema(sales)
        future = pd.read_parquet(self.future_calendar_path)
        future["date"] = pd.to_datetime(future["date"])
        actuals_path = self.sales_path.parent / "actuals_future.parquet"
        actuals = None
        if actuals_path.exists():
            actuals = pd.read_parquet(actuals_path)
            actuals["date"] = pd.to_datetime(actuals["date"])
        logger.info(
            "dataset_loaded",
            source=self.name,
            rows=len(sales),
            items=sales["item_id"].nunique(),
            start=str(sales["date"].min().date()),
            end=str(sales["date"].max().date()),
        )
        return SalesDataset(sales=sales, future_calendar=future, source=self.name,
                            future_actuals=actuals)


class M5RawSource(SalesDataSource):
    """Build the configured subset from the raw M5 CSV files."""

    name = "m5_raw"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.raw_dir = settings.data.raw_dir

    def available(self) -> bool:
        """Whether the three required raw M5 files exist."""
        return all(
            (self.raw_dir / f).exists()
            for f in ("sales_train_evaluation.csv", "calendar.csv", "sell_prices.csv")
        )

    def load(self) -> SalesDataset:
        """Read the raw files, select items and return the long-format subset."""
        cfg = self.settings.data
        wide = self._read_sales_wide(cfg.store_id, cfg.category)
        calendar = pd.read_csv(self.raw_dir / "calendar.csv", parse_dates=["date"])
        if "d" not in calendar.columns:
            # Some M5 mirrors drop the `d` key; rows are ordered d_1 .. d_N.
            calendar = calendar.sort_values("date").reset_index(drop=True)
            calendar["d"] = "d_" + (calendar.index + 1).astype(str)
        day_cols = [c for c in wide.columns if c.startswith("d_")]
        window_cols = day_cols[-cfg.history_days :]

        item_ids = cfg.item_ids or select_items(wide.set_index("item_id")[window_cols],
                                                cfg.n_items)
        wide = wide[wide["item_id"].isin(item_ids)]
        logger.info("m5_items_selected", items=item_ids)

        id_cols = ["item_id", "dept_id", "cat_id", "store_id"]
        long = wide[id_cols + window_cols].melt(id_vars=id_cols, var_name="d", value_name="units")
        snap_col = f"snap_{cfg.state_id}"
        cal = calendar[["d", "date", "wm_yr_wk", "event_name_1", "event_type_1", snap_col]].rename(
            columns={"event_name_1": "event_name", "event_type_1": "event_type", snap_col: "snap"}
        )
        long = long.merge(cal, on="d", how="left")

        prices = self._read_prices(cfg.store_id, item_ids)
        long = long.merge(prices, on=["item_id", "wm_yr_wk"], how="left")
        long["event_name"] = long["event_name"].fillna("")
        long["event_type"] = long["event_type"].fillna("")
        long["snap"] = long["snap"].fillna(0).astype(int)
        sales = long[SALES_COLUMNS].sort_values(["item_id", "date"]).reset_index(drop=True)

        last_day_index = int(window_cols[-1].split("_")[1])
        future = cal[cal["d"].str[2:].astype(int) > last_day_index].copy()
        future["event_name"] = future["event_name"].fillna("")
        future["event_type"] = future["event_type"].fillna("")
        future = future[FUTURE_CALENDAR_COLUMNS].reset_index(drop=True)

        actuals = self._read_future_actuals(cfg.store_id, item_ids, cal)
        return SalesDataset(sales=sales, future_calendar=future, source=self.name,
                            future_actuals=actuals)

    def _read_sales_wide(self, store_id: str, category: str) -> pd.DataFrame:
        chunks = []
        for chunk in pd.read_csv(self.raw_dir / "sales_train_evaluation.csv", chunksize=5000):
            chunks.append(chunk[(chunk["store_id"] == store_id) & (chunk["cat_id"] == category)])
        wide = pd.concat(chunks, ignore_index=True)
        if wide.empty:
            raise ValueError(f"No M5 rows for store={store_id} category={category}")
        return wide

    def _read_prices(self, store_id: str, item_ids: list[str]) -> pd.DataFrame:
        chunks = []
        for chunk in pd.read_csv(self.raw_dir / "sell_prices.csv", chunksize=500_000):
            chunks.append(chunk[(chunk["store_id"] == store_id) & chunk["item_id"].isin(item_ids)])
        prices = pd.concat(chunks, ignore_index=True)
        return prices[["item_id", "wm_yr_wk", "sell_price"]]

    def _read_future_actuals(
        self, store_id: str, item_ids: list[str], cal: pd.DataFrame
    ) -> pd.DataFrame | None:
        path = self.raw_dir / "sales_test_evaluation.csv"
        if not path.exists():
            return None
        test = pd.read_csv(path)
        if "item_id" not in test.columns:
            parts = test["id"].str.rsplit("_", n=3, expand=True)
            test["item_id"] = parts[0]
            test["store_id"] = parts[1] + "_" + parts[2]
        test = test[(test["store_id"] == store_id) & test["item_id"].isin(item_ids)]
        day_cols = [c for c in test.columns if c.startswith("d_")]
        if test.empty or not day_cols:
            return None
        long = test[["item_id", *day_cols]].melt(id_vars="item_id", var_name="d",
                                                 value_name="units")
        long = long.merge(cal[["d", "date"]], on="d", how="left")
        return long[["date", "item_id", "units"]].sort_values(["item_id", "date"]).reset_index(
            drop=True
        )


def select_items(window: pd.DataFrame, n_items: int) -> list[str]:
    """Pick a deterministic mix of high-volume and mid-volume items.

    Two thirds of the items are the highest-volume SKUs (stable demand); the
    remaining third is sampled evenly from mid-ranked SKUs that sell on most
    days, which yields more varied risk profiles for the demo.

    Args:
        window: Wide frame indexed by ``item_id`` with one column per day.
        n_items: Number of items to select.

    Returns:
        List of selected item identifiers.
    """
    totals = window.sum(axis=1).sort_values(ascending=False)
    zero_share = (window == 0).mean(axis=1)
    n_top = min(len(totals), math.ceil(n_items * 2 / 3))
    selected = totals.index[:n_top].tolist()
    remaining = n_items - n_top
    if remaining > 0:
        mid = [i for i in totals.index[40:] if zero_share[i] < 0.4 and i not in selected][:100]
        if mid:
            idx = np.linspace(0, len(mid) - 1, num=min(remaining, len(mid))).round().astype(int)
            selected += [mid[i] for i in idx]
    return selected


class SyntheticSource(SalesDataSource):
    """Generate an M5-shaped dataset with known seasonality, events and spikes."""

    name = "synthetic"

    def __init__(
        self,
        n_items: int = 6,
        history_days: int = 730,
        end_date: str = "2016-05-22",
        seed: int = 42,
        horizon_days: int = 28,
    ) -> None:
        self.n_items = n_items
        self.history_days = history_days
        self.end_date = pd.Timestamp(end_date)
        self.seed = seed
        self.horizon_days = horizon_days

    def load(self) -> SalesDataset:
        """Generate and return the synthetic dataset."""
        from src.data.synthetic import generate_synthetic_sales

        sales, future = generate_synthetic_sales(
            n_items=self.n_items,
            history_days=self.history_days,
            end_date=self.end_date,
            seed=self.seed,
            horizon_days=self.horizon_days,
        )
        logger.warning("synthetic_dataset_used", items=self.n_items, days=self.history_days)
        return SalesDataset(sales=sales, future_calendar=future, source=self.name)


def save_dataset(dataset: SalesDataset, settings: Settings) -> Path:
    """Persist ``dataset`` to the processed directory as parquet files."""
    cfg = settings.data
    cfg.processed_dir.mkdir(parents=True, exist_ok=True)
    dataset.sales.to_parquet(cfg.sales_path, index=False)
    dataset.future_calendar.to_parquet(cfg.future_calendar_path, index=False)
    if dataset.future_actuals is not None:
        dataset.future_actuals.to_parquet(cfg.processed_dir / "actuals_future.parquet",
                                          index=False)
    logger.info("dataset_saved", path=str(cfg.sales_path), rows=len(dataset.sales))
    return cfg.sales_path


def ingest_sales(settings: Settings) -> SalesDataset:
    """Load the dataset from the best available source.

    Order of preference: processed parquet subset, then (optionally) synthetic data.

    Raises:
        FileNotFoundError: if no processed data exists and the synthetic fallback
            is disabled.
    """
    cfg = settings.data
    processed = ProcessedParquetSource(cfg.sales_path, cfg.future_calendar_path)
    if processed.available():
        return processed.load()
    if cfg.allow_synthetic_fallback:
        return SyntheticSource(
            n_items=min(cfg.n_items, 8), history_days=cfg.history_days, seed=cfg.random_seed
        ).load()
    raise FileNotFoundError(
        f"Processed data not found at {cfg.sales_path}. Run `python scripts/build_subset.py`."
    )
