"""Data repositories shared by agents, orchestration and the API.

:class:`Repository` is the read contract the pipeline depends on (dependency
inversion: agents never know where data lives). Two implementations exist:

* :class:`src.db.repository.SQLiteRepository` - production path, reads the database.
* :class:`InMemoryRepository` - wraps a :class:`SalesDataset` (tests, notebooks,
  scenario evaluation on modified data).
"""

from __future__ import annotations

from functools import cached_property
from typing import Protocol, runtime_checkable

import pandas as pd

from src.config.settings import Settings
from src.data.catalog import Product, fallback_product, load_catalog
from src.data.ingestion import SalesDataset, ingest_sales
from src.data.simulate_inventory import (
    load_inventory_params,
    save_inventory_params,
    simulate_inventory_params,
)
from src.logging.logger import get_logger

logger = get_logger(__name__)


@runtime_checkable
class Repository(Protocol):
    """Read access to sales history, calendar, inventory and product metadata."""

    @property
    def source(self) -> str:
        """Label of the data origin (e.g. ``sqlite:m5_raw``)."""

    @property
    def item_ids(self) -> list[str]:
        """Items the pipeline runs on by default (the demo set)."""

    @property
    def date_range(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        """First and last date of the history served to the pipeline."""

    @property
    def future_calendar(self) -> pd.DataFrame:
        """Calendar rows after the last observed sales date."""

    @property
    def inventory_params(self) -> pd.DataFrame:
        """Inventory parameters for every item."""

    def item_sales(self, item_id: str) -> pd.DataFrame:
        """History rows of one item (``SALES_COLUMNS``); ``KeyError`` if unknown."""

    def item_inventory_params(self, item_id: str) -> dict:
        """Inventory parameters of one item; ``KeyError`` if unknown."""

    def product(self, item_id: str) -> Product:
        """Display metadata of one item."""

    def future_actuals(self, item_id: str) -> pd.DataFrame | None:
        """Observed sales after the forecast origin, if available."""

    def to_dataset(self, item_ids: list[str] | None = None) -> SalesDataset:
        """Materialise a :class:`SalesDataset` (used by scenario evaluation)."""


class InMemoryRepository:
    """Cached access to an in-memory dataset and its inventory parameters."""

    def __init__(self, settings: Settings, dataset: SalesDataset | None = None) -> None:
        self.settings = settings
        self._dataset = dataset

    @cached_property
    def dataset(self) -> SalesDataset:
        """The full dataset (loaded lazily on first access)."""
        return self._dataset if self._dataset is not None else ingest_sales(self.settings)

    @property
    def source(self) -> str:
        """Origin of the wrapped dataset."""
        return self.dataset.source

    @property
    def date_range(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        """First and last observed sales date."""
        return pd.Timestamp(self.sales["date"].min()), self.dataset.last_date

    @property
    def sales(self) -> pd.DataFrame:
        """Long-format sales for all items."""
        return self.dataset.sales

    @property
    def future_calendar(self) -> pd.DataFrame:
        """Calendar rows after the last observed date."""
        return self.dataset.future_calendar

    @property
    def item_ids(self) -> list[str]:
        """All item identifiers."""
        return self.dataset.item_ids

    def item_sales(self, item_id: str) -> pd.DataFrame:
        """Raw sales rows of a single item, sorted by date.

        Raises:
            KeyError: if the item is unknown.
        """
        df = self.sales[self.sales["item_id"] == item_id]
        if df.empty:
            raise KeyError(f"Unknown item_id: {item_id}")
        return df.sort_values("date").reset_index(drop=True)

    @cached_property
    def inventory_params(self) -> pd.DataFrame:
        """Simulated inventory parameters (generated and cached on first use)."""
        path = self.settings.data.inventory_params_path
        if path.exists():
            params = load_inventory_params(path)
            if set(self.item_ids).issubset(set(params["item_id"])):
                return params
            logger.warning("inventory_params_stale_regenerating", path=str(path))
        params = simulate_inventory_params(self.sales, seed=self.settings.data.random_seed)
        if self.dataset.source != "synthetic":
            save_inventory_params(params, path)
        return params

    def item_inventory_params(self, item_id: str) -> dict:
        """Inventory parameters for one item as a plain dict.

        Raises:
            KeyError: if the item has no parameters.
        """
        rows = self.inventory_params[self.inventory_params["item_id"] == item_id]
        if rows.empty:
            raise KeyError(f"No inventory parameters for item_id: {item_id}")
        return rows.iloc[0].to_dict()

    @cached_property
    def catalog(self) -> dict[str, Product]:
        """Product display metadata keyed by item id."""
        return load_catalog(self.settings.data.catalog_path)

    def product(self, item_id: str) -> Product:
        """Display metadata for one item (falls back to the item code)."""
        return self.catalog.get(item_id) or fallback_product(item_id)

    def future_actuals(self, item_id: str) -> pd.DataFrame | None:
        """Observed sales after the forecast origin, if available (evaluation only)."""
        actuals = self.dataset.future_actuals
        if actuals is None:
            return None
        df = actuals[actuals["item_id"] == item_id]
        return df.sort_values("date").reset_index(drop=True) if not df.empty else None

    def to_dataset(self, item_ids: list[str] | None = None) -> SalesDataset:
        """The wrapped dataset, optionally restricted to ``item_ids``."""
        if item_ids is None:
            return self.dataset
        keep = set(item_ids)
        actuals = self.dataset.future_actuals
        return SalesDataset(
            sales=self.sales[self.sales["item_id"].isin(keep)].reset_index(drop=True),
            future_calendar=self.future_calendar, source=self.source,
            future_actuals=(actuals[actuals["item_id"].isin(keep)].reset_index(drop=True)
                            if actuals is not None else None))


DataRepository = InMemoryRepository
