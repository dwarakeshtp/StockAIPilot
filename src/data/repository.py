"""Read-only data repository shared by agents, orchestration and dashboard.

The repository loads the dataset and simulated inventory parameters once and
serves per-item slices, hiding storage details from consumers (dependency
inversion: agents depend on this abstraction, not on files).
"""

from __future__ import annotations

from functools import cached_property

import pandas as pd

from src.config.settings import Settings
from src.data.ingestion import SalesDataset, ingest_sales
from src.data.simulate_inventory import (
    load_inventory_params,
    save_inventory_params,
    simulate_inventory_params,
)
from src.logging.logger import get_logger

logger = get_logger(__name__)


class DataRepository:
    """Cached access to sales history, future calendar and inventory parameters."""

    def __init__(self, settings: Settings, dataset: SalesDataset | None = None) -> None:
        self.settings = settings
        self._dataset = dataset

    @cached_property
    def dataset(self) -> SalesDataset:
        """The full dataset (loaded lazily on first access)."""
        return self._dataset if self._dataset is not None else ingest_sales(self.settings)

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

    def future_actuals(self, item_id: str) -> pd.DataFrame | None:
        """Observed sales after the forecast origin, if available (evaluation only)."""
        actuals = self.dataset.future_actuals
        if actuals is None:
            return None
        df = actuals[actuals["item_id"] == item_id]
        return df.sort_values("date").reset_index(drop=True) if not df.empty else None
