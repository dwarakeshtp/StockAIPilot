"""Tests for data ingestion, persistence and the repository."""

from __future__ import annotations

import pandas as pd
import pytest

from src.config.settings import Settings
from src.data.ingestion import (
    SALES_COLUMNS,
    ProcessedParquetSource,
    SalesDataset,
    ingest_sales,
    save_dataset,
    select_items,
    validate_sales_schema,
)
from src.data.repository import DataRepository
from src.data.simulate_inventory import INVENTORY_PARAM_COLUMNS, simulate_inventory_params


def test_synthetic_dataset_schema(synthetic_dataset: SalesDataset) -> None:
    validate_sales_schema(synthetic_dataset.sales)
    assert synthetic_dataset.sales["item_id"].nunique() == 4
    assert len(synthetic_dataset.future_calendar) >= 7
    assert (synthetic_dataset.sales["units"] >= 0).all()


def test_validate_schema_raises_on_missing_columns() -> None:
    with pytest.raises(ValueError, match="missing required columns"):
        validate_sales_schema(pd.DataFrame({"date": [], "units": []}))


def test_save_and_load_roundtrip(settings: Settings, synthetic_dataset: SalesDataset) -> None:
    save_dataset(synthetic_dataset, settings)
    source = ProcessedParquetSource(settings.data.sales_path, settings.data.future_calendar_path)
    assert source.available()
    loaded = source.load()
    assert list(loaded.sales.columns) == SALES_COLUMNS
    assert len(loaded.sales) == len(synthetic_dataset.sales)
    assert loaded.last_date == synthetic_dataset.last_date


def test_ingest_falls_back_to_synthetic(settings: Settings) -> None:
    dataset = ingest_sales(settings)
    assert dataset.source == "synthetic"


def test_ingest_raises_without_fallback(settings: Settings) -> None:
    data = settings.data.model_copy(update={"allow_synthetic_fallback": False})
    with pytest.raises(FileNotFoundError):
        ingest_sales(settings.model_copy(update={"data": data}))


def test_select_items_mixes_volume_levels() -> None:
    days = [f"d_{i}" for i in range(30)]
    window = pd.DataFrame(
        [[100 - i] * 30 for i in range(80)], index=[f"I{i}" for i in range(80)], columns=days
    )
    picked = select_items(window, 6)
    assert len(picked) == 6
    assert picked[:4] == ["I0", "I1", "I2", "I3"]
    assert all(int(p[1:]) >= 40 for p in picked[4:])


def test_simulated_inventory_params(synthetic_dataset: SalesDataset) -> None:
    params = simulate_inventory_params(synthetic_dataset.sales, seed=1)
    assert list(params.columns) == INVENTORY_PARAM_COLUMNS
    assert params["is_simulated"].all()
    assert (params["lead_time_days"] >= 2).all()
    assert (params["moq"] % params["pack_size"] == 0).all()


def test_repository_item_access(repository: DataRepository) -> None:
    item = repository.item_ids[0]
    df = repository.item_sales(item)
    assert df["date"].is_monotonic_increasing
    params = repository.item_inventory_params(item)
    assert params["item_id"] == item
    with pytest.raises(KeyError):
        repository.item_sales("UNKNOWN")
