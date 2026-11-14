"""Tests for the SQLite data layer: loader, repository parity and inventory updates."""

from __future__ import annotations

import pandas as pd
import pytest
from sqlalchemy import inspect, text

from src.data.ingestion import ProcessedParquetSource
from src.data.repository import InMemoryRepository, Repository
from src.db.engine import get_engine
from src.db.loader import init_db, load_from_processed_seed, price_spells, read_meta
from src.db.repository import DatabaseNotInitialisedError, SQLiteRepository
from src.db.stores import RunStore
from tests.conftest import ROOT, make_test_settings


def test_loader_creates_schema_and_counts(sqlite_engine, synthetic_dataset) -> None:
    tables = set(inspect(sqlite_engine).get_table_names())
    assert {"items", "calendar", "daily_sales", "prices", "sales_holdout",
            "inventory_positions", "pipeline_runs", "decisions", "jobs", "meta"} <= tables
    assert "v_daily_sales" in inspect(sqlite_engine).get_view_names()
    with sqlite_engine.connect() as conn:
        n_sales = conn.execute(text("SELECT COUNT(*) FROM daily_sales")).scalar_one()
        n_items = conn.execute(text("SELECT COUNT(*) FROM items")).scalar_one()
        n_cal = conn.execute(text("SELECT COUNT(*) FROM calendar")).scalar_one()
    assert n_sales == len(synthetic_dataset.sales)
    assert n_items == len(synthetic_dataset.item_ids)
    assert n_cal == synthetic_dataset.sales["date"].nunique() + len(
        synthetic_dataset.future_calendar)
    meta = read_meta(sqlite_engine)
    assert meta["source"] == "synthetic"
    assert meta["last_sales_date"] == str(synthetic_dataset.last_date.date())


def test_price_spells_compress_constant_runs() -> None:
    daily = pd.DataFrame({
        "item_id": ["A"] * 6 + ["B"] * 2,
        "date": pd.date_range("2016-01-01", periods=6).tolist()
        + pd.date_range("2016-01-01", periods=2).tolist(),
        "sell_price": [1.0, 1.0, 1.5, 1.5, None, 1.5, 2.0, 2.0],
    })
    spells = price_spells(daily)
    assert spells.to_dict("records") == [
        {"item_id": "A", "start_date": "2016-01-01", "end_date": "2016-01-02", "sell_price": 1.0},
        {"item_id": "A", "start_date": "2016-01-03", "end_date": "2016-01-04", "sell_price": 1.5},
        {"item_id": "A", "start_date": "2016-01-06", "end_date": "2016-01-06", "sell_price": 1.5},
        {"item_id": "B", "start_date": "2016-01-01", "end_date": "2016-01-02", "sell_price": 2.0},
    ]


def test_init_db_is_idempotent_and_rebuild_keeps_runs(settings, synthetic_dataset,
                                                      pipeline_result_factory) -> None:
    engine = get_engine(settings.database.url)
    first = init_db(settings, engine=engine, dataset=synthetic_dataset)
    assert not first.skipped and first.items == 4
    RunStore(engine).save(pipeline_result_factory())
    again = init_db(settings, engine=engine, dataset=synthetic_dataset)
    assert again.skipped and again.source == "synthetic"
    rebuilt = init_db(settings, engine=engine, dataset=synthetic_dataset, rebuild=True)
    assert not rebuilt.skipped
    assert len(RunStore(engine).list()) == 1


def test_sqlite_repository_matches_in_memory(sqlite_repository, repository) -> None:
    assert isinstance(sqlite_repository, Repository)
    assert sqlite_repository.item_ids == repository.item_ids
    assert sqlite_repository.source == "sqlite:synthetic"
    for item in repository.item_ids:
        pd.testing.assert_frame_equal(sqlite_repository.item_sales(item),
                                      repository.item_sales(item), check_dtype=False)
        assert sqlite_repository.item_inventory_params(item) == \
            repository.item_inventory_params(item)
    pd.testing.assert_frame_equal(sqlite_repository.future_calendar, repository.future_calendar,
                                  check_dtype=False)
    assert sqlite_repository.date_range == repository.date_range


def test_sqlite_repository_errors_and_search(sqlite_repository, settings) -> None:
    with pytest.raises(KeyError):
        sqlite_repository.item_sales("UNKNOWN")
    with pytest.raises(KeyError):
        sqlite_repository.item_inventory_params("UNKNOWN")
    rows, total = sqlite_repository.search_items("SYN_002")
    assert total == 1 and rows[0]["item_id"] == "FOODS_SYN_002" and rows[0]["is_demo"]
    assert sqlite_repository.item("UNKNOWN") is None
    assert sqlite_repository.future_actuals(sqlite_repository.item_ids[0]) is None
    empty = SQLiteRepository(settings, get_engine("sqlite:///" + str(
        settings.outputs.root / "empty.db")))
    with pytest.raises(DatabaseNotInitialisedError):
        _ = empty.item_ids


def test_update_inventory(sqlite_repository) -> None:
    item = sqlite_repository.item_ids[0]
    updated = sqlite_repository.update_inventory(item, {"on_hand": 7, "lead_time_days": 3})
    assert updated["on_hand"] == 7 and updated["lead_time_days"] == 3
    assert updated["is_simulated"] is False
    with pytest.raises(ValueError):
        sqlite_repository.update_inventory(item, {"item_id": "X"})


@pytest.mark.skipif(not (ROOT / "data" / "processed" / "sales_subset.parquet").exists(),
                    reason="processed M5 subset not available")
def test_demo_items_identical_to_parquet(tmp_path) -> None:
    """Regression gate: SQLite serves the 12 demo items exactly like the parquet subset."""
    settings = make_test_settings(tmp_path)
    settings = settings.model_copy(update={"data": settings.data.model_copy(update={
        "processed_dir": ROOT / "data" / "processed", "simulated_dir": ROOT / "data" / "simulated"
    })})
    engine = get_engine(settings.database.url)
    load_from_processed_seed(engine, settings)
    db = SQLiteRepository(settings, engine)
    cfg = settings.data
    mem = InMemoryRepository(settings, ProcessedParquetSource(cfg.sales_path,
                                                              cfg.future_calendar_path).load())
    assert db.item_ids == mem.item_ids
    for item in mem.item_ids:
        pd.testing.assert_frame_equal(db.item_sales(item), mem.item_sales(item))
        assert db.item_inventory_params(item) == mem.item_inventory_params(item)
        pd.testing.assert_frame_equal(db.future_actuals(item), mem.future_actuals(item))
    pd.testing.assert_frame_equal(db.future_calendar, mem.future_calendar)
