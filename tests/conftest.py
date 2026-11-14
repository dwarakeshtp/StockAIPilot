"""Shared pytest fixtures (synthetic data only - tests never need the M5 files)."""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
os.environ["LLM_PROVIDER"] = "none"

from src.config.settings import Settings, load_settings  # noqa: E402
from src.data.ingestion import SalesDataset, SyntheticSource  # noqa: E402
from src.data.repository import InMemoryRepository  # noqa: E402


def make_test_settings(tmp_path: Path) -> Settings:
    """Settings tuned for fast tests; outputs and the database live under ``tmp_path``."""
    base = load_settings(load_env=False)
    data = base.data.model_copy(
        update={"simulated_dir": tmp_path / "simulated", "processed_dir": tmp_path / "processed"}
    )
    forecast = base.forecast.model_copy(
        update={"backtest_folds": 2, "arima_order_grid": [(1, 0, 1)],
                "prophet_uncertainty_samples": 100, "arima_fit_window": 200}
    )
    llm = base.llm.model_copy(update={"provider": "none"})
    outputs = base.outputs.model_copy(update={"root": tmp_path / "outputs"})
    database = base.database.model_copy(
        update={"url": f"sqlite:///{tmp_path / 'db' / 'test.db'}", "init_on_startup": False})
    return base.model_copy(update={"data": data, "forecast": forecast, "llm": llm,
                                   "outputs": outputs, "database": database})


@pytest.fixture(scope="session")
def synthetic_dataset() -> SalesDataset:
    """Small deterministic synthetic dataset (4 items x 400 days)."""
    return SyntheticSource(n_items=4, history_days=400, seed=7).load()


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    """Settings tuned for fast tests, writing outputs to a temp directory."""
    return make_test_settings(tmp_path)


@pytest.fixture()
def repository(settings: Settings, synthetic_dataset: SalesDataset) -> InMemoryRepository:
    """In-memory repository backed by the synthetic dataset."""
    return InMemoryRepository(settings, dataset=synthetic_dataset)


@pytest.fixture()
def sqlite_engine(settings: Settings, synthetic_dataset: SalesDataset):
    """Temp SQLite database loaded with the synthetic dataset."""
    from src.db.engine import get_engine
    from src.db.loader import load_dataset

    engine = get_engine(settings.database.url)
    load_dataset(engine, settings, synthetic_dataset)
    yield engine
    engine.dispose()


@pytest.fixture()
def sqlite_repository(settings: Settings, sqlite_engine):
    """SQLite repository over the synthetic database."""
    from src.db.repository import SQLiteRepository

    return SQLiteRepository(settings, sqlite_engine)


@pytest.fixture()
def services(settings: Settings, sqlite_engine) -> Iterator:
    """Service layer over the synthetic database."""
    from src.services import Services

    built = Services.build(settings, sqlite_engine)
    yield built
    built.close()


@pytest.fixture(scope="session")
def base_pipeline_result(synthetic_dataset: SalesDataset, tmp_path_factory):
    """One real pipeline result on synthetic data (computed once per session)."""
    from src.orchestration.runner import PipelineRunner

    settings = make_test_settings(tmp_path_factory.mktemp("base_result"))
    repo = InMemoryRepository(settings, dataset=synthetic_dataset)
    return PipelineRunner(settings, repository=repo).run_item(repo.item_ids[0])


@pytest.fixture()
def pipeline_result_factory(base_pipeline_result):
    """Copies of the base result with a fresh run id / timestamp (and optional item id)."""
    from src.utils.ids import new_run_id, utc_now

    def make(item_id: str | None = None, run_id: str | None = None):
        update = {"run_id": run_id or new_run_id(), "created_at": utc_now()}
        if item_id:
            update["item_id"] = item_id
        return base_pipeline_result.model_copy(update=update)

    return make


@pytest.fixture()
def item_sales(synthetic_dataset: SalesDataset) -> pd.DataFrame:
    """Sales rows of the first synthetic item."""
    item = synthetic_dataset.item_ids[0]
    return synthetic_dataset.sales[synthetic_dataset.sales["item_id"] == item].reset_index(
        drop=True
    )
