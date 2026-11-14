"""Shared pytest fixtures (synthetic data only - tests never need the M5 files)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
os.environ["LLM_PROVIDER"] = "none"

from src.config.settings import Settings, load_settings  # noqa: E402
from src.data.ingestion import SalesDataset, SyntheticSource  # noqa: E402
from src.data.repository import DataRepository  # noqa: E402


@pytest.fixture(scope="session")
def synthetic_dataset() -> SalesDataset:
    """Small deterministic synthetic dataset (4 items x 400 days)."""
    return SyntheticSource(n_items=4, history_days=400, seed=7).load()


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    """Settings tuned for fast tests, writing outputs to a temp directory."""
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
    return base.model_copy(update={"data": data, "forecast": forecast, "llm": llm,
                                   "outputs": outputs})


@pytest.fixture()
def repository(settings: Settings, synthetic_dataset: SalesDataset) -> DataRepository:
    """Repository backed by the synthetic dataset."""
    return DataRepository(settings, dataset=synthetic_dataset)


@pytest.fixture()
def item_sales(synthetic_dataset: SalesDataset) -> pd.DataFrame:
    """Sales rows of the first synthetic item."""
    item = synthetic_dataset.item_ids[0]
    return synthetic_dataset.sales[synthetic_dataset.sales["item_id"] == item].reset_index(
        drop=True
    )
