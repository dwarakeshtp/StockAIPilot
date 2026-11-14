"""Dashboard tests: chart builders (pure functions) and a headless page smoke test."""

from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go
import pytest

from dashboard.components.charts import (
    anomaly_figure,
    backtest_figure,
    components_figure,
    confidence_gauge,
    forecast_figure,
    inventory_projection_figure,
    metrics_comparison_figure,
    trace_timeline_figure,
)
from src.data.cleaning import clean_item_sales
from src.orchestration.runner import PipelineRunner

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def pipeline_output(synthetic_dataset, tmp_path_factory):
    from src.config.settings import load_settings
    from src.data.repository import DataRepository

    base = load_settings(load_env=False)
    tmp = tmp_path_factory.mktemp("dash")
    settings = base.model_copy(update={
        "forecast": base.forecast.model_copy(update={"backtest_folds": 2,
                                                     "arima_order_grid": [(1, 0, 1)]}),
        "llm": base.llm.model_copy(update={"provider": "none"}),
        "data": base.data.model_copy(update={"simulated_dir": tmp / "sim"}),
        "outputs": base.outputs.model_copy(update={"root": tmp / "out"}),
    })
    repo = DataRepository(settings, dataset=synthetic_dataset)
    item = repo.item_ids[0]
    result = PipelineRunner(settings, repository=repo).run_item(item)
    history, _ = clean_item_sales(repo.item_sales(item))
    return result, history


def test_chart_builders_return_figures(pipeline_output) -> None:
    result, history = pipeline_output
    figures = [
        forecast_figure(history, result, show_models=("arima", "prophet", "ensemble")),
        backtest_figure(result),
        anomaly_figure(history, result),
        confidence_gauge(result.risk.confidence_score),
        components_figure(result.risk.confidence_components),
        inventory_projection_figure(result),
        trace_timeline_figure(result.trace),
    ]
    for fig in figures:
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1


def test_forecast_figure_contains_selected_model(pipeline_output) -> None:
    result, history = pipeline_output
    names = [t.name for t in forecast_figure(history, result).data]
    assert f"Forecast ({result.forecast.model_used})" in names
    assert "80% interval" in names


def test_metrics_comparison_figure() -> None:
    import pandas as pd

    frame = pd.DataFrame({"item_id": ["A", "A", "B", "B"], "model": ["arima", "prophet"] * 2,
                          "wape": [0.1, 0.2, 0.3, 0.4]})
    fig = metrics_comparison_figure(frame, "wape")
    assert len(fig.data) == 2


@pytest.mark.skipif(not (ROOT / "data" / "processed" / "sales_subset.parquet").exists(),
                    reason="processed M5 subset not available")
def test_dashboard_pages_render_without_exceptions() -> None:
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "dashboard" / "app.py"), default_timeout=180).run()
    assert not at.exception
    for page in ("views/forecast.py", "views/risk.py", "views/inventory.py",
                 "views/critic.py", "views/trace.py", "views/metrics.py"):
        at.switch_page(page).run()
        assert not at.exception, f"{page}: {[e.value for e in at.exception]}"
