"""Shared dashboard state: cached runner, results and sidebar controls."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.agents.schemas import PipelineResult
from src.config.settings import Settings, load_settings
from src.data.cleaning import clean_item_sales
from src.logging.logger import configure_logging
from src.orchestration.decisions import DecisionStore
from src.orchestration.runner import PipelineRunner, load_latest_results

RESULTS_KEY = "results"
WHATIF_KEY = "what_if_results"
ITEM_KEY = "selected_item"


@st.cache_resource(show_spinner=False)
def get_settings() -> Settings:
    """Application settings (cached for the server lifetime)."""
    settings = load_settings()
    configure_logging(settings.outputs.logs_dir, level="WARNING")
    return settings


@st.cache_resource(show_spinner="Loading data and agents ...")
def get_runner() -> PipelineRunner:
    """The pipeline runner (agents, graph, repository) - created once."""
    return PipelineRunner(get_settings())


def get_decision_store() -> DecisionStore:
    """Decision store backed by ``outputs/decisions.jsonl``."""
    return DecisionStore(get_settings().outputs.decisions_file)


@st.cache_data(show_spinner=False)
def get_history(item_id: str) -> pd.DataFrame:
    """Cleaned daily history of an item (with stockout flags)."""
    clean, _ = clean_item_sales(get_runner().repository.item_sales(item_id))
    return clean


@st.cache_data(show_spinner=False)
def get_future_actuals(item_id: str) -> pd.DataFrame | None:
    """Observed sales after the forecast origin (evaluation only)."""
    return get_runner().repository.future_actuals(item_id)


def results() -> dict[str, PipelineResult]:
    """Results by item, loaded from the last saved batch on first access."""
    if RESULTS_KEY not in st.session_state:
        st.session_state[RESULTS_KEY] = {
            r.item_id: r for r in load_latest_results(get_settings())
        }
    return st.session_state[RESULTS_KEY]


def what_if_results() -> dict[str, PipelineResult]:
    """What-if results by item (session only)."""
    return st.session_state.setdefault(WHATIF_KEY, {})


def store_results(new_results: list[PipelineResult], persist: bool = True) -> None:
    """Merge new results into the session and optionally persist the full batch."""
    current = results()
    for r in new_results:
        current[r.item_id] = r
        what_if_results().pop(r.item_id, None)
    if persist and current:
        get_runner().save_results(list(current.values()))


def selected_item() -> str | None:
    """Currently selected item id."""
    return st.session_state.get(ITEM_KEY)


def selected_result() -> PipelineResult | None:
    """Pipeline result of the selected item, if available."""
    item = selected_item()
    return results().get(item) if item else None


def run_pipeline(item_ids: list[str]) -> None:
    """Run the agent pipeline for ``item_ids`` with a progress bar."""
    runner = get_runner()
    bar = st.progress(0.0, text="Starting agents ...")
    done: list[PipelineResult] = []
    for idx, item in enumerate(item_ids, start=1):
        bar.progress((idx - 1) / len(item_ids), text=f"Running agents for {item} ({idx}/"
                                                       f"{len(item_ids)}) ...")
        done.append(runner.run_item(item))
    bar.progress(1.0, text="Done")
    store_results(done)
    bar.empty()


def render_sidebar() -> None:
    """Sidebar with item picker, run controls and system status."""
    runner = get_runner()
    items = runner.repository.item_ids
    with st.sidebar:
        st.markdown("### Controls")
        if ITEM_KEY not in st.session_state or st.session_state[ITEM_KEY] not in items:
            st.session_state[ITEM_KEY] = items[0]
        st.selectbox("Item (SKU)", items, key=ITEM_KEY)
        c1, c2 = st.columns(2)
        if c1.button("Run item", help="Run all four agents for the selected item",
                     type="primary"):
            run_pipeline([st.session_state[ITEM_KEY]])
            st.rerun()
        if c2.button("Run all", help=f"Run the pipeline for all {len(items)} items (~30s)"):
            run_pipeline(items)
            st.rerun()

        st.divider()
        dataset = runner.repository.dataset
        st.caption("**Data**")
        st.caption(f"Source: `{dataset.source}` - {len(items)} items, "
                   f"{dataset.sales['date'].min().date()} to {dataset.last_date.date()}")
        st.caption(f"Forecast origin: {dataset.last_date.date()} | horizon "
                   f"{get_settings().forecast.horizon} days")
        st.caption("**LLM**")
        st.caption(f"Provider chain: `{runner.llm_description}`")
        st.badge("Inventory inputs are simulated", color="orange", icon=":material/science:")
        n = len(results())
        st.caption(f"{n} item result(s) loaded" if n else "No results yet - click **Run all**.")


def require_result() -> PipelineResult | None:
    """Return the selected result or render a hint when missing."""
    result = selected_result()
    if result is None:
        st.info("No agent results for this item yet. Use **Run item** or **Run all** in the "
                "sidebar (or run `python scripts/run_pipeline.py`).")
    return result
