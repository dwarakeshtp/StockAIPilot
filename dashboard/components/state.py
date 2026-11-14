"""Shared dashboard state: API access, cached results and sidebar controls.

All data comes from the FastAPI backend through :mod:`dashboard.components.api_client`.
"""

from __future__ import annotations

import time

import pandas as pd
import streamlit as st

from dashboard.components.api_client import ApiError, StockPilotClient, get_client
from src.agents.schemas import PipelineResult
from src.config.settings import Settings, load_settings
from src.data.catalog import Product, fallback_product

RESULTS_KEY = "results"
WHATIF_KEY = "what_if_results"
ITEM_KEY = "selected_item"
EXTRA_ITEMS_KEY = "extra_items"
PRODUCTS_KEY = "products"
SEARCH_PICK_KEY = "search_pick"
JOB_POLL_SECONDS = 1.0


@st.cache_resource(show_spinner=False)
def get_settings() -> Settings:
    """Application settings (UI defaults only; data comes from the API)."""
    return load_settings()


def api() -> StockPilotClient:
    """The API client."""
    return get_client()


def system_info() -> dict:
    """Backend info, fetched once per session."""
    if "system_info" not in st.session_state:
        st.session_state["system_info"] = api().system_info()
    return st.session_state["system_info"]


def _products() -> dict[str, Product]:
    return st.session_state.setdefault(PRODUCTS_KEY, {})


def _remember(rows: list[dict]) -> None:
    fields = set(Product.model_fields)
    for row in rows:
        _products()[row["item_id"]] = Product(**{k: v for k, v in row.items() if k in fields})


def demo_items() -> list[str]:
    """Demo item ids (cached per session)."""
    if "demo_items" not in st.session_state:
        page = api().items(demo_only=True, limit=500)
        _remember(page["items"])
        st.session_state["demo_items"] = [r["item_id"] for r in page["items"]]
    return st.session_state["demo_items"]


def get_product(item_id: str) -> Product:
    """Display metadata (name, department) for an item code."""
    product = _products().get(item_id)
    if product is None:
        try:
            _remember([api().item(item_id)["item"]])
            product = _products()[item_id]
        except ApiError:
            product = fallback_product(item_id)
    return product


def product_label(item_id: str) -> str:
    """Name plus item code, e.g. ``Bananas (FOODS_3_090)``."""
    return get_product(item_id).label


@st.cache_data(show_spinner=False, ttl=600)
def get_history(item_id: str) -> pd.DataFrame:
    """Cleaned daily history of an item (with stockout flags)."""
    return api().history(item_id)


@st.cache_data(show_spinner=False, ttl=600)
def get_future_actuals(item_id: str) -> pd.DataFrame | None:
    """Observed sales after the forecast origin (evaluation only)."""
    return api().actuals(item_id)


def results() -> dict[str, PipelineResult]:
    """Latest stored run per item, loaded from the API on first access."""
    if RESULTS_KEY not in st.session_state:
        st.session_state[RESULTS_KEY] = api().latest_runs()
    return st.session_state[RESULTS_KEY]


def what_if_results() -> dict[str, PipelineResult]:
    """What-if results by item (session only; also stored server-side)."""
    return st.session_state.setdefault(WHATIF_KEY, {})


def store_results(new_results: list[PipelineResult]) -> None:
    """Merge freshly fetched results into the session."""
    current = results()
    for r in new_results:
        current[r.item_id] = r
        what_if_results().pop(r.item_id, None)


def selected_item() -> str | None:
    """Currently selected item id."""
    return st.session_state.get(ITEM_KEY)


def selected_result() -> PipelineResult | None:
    """Pipeline result of the selected item, if available."""
    item = selected_item()
    return results().get(item) if item else None


def run_pipeline(item_ids: list[str]) -> None:
    """Queue a background job on the API and poll it to drive a progress bar."""
    client = api()
    job = client.submit_job(item_ids)
    bar = st.progress(0.0, text="Job queued ...")
    while job["status"] in ("queued", "running"):
        time.sleep(JOB_POLL_SECONDS)
        job = client.job(job["job_id"])
        current = job["item_ids"][min(job["done"], job["total"] - 1)]
        bar.progress(job["progress"], text=f"Running agents for {product_label(current)} "
                                           f"({min(job['done'] + 1, job['total'])}/"
                                           f"{job['total']}) ...")
    bar.progress(1.0, text="Done")
    store_results(list(client.latest_runs(job["item_ids"]).values()))
    bar.empty()
    for error in job["errors"]:
        st.sidebar.error(error)


def _pick_search_result() -> None:
    """Callback: add the picked search result to the product list and select it."""
    item = st.session_state.get(SEARCH_PICK_KEY)
    if item:
        extra = st.session_state.setdefault(EXTRA_ITEMS_KEY, [])
        if item not in extra:
            extra.append(item)
        st.session_state[ITEM_KEY] = item


def _render_search(total_items: int) -> None:
    query = st.text_input(f"Search all {total_items:,} items", placeholder="e.g. FOODS_3_2 "
                          "or milk", help="Any item of the store/category can be analysed.")
    if not query:
        return
    page = api().items(search=query, limit=25)
    _remember(page["items"])
    ids = [r["item_id"] for r in page["items"]]
    if not ids:
        st.caption("No matching items.")
        return
    st.selectbox(f"{page['total']} match(es)", [None, *ids], key=SEARCH_PICK_KEY,
                 format_func=lambda i: "Choose an item ..." if i is None else product_label(i),
                 on_change=_pick_search_result)


def _api_unavailable(exc: ApiError) -> None:
    with st.sidebar:
        st.error("Backend API unavailable", icon=":material/cloud_off:")
    st.error(f"**The StockPilot API is not reachable.** {exc.detail}")
    st.markdown("Start the API in another terminal, then reload this page:")
    st.code("uvicorn src.api.main:app --port 8000", language="bash")
    st.caption("Or run both together with `scripts/dev.sh`. Set `STOCKPILOT_API_URL` if the "
               "API runs elsewhere.")
    st.stop()


def render_sidebar() -> None:
    """Sidebar with item picker, search, run controls and system status."""
    try:
        info = system_info()
        demo = demo_items()
    except ApiError as exc:
        _api_unavailable(exc)
        return
    items = demo + [i for i in st.session_state.get(EXTRA_ITEMS_KEY, []) if i not in demo]
    with st.sidebar:
        st.markdown("### Controls")
        if ITEM_KEY not in st.session_state or st.session_state[ITEM_KEY] not in items:
            st.session_state[ITEM_KEY] = items[0]
        st.selectbox("Product", items, key=ITEM_KEY, format_func=product_label)
        _render_search(info["items"])
        c1, c2 = st.columns(2)
        if c1.button("Run item", help="Run all four agents for the selected item",
                     type="primary"):
            run_pipeline([st.session_state[ITEM_KEY]])
            st.rerun()
        if c2.button("Run all", help=f"Run the pipeline for the {len(demo)} demo items "
                                     "as a background job (~30s)"):
            run_pipeline(demo)
            st.rerun()

        st.divider()
        st.caption("**Data**")
        st.caption(f"Source: `{info['data_source']}` - {info['items']:,} items "
                   f"({len(demo)} demo), {info['history_start']} to {info['history_end']}")
        st.caption(f"Forecast origin: {info['forecast_origin']} | horizon "
                   f"{info['horizon_days']} days")
        st.caption("**Backend**")
        st.caption(f"API: `{api().base_url}` | LLM: `{info['llm']}`")
        st.badge("Inventory inputs are simulated", color="orange", icon=":material/science:")
        st.caption("Product names are illustrative: M5 item codes are anonymised, so each "
                   "demo code is labelled with a food that fits its sales pattern.")
        n = len(results())
        st.caption(f"{n} item result(s) loaded" if n else "No results yet - click **Run all**.")


def require_result() -> PipelineResult | None:
    """Return the selected result or render a hint when missing."""
    result = selected_result()
    if result is None:
        st.info("No agent results for this item yet. Use **Run item** or **Run all** in the "
                "sidebar (or `POST /jobs` on the API).")
        return None
    product = get_product(result.item_id)
    st.subheader(product.name)
    st.caption(f"{product.department} | sold per {product.unit} | item code "
               f"`{product.item_id}`" + (f" | {product.profile_note}"
                                          if product.profile_note else ""))
    return result
