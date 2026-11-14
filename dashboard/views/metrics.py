"""Performance metrics page: forecast accuracy and evaluation results."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.components.charts import metrics_comparison_figure
from dashboard.components.state import get_product, get_settings, results
from src.utils.io import read_json

st.title("Performance Metrics")


def with_product(df: pd.DataFrame) -> pd.DataFrame:
    """Insert a product-name column next to ``item_id``."""
    if "item_id" in df.columns:
        df.insert(int(df.columns.get_loc("item_id")), "product",
                  df["item_id"].map(lambda i: get_product(i).name))
    return df


all_results = results()
if not all_results:
    st.info("Run the pipeline first (sidebar -> Run all).")
    st.stop()

rows = []
for r in all_results.values():
    for model, m in r.forecast.metrics_by_model.items():
        rows.append({"item_id": get_product(r.item_id).name, "model": model,
                     "selected": model == r.forecast.model_used, **m.model_dump()})
frame = pd.DataFrame(rows)

st.subheader("Forecast accuracy (rolling-origin backtest, 4 x 7 days)")
summary = frame.groupby("model")[["mae", "rmse", "mape", "wape"]].mean().sort_values("mae")
selected = frame[frame["selected"]][["mae", "rmse", "mape", "wape"]].mean().rename("agent-selected")
summary = pd.concat([summary, selected.to_frame().T])
c1, c2 = st.columns([2, 3])
with c1:
    st.dataframe(summary.style.format({"mae": "{:.2f}", "rmse": "{:.2f}", "mape": "{:.1%}",
                                       "wape": "{:.1%}"}, na_rep="-")
                 .highlight_min(subset=["mae", "rmse", "wape"], color="#CDEBD3"))
    counts = pd.Series([r.forecast.model_used for r in all_results.values()]).value_counts()
    st.markdown("**Models selected by the Forecast Agent**")
    st.bar_chart(counts, horizontal=True, height=160)
with c2:
    metric = st.segmented_control("Metric", ["mae", "rmse", "wape", "mape"], default="wape")
    st.plotly_chart(metrics_comparison_figure(frame, metric or "wape"))

eval_dir = get_settings().outputs.evaluation_dir
summary_path = eval_dir / "evaluation_summary.json"
st.divider()
st.subheader("Evaluation results")
if not summary_path.exists():
    st.info("Run `python scripts/run_evaluation.py` to generate the hold-out, inventory "
            "simulation and scenario evaluations.")
    st.stop()

ev = read_json(summary_path)
st.caption(f"Generated {ev.get('generated_at', '')[:19]} on data source `{ev.get('data_source')}`.")

holdout = ev.get("holdout_accuracy")
if holdout:
    st.markdown("**Out-of-sample accuracy on the 7 days after the forecast origin (M5 actuals)**")
    st.dataframe(pd.DataFrame(holdout["by_model"]).T.style.format(
        {"mae": "{:.2f}", "rmse": "{:.2f}", "mape": "{:.1%}", "wape": "{:.1%}"}, na_rep="-"))

sim = ev.get("inventory_simulation")
if sim:
    st.markdown(f"**Inventory policy simulation** - replay of the last {sim['days']} days with "
                "actual demand (agentic SS/ROP policy vs naive fixed-reorder policy)")
    totals = pd.DataFrame(sim["totals"]).T
    k = st.columns(4)
    a, n = totals.loc["agentic"], totals.loc["naive"]
    k[0].metric("Stockout days", int(a["stockout_days"]), delta=int(a["stockout_days"] -
                                                                    n["stockout_days"]),
                delta_color="inverse")
    k[1].metric("Fill rate", f"{a['fill_rate']:.1%}",
                delta=f"{a['fill_rate'] - n['fill_rate']:+.1%}")
    k[2].metric("Avg inventory (units)", f"{a['avg_inventory']:.0f}",
                delta=f"{a['avg_inventory'] - n['avg_inventory']:+.0f}", delta_color="inverse")
    k[3].metric("Lost sales (units)", f"{a['lost_units']:.0f}",
                delta=f"{a['lost_units'] - n['lost_units']:+.0f}", delta_color="inverse")
    st.dataframe(with_product(pd.DataFrame(sim["by_item"])), hide_index=True)

scen = ev.get("scenario_evaluation")
if scen:
    st.markdown(f"**Critic scenario evaluation** - {scen['n_scenarios']} injected scenarios")
    k = st.columns(5)
    without = scen.get("recall_without_critic")
    k[0].metric("Detection recall", f"{scen['recall']:.0%}",
                delta=f"{scen['recall'] - without:+.0%} vs Risk Agent alone"
                if without is not None else None,
                help="Share of expected issues surfaced by the multi-agent system.")
    k[1].metric("Detection precision", f"{scen['precision']:.0%}",
                help="Flags that are neither expected nor an accepted side effect count as "
                     f"false positives. Strict precision: {scen.get('strict_precision', 0):.0%}.")
    k[2].metric("Verdict match", f"{scen['verdict_accuracy']:.0%}")
    k[3].metric("Grounding pass rate", f"{scen['grounding_pass_rate']:.0%}")
    k[4].metric("Scenarios", scen["n_scenarios"])
    st.dataframe(with_product(pd.DataFrame(scen["scenarios"])), hide_index=True)
