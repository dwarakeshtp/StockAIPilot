"""Overview: portfolio KPIs, recommendations table and the agent workflow."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.components.state import get_decision_store, results

st.title("Agentic Demand Forecasting & Inventory Reorder Recommendations")
st.caption(
    "Four collaborating agents - Forecast, Risk & Uncertainty, Inventory Planning and Critic - "
    "orchestrated with LangGraph on the M5 (Walmart) dataset, store CA_1, FOODS category. "
    "Every recommendation requires human approval."
)

WORKFLOW_DOT = """
digraph {
  rankdir=LR; node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11];
  ingest [label="Data ingestion", fillcolor="#EEEEEE"];
  clean [label="Cleaning +\\nquality check", fillcolor="#EEEEEE"];
  feat [label="Feature\\nengineering", fillcolor="#EEEEEE"];
  fa [label="Forecast Agent\\nARIMA / Prophet", fillcolor="#D6E8F7"];
  ra [label="Risk & Uncertainty\\nAgent", fillcolor="#FCE6C3"];
  ia [label="Inventory Planning\\nAgent", fillcolor="#D5EFD9"];
  ca [label="Critic Agent\\nvalidate + explain", fillcolor="#E6DAF3"];
  hu [label="Planner\\napprove / modify / reject", shape=ellipse, fillcolor="#FFFFFF"];
  ingest -> clean -> feat -> fa -> ra -> ia -> ca -> hu;
  ca -> fa [label="revise (max 1)", style=dashed, color="#7B3FBF", fontsize=9];
}
"""

all_results = results()
if not all_results:
    st.info("No agent results yet. Click **Run all** in the sidebar to run the pipeline for "
            "every item (about 30 seconds).")
    st.graphviz_chart(WORKFLOW_DOT)
    st.stop()

decisions = get_decision_store().latest_by_item()
rows = []
for r in all_results.values():
    rec = r.critic.recommendation
    decision = decisions.get(r.item_id)
    rows.append({
        "Item": r.item_id,
        "Action": rec.action.value,
        "Qty": rec.quantity,
        "Order value": rec.order_value,
        "Risk": r.risk.risk_level.value,
        "Confidence": r.risk.confidence_score,
        "Stockout prob.": r.inventory.stockout_probability,
        "Days of cover": r.inventory.days_of_cover,
        "Model": r.forecast.model_used,
        "WAPE": r.forecast.selected_metrics.wape,
        "Verdict": r.critic.verdict.value,
        "Revisions": r.revision_count,
        "Decision": decision.decision.value if decision else "PENDING",
    })
table = pd.DataFrame(rows).sort_values(["Verdict", "Stockout prob."], ascending=[False, False])

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Items analysed", len(table))
c2.metric("Orders proposed", int((table["Action"] != "HOLD").sum()),
          help="REORDER or EXPEDITE recommendations")
c3.metric("Expedite", int((table["Action"] == "EXPEDITE").sum()))
c4.metric("Need human review", int((table["Verdict"] == "NEEDS_HUMAN_REVIEW").sum()))
c5.metric("Proposed order value", f"${table['Order value'].sum():,.0f}",
          help="Using simulated unit costs")

st.subheader("Recommendations")
st.dataframe(
    table,
    hide_index=True,
    column_config={
        "Order value": st.column_config.NumberColumn(format="$%.0f"),
        "Confidence": st.column_config.ProgressColumn(min_value=0, max_value=100,
                                                      format="%.0f"),
        "Stockout prob.": st.column_config.NumberColumn(format="percent"),
        "WAPE": st.column_config.NumberColumn(format="percent"),
        "Days of cover": st.column_config.NumberColumn(format="%.1f"),
    },
)

left, right = st.columns([3, 2])
with left:
    st.subheader("Agent workflow")
    st.graphviz_chart(WORKFLOW_DOT)
with right:
    st.subheader("Distribution")
    st.bar_chart(table["Risk"].value_counts(), horizontal=True, height=150)
    st.bar_chart(table["Verdict"].value_counts(), horizontal=True, height=180)

st.caption("Select an item in the sidebar and walk through the agent pages (1-4) to see how "
           "each agent contributed to its recommendation.")
