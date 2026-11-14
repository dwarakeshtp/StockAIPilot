"""Inventory Planning Agent page with a What-if panel."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.components.charts import inventory_projection_figure
from dashboard.components.formatting import action_badge, verdict_badge
from dashboard.components.state import get_runner, get_settings, require_result, what_if_results

st.title("3. Inventory Planning Agent")
st.caption("Rule-based policy: safety stock, reorder point, order-up-to reorder quantity "
           "(rounded to pack size / MOQ) and stockout probability over the lead time.")

result = require_result()
if result is None:
    st.stop()
inv = result.inventory

head = st.columns([1, 5])
with head[0]:
    action_badge(inv.action)
with head[1]:
    st.markdown(f"**{result.critic.recommendation.headline}**")

c = st.columns(6)
c[0].metric("Safety stock", f"{inv.safety_stock:,.0f}")
c[1].metric("Reorder point", f"{inv.reorder_point:,.0f}")
c[2].metric("Inventory position", f"{inv.inventory_position:,.0f}")
c[3].metric("Reorder qty", f"{inv.reorder_qty:,}")
c[4].metric("Stockout prob.", f"{inv.stockout_probability:.0%}")
c[5].metric("Days of cover", f"{inv.days_of_cover:.1f}")

left, right = st.columns([3, 2])
with left:
    st.plotly_chart(inventory_projection_figure(result))
with right:
    st.markdown("**Inputs** :orange-badge[SIMULATED]")
    p = inv.params
    st.dataframe(pd.DataFrame({
        "parameter": ["On hand", "On order", "Open order ETA (day)", "Lead time (days)",
                      "Lead time std", "MOQ", "Pack size", "Unit cost", "Service level",
                      "Review period (days)"],
        "value": [p.on_hand, p.on_order, p.on_order_eta_days, p.lead_time_days, p.lead_time_std,
                  p.moq, p.pack_size, p.unit_cost, inv.service_level, inv.review_period_days],
    }).astype({"value": str}), hide_index=True)
    st.caption(f"Mean daily demand {inv.mean_daily_demand:.1f}; demand std {inv.demand_std:.2f} "
               f"x risk multiplier = {inv.adjusted_demand_std:.2f}; EOQ (reference) "
               f"{inv.eoq:.0f}.")

with st.expander("Agent reasoning", expanded=False):
    for step in inv.reasoning:
        st.markdown(f"- {step}")
    if inv.narrative:
        st.info(inv.narrative, icon=":material/smart_toy:")

st.divider()
st.subheader("What-if analysis")
st.caption("Re-runs the Inventory Planning and Critic agents (LangGraph what-if graph) with "
           "new assumptions; the forecast and risk assessment are reused.")
settings = get_settings()
with st.form("what_if"):
    f = st.columns(4)
    lead_time = f[0].slider("Lead time (days)", 1, 21, int(round(p.lead_time_days)))
    service = f[1].slider("Service level", 0.80, 0.99, float(settings.inventory.service_level),
                          step=0.01)
    demand_mult = f[2].slider("Demand multiplier", 0.5, 3.0, 1.0, step=0.1,
                              help="e.g. 1.5 = a promotion lifting demand by 50%")
    on_hand = f[3].number_input("On hand", min_value=0, value=int(p.on_hand), step=10)
    submitted = st.form_submit_button("Run what-if", type="primary")

if submitted:
    overrides = {"lead_time_days": float(lead_time), "service_level": float(service),
                 "demand_multiplier": float(demand_mult), "on_hand": float(on_hand)}
    with st.spinner("Re-planning ..."):
        what_if_results()[result.item_id] = get_runner().what_if(result, overrides)

scenario = what_if_results().get(result.item_id)
if scenario is not None:
    s_inv = scenario.inventory
    comparison = pd.DataFrame({
        "metric": ["Action", "Reorder qty", "Safety stock", "Reorder point", "Stockout prob.",
                   "Days of cover", "Critic verdict"],
        "baseline": [inv.action.value, inv.reorder_qty, round(inv.safety_stock, 1),
                     round(inv.reorder_point, 1), f"{inv.stockout_probability:.0%}",
                     round(inv.days_of_cover, 1), result.critic.verdict.value],
        "what-if": [s_inv.action.value, s_inv.reorder_qty, round(s_inv.safety_stock, 1),
                    round(s_inv.reorder_point, 1), f"{s_inv.stockout_probability:.0%}",
                    round(s_inv.days_of_cover, 1), scenario.critic.verdict.value],
    }).astype(str)
    w1, w2 = st.columns([2, 3])
    with w1:
        st.markdown(f"**Scenario:** `{scenario.overrides}`")
        st.dataframe(comparison, hide_index=True)
        verdict_badge(scenario.critic.verdict)
    with w2:
        st.plotly_chart(inventory_projection_figure(scenario), key="what_if_chart")
    st.info(scenario.critic.explanation, icon=":material/gavel:")
