"""Critic Agent page: validation, explanation and human approval."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.components.formatting import SEVERITY_ICONS, action_badge, risk_badge, verdict_badge
from dashboard.components.state import (
    get_decision_store,
    get_product,
    product_label,
    require_result,
    results,
)
from src.orchestration.decisions import DecisionType

st.title("4. Critic Agent & Human Approval")
st.caption("Validates all agent outputs with 13 rules, can send the Forecast Agent back for one "
           "revision, writes a grounded business explanation and hands the decision to a "
           "planner.")

result = require_result()
if result is None:
    st.stop()
critic = result.critic
rec = critic.recommendation

with st.container(border=True):
    b = st.columns([1, 1, 1, 4])
    with b[0]:
        verdict_badge(critic.verdict)
    with b[1]:
        action_badge(rec.action)
    with b[2]:
        risk_badge(rec.risk_level)
    st.subheader(rec.headline)
    st.write(rec.rationale)
    source = critic.explanation_source
    st.markdown(f"**Explanation** <small>(source: `{source}`)</small>", unsafe_allow_html=True)
    st.info(critic.explanation, icon=":material/smart_toy:" if source.startswith("llm")
            else ":material/description:")
    if not critic.grounding_passed:
        st.warning(f"LLM explanation rejected by the grounding check (unsupported numbers: "
                   f"{', '.join(critic.ungrounded_numbers[:5])}); template used instead.")
    if result.revision_count:
        st.success(f"The Critic requested {result.revision_count} forecast revision(s); final "
                   f"model: **{result.forecast.model_used}**.", icon=":material/autorenew:")

st.subheader("Validation findings")
if critic.issues:
    for issue in critic.issues:
        st.markdown(f"{SEVERITY_ICONS[issue.severity]} **{issue.severity}** `{issue.rule}` "
                    f"({issue.agent}) - {issue.message}"
                    + (f" -> *{issue.revision_hint.value}*" if issue.revision_hint else ""))
else:
    st.caption("No issues found.")

with st.expander("Critic reasoning"):
    for step in critic.reasoning:
        st.markdown(f"- {step}")

st.divider()
st.subheader("Planner decision")
store = get_decision_store()
with st.form("decision"):
    choice = st.radio("Decision", [d.value for d in DecisionType], horizontal=True)
    qty = st.number_input("Final quantity (used when MODIFIED)", min_value=0,
                          value=int(rec.quantity), step=max(result.inventory.params.pack_size, 1))
    reason = st.text_area("Reason (required)", placeholder="e.g. Supplier confirmed lead time")
    reviewer = st.text_input("Reviewer", value="planner")
    if st.form_submit_button("Submit decision", type="primary"):
        try:
            decision = store.record(result, DecisionType(choice), reason.strip(),
                                    final_qty=int(qty), reviewer=reviewer or "planner")
            st.success(f"Recorded {decision.decision.value} for {product_label(decision.item_id)} "
                       f"(final qty {decision.final_qty}).")
        except ValueError as exc:
            st.error(f"Decision not recorded: {exc}")

history = store.load()
if history:
    df = pd.DataFrame([d.model_dump(mode="json") for d in history])
    df.insert(0, "product", df["item_id"].map(lambda i: get_product(i).name))
    st.markdown("**Decision log**")
    columns = ["decided_at", "product", "item_id", "recommended_action", "recommended_qty",
               "verdict", "decision", "final_qty", "reason", "reviewer"]
    st.dataframe(df[columns].iloc[::-1], hide_index=True)

    latest = store.latest_by_item()
    po = [
        {"item_id": d.item_id, "product": get_product(d.item_id).name,
         "department": get_product(d.item_id).department, "quantity": d.final_qty,
         "unit_cost": results()[d.item_id].inventory.params.unit_cost
         if d.item_id in results() else None,
         "decision": d.decision.value, "reviewer": d.reviewer, "reason": d.reason}
        for d in latest.values()
        if d.decision != DecisionType.REJECTED and d.final_qty > 0
    ]
    if po:
        st.download_button("Download approved purchase-order draft (CSV)",
                           pd.DataFrame(po).to_csv(index=False), file_name="po_draft.csv",
                           mime="text/csv")
