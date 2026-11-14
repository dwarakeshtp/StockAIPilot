# Presentation Outline (10-12 slides)

1. **Title.** StockPilot AI: Agentic Demand Forecasting & Reorder Recommendations (team,
   course).
2. **Problem.** Weekly replenishment decisions across hundreds of SKUs. Spreadsheet rules
   ignore uncertainty, and ML forecasts are opaque. Cost of stockouts vs overstock.
3. **Objective and scope.** POC: one store (CA_1), FOODS, 12 SKUs, 7-day horizon,
   rule-based SS/ROP/ROQ, human approval.
4. **Data.** M5 Walmart dataset; item profile table; censored-demand handling; simulated
   inventory (be explicit). Source: `docs/dataset.md`.
5. **Architecture.** Layered diagram from `docs/architecture.md` section 1.
6. **Agent workflow.** LangGraph diagram with the Critic revision loop and human decision
   (section 2).
7. **The four agents.** One row each: input, output, technique (table from
   `agent_design.md`).
8. **Trust by design.** Grounding check, template fallback, 13 Critic rules, verdicts,
   trace. Show a trace screenshot.
9. **Live demo.** Follow `docs/demo_script.md`: Overview, Forecast, Inventory what-if,
   Critic decision, Trace.
10. **Results.** Hold-out WAPE 35.8% vs 42.5% for naive; stockout days 8 vs 51; fill rate
    99.2% vs 90.5% with 12% less stock; Critic recall 100% vs 18% without it.
11. **Lessons and limitations.** Exploding ARIMA caught by sanity checks; sales ≠ demand;
    simulated inventory; free LLM rate limits.
12. **Future work and Q&A.** Global ML model, multi-store, weather and promotions, ERP feed,
    learning from planner decisions.

Screenshots to capture: Overview KPIs, forecast chart with interval, inventory projection
plus what-if comparison, Critic verdict card, trace timeline, metrics page.
