# Demo Script (about 8 minutes)

**Setup:** run `streamlit run dashboard/app.py`. Optionally add an OpenRouter or Cohere key
to `.env` beforehand so the explanations come from the LLM; otherwise templates are used.

## 1. Problem and data (1 min) - Overview page

* "A store planner must decide every week what to reorder for hundreds of SKUs. Forecasts
  alone are not enough: they need risk, policy and a sanity check."
* Point to the sidebar: M5 Walmart data, store CA_1, FOODS, 12 SKUs. Inventory inputs are
  **simulated**, which is labelled.
* Click **Run all**. Four agents run for 12 items in about 25 s.
* Walk the KPI row (orders proposed, expedites, items needing human review, order value),
  the workflow diagram and the recommendations table.

## 2. Forecast Agent (1.5 min) - Forecast page, item `FOODS_3_064`

* The chart shows history, suspected stockout spans, events, the 7-day forecast with its
  80% band, and overlays for ARIMA, Prophet and naive.
* Toggle **Show actual hold-out sales**: overlay the real next-week sales on the forecast
  band.
* The backtest chart and metrics table (MAE / RMSE / MAPE / WAPE per model) show why ARIMA
  was selected and its skill versus naive.

## 3. Risk Agent (1 min) - Risk page

* The confidence gauge and its four components (accuracy, interval precision, model
  agreement, data quality).
* Switch to `FOODS_3_519` (intermittent): low confidence, HIGH risk, anomalies flagged.

## 4. Inventory Agent and what-if (1.5 min) - Inventory page, `FOODS_3_120`

* EXPEDITE: the projected stock line crosses zero before the lead time ends; SS, ROP and
  order quantity are shown with their formulas.
* What-if: set lead time to 12 days and service level to 0.98, then click **Run what-if**.
  The side-by-side table shows safety stock and order quantity rising, and the Critic
  re-validates the result.

## 5. Critic Agent and human decision (2 min) - Critic page

* `FOODS_3_090`: **NEEDS_HUMAN_REVIEW**. The open order arrives after the projected stockout,
  so the Critic escalates even though the Inventory Agent said HOLD. Read the explanation
  aloud.
* `FOODS_3_519`: the Critic asked the Forecast Agent to **revise**. Backtest WAPE was 63% and
  ARIMA and Prophet disagreed by 70%, so it requested the ensemble. After one revision the
  issues remain (the order relies on a low-confidence forecast), so it escalates to a human.
* Record a decision: **Modify**, quantity 240, reason "Expedite open order instead". It
  appears in the decision log. Download the PO draft CSV.

## 6. Agent trace (30 s) - Agent Trace page

* The timeline of nodes, including the forecast → critic → forecast loop for
  `FOODS_3_519`. Expand a step to show its inputs, outputs, reasoning and the LLM provider.
  Download the JSON.

## 7. Evaluation (1 min) - Performance Metrics page

* Hold-out: agent-selected WAPE 35.8% vs 42.5% for naive.
* Inventory simulation: 8 vs 51 stockout days, fill rate 99.2% vs 90.5%, with less
  inventory.
* Critic scenarios: 11 injected issues, 100% recall and 100% verdict accuracy; the Risk
  Agent alone would catch 18%.

**Closing line:** "Numbers come from code, words from the LLM, and nothing reaches the
planner unchecked."
