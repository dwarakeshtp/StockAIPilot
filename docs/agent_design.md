# Agent Design

All agents extend `BaseAgent` (`src/agents/base_agent.py`). It provides `run(...)`, which
returns a pydantic model, and `narrate(task, facts)`. `narrate` asks the configured LLM
for one or two sentences and returns them only if they pass the grounding check. Thresholds
live in `config/settings.yaml`.

## 1. Forecast Agent - `src/agents/forecast_agent.py`

**Input:** engineered features (history), the future frame (calendar, prices, SNAP, events),
the holiday table and an optional `RevisionRequest`.
**Output:** `ForecastOutput`.

1. **Candidate models.** `SeasonalNaiveForecaster` (baseline), `ARIMAForecaster` (SARIMAX
   with weekly seasonality, AIC order selection over sane fits) and `ProphetForecaster`
   (weekly and yearly seasonality, M5 events as holidays, SNAP and price ratio as
   regressors).
2. **Rolling-origin backtest.** Each model is evaluated over 4 folds of 7 days, giving MAE,
   RMSE, MAPE (non-zero days only) and WAPE. The ensemble (mean of ARIMA and Prophet) is
   scored from the same folds.
3. **Plausibility filter.** Models whose forecasts are non-finite or more than 10× the
   history maximum are excluded.
4. **Selection.** The lowest MAE among ARIMA and Prophet wins. On revision, `USE_ENSEMBLE`
   switches to the ensemble and `USE_ALTERNATIVE_MODEL` switches to the other model.
5. **Outputs.** A 7-day forecast with an 80% interval, metrics for every model, skill
   versus naive, ARIMA/Prophet agreement on the 7-day total, and backtest points for
   charts.

Design decision: global ML models such as LightGBM were deliberately left out. The brief
specifies ARIMA and Prophet, and per-item statistical models are explainable to a planner.

## 2. Risk & Uncertainty Agent - `src/agents/risk_agent.py`

**Input:** features, `ForecastOutput` and `DataQualityReport`. **Output:** `RiskOutput`.

Confidence score (0-100):

\[
C = 100\,(0.4\,A + 0.3\,P + 0.2\,G + 0.1\,Q) - \min(15,\ 5 \cdot n_{\text{recent anomalies}})
\]

| Component | Definition |
|---|---|
| Accuracy \(A\) | \(1 - \min(1, \text{WAPE})\) of the selected model |
| Precision \(P\) | \(1 - \min(1, (\text{mean interval width} / \text{mean forecast}) / 2)\) |
| Agreement \(G\) | \(1 - \lvert \text{ARIMA}_{7d} - \text{Prophet}_{7d} \rvert / \max(\cdot)\) |
| Data quality \(Q\) | Quality score from cleaning (missing or imputed days, stockout streaks) |

* **Anomalies.** A rolling 28-day z-score above 3 or a rolling IQR check (3×) over the
  last 90 days. Suspected stockout days are skipped. Anomalies in the last 14 days
  count towards the penalty.
* **Level shift.** If the mean of the last 7 days is at least 1.5× (spike) or at most
  1/1.5 (drop) of the preceding 28-day mean, it is flagged.
* **Risk level.** Below 45 is HIGH, 70 or above is LOW, anything between is MEDIUM. A spike
  or drop escalates the level by one.
* **Uncertainty multiplier** for safety stock: LOW 1.0, MEDIUM 1.2, HIGH 1.4.

## 3. Inventory Planning Agent - `src/agents/inventory_agent.py`

**Input:** forecast, risk, the item's (simulated) inventory parameters and optional what-if
overrides. **Output:** `InventoryOutput`.

| Quantity | Formula |
|---|---|
| Mean demand \(\mu_d\) | Mean of the 7-day forecast (× demand multiplier override) |
| Demand std \(\sigma_d\) | Backtest RMSE × risk multiplier |
| Safety stock | \(SS = z\sqrt{L\sigma_d^2 + \mu_d^2\sigma_L^2}\) |
| Reorder point | \(ROP = \mu_d L + SS\) |
| Order quantity | \(\max(0,\ \mu_d (L+R) + SS - IP)\), rounded up to pack size, at least MOQ |
| Stockout probability | \(1 - \Phi\big((IP - \mu_d L)/\sigma_{DL}\big)\) |
| Projection | Daily on-hand over \(\max(7, L)\) days, including open-order arrival |

Here \(IP\) = on hand + on order, \(R\) = review period (7 days) and \(z\) comes from the
service level (default 0.95).

* **Action.** EXPEDITE if the projection runs out of stock within the lead time; REORDER if
  \(IP \le ROP\); otherwise HOLD.
* **Also reported:** days of cover and an EOQ reference.
* **Overrides.** Supported keys are `lead_time_days`, `lead_time_std`, `service_level`,
  `demand_multiplier` and `on_hand`. Unknown keys raise an error.

## 4. Critic Agent - `src/agents/critic_agent.py`, `validation_rules.py`

**Input:** all upstream outputs plus data quality. **Output:** `CriticOutput`.

| Rule id | Fires when | Severity | Effect |
|---|---|---|---|
| `forecast_sanity` | Forecast is non-finite, negative or more than 50× the history max | CRITICAL | Revise: alternative model |
| `interval_consistency` | lower ≤ yhat ≤ upper is violated | CRITICAL | Human review |
| `forecast_accuracy` | Backtest WAPE > 0.6 | WARNING | Revise: ensemble if it backtests better |
| `model_disagreement` | ARIMA and Prophet 7-day totals differ by more than 35% | WARNING | Revise: ensemble |
| `no_skill_vs_naive` | Selected model is no better than seasonal-naive | WARNING | - |
| `excessive_order` | Cover after the order arrives is more than 30 days | WARNING | - |
| `safety_stock_sanity` | SS > 1.5 × lead-time demand | WARNING | - |
| `risk_action_mismatch` | HIGH risk with HOLD while stockout probability > 0.2 | WARNING | Escalates |
| `stockout_before_open_order` | Projected stockout before the open order arrives | WARNING | Escalates |
| `low_confidence_large_order` | Confidence < 45 and order > 14 days of demand | WARNING | Escalates |
| `data_quality` | Quality score < 0.8 | WARNING | - |
| `high_stockout_probability` | Stockout probability > 0.5 | WARNING | - |
| `simulated_inputs` | Inventory parameters are simulated | INFO | - |

**Verdict:**

1. Any issue carrying a revision hint, while the revision budget (1) remains, gives
   **NEEDS_REVISION**. LangGraph loops back to the Forecast Agent with that hint.
2. Otherwise, any CRITICAL issue or escalating rule gives **NEEDS_HUMAN_REVIEW**.
3. Otherwise, any WARNING gives **APPROVED_WITH_WARNINGS**.
4. Otherwise the verdict is **APPROVED**.

**Explanation.** The Critic builds a facts dict (forecast, model, metrics, confidence, risk,
SS, ROP, quantity, stockout probability, issues) and asks the LLM for a short business
explanation. The grounding check (`src/agents/grounding.py`) extracts every number from the
text; each must match a fact within max(0.5, 2%), or a fact × 100 for percentages, after
dates and SKU codes are stripped. Text that fails falls back to a deterministic template.
`grounding_passed` and `explanation_source` are recorded.

## 5. Agent collaboration

* **Forward flow.** Forecast feeds Risk (accuracy, interval, agreement), Risk feeds
  Inventory (uncertainty multiplier), and all three feed the Critic.
* **Feedback loop.** The Critic sends a `RevisionRequest` back to the Forecast Agent, at
  most once, so every run terminates.
* **Human in the loop.** The Critic's verdict sets the planner's review priority. Every
  decision (Approve, Modify with quantity and reason, Reject with reason) is logged with
  the run id.
