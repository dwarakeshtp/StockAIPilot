# Evaluation Report

_Generated 2026-11-14T09:37:40 UTC from data source `m5_processed` (12 items). Regenerate with `python scripts/run_evaluation.py`._

## 1. Forecast accuracy on unseen data

Each model's 7-day forecast is compared with the real M5 sales for the 7 days after the forecast origin (never seen during training or model selection). MAPE is computed on non-zero days only; WAPE = sum|error| / sum(actual).

|  | mae | rmse | mape | wape | n |
|---|---|---|---|---|---|
| agent_selected | 8.558 | 14.558 | 1.090 | 0.358 | 84 |
| ensemble | 8.618 | 13.956 | 1.060 | 0.360 | 84 |
| prophet | 8.750 | 14.080 | 1.004 | 0.366 | 84 |
| arima | 9.073 | 14.869 | 1.140 | 0.379 | 84 |
| seasonal_naive | 10.167 | 18.285 | 0.939 | 0.425 | 84 |

Lowest MAE: **agent_selected**. `agent_selected` is the model the Forecast Agent chose per item from the rolling-origin backtest (after any Critic revision).

Per item (agent-selected model):

| item_id | actual_7d | forecast_7d | mae | wape |
|---|---|---|---|---|
| FOODS_3_064 | 195.00 | 192.60 | 4.88 | 0.17 |
| FOODS_3_090 | 387.00 | 328.70 | 20.54 | 0.37 |
| FOODS_3_120 | 136.00 | 345.60 | 31.27 | 1.61 |
| FOODS_3_148 | 28.00 | 39.40 | 2.56 | 0.64 |
| FOODS_3_252 | 301.00 | 269.30 | 8.64 | 0.20 |
| FOODS_3_282 | 135.00 | 147.30 | 2.24 | 0.12 |
| FOODS_3_351 | 55.00 | 69.30 | 4.33 | 0.55 |
| FOODS_3_455 | 29.00 | 59.70 | 4.38 | 1.06 |
| FOODS_3_519 | 118.00 | 141.60 | 7.08 | 0.42 |
| FOODS_3_586 | 238.00 | 271.10 | 7.21 | 0.21 |
| FOODS_3_587 | 238.00 | 221.90 | 3.49 | 0.10 |
| FOODS_3_714 | 149.00 | 127.00 | 6.07 | 0.29 |

## 2. Inventory policy simulation

Replay of the last 56 days of actual demand, all items, same starting stock and lead time for both policies. Unmet demand is lost.

* **Agentic** - weekly refit of the selected model; SS / ROP / order-up-to from the Inventory Planning Agent's formulas with the Risk Agent's uncertainty multiplier.
* **Naive** - reorder 14 days of average sales whenever stock drops below 7 days of average sales (fixed at the start of the window).

|  | stockout_days | lost_units | fill_rate | avg_inventory | n_orders |
|---|---|---|---|---|---|
| agentic | 8.000 | 130.200 | 0.992 | 2086.100 | 83.000 |
| naive | 51.000 | 1590.800 | 0.905 | 2379.800 | 48.000 |

The agentic policy had **8 stockout days vs 51**, fill rate **99.2% vs 90.5%**, and -294 units of average inventory (summed over items) relative to the naive rule.

| item_id | model | stockout_days_agentic | stockout_days_naive | fill_rate_agentic | fill_rate_naive | avg_inv_agentic | avg_inv_naive | orders_agentic | orders_naive |
|---|---|---|---|---|---|---|---|---|---|
| FOODS_3_064 | arima | 0 | 0 | 1.000 | 1.000 | 136.300 | 279.400 | 7 | 5 |
| FOODS_3_090 | prophet | 1 | 7 | 0.992 | 0.923 | 358.400 | 361.500 | 8 | 4 |
| FOODS_3_120 | arima | 2 | 16 | 0.976 | 0.647 | 438.000 | 124.800 | 6 | 4 |
| FOODS_3_148 | arima | 0 | 0 | 1.000 | 1.000 | 54.100 | 79.000 | 5 | 4 |
| FOODS_3_252 | arima | 0 | 0 | 1.000 | 1.000 | 204.200 | 279.100 | 7 | 5 |
| FOODS_3_282 | arima | 2 | 8 | 0.978 | 0.907 | 155.000 | 129.800 | 6 | 4 |
| FOODS_3_351 | arima | 0 | 0 | 1.000 | 1.000 | 65.800 | 122.400 | 5 | 4 |
| FOODS_3_455 | arima | 1 | 0 | 0.997 | 1.000 | 51.700 | 107.200 | 7 | 4 |
| FOODS_3_519 | ensemble | 1 | 12 | 0.979 | 0.597 | 84.000 | 40.100 | 12 | 4 |
| FOODS_3_586 | arima | 0 | 0 | 1.000 | 1.000 | 204.400 | 358.300 | 7 | 4 |
| FOODS_3_587 | arima | 1 | 0 | 0.995 | 1.000 | 157.600 | 378.400 | 6 | 3 |
| FOODS_3_714 | arima | 0 | 8 | 1.000 | 0.847 | 176.600 | 119.800 | 7 | 3 |

## 3. Critic scenario evaluation

11 scenarios inject demand anomalies, data-quality problems and supply issues into real items and run the full LangGraph pipeline.

| Metric | Value |
|---|---|
| Detection recall | 100% |
| Detection precision (accepted side effects excluded) | 100% |
| Strict precision | 69% |
| Verdict accuracy | 100% |
| Grounding pass rate | 100% |
| Recall with Risk Agent signals only (no Critic) | 18% |

| scenario | item_id | expected_flags | detected_flags | unexpected | action | verdict | verdict_ok |
|---|---|---|---|---|---|---|---|
| clean_baseline | FOODS_3_252 | - | - | - | HOLD | APPROVED | True |
| demand_spike | FOODS_3_586 | risk:spike | risk:spike | - | HOLD | APPROVED | True |
| demand_drop | FOODS_3_064 | risk:drop | risk:drop | - | HOLD | APPROVED | True |
| data_gaps | FOODS_3_714 | data_quality | data_quality | - | HOLD | APPROVED_WITH_WARNINGS | True |
| noisy_history | FOODS_3_252 | forecast_accuracy | forecast_accuracy | - | HOLD | APPROVED_WITH_WARNINGS | True |
| intermittent_item | FOODS_3_519 | forecast_accuracy, model_disagreement, revision | forecast_accuracy, high_stockout_probability, model_disagreement, revision, risk:spike | - | EXPEDITE | NEEDS_HUMAN_REVIEW | True |
| long_lead_time | FOODS_3_282 | high_stockout_probability | excessive_order, high_stockout_probability | - | EXPEDITE | APPROVED_WITH_WARNINGS | True |
| zero_stock | FOODS_3_587 | high_stockout_probability | high_stockout_probability, risk:spike | - | EXPEDITE | APPROVED_WITH_WARNINGS | True |
| huge_moq | FOODS_3_455 | excessive_order | excessive_order, high_stockout_probability | - | EXPEDITE | APPROVED_WITH_WARNINGS | True |
| late_open_order | FOODS_3_120 | stockout_before_open_order | stockout_before_open_order | - | HOLD | NEEDS_HUMAN_REVIEW | True |
| overstocked | FOODS_3_090 | - | - | - | HOLD | APPROVED | True |

## Limitations

* Inventory positions, lead times, MOQs and costs are simulated (M5 has no inventory data); every recommendation is labelled accordingly.
* M5 records sales, not demand - suspected stockout streaks are excluded from training but the simulation still replays observed sales as demand.
* One store, one category, ~12 SKUs and a single forecast origin for the hold-out check - results are indicative, not statistically conclusive.
