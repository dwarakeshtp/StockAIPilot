"""Render ``evaluation_summary.json`` as a markdown report."""

from __future__ import annotations

import pandas as pd


def _table(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    """Markdown table without the optional ``tabulate`` dependency."""

    def cell(v: object) -> str:
        if isinstance(v, float):
            return format(v, floatfmt)
        return str(v)

    header = [df.index.name or "", *map(str, df.columns)]
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for row in df.itertuples(name=None):
        out.append("| " + " | ".join([str(row[0]), *(cell(v) for v in row[1:])]) + " |")
    return "\n".join(out)


def render_report(summary: dict) -> str:
    """Build the evaluation report markdown from an evaluation summary dict."""
    lines = [
        "# Evaluation Report",
        "",
        f"_Generated {str(summary.get('generated_at', ''))[:19]} UTC from data source "
        f"`{summary.get('data_source')}` ({summary.get('n_items')} items). "
        "Regenerate with `python scripts/run_evaluation.py`._",
        "",
    ]

    holdout = summary.get("holdout_accuracy")
    if holdout:
        df = pd.DataFrame(holdout["by_model"]).T[["mae", "rmse", "mape", "wape", "n"]]
        df = df.sort_values("mae").astype({"n": int})
        best = df.index[0]
        lines += [
            "## 1. Forecast accuracy on unseen data",
            "",
            "Each model's 7-day forecast is compared with the real M5 sales for the 7 days "
            "after the forecast origin (never seen during training or model selection). "
            "MAPE is computed on non-zero days only; WAPE = sum|error| / sum(actual).",
            "",
            _table(df),
            "",
            f"Lowest MAE: **{best}**. `agent_selected` is the model the Forecast Agent chose "
            "per item from the rolling-origin backtest (after any Critic revision).",
            "",
        ]
        if holdout.get("by_item"):
            items = pd.DataFrame(holdout["by_item"]).set_index("item_id")
            lines += ["Per item (agent-selected model):", "",
                      _table(items[["actual_7d", "forecast_7d", "mae", "wape"]], ".2f"), ""]

    sim = summary.get("inventory_simulation")
    if sim:
        totals = pd.DataFrame(sim["totals"]).T
        a, n = totals.loc["agentic"], totals.loc["naive"]
        lines += [
            "## 2. Inventory policy simulation",
            "",
            f"Replay of the last {sim['days']} days of actual demand, all items, same starting "
            "stock and lead time for both policies. Unmet demand is lost.",
            "",
            "* **Agentic** - weekly refit of the selected model; SS / ROP / order-up-to from the "
            "Inventory Planning Agent's formulas with the Risk Agent's uncertainty multiplier.",
            "* **Naive** - reorder 14 days of average sales whenever stock drops below 7 days of "
            "average sales (fixed at the start of the window).",
            "",
            _table(totals, ".3f"),
            "",
            f"The agentic policy had **{int(a['stockout_days'])} stockout days vs "
            f"{int(n['stockout_days'])}**, fill rate **{a['fill_rate']:.1%} vs "
            f"{n['fill_rate']:.1%}**, and {a['avg_inventory'] - n['avg_inventory']:+.0f} units of "
            "average inventory (summed over items) relative to the naive rule.",
            "",
            _table(pd.DataFrame(sim["by_item"]).set_index("item_id"), ".3f"),
            "",
        ]

    scen = summary.get("scenario_evaluation")
    if scen:
        rows = pd.DataFrame(scen["scenarios"]).set_index("scenario")
        cols = ["item_id", "expected_flags", "detected_flags", "unexpected", "action", "verdict",
                "verdict_ok"]
        lines += [
            "## 3. Critic scenario evaluation",
            "",
            f"{scen['n_scenarios']} scenarios inject demand anomalies, data-quality problems and "
            "supply issues into real items and run the full LangGraph pipeline.",
            "",
            "| Metric | Value |",
            "|---|---|",
            f"| Detection recall | {scen['recall']:.0%} |",
            f"| Detection precision (accepted side effects excluded) | {scen['precision']:.0%} |",
            f"| Strict precision | {scen.get('strict_precision', 0):.0%} |",
            f"| Verdict accuracy | {scen['verdict_accuracy']:.0%} |",
            f"| Grounding pass rate | {scen['grounding_pass_rate']:.0%} |",
            f"| Recall with Risk Agent signals only (no Critic) | "
            f"{scen.get('recall_without_critic', 0):.0%} |",
            "",
            _table(rows[[c for c in cols if c in rows.columns]]),
            "",
        ]

    lines += [
        "## Limitations",
        "",
        "* Inventory positions, lead times, MOQs and costs are simulated (M5 has no inventory "
        "data); every recommendation is labelled accordingly.",
        "* M5 records sales, not demand - suspected stockout streaks are excluded from training "
        "but the simulation still replays observed sales as demand.",
        "* One store, one category, ~12 SKUs and a single forecast origin for the hold-out "
        "check - results are indicative, not statistically conclusive.",
        "",
    ]
    return "\n".join(lines)
