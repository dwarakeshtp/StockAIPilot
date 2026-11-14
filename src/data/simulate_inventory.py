"""Simulated inventory and supplier parameters.

M5 contains sales and prices only - no stock levels or lead times. To exercise
the inventory logic we generate *clearly labelled* simulated parameters with a
fixed seed. On-hand stock is drawn relative to each item's recent demand so that
the demo contains a mix of healthy, at-risk and over-stocked items.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.logging.logger import get_logger

logger = get_logger(__name__)

INVENTORY_PARAM_COLUMNS: list[str] = [
    "item_id",
    "on_hand",
    "on_order",
    "on_order_eta_days",
    "lead_time_days",
    "lead_time_std",
    "moq",
    "pack_size",
    "unit_cost",
    "holding_cost_rate",
    "is_simulated",
]


def simulate_inventory_params(sales: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Generate one row of inventory parameters per item.

    Args:
        sales: Long-format sales with ``item_id``, ``date``, ``units``, ``sell_price``.
        seed: Random seed.

    Returns:
        DataFrame following :data:`INVENTORY_PARAM_COLUMNS`.
    """
    rng = np.random.default_rng(seed)
    last_date = sales["date"].max()
    recent = sales[sales["date"] > last_date - pd.Timedelta(days=28)]
    rows = []
    cover_profiles = [0.4, 0.7, 1.0, 1.5, 2.5, 4.0]
    for idx, (item_id, grp) in enumerate(recent.groupby("item_id", sort=True)):
        mean_daily = max(float(grp["units"].mean()), 0.5)
        price = float(grp["sell_price"].dropna().iloc[-1]) if grp["sell_price"].notna().any() else 3
        lead_time = int(rng.integers(2, 11))
        cover_weeks = cover_profiles[idx % len(cover_profiles)] * rng.uniform(0.85, 1.15)
        on_hand = int(round(mean_daily * 7 * cover_weeks))
        has_open_order = idx % 4 == 1
        on_order = int(round(mean_daily * lead_time)) if has_open_order else 0
        pack_size = int(rng.choice([6, 12, 24]))
        rows.append(
            {
                "item_id": item_id,
                "on_hand": on_hand,
                "on_order": on_order,
                "on_order_eta_days": int(rng.integers(1, lead_time + 1)) if has_open_order else 0,
                "lead_time_days": lead_time,
                "lead_time_std": round(float(rng.uniform(0.3, 0.3 * lead_time)), 2),
                "moq": int(pack_size * rng.choice([1, 2, 4])),
                "pack_size": pack_size,
                "unit_cost": round(price * 0.65, 2),
                "holding_cost_rate": 0.25,
                "is_simulated": True,
            }
        )
    params = pd.DataFrame(rows, columns=INVENTORY_PARAM_COLUMNS)
    logger.info("inventory_params_simulated", items=len(params), seed=seed)
    return params


def save_inventory_params(params: pd.DataFrame, path: Path) -> Path:
    """Write simulated parameters to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    params.to_csv(path, index=False)
    return path


def load_inventory_params(path: Path) -> pd.DataFrame:
    """Read simulated parameters from CSV."""
    return pd.read_csv(path)
