"""Relational schema of the StockPilot database.

Data tables (rebuilt from the M5 CSVs by the loader):

* ``items`` - one row per SKU of the store/category, with catalog display names.
* ``calendar`` - one row per day (events, SNAP flag, M5 week key).
* ``daily_sales`` - observed units per item and day.
* ``prices`` - price spells: a constant ``sell_price`` between two dates.
* ``sales_holdout`` - units sold after the forecast origin (evaluation only).
* ``inventory_positions`` - on-hand stock, open orders and supplier terms.

Application tables (kept across rebuilds): ``pipeline_runs``, ``decisions``, ``jobs``.
Dates are stored as ISO ``YYYY-MM-DD`` text, which sorts and compares correctly.
"""

from __future__ import annotations

from sqlalchemy import (
    Boolean,
    Column,
    Engine,
    Float,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    text,
)

SCHEMA_VERSION = "1"

metadata = MetaData()

meta = Table(
    "meta", metadata,
    Column("key", String, primary_key=True),
    Column("value", Text),
)

items = Table(
    "items", metadata,
    Column("item_id", String, primary_key=True),
    Column("dept_id", String),
    Column("cat_id", String),
    Column("store_id", String),
    Column("product_name", String),
    Column("department", String),
    Column("unit", String),
    Column("perishable", Boolean, nullable=False, default=False),
    Column("profile_note", Text),
    Column("is_demo", Boolean, nullable=False, default=False),
    Index("ix_items_is_demo", "is_demo"),
)

calendar = Table(
    "calendar", metadata,
    Column("date", String, primary_key=True),
    Column("d", String),
    Column("wm_yr_wk", Integer),
    Column("event_name", String, nullable=False, default=""),
    Column("event_type", String, nullable=False, default=""),
    Column("snap", Integer, nullable=False, default=0),
)

daily_sales = Table(
    "daily_sales", metadata,
    Column("item_id", String, primary_key=True),
    Column("date", String, primary_key=True),
    Column("units", Integer, nullable=False),
    sqlite_with_rowid=False,
)

prices = Table(
    "prices", metadata,
    Column("item_id", String, primary_key=True),
    Column("start_date", String, primary_key=True),
    Column("end_date", String, nullable=False),
    Column("sell_price", Float, nullable=False),
    sqlite_with_rowid=False,
)

sales_holdout = Table(
    "sales_holdout", metadata,
    Column("item_id", String, primary_key=True),
    Column("date", String, primary_key=True),
    Column("units", Integer, nullable=False),
    sqlite_with_rowid=False,
)

inventory_positions = Table(
    "inventory_positions", metadata,
    Column("item_id", String, primary_key=True),
    Column("on_hand", Integer, nullable=False),
    Column("on_order", Integer, nullable=False),
    Column("on_order_eta_days", Integer, nullable=False),
    Column("lead_time_days", Integer, nullable=False),
    Column("lead_time_std", Float, nullable=False),
    Column("moq", Integer, nullable=False),
    Column("pack_size", Integer, nullable=False),
    Column("unit_cost", Float, nullable=False),
    Column("holding_cost_rate", Float, nullable=False),
    Column("is_simulated", Boolean, nullable=False),
    Column("updated_at", String),
)

pipeline_runs = Table(
    "pipeline_runs", metadata,
    Column("run_id", String, primary_key=True),
    Column("kind", String, nullable=False),
    Column("item_id", String, nullable=False),
    Column("created_at", String, nullable=False),
    Column("action", String),
    Column("qty", Integer),
    Column("verdict", String),
    Column("revision_count", Integer),
    Column("result_json", Text, nullable=False),
    Index("ix_runs_item_kind_created", "item_id", "kind", "created_at"),
)

decisions = Table(
    "decisions", metadata,
    Column("decision_id", String, primary_key=True),
    Column("run_id", String, nullable=False),
    Column("item_id", String, nullable=False),
    Column("recommended_action", String, nullable=False),
    Column("recommended_qty", Integer, nullable=False),
    Column("verdict", String, nullable=False),
    Column("decision", String, nullable=False),
    Column("final_qty", Integer, nullable=False),
    Column("reason", Text, nullable=False),
    Column("reviewer", String, nullable=False),
    Column("decided_at", String, nullable=False),
    Index("ix_decisions_item_decided", "item_id", "decided_at"),
)

jobs = Table(
    "jobs", metadata,
    Column("job_id", String, primary_key=True),
    Column("status", String, nullable=False),
    Column("total", Integer, nullable=False),
    Column("done", Integer, nullable=False, default=0),
    Column("item_ids", Text, nullable=False),
    Column("run_ids", Text, nullable=False, default="[]"),
    Column("errors", Text, nullable=False, default="[]"),
    Column("created_at", String, nullable=False),
    Column("started_at", String),
    Column("finished_at", String),
)

DATA_TABLES: tuple[Table, ...] = (
    items, calendar, daily_sales, prices, sales_holdout, inventory_positions,
)

V_DAILY_SALES = """
CREATE VIEW IF NOT EXISTS v_daily_sales AS
SELECT s.date, s.item_id, i.dept_id, i.cat_id, i.store_id, s.units, p.sell_price,
       c.event_name, c.event_type, c.snap
FROM daily_sales s
JOIN items i ON i.item_id = s.item_id
JOIN calendar c ON c.date = s.date
LEFT JOIN prices p
  ON p.item_id = s.item_id AND s.date BETWEEN p.start_date AND p.end_date
"""


def create_schema(engine: Engine) -> None:
    """Create every table and the ``v_daily_sales`` view if missing."""
    metadata.create_all(engine)
    with engine.begin() as conn:
        conn.execute(text(V_DAILY_SALES))


def clear_data_tables(conn) -> None:
    """Delete all rows of the data tables (application tables are untouched)."""
    for table in DATA_TABLES:
        conn.execute(table.delete())
    conn.execute(meta.delete())
