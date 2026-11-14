"""SQLite-backed implementation of :class:`src.data.repository.Repository`."""

from __future__ import annotations

import threading
from functools import cached_property
from typing import Any

import pandas as pd
from sqlalchemy import Engine, func, or_, select, text, update

from src.config.settings import Settings
from src.data.catalog import Product, fallback_product
from src.data.ingestion import FUTURE_CALENDAR_COLUMNS, SALES_COLUMNS, SalesDataset
from src.data.simulate_inventory import INVENTORY_PARAM_COLUMNS
from src.db.engine import get_engine
from src.db.loader import read_meta
from src.db.schema import inventory_positions, items
from src.utils.ids import utc_now

EDITABLE_INVENTORY_FIELDS = (
    "on_hand", "on_order", "on_order_eta_days", "lead_time_days", "lead_time_std", "moq",
    "pack_size", "unit_cost", "holding_cost_rate",
)
_SALES_SQL = text(
    "SELECT date, item_id, dept_id, cat_id, store_id, units, sell_price, event_name, "
    "event_type, snap FROM v_daily_sales "
    "WHERE item_id = :item_id AND date BETWEEN :start AND :end ORDER BY date"
)


class DatabaseNotInitialisedError(RuntimeError):
    """Raised when the database has no data (run ``scripts/init_db.py``)."""


def _product_from_row(row: Any) -> Product:
    if not row.product_name:
        return fallback_product(row.item_id)
    return Product(item_id=row.item_id, name=row.product_name,
                   department=row.department or "Unknown", unit=row.unit or "unit",
                   perishable=bool(row.perishable), profile_note=row.profile_note or "")


def _item_dict(row: Any) -> dict[str, Any]:
    product = _product_from_row(row)
    return {**product.model_dump(), "label": product.label, "dept_id": row.dept_id,
            "store_id": row.store_id, "is_demo": bool(row.is_demo)}


class SQLiteRepository:
    """Serves per-item slices from the database with a small per-item cache.

    ``item_sales`` returns the last ``data.history_days`` days up to the last
    observed sales date, so the pipeline sees the same window as the original
    parquet subset while the database keeps the full history.
    """

    def __init__(self, settings: Settings, engine: Engine | None = None) -> None:
        self.settings = settings
        self.engine = engine or get_engine(settings.database.url)
        self._sales_cache: dict[str, pd.DataFrame] = {}
        self._lock = threading.Lock()

    @cached_property
    def meta(self) -> dict[str, str]:
        """Load metadata written by the loader."""
        entries = read_meta(self.engine)
        if not entries.get("source"):
            raise DatabaseNotInitialisedError(
                f"Database {self.settings.database.url} is empty. "
                "Run `python scripts/init_db.py`.")
        return entries

    def clear_cache(self) -> None:
        """Forget cached slices (call after reloading the data tables)."""
        with self._lock:
            self._sales_cache.clear()
            for name in ("meta", "item_ids", "future_calendar", "history_window"):
                self.__dict__.pop(name, None)

    @property
    def source(self) -> str:
        """``sqlite:<loader source>``, e.g. ``sqlite:m5_raw``."""
        return f"sqlite:{self.meta['source']}"

    @property
    def last_date(self) -> pd.Timestamp:
        """Last observed sales date (the forecast origin)."""
        return pd.Timestamp(self.meta["last_sales_date"])

    @cached_property
    def history_window(self) -> tuple[str, str]:
        """ISO start/end dates of the history served to the pipeline."""
        first = pd.Timestamp(self.meta["first_sales_date"])
        start = max(first, self.last_date - pd.Timedelta(days=self.settings.data.history_days - 1))
        return start.strftime("%Y-%m-%d"), self.last_date.strftime("%Y-%m-%d")

    @property
    def date_range(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        """First and last date of the history window served to the pipeline."""
        start, end = self.history_window
        return pd.Timestamp(start), pd.Timestamp(end)

    @cached_property
    def item_ids(self) -> list[str]:
        """Demo items (or every item if none is flagged)."""
        self.meta  # noqa: B018 - raise early when the database is empty
        with self.engine.connect() as conn:
            demo = conn.execute(select(items.c.item_id).where(items.c.is_demo)
                                .order_by(items.c.item_id)).scalars().all()
            if demo:
                return list(demo)
            return list(conn.execute(select(items.c.item_id).order_by(items.c.item_id))
                        .scalars().all())

    def has_item(self, item_id: str) -> bool:
        """Whether ``item_id`` exists in the item master."""
        with self.engine.connect() as conn:
            return conn.execute(select(items.c.item_id).where(items.c.item_id == item_id)
                                ).first() is not None

    def search_items(self, search: str | None = None, department: str | None = None,
                     demo_only: bool = False, limit: int = 50,
                     offset: int = 0) -> tuple[list[dict[str, Any]], int]:
        """Page through the item master.

        Returns:
            ``(rows, total)`` where rows carry product metadata and ``is_demo``.
        """
        conditions = []
        if search:
            pattern = f"%{search.strip()}%"
            conditions.append(or_(items.c.item_id.ilike(pattern),
                                  items.c.product_name.ilike(pattern)))
        if department:
            conditions.append(or_(items.c.department == department,
                                  items.c.dept_id == department))
        if demo_only:
            conditions.append(items.c.is_demo)
        query = select(items).where(*conditions)
        with self.engine.connect() as conn:
            total = conn.execute(select(func.count()).select_from(query.subquery())).scalar_one()
            rows = conn.execute(query.order_by(items.c.is_demo.desc(), items.c.item_id)
                                .limit(limit).offset(offset)).all()
        return [_item_dict(r) for r in rows], int(total)

    def item(self, item_id: str) -> dict[str, Any] | None:
        """Item master row with display metadata (``None`` if unknown)."""
        with self.engine.connect() as conn:
            row = conn.execute(select(items).where(items.c.item_id == item_id)).first()
        return _item_dict(row) if row is not None else None

    def product(self, item_id: str) -> Product:
        """Display metadata (falls back to the item code for uncatalogued items)."""
        with self.engine.connect() as conn:
            row = conn.execute(select(items).where(items.c.item_id == item_id)).first()
        return _product_from_row(row) if row is not None else fallback_product(item_id)

    def item_sales(self, item_id: str) -> pd.DataFrame:
        """History rows of one item (copy of the cached frame).

        Raises:
            KeyError: if the item has no sales rows.
        """
        with self._lock:
            cached = self._sales_cache.get(item_id)
        if cached is None:
            start, end = self.history_window
            with self.engine.connect() as conn:
                df = pd.read_sql(_SALES_SQL, conn,
                                 params={"item_id": item_id, "start": start, "end": end})
            if df.empty:
                raise KeyError(f"Unknown item_id: {item_id}")
            df["date"] = pd.to_datetime(df["date"])
            df["event_name"] = df["event_name"].fillna("")
            df["event_type"] = df["event_type"].fillna("")
            df["snap"] = df["snap"].fillna(0).astype("int64")
            df["sell_price"] = df["sell_price"].astype("float64")
            cached = df[SALES_COLUMNS]
            with self._lock:
                self._sales_cache[item_id] = cached
        return cached.copy()

    @cached_property
    def future_calendar(self) -> pd.DataFrame:
        """Calendar rows after the last observed sales date."""
        with self.engine.connect() as conn:
            df = pd.read_sql(text("SELECT date, event_name, event_type, snap FROM calendar "
                                  "WHERE date > :last ORDER BY date"),
                             conn, params={"last": self.history_window[1]})
        df["date"] = pd.to_datetime(df["date"])
        df["snap"] = df["snap"].astype("int64")
        return df[FUTURE_CALENDAR_COLUMNS]

    @property
    def inventory_params(self) -> pd.DataFrame:
        """Current inventory positions of every item (read fresh: they are editable)."""
        with self.engine.connect() as conn:
            df = pd.read_sql(select(inventory_positions).order_by(inventory_positions.c.item_id),
                             conn)
        df["is_simulated"] = df["is_simulated"].astype(bool)
        return df[INVENTORY_PARAM_COLUMNS]

    def item_inventory_params(self, item_id: str) -> dict:
        """Inventory parameters of one item.

        Raises:
            KeyError: if the item has no inventory row.
        """
        with self.engine.connect() as conn:
            row = conn.execute(select(inventory_positions)
                               .where(inventory_positions.c.item_id == item_id)).first()
        if row is None:
            raise KeyError(f"No inventory parameters for item_id: {item_id}")
        params = {c: getattr(row, c) for c in INVENTORY_PARAM_COLUMNS}
        params["is_simulated"] = bool(params["is_simulated"])
        return params

    def inventory_updated_at(self, item_id: str) -> str | None:
        """When the inventory row was last written."""
        with self.engine.connect() as conn:
            return conn.execute(select(inventory_positions.c.updated_at)
                                .where(inventory_positions.c.item_id == item_id)).scalar()

    def update_inventory(self, item_id: str, changes: dict[str, Any]) -> dict:
        """Update editable inventory fields (e.g. after a stock count).

        Raises:
            KeyError: if the item has no inventory row.
            ValueError: if a field is not editable.
        """
        unknown = set(changes) - set(EDITABLE_INVENTORY_FIELDS)
        if unknown:
            raise ValueError(f"Fields not editable: {sorted(unknown)}")
        self.item_inventory_params(item_id)
        if changes:
            with self.engine.begin() as conn:
                conn.execute(update(inventory_positions)
                             .where(inventory_positions.c.item_id == item_id)
                             .values(**changes, is_simulated=False,
                                     updated_at=str(utc_now())))
        return self.item_inventory_params(item_id)

    def future_actuals(self, item_id: str) -> pd.DataFrame | None:
        """Holdout sales after the forecast origin, if loaded."""
        with self.engine.connect() as conn:
            df = pd.read_sql(text("SELECT date, item_id, units FROM sales_holdout "
                                  "WHERE item_id = :item_id ORDER BY date"),
                             conn, params={"item_id": item_id})
        if df.empty:
            return None
        df["date"] = pd.to_datetime(df["date"])
        return df

    def item_stats(self, item_id: str) -> dict[str, Any]:
        """Headline numbers for one item over the served history window."""
        sales = self.item_sales(item_id)
        recent = sales.tail(28)
        price = sales["sell_price"].dropna()
        return {
            "history_start": sales["date"].min().strftime("%Y-%m-%d"),
            "history_end": sales["date"].max().strftime("%Y-%m-%d"),
            "history_days": len(sales),
            "total_units": float(sales["units"].sum()),
            "mean_daily_units_28d": round(float(recent["units"].mean()), 2),
            "zero_sales_share": round(float((sales["units"] == 0).mean()), 3),
            "last_price": float(price.iloc[-1]) if not price.empty else None,
        }

    def counts(self) -> dict[str, int]:
        """Row counts of the data tables."""
        names = ("items", "daily_sales", "prices", "calendar", "sales_holdout",
                 "inventory_positions", "pipeline_runs", "decisions", "jobs")
        with self.engine.connect() as conn:
            return {n: int(conn.execute(text(f"SELECT COUNT(*) FROM {n}")).scalar_one())
                    for n in names}

    def to_dataset(self, item_ids: list[str] | None = None) -> SalesDataset:
        """Materialise the served window of ``item_ids`` (default: demo items)."""
        ids = item_ids or self.item_ids
        sales = pd.concat([self.item_sales(i) for i in ids], ignore_index=True)
        actual_frames = [a for a in (self.future_actuals(i) for i in ids) if a is not None]
        return SalesDataset(
            sales=sales, future_calendar=self.future_calendar.copy(), source=self.source,
            future_actuals=pd.concat(actual_frames, ignore_index=True) if actual_frames else None)
