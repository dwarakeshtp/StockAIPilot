"""Ingestion layer: load sales, prices, calendar and inventory into SQLite.

Three entry points feed the same normalised tables (see :mod:`src.db.schema`):

* :func:`load_from_m5_raw` - the whole store/category from the raw M5 CSVs
  (CA_1 FOODS: ~1,400 items x 1,941 days), plus the 28-day evaluation holdout.
* :func:`load_from_processed_seed` - the committed 12-item parquet subset, used
  when the raw CSVs are not downloaded.
* :func:`load_dataset` - any in-memory :class:`SalesDataset` (tests, synthetic data).

:func:`init_db` picks a source and is idempotent: it does nothing when the
database already holds data, unless ``rebuild=True``.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Literal

import pandas as pd
from sqlalchemy import Engine, inspect, select

from src.config.settings import Settings
from src.data.catalog import load_catalog
from src.data.ingestion import (
    M5RawSource,
    ProcessedParquetSource,
    SalesDataset,
    SyntheticSource,
    select_items,
)
from src.data.simulate_inventory import (
    INVENTORY_PARAM_COLUMNS,
    load_inventory_params,
    simulate_inventory_params,
)
from src.db.engine import get_engine
from src.db.schema import SCHEMA_VERSION, clear_data_tables, create_schema, meta
from src.logging.logger import get_logger
from src.utils.ids import utc_now

logger = get_logger(__name__)

InitSource = Literal["auto", "raw", "processed"]
M5_DAY_ONE = pd.Timestamp("2011-01-29")
ITEM_COLUMNS = ["item_id", "dept_id", "cat_id", "store_id", "product_name", "department",
                "unit", "perishable", "profile_note", "is_demo"]
CALENDAR_COLUMNS = ["date", "d", "wm_yr_wk", "event_name", "event_type", "snap"]


@dataclass
class LoadReport:
    """Outcome of a database load."""

    source: str
    items: int = 0
    demo_items: list[str] = field(default_factory=list)
    sales_rows: int = 0
    price_spells: int = 0
    calendar_days: int = 0
    holdout_rows: int = 0
    first_date: str | None = None
    last_date: str | None = None
    seconds: float = 0.0
    skipped: bool = False

    def to_dict(self) -> dict:
        """Plain-dict form (logged and stored in ``meta``)."""
        return asdict(self)


def _iso(dates: pd.Series) -> pd.Series:
    return pd.to_datetime(dates).dt.strftime("%Y-%m-%d")


def price_spells(daily: pd.DataFrame) -> pd.DataFrame:
    """Compress daily prices into spells of consecutive days with the same price.

    Args:
        daily: Frame with ``item_id``, ``date`` and ``sell_price`` (NaN = no price).

    Returns:
        Frame with ``item_id``, ``start_date``, ``end_date`` (ISO text), ``sell_price``.
    """
    df = daily[["item_id", "date", "sell_price"]].dropna(subset=["sell_price"])
    if df.empty:
        return pd.DataFrame(columns=["item_id", "start_date", "end_date", "sell_price"])
    df = df.assign(date=pd.to_datetime(df["date"])).sort_values(["item_id", "date"])
    starts = (df["item_id"].ne(df["item_id"].shift())
              | df["sell_price"].ne(df["sell_price"].shift())
              | df["date"].diff().ne(pd.Timedelta(days=1)))
    spells = df.groupby(starts.cumsum().to_numpy()).agg(
        item_id=("item_id", "first"), start_date=("date", "min"), end_date=("date", "max"),
        sell_price=("sell_price", "first"))
    spells["start_date"] = spells["start_date"].dt.strftime("%Y-%m-%d")
    spells["end_date"] = spells["end_date"].dt.strftime("%Y-%m-%d")
    return spells.reset_index(drop=True)


def _calendar_frame(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Union of daily calendar rows; derives M5-aligned ``d``/``wm_yr_wk`` when absent."""
    cal = pd.concat(frames, ignore_index=True)
    cal["date"] = pd.to_datetime(cal["date"])
    cal = cal.drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)
    day_index = (cal["date"] - M5_DAY_ONE).dt.days
    if "d" not in cal.columns:
        cal["d"] = "d_" + (day_index + 1).astype(str)
    if "wm_yr_wk" not in cal.columns:
        cal["wm_yr_wk"] = day_index // 7
    cal["event_name"] = cal["event_name"].fillna("")
    cal["event_type"] = cal["event_type"].fillna("")
    cal["snap"] = cal["snap"].fillna(0).astype(int)
    cal["date"] = cal["date"].dt.strftime("%Y-%m-%d")
    return cal[CALENDAR_COLUMNS]


def _items_frame(ids: pd.DataFrame, demo_ids: list[str], settings: Settings) -> pd.DataFrame:
    """Item master rows enriched with catalog display metadata."""
    catalog = load_catalog(settings.data.catalog_path)
    out = ids.drop_duplicates("item_id").sort_values("item_id").reset_index(drop=True)
    product = out["item_id"].map(catalog.get)
    out["product_name"] = product.map(lambda p: p.name if p else None)
    out["department"] = product.map(lambda p: p.department if p else None)
    out["unit"] = product.map(lambda p: p.unit if p else None)
    out["perishable"] = product.map(lambda p: bool(p.perishable) if p else False)
    out["profile_note"] = product.map(lambda p: p.profile_note if p else None)
    out["is_demo"] = out["item_id"].isin(demo_ids)
    return out[ITEM_COLUMNS]


def choose_demo_items(settings: Settings, item_ids: list[str],
                      window: pd.DataFrame | None = None) -> list[str]:
    """Demo SKUs shown by default: configured ids, else catalog items, else a selection."""
    available = set(item_ids)
    configured = [i for i in settings.data.item_ids if i in available]
    if configured:
        return sorted(configured)
    catalogued = [i for i in load_catalog(settings.data.catalog_path) if i in available]
    if catalogued:
        return sorted(catalogued)
    if window is not None and len(item_ids) > settings.data.n_items:
        return sorted(select_items(window, settings.data.n_items))
    return sorted(item_ids)


def _inventory_frame(settings: Settings, sales: pd.DataFrame, demo_ids: list[str],
                     inventory: pd.DataFrame | None = None) -> pd.DataFrame:
    """Simulated positions for every item; demo items keep the committed CSV values."""
    if inventory is not None:
        params = inventory.copy()
    else:
        params = simulate_inventory_params(sales, seed=settings.data.random_seed)
        path = settings.data.inventory_params_path
        if path.exists():
            seeded = load_inventory_params(path)
            seeded = seeded[seeded["item_id"].isin(demo_ids)]
            if set(demo_ids).issubset(set(seeded["item_id"])):
                params = pd.concat([params[~params["item_id"].isin(demo_ids)], seeded],
                                   ignore_index=True)
    params = params[INVENTORY_PARAM_COLUMNS].sort_values("item_id").reset_index(drop=True)
    params["is_simulated"] = params["is_simulated"].astype(bool)
    params["updated_at"] = str(utc_now())
    return params


def write_tables(engine: Engine, *, source: str, items: pd.DataFrame, calendar: pd.DataFrame,
                 sales: pd.DataFrame, spells: pd.DataFrame, holdout: pd.DataFrame | None,
                 inventory: pd.DataFrame, settings: Settings, started: float) -> LoadReport:
    """Replace every data table in a single transaction and record ``meta``."""
    holdout = holdout if holdout is not None else pd.DataFrame(columns=["item_id", "date",
                                                                         "units"])
    report = LoadReport(
        source=source, items=len(items),
        demo_items=items.loc[items["is_demo"], "item_id"].tolist(),
        sales_rows=len(sales), price_spells=len(spells), calendar_days=len(calendar),
        holdout_rows=len(holdout),
        first_date=str(sales["date"].min()) if len(sales) else None,
        last_date=str(sales["date"].max()) if len(sales) else None,
    )
    create_schema(engine)
    with engine.begin() as conn:
        clear_data_tables(conn)
        for name, frame in (("items", items), ("calendar", calendar),
                            ("daily_sales", sales[["item_id", "date", "units"]]),
                            ("prices", spells), ("sales_holdout", holdout[["item_id", "date",
                                                                           "units"]]),
                            ("inventory_positions", inventory)):
            frame.to_sql(name, conn, if_exists="append", index=False, chunksize=50_000)
        report.seconds = round(time.perf_counter() - started, 2)
        entries = {
            "schema_version": SCHEMA_VERSION, "source": source, "loaded_at": str(utc_now()),
            "store_id": settings.data.store_id, "category": settings.data.category,
            "first_sales_date": report.first_date or "", "last_sales_date": report.last_date or "",
            "load_report": json.dumps(report.to_dict()),
        }
        conn.execute(meta.insert(), [{"key": k, "value": v} for k, v in entries.items()])
    logger.info("database_loaded", **{k: v for k, v in report.to_dict().items()
                                      if k != "demo_items"})
    return report


def _units(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame[["item_id", "date", "units"]].copy()
    out["date"] = _iso(out["date"])
    out["units"] = out["units"].round().astype("int64")
    return out


def load_dataset(engine: Engine, settings: Settings, dataset: SalesDataset,
                 source: str | None = None, inventory: pd.DataFrame | None = None,
                 demo_ids: list[str] | None = None) -> LoadReport:
    """Load an in-memory dataset (synthetic, processed seed, ...) into the database."""
    started = time.perf_counter()
    sales = dataset.sales.copy()
    sales["date"] = pd.to_datetime(sales["date"])
    item_ids = sorted(sales["item_id"].unique().tolist())
    demo_ids = demo_ids if demo_ids is not None else choose_demo_items(settings, item_ids)
    calendar = _calendar_frame([sales[["date", "event_name", "event_type", "snap"]],
                                dataset.future_calendar])
    return write_tables(
        engine, source=source or dataset.source,
        items=_items_frame(sales[["item_id", "dept_id", "cat_id", "store_id"]], demo_ids,
                           settings),
        calendar=calendar, sales=_units(sales), spells=price_spells(sales),
        holdout=_units(dataset.future_actuals) if dataset.future_actuals is not None else None,
        inventory=_inventory_frame(settings, sales, demo_ids, inventory),
        settings=settings, started=started)


def load_from_processed_seed(engine: Engine, settings: Settings) -> LoadReport:
    """Load the committed 12-item parquet subset (no raw CSVs required)."""
    cfg = settings.data
    source = ProcessedParquetSource(cfg.sales_path, cfg.future_calendar_path)
    if not source.available():
        raise FileNotFoundError(f"Processed seed not found at {cfg.sales_path}")
    return load_dataset(engine, settings, source.load(), source="m5_processed")


def load_from_m5_raw(engine: Engine, settings: Settings) -> LoadReport:
    """Load the whole configured store/category from the raw M5 CSV files."""
    started = time.perf_counter()
    cfg = settings.data
    raw = M5RawSource(settings)
    if not raw.available():
        raise FileNotFoundError(f"Raw M5 CSVs not found in {cfg.raw_dir}")
    wide = raw._read_sales_wide(cfg.store_id, cfg.category)
    cal = pd.read_csv(cfg.raw_dir / "calendar.csv", parse_dates=["date"])
    if "d" not in cal.columns:
        cal = cal.sort_values("date").reset_index(drop=True)
        cal["d"] = "d_" + (cal.index + 1).astype(str)
    snap_col = f"snap_{cfg.state_id}"
    cal = cal[["date", "d", "wm_yr_wk", "event_name_1", "event_type_1", snap_col]].rename(
        columns={"event_name_1": "event_name", "event_type_1": "event_type", snap_col: "snap"})
    calendar = _calendar_frame([cal])
    d_to_date = dict(zip(calendar["d"], calendar["date"], strict=True))

    item_ids = sorted(wide["item_id"].tolist())
    day_cols = [c for c in wide.columns if c.startswith("d_")]
    window = wide.set_index("item_id")[day_cols[-cfg.history_days:]]
    demo_ids = choose_demo_items(settings, item_ids, window)

    stacked = wide.set_index("item_id")[day_cols].stack()
    sales = pd.DataFrame({
        "item_id": stacked.index.get_level_values(0),
        "date": stacked.index.get_level_values(1).map(d_to_date),
        "units": stacked.to_numpy().astype("int64"),
    })
    logger.info("m5_raw_sales_read", items=len(item_ids), rows=len(sales))

    weekly = raw._read_prices(cfg.store_id, item_ids)
    daily_prices = weekly.merge(calendar[["date", "wm_yr_wk"]], on="wm_yr_wk", how="inner")
    spells = price_spells(daily_prices)

    holdout = raw._read_future_actuals(cfg.store_id, item_ids, cal)
    sim_sales = sales.assign(date=pd.to_datetime(sales["date"]))
    last = sim_sales["date"].max()
    recent = sim_sales[sim_sales["date"] > last - pd.Timedelta(days=28)]
    recent = recent.merge(daily_prices.assign(date=pd.to_datetime(daily_prices["date"])),
                          on=["item_id", "date"], how="left")
    return write_tables(
        engine, source="m5_raw",
        items=_items_frame(wide[["item_id", "dept_id", "cat_id", "store_id"]], demo_ids,
                           settings),
        calendar=calendar, sales=sales, spells=spells,
        holdout=_units(holdout) if holdout is not None else None,
        inventory=_inventory_frame(settings, recent, demo_ids),
        settings=settings, started=started)


def read_meta(engine: Engine) -> dict[str, str]:
    """All ``meta`` entries (empty when the database is not initialised)."""
    if not inspect(engine).has_table("meta"):
        return {}
    with engine.connect() as conn:
        return {row.key: row.value for row in conn.execute(select(meta))}


def is_loaded(engine: Engine) -> bool:
    """Whether the data tables have been populated."""
    return bool(read_meta(engine).get("source"))


def init_db(settings: Settings, source: InitSource = "auto", rebuild: bool = False,
            engine: Engine | None = None, dataset: SalesDataset | None = None) -> LoadReport:
    """Create the schema and load data unless the database is already populated.

    Args:
        settings: Application settings (database URL, data paths).
        source: ``raw`` (M5 CSVs), ``processed`` (committed seed) or ``auto``
            (raw when available, else the seed, else synthetic if allowed).
        rebuild: Reload the data tables even if they already hold data. Runs,
            decisions and jobs are preserved.
        engine: Engine to use (defaults to ``settings.database.url``).
        dataset: Load this in-memory dataset instead of reading files.

    Raises:
        FileNotFoundError: if the requested source is unavailable.
    """
    engine = engine or get_engine(settings.database.url)
    create_schema(engine)
    current = read_meta(engine)
    if current.get("source") and not rebuild:
        report = LoadReport(**json.loads(current.get("load_report", "{}") or "{}"))
        report.skipped = True
        return report
    if dataset is not None:
        return load_dataset(engine, settings, dataset)
    if source in ("auto", "raw") and M5RawSource(settings).available():
        return load_from_m5_raw(engine, settings)
    if source == "raw":
        raise FileNotFoundError(f"Raw M5 CSVs not found in {settings.data.raw_dir}. "
                                "Run `python scripts/download_data.py` first.")
    cfg = settings.data
    if ProcessedParquetSource(cfg.sales_path, cfg.future_calendar_path).available():
        return load_from_processed_seed(engine, settings)
    if source == "auto" and cfg.allow_synthetic_fallback:
        synthetic = SyntheticSource(n_items=min(cfg.n_items, 8), history_days=cfg.history_days,
                                    seed=cfg.random_seed).load()
        return load_dataset(engine, settings, synthetic)
    raise FileNotFoundError(f"No data source available (looked in {cfg.raw_dir} and "
                            f"{cfg.processed_dir}).")
