# Dataset

## Source

**M5 Forecasting - Accuracy** (Walmart, Kaggle 2020): daily unit sales for 3,049 products
across 10 US stores, 2011-01-29 to 2016-06-19, with calendar events, SNAP days and weekly
prices. License: competition data for research and education.

Files used: `sales_train_evaluation.csv`, `sales_test_evaluation.csv` (the 28 days after
training, used only for the hold-out evaluation), `calendar.csv` and `sell_prices.csv`.
`scripts/download_m5.py` tries the Kaggle CLI first and then a public GitHub mirror
(Nixtla `m5-forecasts`).

## Runtime database (`data/db/stockpilot.db`)

At runtime every layer reads the data from **SQLite**, not from CSV or parquet files.
`scripts/init_db.py` (or the API on its first start) loads the database:

| Source (`--source`) | When | Contents |
|---|---|---|
| `raw` | `data/raw/` has the M5 CSVs (`auto` picks it first) | Whole CA_1 FOODS category: 1,437 items x 1,941 days = 2,789,217 sales rows, 1,969 calendar days, 5,570 price spells, 40,236 hold-out rows. Loads in about 10 s |
| `processed` | Raw CSVs absent (fresh clone) | The committed 12-item parquet seed described below (8,760 sales rows, 336 hold-out rows) |
| synthetic | Neither available and `allow_synthetic_fallback: true` | Generated M5-shaped data (development only) |

Tables: `items` (with catalog names and an `is_demo` flag), `calendar`, `daily_sales`,
`prices` (price spells: one row per run of days with the same price), `sales_holdout`,
`inventory_positions`, plus the application tables `pipeline_runs`, `decisions` and `jobs`.
See the ER diagram in [architecture.md](architecture.md#5-data-layer-and-persistence).

* **Demo items.** The 12 SKUs below are flagged `is_demo` and are the default set for
  "Run all" and `POST /jobs {"all_demo": true}`. Any other SKU can be analysed through the
  dashboard search, `POST /runs` or `scripts/run_pipeline.py --items`.
* **History window.** `SQLiteRepository` serves the last `data.history_days` (730) per item
  from the `v_daily_sales` view, ending at the forecast origin 2016-05-22. The database keeps
  the full history back to 2011-01-29.
* **Parity.** For the demo items, the SQLite frames are identical to the parquet frames
  (values and dtypes; enforced by `tests/test_db.py`), so forecasts, recommendations and
  evaluation numbers are unchanged.

## POC subset (demo items and fallback seed)

| Dimension | Value | Why |
|---|---|---|
| Store | `CA_1` | Single store, as scoped |
| Category | `FOODS` | Highest volume, clear weekly and SNAP patterns |
| Items | 12 SKUs | 8 high-volume plus 4 mid-volume or intermittent, for variety |
| History | 2014-05-24 to 2016-05-22 (730 days) | Two years, so yearly seasonality is learnable |
| Forecast origin | 2016-05-22 | Last day of history |
| Horizon | 7 days (2016-05-23 to 2016-05-29) | As scoped |
| Future calendar | 28 days | Events, SNAP and prices for the forecast window |
| Hold-out actuals | 28 days × 12 items (336 rows) | Out-of-sample accuracy check only |

Item selection (`src/data/ingestion.py::select_items`): rank all `CA_1` FOODS items by total
units, take the top items (two-thirds of the slots) and fill the rest with mid-ranked items
(rank 40 or lower) whose zero-sales share is below 40%. This keeps the subset realistic
rather than cherry-picking smooth series.

Rebuild with `python scripts/build_subset.py --download`.

## Pipeline input schema (`v_daily_sales` / `sales_subset.parquet`)

| Column | Type | Description |
|---|---|---|
| `date` | datetime | Calendar day |
| `item_id`, `dept_id`, `cat_id`, `store_id` | str | Hierarchy |
| `units` | float | Units sold |
| `sell_price` | float | Price on that day (M5 weekly price, stored as price spells) |
| `event_name`, `event_type` | str | M5 calendar events (30 distinct in the window; empty if none) |
| `snap` | int | SNAP benefits day for California (33% of days) |

`calendar_future.parquet` has `date, event_name, event_type, snap`.
`actuals_future.parquet` has `date, item_id, units`.

## Item profile

| Item | Product (illustrative) | Mean units/day | Max | Zero-day share | Avg price | Suspected stockout days | Quality score |
|---|---|---|---|---|---|---|---|
| FOODS_3_064 | White Sandwich Bread | 25.2 | 71 | 1.0% | 1.14 | 0 | 1.00 |
| FOODS_3_090 | Bananas | 54.9 | 260 | 20.1% | 1.50 | 145 | 0.80 |
| FOODS_3_120 | Ground Beef | 38.5 | 157 | 11.5% | 4.98 | 16 | 0.89 |
| FOODS_3_148 | Canned Black Beans | 5.0 | 38 | 30.1% | 1.02 | 203 | 0.76 |
| FOODS_3_252 | Large Eggs (Dozen) | 41.7 | 116 | 0.3% | 1.58 | 0 | 1.00 |
| FOODS_3_282 | Whole Milk (Half Gallon) | 27.9 | 92 | 4.5% | 2.51 | 23 | 0.88 |
| FOODS_3_351 | Canned Sweet Corn | 6.6 | 38 | 27.4% | 1.02 | 196 | 0.77 |
| FOODS_3_455 | Shredded Cheddar Cheese | 10.2 | 40 | 2.1% | 2.24 | 0 | 1.00 |
| FOODS_3_519 | Hass Avocado | 4.1 | 55 | 30.7% | 1.00 | 34 | 0.88 |
| FOODS_3_586 | Roma Tomatoes | 44.9 | 106 | 0.3% | 1.67 | 0 | 1.00 |
| FOODS_3_587 | Orange Juice | 25.2 | 80 | 11.0% | 2.48 | 78 | 0.85 |
| FOODS_3_714 | Spaghetti Pasta | 25.2 | 78 | 0.3% | 1.58 | 0 | 1.00 |

## Product names (`data/reference/product_catalog.csv`)

M5 anonymises products: `FOODS_3_090` means "item 90 of FOODS department 3" and the real
product is not published. To make the demo readable, each code is mapped to an
**illustrative** food chosen to match its sales profile. Cheap, high-volume items with empty-shelf
gaps become Bananas; slow, erratic items become Hass Avocado or canned goods; the priciest item
becomes Ground Beef. The catalog also stores the department, the selling unit and a
perishable flag. Codes missing from the catalog are shown as-is, and the names never
contain digits because the LLM grounding check treats every number as a fact to verify.

## Data quality and cleaning (`src/data/quality.py`, `cleaning.py`)

1. Remove duplicate dates and reindex to a continuous daily calendar. Missing days are
   marked `is_imputed` and filled with 0 units.
2. Clip negative units. Forward and back fill prices; fill events with "" and SNAP with 0.
3. **Censored demand.** A zero streak of 7 or more days on an item averaging at least 3
   units/day is a suspected stockout (shelf empty, not zero demand). Those days get
   `y = NaN`, so the models ignore them; `units` keeps the observed value.
4. **Quality score** (0-1): penalises missing, imputed and suspected stockout days. Below 0.8
   the Critic raises a `data_quality` warning.

## Features (`src/data/features.py`)

Calendar features (day of week, month, weekend, event flag), price ratio versus the 28-day
average, price change, lagged rolling mean and std (7 and 28 days, shifted by one day to
avoid leakage) and lags at 7 and 28 days. Prophet uses `snap` and `price_ratio` as
regressors and the M5 events as holidays. In the future frame, the last observed price is
carried forward.

## Train / test strategy

* **Model selection:** a rolling-origin backtest over the last 4 × 7 days of history. Each
  fold trains only on data before it.
* **Hold-out:** the real 7 days after the origin (`actuals_future`) are never used for
  training or selection, only by `scripts/run_evaluation.py`.
* **Inventory simulation:** a replay of the last 56 days with weekly refits on data
  available at each decision point.

## Simulated inventory parameters (`data/simulated/inventory_params.csv`)

M5 contains no inventory, so `src/data/simulate_inventory.py` generates seeded, reproducible
values (seed 42). The loader writes them to the `inventory_positions` table: demo items keep
the committed CSV values, and every other item is simulated with the same generator.
`PUT /items/{id}/inventory` overwrites a row (for example after a stock count) and sets
`is_simulated` to `False`.

| Field | Generation |
|---|---|
| `on_hand` | Mean daily demand × a cover profile in {0.4, 0.7, 1.0, 1.5, 2.5, 4.0} weeks (±15%), giving a mix of urgent and overstocked items |
| `on_order`, `on_order_eta_days` | Open order for some items, with an ETA within the lead time |
| `lead_time_days`, `lead_time_std` | 2-10 days; std uniform between 0.3 and 0.3 × lead time |
| `moq`, `pack_size` | Case packs of 6, 12 or 24; MOQ of 1, 2 or 4 packs |
| `unit_cost`, `holding_cost_rate` | 0.65 × average shelf price; 25% per year |
| `is_simulated` | `True` when loaded; surfaced in the UI and as a Critic INFO issue |

## Synthetic fallback

`src/data/synthetic.py` generates M5-shaped data (weekly seasonality, trend, events, SNAP,
one spike item and one intermittent item). Tests load it into a temporary SQLite database
(`load_dataset`), and `init_db` falls back to it if neither the raw CSVs nor the processed
seed exist, so the project runs offline.
