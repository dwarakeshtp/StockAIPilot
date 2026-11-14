"""Build the processed M5 subset and simulated inventory parameters.

Usage:
    python scripts/build_subset.py            # build from data/raw (M5 CSVs)
    python scripts/build_subset.py --download # download M5 first (see download_m5.py)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.settings import load_settings  # noqa: E402
from src.data.ingestion import M5RawSource, save_dataset  # noqa: E402
from src.data.simulate_inventory import (  # noqa: E402
    save_inventory_params,
    simulate_inventory_params,
)
from src.logging.logger import configure_logging, get_logger  # noqa: E402


def main() -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Download raw M5 data first")
    args = parser.parse_args()

    settings = load_settings()
    configure_logging(settings.outputs.logs_dir)
    logger = get_logger("build_subset")

    if args.download:
        from scripts.download_m5 import download_m5

        download_m5(settings.data.raw_dir)

    source = M5RawSource(settings)
    if not source.available():
        logger.error("raw_m5_missing", raw_dir=str(settings.data.raw_dir))
        print("Raw M5 files not found. Run `python scripts/download_m5.py` first.")
        return 1

    dataset = source.load()
    save_dataset(dataset, settings)
    params = simulate_inventory_params(dataset.sales, seed=settings.data.random_seed)
    save_inventory_params(params, settings.data.inventory_params_path)
    print(
        f"Saved {len(dataset.sales):,} rows for {dataset.sales['item_id'].nunique()} items "
        f"({dataset.sales['date'].min().date()} .. {dataset.sales['date'].max().date()})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
