"""Download the raw M5 Forecasting files into ``data/raw``.

Two strategies are tried in order:

1. Kaggle API (requires ``~/.kaggle/kaggle.json`` and accepting the competition rules).
2. Public GitHub mirror maintained by Nixtla (``Nixtla/m5-forecasts``).

Usage:
    python scripts/download_m5.py
"""

from __future__ import annotations

import io
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_FILES = ("sales_train_evaluation.csv", "calendar.csv", "sell_prices.csv")
GITHUB_MIRROR = "https://codeload.github.com/Nixtla/m5-forecasts/zip/refs/heads/main"


def _has_files(raw_dir: Path) -> bool:
    return all((raw_dir / f).exists() for f in REQUIRED_FILES)


def _try_kaggle(raw_dir: Path) -> bool:
    if shutil.which("kaggle") is None or not (Path.home() / ".kaggle" / "kaggle.json").exists():
        return False
    result = subprocess.run(
        ["kaggle", "competitions", "download", "-c", "m5-forecasting-accuracy", "-p",
         str(raw_dir)],
        check=False,
    )
    archive = raw_dir / "m5-forecasting-accuracy.zip"
    if result.returncode != 0 or not archive.exists():
        return False
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(raw_dir)
    archive.unlink()
    return _has_files(raw_dir)


def _try_github_mirror(raw_dir: Path) -> bool:
    print(f"Downloading M5 mirror from {GITHUB_MIRROR} (~190 MB) ...")
    with urllib.request.urlopen(GITHUB_MIRROR, timeout=600) as response:
        outer = zipfile.ZipFile(io.BytesIO(response.read()))
    inner_name = next(n for n in outer.namelist() if n.endswith("datasets/m5.zip"))
    with zipfile.ZipFile(io.BytesIO(outer.read(inner_name))) as inner:
        inner.extractall(raw_dir)
    return _has_files(raw_dir)


def download_m5(raw_dir: Path = ROOT / "data" / "raw") -> Path:
    """Ensure the raw M5 files exist in ``raw_dir`` and return the directory."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    if _has_files(raw_dir):
        print("Raw M5 files already present.")
        return raw_dir
    if _try_kaggle(raw_dir) or _try_github_mirror(raw_dir):
        print(f"M5 files available in {raw_dir}")
        return raw_dir
    raise RuntimeError("Could not download M5 data via Kaggle or the GitHub mirror.")


if __name__ == "__main__":
    download_m5()
    sys.exit(0)
