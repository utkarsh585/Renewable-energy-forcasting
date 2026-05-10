"""
Data Acquisition Module

Downloads hourly weather data from NASA POWER API for all configured regions
and saves one CSV per region in `nasa_power_data/`.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime

import pandas as pd
import requests

from config import (
    API_COMMUNITY,
    API_END_DATE,
    API_PARAMETERS,
    API_START_DATE,
    API_TIMEOUT_SECONDS,
    COLUMN_RENAME,
    RAW_DATA_DIR,
    REGIONS,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


def expected_hourly_rows(start_yyyymmdd: str, end_yyyymmdd: str) -> int:
    """Return expected number of hourly rows for an inclusive date range."""
    start = datetime.strptime(start_yyyymmdd, "%Y%m%d")
    end = datetime.strptime(end_yyyymmdd, "%Y%m%d")
    days = (end - start).days + 1
    return days * 24


def build_api_url(latitude: float, longitude: float) -> str:
    """Build NASA POWER hourly endpoint URL for one point."""
    return (
        "https://power.larc.nasa.gov/api/temporal/hourly/point?"
        f"parameters={API_PARAMETERS}"
        f"&community={API_COMMUNITY}"
        f"&longitude={longitude}"
        f"&latitude={latitude}"
        f"&start={API_START_DATE}"
        f"&end={API_END_DATE}"
        "&format=JSON"
    )


def fetch_region_dataframe(region: str, latitude: float, longitude: float) -> pd.DataFrame:
    """Fetch and normalize one region's weather dataframe."""
    url = build_api_url(latitude, longitude)
    response = requests.get(url, timeout=API_TIMEOUT_SECONDS)
    response.raise_for_status()

    payload = response.json()
    parameters = payload["properties"]["parameter"]
    df = pd.DataFrame(parameters)

    # NASA keys are timestamp strings in YYYYMMDDHH format.
    df.index = pd.to_datetime(df.index, format="%Y%m%d%H")
    df = df.sort_index()
    df = df.rename_axis("datetime").reset_index()
    df = df.rename(columns=COLUMN_RENAME)

    expected_cols = {"datetime", *COLUMN_RENAME.values()}
    missing = expected_cols.difference(df.columns)
    if missing:
        raise ValueError(f"{region}: Missing expected columns: {sorted(missing)}")

    return df[["datetime", *COLUMN_RENAME.values()]]


def run_data_download() -> None:
    """Download data for all configured regions and validate output shape."""
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    expected_rows = expected_hourly_rows(API_START_DATE, API_END_DATE)

    print("=" * 60)
    print("  STAGE 0: DATA ACQUISITION (NASA POWER)")
    print("=" * 60)
    print(f"  Date range: {API_START_DATE} to {API_END_DATE}")
    print(f"  Expected hourly rows per region: {expected_rows}")

    for region, (lat, lon) in REGIONS.items():
        print(f"\n▸ Downloading {region} ({lat}, {lon})...")
        df = fetch_region_dataframe(region, lat, lon)

        out_path = os.path.join(RAW_DATA_DIR, f"{region}.csv")
        df.to_csv(out_path, index=False)

        completeness = (len(df) / expected_rows) * 100
        print(f"  Shape: {df.shape}")
        print(f"  Date range: {df['datetime'].min()} → {df['datetime'].max()}")
        print(f"  Completeness: {completeness:.2f}%")
        print(f"  ✓ Saved to {out_path}")

    print("\n✅ Data acquisition complete for all regions.\n")


if __name__ == "__main__":
    run_data_download()
