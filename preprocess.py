"""
Data Preprocessing Module
─────────────────────────
• Loads raw CSV files from nasa_power_data/
• Converts timestamps to datetime
• Handles missing values (forward fill + interpolation)
• Computes synthetic target variables (solar_power, wind_power)
• Saves cleaned data to processed_data/
"""

import pandas as pd
import numpy as np
import os
import sys
from config import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR, REGIONS,
    FEATURE_COLS, SOLAR_EFFICIENCY
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


def load_raw_data(region: str) -> pd.DataFrame:
    """Load raw CSV for a given region."""
    path = os.path.join(RAW_DATA_DIR, f"{region}.csv")
    df = pd.read_csv(path)
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full preprocessing pipeline:
    1. Parse datetime
    2. Sort chronologically
    3. Replace sentinel values (-999) with NaN
    4. Forward-fill then interpolate missing values
    5. Compute synthetic target variables
    """
    required_cols = {"datetime", *FEATURE_COLS}
    missing_cols = required_cols.difference(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {sorted(missing_cols)}")

    # 1. Parse datetime
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    # 2. Replace NASA POWER sentinel values (-999.0) with NaN
    for col in FEATURE_COLS:
        df[col] = df[col].replace(-999.0, np.nan)

    # 3. Handle missing values
    df[FEATURE_COLS] = df[FEATURE_COLS].ffill()
    df[FEATURE_COLS] = df[FEATURE_COLS].interpolate(method="linear")
    # Drop any remaining NaN rows at the very start
    df = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)

    # 4. Compute synthetic target variables
    # Solar power: irradiance × panel efficiency
    df["solar_power"] = df["solar_irradiance"] * SOLAR_EFFICIENCY
    # Wind power: proportional to cube of wind speed
    df["wind_power"] = df["wind_speed"] ** 3

    return df


def run_preprocessing():
    """Preprocess all regions and save to disk."""
    print("=" * 60)
    print("  STAGE 1: DATA PREPROCESSING")
    print("=" * 60)

    for region in REGIONS:
        print(f"\n▸ Processing {region}...")
        df = load_raw_data(region)
        sentinel_count = int((df[FEATURE_COLS] == -999.0).sum().sum())
        df = preprocess(df)

        out_path = os.path.join(PROCESSED_DATA_DIR, f"{region}_processed.csv")
        df.to_csv(out_path, index=False)

        print(f"  Shape: {df.shape}")
        print(f"  Date range: {df['datetime'].min()} → {df['datetime'].max()}")
        print(f"  Sentinel values replaced: {sentinel_count}")
        print(f"  Missing values: {df[FEATURE_COLS].isnull().sum().sum()}")
        print(f"  Solar power range: {df['solar_power'].min():.2f} – {df['solar_power'].max():.2f}")
        print(f"  Wind power range:  {df['wind_power'].min():.2f} – {df['wind_power'].max():.2f}")
        print(f"  ✓ Saved to {out_path}")

    print("\n✅ Preprocessing complete for all regions.\n")


if __name__ == "__main__":
    run_preprocessing()
