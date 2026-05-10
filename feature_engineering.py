"""
Feature Engineering Module
──────────────────────────
• Extracts time-based features (hour, day, month, day_of_week)
• Adds cyclical encodings for hour and month
• Creates lag features for all weather variables
• Computes rolling averages for smoothing
• Saves feature-engineered datasets
"""

import pandas as pd
import numpy as np
import os
import sys
from config import (
    PROCESSED_DATA_DIR, REGIONS, FEATURE_COLS,
    LAG_HOURS, ROLLING_WINDOWS, FORECAST_HORIZON_HOURS
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract temporal features from datetime column."""
    df["hour"] = df["datetime"].dt.hour
    df["day"] = df["datetime"].dt.day
    df["month"] = df["datetime"].dt.month
    df["day_of_week"] = df["datetime"].dt.dayofweek
    df["day_of_year"] = df["datetime"].dt.dayofyear

    # Cyclical encoding for hour (0-23) and month (1-12)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create lag features for each weather variable."""
    for col in FEATURE_COLS:
        for lag in LAG_HOURS:
            df[f"{col}_lag_{lag}h"] = df[col].shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling mean and std for smoothing."""
    for col in FEATURE_COLS:
        for window in ROLLING_WINDOWS:
            df[f"{col}_roll_mean_{window}h"] = (
                df[col].rolling(window=window, min_periods=1).mean()
            )
            df[f"{col}_roll_std_{window}h"] = (
                df[col].rolling(window=window, min_periods=1).std()
            )
    return df


def add_forecast_targets(df: pd.DataFrame, horizon_hours: int) -> pd.DataFrame:
    """
    Shift targets forward so each feature row at time t predicts target at t+h.

    Example for h=1:
    - features from 10:00
    - target value from 11:00
    """
    df["solar_power"] = df["solar_power"].shift(-horizon_hours)
    df["wind_power"] = df["wind_power"].shift(-horizon_hours)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Full feature engineering pipeline."""
    df = add_time_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_forecast_targets(df, FORECAST_HORIZON_HOURS)

    # Drop rows with NaN from lag/rolling/target shifting.
    df = df.dropna().reset_index(drop=True)

    return df


def run_feature_engineering():
    """Apply feature engineering to all regions."""
    print("=" * 60)
    print("  STAGE 2: FEATURE ENGINEERING")
    print("=" * 60)
    print(f"  Forecast horizon: +{FORECAST_HORIZON_HOURS} hour(s)")

    for region in REGIONS:
        print(f"\n▸ Engineering features for {region}...")
        path = os.path.join(PROCESSED_DATA_DIR, f"{region}_processed.csv")
        df = pd.read_csv(path, parse_dates=["datetime"])

        df = engineer_features(df)

        out_path = os.path.join(PROCESSED_DATA_DIR, f"{region}_featured.csv")
        df.to_csv(out_path, index=False)

        n_features = len([c for c in df.columns if c not in
                         ["datetime", "solar_power", "wind_power"]])
        print(f"  Shape: {df.shape}")
        print(f"  Total features: {n_features}")
        print("  Target alignment: features at time t -> targets at time t+h")
        print(f"  ✓ Saved to {out_path}")

    print("\n✅ Feature engineering complete for all regions.\n")


if __name__ == "__main__":
    run_feature_engineering()
