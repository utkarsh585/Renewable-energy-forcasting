"""
Shared inference utilities for API and dashboard.

Supports:
1) Latest +1h prediction using local engineered dataset
2) Custom generation timestamp prediction using:
   - historical processed data (when in-range)
   - live/recent weather from Open-Meteo for real-time forecasting
"""

from __future__ import annotations

import json
import pickle
from functools import lru_cache
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import requests

from config import (
    FEATURE_COLS,
    FORECAST_HORIZON_HOURS,
    LAG_HOURS,
    PROCESSED_DATA_DIR,
    REGIONS,
    RESULTS_DIR,
    ROLLING_WINDOWS,
)


TARGETS = ["solar_power", "wind_power"]
EXCLUDE_COLS = ["datetime", "solar_power", "wind_power"]
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_TIMEOUT = 45
OPEN_METEO_TZ = "Asia/Kolkata"


@lru_cache(maxsize=1)
def load_manifest() -> Dict:
    path = Path(RESULTS_DIR) / "model_manifest.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing model manifest at {path}. "
            "Run `python model_selection.py` or `python models.py` first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=64)
def load_model(model_path: str):
    with open(model_path, "rb") as f:
        return pickle.load(f)


@lru_cache(maxsize=8)
def load_feature_columns(region: str) -> list[str]:
    path = Path(PROCESSED_DATA_DIR) / f"{region}_featured.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing featured dataset: {path}")
    df = pd.read_csv(path, nrows=1)
    return [c for c in df.columns if c not in EXCLUDE_COLS]


@lru_cache(maxsize=8)
def load_processed_weather(region: str) -> pd.DataFrame:
    path = Path(PROCESSED_DATA_DIR) / f"{region}_processed.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing processed dataset: {path}")
    df = pd.read_csv(path, parse_dates=["datetime"])
    return df[["datetime", *FEATURE_COLS]].sort_values("datetime").reset_index(drop=True)


def _normalize_hour(ts_like) -> pd.Timestamp:
    ts = pd.Timestamp(ts_like)
    if ts.tzinfo is not None:
        # Convert to India local time then drop tz for alignment with stored data.
        ts = ts.tz_convert(OPEN_METEO_TZ).tz_localize(None)
    return ts.floor("h")


@lru_cache(maxsize=16)
def fetch_open_meteo_recent_forecast(region: str) -> pd.DataFrame:
    """
    Pull recent+near-future hourly weather using Open-Meteo.
    We fetch both past and forecast windows to support lag/rolling features.
    """
    lat, lon = REGIONS[region]
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "shortwave_radiation,wind_speed_10m,temperature_2m,relative_humidity_2m",
        "past_days": 7,
        "forecast_days": 16,
        "timezone": OPEN_METEO_TZ,
    }

    resp = requests.get(OPEN_METEO_URL, params=params, timeout=OPEN_METEO_TIMEOUT)
    resp.raise_for_status()
    payload = resp.json()

    if "hourly" not in payload or "time" not in payload["hourly"]:
        raise ValueError("Open-Meteo response missing hourly data.")

    hourly = payload["hourly"]
    df = pd.DataFrame(
        {
            "datetime": pd.to_datetime(hourly["time"]),
            "solar_irradiance": hourly.get("shortwave_radiation"),
            "wind_speed": hourly.get("wind_speed_10m"),
            "temperature": hourly.get("temperature_2m"),
            "humidity": hourly.get("relative_humidity_2m"),
        }
    )

    # Fill occasional missing values safely for feature generation.
    df[FEATURE_COLS] = df[FEATURE_COLS].interpolate(method="linear", limit_direction="both")
    df[FEATURE_COLS] = df[FEATURE_COLS].ffill().bfill()

    return df.sort_values("datetime").reset_index(drop=True)


def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour"] = df["datetime"].dt.hour
    df["day"] = df["datetime"].dt.day
    df["month"] = df["datetime"].dt.month
    df["day_of_week"] = df["datetime"].dt.dayofweek
    df["day_of_year"] = df["datetime"].dt.dayofyear
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    return df


def _add_lag_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in FEATURE_COLS:
        for lag in LAG_HOURS:
            df[f"{col}_lag_{lag}h"] = df[col].shift(lag)
    for col in FEATURE_COLS:
        for window in ROLLING_WINDOWS:
            df[f"{col}_roll_mean_{window}h"] = df[col].rolling(window=window, min_periods=1).mean()
            df[f"{col}_roll_std_{window}h"] = df[col].rolling(window=window, min_periods=1).std()
    return df


def build_feature_row_for_generation_time(
    region: str, generation_timestamp
) -> Tuple[pd.DataFrame, pd.Timestamp, str]:
    """
    Build one model-ready feature row to predict generation at `generation_timestamp`.
    Since model predicts t+1 from features at t, feature_time = generation_time - horizon.
    """
    generation_ts = _normalize_hour(generation_timestamp)
    feature_ts = generation_ts - pd.Timedelta(hours=FORECAST_HORIZON_HOURS)

    hist = load_processed_weather(region)
    hist_end = hist["datetime"].max()

    if feature_ts <= hist_end:
        weather = hist.copy()
        weather_source = "historical_processed_data"
    else:
        weather = fetch_open_meteo_recent_forecast(region)
        weather_source = "open_meteo_recent_forecast"

    weather = weather.sort_values("datetime").drop_duplicates("datetime", keep="last").reset_index(drop=True)
    feat_df = _add_time_features(weather)
    feat_df = _add_lag_rolling_features(feat_df)

    target_row = feat_df[feat_df["datetime"] == feature_ts]
    if target_row.empty:
        available_min = feat_df["datetime"].min()
        available_max = feat_df["datetime"].max()
        raise ValueError(
            "Selected timestamp is outside available inference weather window. "
            f"Requested feature timestamp: {feature_ts}. "
            f"Available: {available_min} to {available_max}."
        )

    return target_row.iloc[[0]], generation_ts, weather_source


def predict_generation(region: str, generation_timestamp) -> Dict:
    if region not in REGIONS:
        raise ValueError(f"Unknown region: {region}")

    manifest = load_manifest()
    if region not in manifest.get("regions", {}):
        raise ValueError(f"Region '{region}' not found in model manifest.")

    row_df, generation_ts, weather_source = build_feature_row_for_generation_time(region, generation_timestamp)
    feature_cols = load_feature_columns(region)

    missing_cols = [c for c in feature_cols if c not in row_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required feature columns: {missing_cols[:10]}")

    X = row_df[feature_cols].copy()
    if X.isnull().any().any():
        bad_cols = X.columns[X.isnull().any()].tolist()
        raise ValueError(
            "Cannot predict because lag/rolling features are incomplete for the selected timestamp. "
            f"Missing columns: {bad_cols[:10]}"
        )

    X_np = X.to_numpy(dtype=float)
    feature_ts = _normalize_hour(generation_ts - pd.Timedelta(hours=FORECAST_HORIZON_HOURS))

    out = {
        "region": region,
        "generation_timestamp": generation_ts.isoformat(),
        "feature_timestamp": feature_ts.isoformat(),
        "forecast_horizon_hours": FORECAST_HORIZON_HOURS,
        "weather_source": weather_source,
        "predictions": {},
    }

    for target in TARGETS:
        cfg = manifest["regions"][region][target]["best_api"]
        model_name = cfg["model"]
        model_path = cfg["artifact_path"]
        model = load_model(model_path)
        y_pred = float(model.predict(X_np)[0])
        # Physical guardrail: generation cannot be negative.
        y_pred = max(0.0, y_pred)
        out["predictions"][target] = {
            "model": model_name,
            "prediction": y_pred,
            "metrics": cfg["metrics"],
        }

    return out


def predict_latest(region: str) -> Dict:
    featured_path = Path(PROCESSED_DATA_DIR) / f"{region}_featured.csv"
    if not featured_path.exists():
        raise FileNotFoundError(f"Missing featured dataset: {featured_path}")
    df = pd.read_csv(featured_path, parse_dates=["datetime"])
    if df.empty:
        raise ValueError(f"No rows in featured dataset for region: {region}")
    latest_feature_time = df.iloc[-1]["datetime"]
    generation_ts = pd.Timestamp(latest_feature_time) + pd.Timedelta(hours=FORECAST_HORIZON_HOURS)
    return predict_generation(region, generation_ts)
