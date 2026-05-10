"""
Configuration module for the Renewable Energy Forecasting Pipeline.
Centralizes all constants, paths, and hyperparameters.
"""

import os

# ──────────────────────────── Paths ────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_DIR = os.path.join(BASE_DIR, "nasa_power_data")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "processed_data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")

for d in [PROCESSED_DATA_DIR, MODELS_DIR, RESULTS_DIR, PLOTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ──────────────────────────── Regions ────────────────────────────
REGIONS = {
    "rajasthan": (26.9124, 70.9120),
    "tamil_nadu": (8.2572, 77.5660),
    "odisha": (20.2961, 85.8245),
    "maharashtra": (21.1458, 79.0882),
    "himachal": (31.1048, 77.1734),
}

# ──────────────────────────── NASA POWER API ────────────────────────────
API_PARAMETERS = "ALLSKY_SFC_SW_DWN,WS10M,T2M,RH2M"
API_START_DATE = "20200101"
API_END_DATE = "20241231"
API_COMMUNITY = "RE"
API_TIMEOUT_SECONDS = 60

# Column mapping from API names to readable names
COLUMN_RENAME = {
    "ALLSKY_SFC_SW_DWN": "solar_irradiance",
    "WS10M": "wind_speed",
    "T2M": "temperature",
    "RH2M": "humidity",
}

FEATURE_COLS = ["solar_irradiance", "wind_speed", "temperature", "humidity"]

# ──────────────────────────── Target Variables ────────────────────────────
# Synthetic power formulas
SOLAR_EFFICIENCY = 0.2  # solar_power = solar_irradiance * 0.2
# wind_power = wind_speed ** 3

# ──────────────────────────── Feature Engineering ────────────────────────────
LAG_HOURS = [1, 2, 3, 6, 12, 24]
ROLLING_WINDOWS = [3, 6, 12, 24]
FORECAST_HORIZON_HOURS = 1  # Predict one hour ahead: target at t+1 using features at t

# ──────────────────────────── Model Training ────────────────────────────
TEST_SPLIT_RATIO = 0.2  # Last 20% of data for testing (time-series aware)
RANDOM_STATE = 42

# LSTM hyperparameters
LSTM_SEQUENCE_LENGTH = 24  # 24 hours lookback
LSTM_HIDDEN_SIZE = 64
LSTM_NUM_LAYERS = 2
LSTM_DROPOUT = 0.2
LSTM_EPOCHS = 30
LSTM_BATCH_SIZE = 64
LSTM_LEARNING_RATE = 0.001
