"""
Model Development Module
────────────────────────
• Time-series aware train/test split (no shuffling)
• Models: Linear Regression, Random Forest, XGBoost, LSTM
• Proper scaling for deep learning
• Saves trained models and predictions
"""

import pandas as pd
import numpy as np
import os
import json
import pickle
import warnings
import random
import sys

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config import (
    PROCESSED_DATA_DIR, MODELS_DIR, RESULTS_DIR, REGIONS,
    TEST_SPLIT_RATIO, RANDOM_STATE,
    LSTM_SEQUENCE_LENGTH, LSTM_HIDDEN_SIZE, LSTM_NUM_LAYERS,
    LSTM_DROPOUT, LSTM_EPOCHS, LSTM_BATCH_SIZE, LSTM_LEARNING_RATE,
    FORECAST_HORIZON_HOURS
)
from model_selection import save_model_manifest

warnings.filterwarnings("ignore")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

SKIP_LSTM = os.getenv("SKIP_LSTM", "0") == "1"

# ──────────────────────── Target columns ────────────────────────
TARGETS = ["solar_power", "wind_power"]
EXCLUDE_COLS = ["datetime", "solar_power", "wind_power"]


# ──────────────────────── LSTM Model Definition ────────────────────────
class LSTMModel(nn.Module):
    """LSTM network for time-series regression."""

    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        # Use last time-step output
        out = self.fc(lstm_out[:, -1, :])
        return out


# ──────────────────────── Helper Functions ────────────────────────
def get_feature_columns(df):
    """Return feature column names (everything except datetime and targets)."""
    return [c for c in df.columns if c not in EXCLUDE_COLS]


def time_series_split(df, test_ratio=TEST_SPLIT_RATIO):
    """Split data chronologically — no shuffling."""
    split_idx = int(len(df) * (1 - test_ratio))
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def evaluate(y_true, y_pred):
    """Compute MAE, RMSE, R² metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"MAE": round(mae, 4), "RMSE": round(rmse, 4), "R2": round(r2, 4)}


def create_lstm_sequences(X, y, seq_len):
    """Create sliding window sequences for LSTM."""
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i: i + seq_len])
        ys.append(y[i + seq_len])
    return np.array(Xs), np.array(ys)


def set_random_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ──────────────────────── Training Functions ────────────────────────
def train_sklearn_model(model, X_train, y_train, X_test, y_test):
    """Train and evaluate a scikit-learn style model."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = evaluate(y_test, y_pred)
    return model, y_pred, metrics


def train_lstm(X_train, y_train, X_test, y_test, input_size):
    """Train and evaluate an LSTM model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Scale data
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train_sc = scaler_X.fit_transform(X_train)
    X_test_sc = scaler_X.transform(X_test)
    y_train_sc = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test_sc = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

    # Create sequences
    X_train_seq, y_train_seq = create_lstm_sequences(
        X_train_sc, y_train_sc, LSTM_SEQUENCE_LENGTH
    )
    X_test_seq, y_test_seq = create_lstm_sequences(
        X_test_sc, y_test_sc, LSTM_SEQUENCE_LENGTH
    )

    if len(X_train_seq) == 0 or len(X_test_seq) == 0:
        print("    ⚠ Not enough data for LSTM sequences, skipping...")
        return None, None, {"MAE": None, "RMSE": None, "R2": None}, scaler_X, scaler_y

    # Convert to tensors
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train_seq),
        torch.FloatTensor(y_train_seq),
    )
    train_loader = DataLoader(
        train_dataset, batch_size=LSTM_BATCH_SIZE, shuffle=False
    )

    # Model
    model = LSTMModel(
        input_size=input_size,
        hidden_size=LSTM_HIDDEN_SIZE,
        num_layers=LSTM_NUM_LAYERS,
        dropout=LSTM_DROPOUT,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LSTM_LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )

    # Training loop
    model.train()
    for epoch in range(LSTM_EPOCHS):
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            output = model(batch_X).squeeze()
            loss = criterion(output, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        scheduler.step(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{LSTM_EPOCHS} — Loss: {avg_loss:.6f}")

    # Evaluation
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test_seq).to(device)
        y_pred_sc = model(X_test_tensor).squeeze().cpu().numpy()

    # Inverse scale
    y_pred = scaler_y.inverse_transform(y_pred_sc.reshape(-1, 1)).flatten()
    y_true = scaler_y.inverse_transform(y_test_seq.reshape(-1, 1)).flatten()

    metrics = evaluate(y_true, y_pred)
    return model, y_pred, metrics, scaler_X, scaler_y


# ──────────────────────── Main Training Pipeline ────────────────────────
def train_all_models():
    """Train all models for all regions and both targets."""
    set_random_seed(RANDOM_STATE)

    print("=" * 60)
    print("  STAGE 3: MODEL TRAINING & EVALUATION")
    print("=" * 60)
    print(f"  Forecast horizon: +{FORECAST_HORIZON_HOURS} hour(s)")
    print(f"  LSTM enabled: {not SKIP_LSTM}")

    all_results = {}

    for region in REGIONS:
        print(f"\n{'─' * 50}")
        print(f"  Region: {region.upper()}")
        print(f"{'─' * 50}")

        # Load featured data
        path = os.path.join(PROCESSED_DATA_DIR, f"{region}_featured.csv")
        df = pd.read_csv(path, parse_dates=["datetime"])

        feature_cols = get_feature_columns(df)
        train_df, test_df = time_series_split(df)

        print(f"  Train size: {len(train_df)} | Test size: {len(test_df)}")

        region_results = {}

        for target in TARGETS:
            print(f"\n  ── Target: {target} ──")

            X_train = train_df[feature_cols].values
            y_train = train_df[target].values
            X_test = test_df[feature_cols].values
            y_test = test_df[target].values

            target_results = {}
            predictions = {}

            # 1. Linear Regression
            print("    ▸ Linear Regression...")
            lr_model, lr_pred, lr_metrics = train_sklearn_model(
                LinearRegression(), X_train, y_train, X_test, y_test
            )
            target_results["Linear Regression"] = lr_metrics
            predictions["Linear Regression"] = lr_pred
            print(f"      {lr_metrics}")

            # 2. Random Forest
            print("    ▸ Random Forest...")
            rf_model, rf_pred, rf_metrics = train_sklearn_model(
                RandomForestRegressor(
                    n_estimators=100, max_depth=15,
                    random_state=RANDOM_STATE, n_jobs=-1
                ),
                X_train, y_train, X_test, y_test
            )
            target_results["Random Forest"] = rf_metrics
            predictions["Random Forest"] = rf_pred
            print(f"      {rf_metrics}")

            # 3. XGBoost
            print("    ▸ XGBoost...")
            xgb_model, xgb_pred, xgb_metrics = train_sklearn_model(
                xgb.XGBRegressor(
                    n_estimators=200, max_depth=8, learning_rate=0.05,
                    random_state=RANDOM_STATE, n_jobs=-1,
                    tree_method="hist"
                ),
                X_train, y_train, X_test, y_test
            )
            target_results["XGBoost"] = xgb_metrics
            predictions["XGBoost"] = xgb_pred
            print(f"      {xgb_metrics}")

            # 4. LSTM
            lstm_model = None
            if SKIP_LSTM:
                lstm_metrics = {"MAE": None, "RMSE": None, "R2": None}
                target_results["LSTM"] = lstm_metrics
                print("    ▸ LSTM... skipped (SKIP_LSTM=1)")
                print(f"      {lstm_metrics}")
            else:
                print("    ▸ LSTM...")
                lstm_result = train_lstm(
                    X_train, y_train, X_test, y_test,
                    input_size=len(feature_cols)
                )
                lstm_model, lstm_pred, lstm_metrics = lstm_result[0], lstm_result[1], lstm_result[2]
                target_results["LSTM"] = lstm_metrics
                if lstm_pred is not None:
                    predictions["LSTM"] = lstm_pred
                print(f"      {lstm_metrics}")

            region_results[target] = target_results

            # Save predictions
            pred_df = test_df[["datetime"]].copy().reset_index(drop=True)
            pred_df["datetime"] = pred_df["datetime"] + pd.Timedelta(hours=FORECAST_HORIZON_HOURS)
            pred_df["actual"] = y_test
            for model_name, preds in predictions.items():
                if model_name == "LSTM" and preds is not None:
                    # LSTM predictions are shorter due to sequence length
                    padded = np.full(len(y_test), np.nan)
                    padded[LSTM_SEQUENCE_LENGTH:LSTM_SEQUENCE_LENGTH + len(preds)] = preds
                    pred_df[model_name] = padded
                else:
                    pred_df[model_name] = preds

            pred_path = os.path.join(
                RESULTS_DIR, f"{region}_{target}_predictions.csv"
            )
            pred_df.to_csv(pred_path, index=False)

            # Save sklearn models
            for name, mdl in [("lr", lr_model), ("rf", rf_model), ("xgb", xgb_model)]:
                model_path = os.path.join(
                    MODELS_DIR, f"{region}_{target}_{name}.pkl"
                )
                with open(model_path, "wb") as f:
                    pickle.dump(mdl, f)

            # Save LSTM model
            if lstm_model is not None:
                lstm_path = os.path.join(
                    MODELS_DIR, f"{region}_{target}_lstm.pt"
                )
                torch.save(lstm_model.state_dict(), lstm_path)

        all_results[region] = region_results

    # Save all metrics
    metrics_path = os.path.join(RESULTS_DIR, "all_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✅ All metrics saved to {metrics_path}")

    manifest_path = save_model_manifest(all_results)
    print(f"✅ Model manifest saved to {manifest_path}")

    return all_results


if __name__ == "__main__":
    results = train_all_models()
