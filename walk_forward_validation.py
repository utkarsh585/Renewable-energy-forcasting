"""
Walk-forward validation for stronger time-series reliability checks.

Examples:
    python walk_forward_validation.py
    python walk_forward_validation.py --regions rajasthan --targets solar_power --models "Linear Regression" --folds 2 --test-hours 72
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb

from config import PROCESSED_DATA_DIR, RANDOM_STATE, REGIONS, RESULTS_DIR


EXCLUDE_COLS = ["datetime", "solar_power", "wind_power"]
ALL_TARGETS = ["solar_power", "wind_power"]
SUPPORTED_MODELS = ["Linear Regression", "Random Forest", "XGBoost"]


def build_model(model_name: str):
    if model_name == "Linear Regression":
        return LinearRegression()
    if model_name == "Random Forest":
        return RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    if model_name == "XGBoost":
        return xgb.XGBRegressor(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.05,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            tree_method="hist",
        )
    raise ValueError(f"Unsupported walk-forward model: {model_name}")


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def aggregate_metrics(fold_metrics: List[Dict[str, float]]) -> Dict[str, float]:
    out = {}
    for key in ["MAE", "RMSE", "R2"]:
        vals = [m[key] for m in fold_metrics]
        out[f"{key}_mean"] = round(mean(vals), 6)
        out[f"{key}_std"] = round(pstdev(vals), 6)
    return out


def load_manifest_models() -> Dict[str, Dict[str, str]]:
    manifest_path = Path(RESULTS_DIR) / "model_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing {manifest_path}. Run training or `python model_selection.py` first."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    out: Dict[str, Dict[str, str]] = {}
    for region, region_data in manifest["regions"].items():
        out[region] = {}
        for target, target_data in region_data.items():
            out[region][target] = target_data["best_api"]["model"]
    return out


def run_walk_forward(
    regions: List[str],
    targets: List[str],
    folds: int,
    test_hours: int,
    override_models: List[str] | None,
) -> Dict:
    manifest_models = load_manifest_models()
    results = {}

    for region in regions:
        path = Path(PROCESSED_DATA_DIR) / f"{region}_featured.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing featured dataset: {path}")

        df = pd.read_csv(path, parse_dates=["datetime"])
        feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
        X = df[feature_cols].to_numpy()

        results[region] = {}
        for target in targets:
            y = df[target].to_numpy()
            target_out = {}

            models_to_run = override_models if override_models else [manifest_models[region][target]]
            for model_name in models_to_run:
                if model_name not in SUPPORTED_MODELS:
                    continue

                min_required = (folds + 1) * test_hours
                if len(df) <= min_required:
                    raise ValueError(
                        f"{region}/{target}: Not enough rows ({len(df)}) for folds={folds}, test_hours={test_hours}"
                    )

                splitter = TimeSeriesSplit(n_splits=folds, test_size=test_hours)
                fold_metrics = []

                for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(X), start=1):
                    model = build_model(model_name)
                    model.fit(X[train_idx], y[train_idx])
                    y_pred = model.predict(X[test_idx])
                    metrics = evaluate(y[test_idx], y_pred)
                    metrics["fold"] = fold_idx
                    fold_metrics.append(metrics)

                target_out[model_name] = {
                    "folds": fold_metrics,
                    "summary": aggregate_metrics(fold_metrics),
                }

            results[region][target] = target_out

    return results


def parse_csv_list(raw: str | None, default: List[str]) -> List[str]:
    if raw is None or not raw.strip():
        return default
    return [s.strip() for s in raw.split(",") if s.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward backtesting for renewable forecasts.")
    parser.add_argument("--regions", type=str, default=None, help="Comma-separated region keys")
    parser.add_argument("--targets", type=str, default=None, help="Comma-separated targets")
    parser.add_argument("--models", type=str, default=None, help="Comma-separated models (default: best_api from manifest)")
    parser.add_argument("--folds", type=int, default=3, help="Number of walk-forward splits")
    parser.add_argument("--test-hours", type=int, default=24 * 7, help="Test window size (hours) per fold")
    args = parser.parse_args()

    regions = parse_csv_list(args.regions, list(REGIONS.keys()))
    targets = parse_csv_list(args.targets, ALL_TARGETS)
    models = parse_csv_list(args.models, []) or None

    if models:
        bad = [m for m in models if m not in SUPPORTED_MODELS]
        if bad:
            raise ValueError(f"Unsupported models requested: {bad}")

    print("=" * 60)
    print("WALK-FORWARD VALIDATION")
    print("=" * 60)
    print(f"Regions: {regions}")
    print(f"Targets: {targets}")
    print(f"Folds: {args.folds}")
    print(f"Test hours/fold: {args.test_hours}")
    print(f"Models: {models if models else 'best_api from manifest'}")

    results = run_walk_forward(
        regions=regions,
        targets=targets,
        folds=args.folds,
        test_hours=args.test_hours,
        override_models=models,
    )

    out_path = Path(RESULTS_DIR) / "walk_forward_metrics.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved walk-forward metrics to {out_path}")


if __name__ == "__main__":
    main()

