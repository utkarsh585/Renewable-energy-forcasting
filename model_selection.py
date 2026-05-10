"""
Model selection utilities.

Builds a manifest that captures:
- best overall model (by metric policy)
- best API-compatible model (sklearn tree/linear models only)
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable
import json

from config import FORECAST_HORIZON_HOURS, MODELS_DIR, RESULTS_DIR


ALL_MODEL_SUFFIX = {
    "Linear Regression": "lr",
    "Random Forest": "rf",
    "XGBoost": "xgb",
    "LSTM": "lstm",
}

# API is intentionally limited to stateless sklearn models.
API_SUPPORTED_MODELS = ["Linear Regression", "Random Forest", "XGBoost"]


def _metric_tuple(metrics: Dict[str, float | None]) -> tuple[float, float, float]:
    """
    Sort key for model selection:
    1) Maximize R2
    2) Minimize RMSE
    3) Minimize MAE
    """
    r2 = metrics.get("R2")
    rmse = metrics.get("RMSE")
    mae = metrics.get("MAE")

    if r2 is None or rmse is None or mae is None:
        return (float("-inf"), float("inf"), float("inf"))
    return (float(r2), float(rmse), float(mae))


def _choose_best(model_metrics: Dict[str, Dict[str, float | None]], allowed: Iterable[str] | None = None) -> str:
    candidates = list(model_metrics.keys()) if allowed is None else [m for m in allowed if m in model_metrics]
    if not candidates:
        raise ValueError("No candidate models available for selection.")

    # max R2, min RMSE, min MAE
    candidates = sorted(
        candidates,
        key=lambda name: (
            -_metric_tuple(model_metrics[name])[0],
            _metric_tuple(model_metrics[name])[1],
            _metric_tuple(model_metrics[name])[2],
        ),
    )
    return candidates[0]


def build_model_manifest(
    all_metrics: Dict[str, Dict[str, Dict[str, Dict[str, float | None]]]]
) -> Dict:
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "forecast_horizon_hours": FORECAST_HORIZON_HOURS,
        "selection_policy": {
            "primary_metric": "R2",
            "tie_breakers": ["RMSE", "MAE"],
        },
        "api_supported_models": API_SUPPORTED_MODELS,
        "regions": {},
    }

    for region, region_data in all_metrics.items():
        manifest["regions"][region] = {}
        for target, target_metrics in region_data.items():
            best_overall = _choose_best(target_metrics)
            best_api = _choose_best(target_metrics, allowed=API_SUPPORTED_MODELS)

            api_suffix = ALL_MODEL_SUFFIX[best_api]
            api_artifact = f"{region}_{target}_{api_suffix}.pkl"

            manifest["regions"][region][target] = {
                "best_overall": {
                    "model": best_overall,
                    "metrics": target_metrics[best_overall],
                    "artifact": (
                        f"{region}_{target}_{ALL_MODEL_SUFFIX[best_overall]}.pt"
                        if best_overall == "LSTM"
                        else f"{region}_{target}_{ALL_MODEL_SUFFIX[best_overall]}.pkl"
                    ),
                },
                "best_api": {
                    "model": best_api,
                    "metrics": target_metrics[best_api],
                    "artifact": api_artifact,
                    "artifact_path": str(Path(MODELS_DIR) / api_artifact),
                },
            }

    return manifest


def save_model_manifest(all_metrics: Dict, output_path: str | Path | None = None) -> Path:
    output = Path(output_path) if output_path is not None else Path(RESULTS_DIR) / "model_manifest.json"
    output.parent.mkdir(parents=True, exist_ok=True)

    manifest = build_model_manifest(all_metrics)
    output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return output


def main() -> None:
    metrics_path = Path(RESULTS_DIR) / "all_metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_path}")

    all_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    out = save_model_manifest(all_metrics)
    print(f"Saved model manifest: {out}")


if __name__ == "__main__":
    main()

