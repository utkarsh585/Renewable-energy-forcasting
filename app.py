"""
FastAPI inference service for renewable forecasting.

Run:
    uvicorn api:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query

from config import FORECAST_HORIZON_HOURS, REGIONS
from inference_service import load_manifest, predict_generation, predict_latest


APP_TITLE = "Renewable Forecast API"
APP_VERSION = "1.1.0"

app = FastAPI(title=APP_TITLE, version=APP_VERSION)


@app.get("/")
def root():
    return {
        "message": "Renewable Forecast API is running.",
        "docs": "/docs",
        "health": "/health",
        "predict_latest": "/predict/latest?region=rajasthan",
        "predict_at_datetime": "/predict/at-datetime?region=rajasthan&generation_datetime=2026-04-19T14:00",
    }


@app.get("/health")
def health():
    try:
        manifest = load_manifest()
    except Exception as exc:
        return {"status": "degraded", "error": str(exc)}

    return {
        "status": "ok",
        "app": APP_TITLE,
        "version": APP_VERSION,
        "regions": list(manifest.get("regions", {}).keys()),
        "forecast_horizon_hours": FORECAST_HORIZON_HOURS,
    }


@app.get("/manifest")
def manifest():
    try:
        return load_manifest()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/predict/latest")
def predict_latest_endpoint(
    region: str = Query(..., description="Region key (e.g., rajasthan, tamil_nadu)")
):
    if region not in REGIONS:
        raise HTTPException(status_code=400, detail=f"Unknown region: {region}")

    try:
        return predict_latest(region)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/predict/latest/all")
def predict_latest_all():
    results = {}
    errors = {}

    for region in REGIONS:
        try:
            results[region] = predict_latest(region)
        except Exception as exc:
            errors[region] = str(exc)

    return {
        "forecast_horizon_hours": FORECAST_HORIZON_HOURS,
        "results": results,
        "errors": errors,
    }


@app.get("/predict/at-datetime")
def predict_at_datetime(
    region: str = Query(..., description="Region key"),
    generation_datetime: str = Query(
        ...,
        description="Generation timestamp in ISO format (e.g., 2026-04-19T14:00)",
    ),
):
    if region not in REGIONS:
        raise HTTPException(status_code=400, detail=f"Unknown region: {region}")

    try:
        return predict_generation(region, generation_datetime)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

