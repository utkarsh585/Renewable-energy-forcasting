# API and Operations Guide

## 1) Build/refresh model manifest
```bash
python model_selection.py
```
Output:
- `results/model_manifest.json`

## 2) Run inference API
```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

Endpoints:
- `GET /health`
- `GET /manifest`
- `GET /predict/latest?region=rajasthan`
- `GET /predict/latest/all`
- `GET /predict/at-datetime?region=rajasthan&generation_datetime=2026-04-19T14:00`

Notes:
- API uses best API-compatible model from `results/model_manifest.json`.
- Predictions are `+1 hour` ahead using the latest row in `processed_data/<region>_featured.csv`.
- For `predict/at-datetime`, the API predicts generation at the selected timestamp using
  feature time = selected timestamp minus 1 hour.
- Real-time custom timestamp predictions use recent/forecast weather from Open-Meteo when
  the selected timestamp is beyond local historical data range.

## 3) Run walk-forward validation
Default (best API model per region/target from manifest):
```bash
python walk_forward_validation.py
```

Quick smoke:
```bash
python walk_forward_validation.py --regions rajasthan --targets solar_power --models "Linear Regression" --folds 2 --test-hours 72
```

Output:
- `results/walk_forward_metrics.json`

## 4) Run full refresh manually
```bash
python refresh_pipeline.py --prepare-deploy-bundle
```

Faster automation run (skip LSTM):
```bash
python refresh_pipeline.py --skip-lstm --prepare-deploy-bundle
```

## 5) Scheduled free automation (GitHub Actions)
Workflow file:
- `.github/workflows/free-refresh.yml`

Behavior:
- Weekly scheduled refresh
- Rebuilds metrics/predictions/manifest and deploy bundle
- Commits updated artifacts automatically
