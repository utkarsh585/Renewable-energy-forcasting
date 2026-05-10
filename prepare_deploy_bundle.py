"""
Prepare a lightweight Streamlit deployment bundle for free hosting.

The bundle contains:
- dashboard app (standalone)
- compact daily weather data
- prediction + metrics outputs
- minimal dependency list

Run:
    python prepare_deploy_bundle.py
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

from config import FORECAST_HORIZON_HOURS, REGIONS


BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
PROCESSED_DIR = BASE_DIR / "processed_data"
DEPLOY_DIR = BASE_DIR / "deploy_streamlit_free"
DEPLOY_DATA_RESULTS = DEPLOY_DIR / "data" / "results"
DEPLOY_DATA_DAILY = DEPLOY_DIR / "data" / "daily"

REQUIRED_RESULT_FILES = ["all_metrics.json"] + [
    f"{region}_{target}_predictions.csv"
    for region in REGIONS
    for target in ("solar_power", "wind_power")
]


APP_TEMPLATE = """\
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

FORECAST_HORIZON_HOURS = __FORECAST_HORIZON_HOURS__
REGIONS = __REGIONS__
TARGETS = ["solar_power", "wind_power"]
MODEL_NAMES = ["Linear Regression", "Random Forest", "XGBoost", "LSTM"]

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "data" / "results"
DAILY_DIR = BASE_DIR / "data" / "daily"

st.set_page_config(
    page_title="Renewable Forecast (Free Deploy)",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("⚡ Renewable Energy Forecast Dashboard")
st.caption(
    "India regions · NASA POWER weather data · synthetic targets · "
    f"forecast horizon +{FORECAST_HORIZON_HOURS} hour(s)"
)


@st.cache_data
def load_metrics():
    path = RESULTS_DIR / "all_metrics.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_predictions(region: str, target: str) -> pd.DataFrame:
    path = RESULTS_DIR / f"{region}_{target}_predictions.csv"
    return pd.read_csv(path, parse_dates=["datetime"])


@st.cache_data
def load_daily(region: str) -> pd.DataFrame:
    path = DAILY_DIR / f"{region}_daily.csv"
    return pd.read_csv(path, parse_dates=["date"])


metrics = load_metrics()

with st.sidebar:
    st.subheader("Controls")
    region = st.selectbox(
        "Region",
        options=list(REGIONS.keys()),
        format_func=lambda x: x.replace("_", " ").title(),
    )
    target = st.selectbox(
        "Target",
        options=TARGETS,
        format_func=lambda x: x.replace("_", " ").title(),
    )
    st.markdown("---")
    st.caption("Tip: this deployment bundle is optimized for free hosting.")

if region in metrics and target in metrics[region]:
    st.subheader("Model Metrics")
    cols = st.columns(4)
    for col, model_name in zip(cols, MODEL_NAMES):
        m = metrics[region][target].get(model_name, {})
        col.metric(
            label=model_name,
            value=f"R²: {m.get('R2', 'N/A')}",
            delta=f"MAE: {m.get('MAE', 'N/A')} | RMSE: {m.get('RMSE', 'N/A')}",
        )

st.subheader("Actual vs Predicted")
pred_df = load_predictions(region, target)
n_points = st.slider(
    "Last N hours",
    min_value=100,
    max_value=min(4000, len(pred_df)),
    value=500,
    step=50,
)
plot_df = pred_df.tail(n_points)

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=plot_df["datetime"],
        y=plot_df["actual"],
        mode="lines",
        name="Actual",
        line=dict(width=1.0),
    )
)

model_colors = {
    "Linear Regression": "#1f77b4",
    "Random Forest": "#2ca02c",
    "XGBoost": "#ff7f0e",
    "LSTM": "#9467bd",
}

for model_name, color in model_colors.items():
    if model_name in plot_df.columns:
        fig.add_trace(
            go.Scatter(
                x=plot_df["datetime"],
                y=plot_df[model_name],
                mode="lines",
                name=model_name,
                line=dict(width=1.2, color=color),
            )
        )

fig.update_layout(height=450, margin=dict(l=20, r=20, t=20, b=20))
st.plotly_chart(fig, use_container_width=True)

st.subheader("Cross-Region Comparison")
metric_choice = st.radio("Metric", ["R2", "MAE", "RMSE"], horizontal=True)

rows = []
for r in REGIONS:
    if r in metrics and target in metrics[r]:
        for model_name in MODEL_NAMES:
            value = metrics[r][target].get(model_name, {}).get(metric_choice)
            if value is not None:
                rows.append(
                    {
                        "Region": r.replace("_", " ").title(),
                        "Model": model_name,
                        metric_choice: value,
                    }
                )

if rows:
    comp_df = pd.DataFrame(rows)
    bar = px.bar(
        comp_df,
        x="Region",
        y=metric_choice,
        color="Model",
        barmode="group",
        title=f"{metric_choice} by Region and Model ({target.replace('_', ' ').title()})",
    )
    bar.update_layout(height=420, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(bar, use_container_width=True)

st.subheader("Daily Weather Overview")
daily_df = load_daily(region)
cols = st.columns(2)

for i, col_name in enumerate(["solar_irradiance", "wind_speed", "temperature", "humidity"]):
    with cols[i % 2]:
        f = go.Figure()
        f.add_trace(
            go.Scatter(
                x=daily_df["date"],
                y=daily_df[col_name],
                mode="lines",
                fill="tozeroy",
                name=col_name,
            )
        )
        f.update_layout(
            height=280,
            margin=dict(l=20, r=20, t=30, b=20),
            title=col_name.replace("_", " ").title(),
        )
        st.plotly_chart(f, use_container_width=True)

st.markdown("---")
st.caption(
    "Data source: NASA POWER API · "
    "Models: Linear Regression, Random Forest, XGBoost, LSTM"
)
"""


README_TEMPLATE = """\
# Renewable Forecast - Free Streamlit Deploy Bundle

This folder is generated automatically for free deployment.

## Contents
- `app.py`: standalone Streamlit dashboard
- `data/results/`: model metrics + prediction outputs
- `data/daily/`: compact daily weather files per region
- `requirements.txt`: minimal dependencies for cloud deployment

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy for free (Streamlit Community Cloud)
1. Create a new GitHub repo and upload only the contents of this folder.
2. In Streamlit Community Cloud, click **New app**.
3. Select your repo, branch, and `app.py`.
4. Click **Deploy**.
"""


REQUIREMENTS_TEXT = """\
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.18.0
"""


def validate_input_artifacts() -> None:
    missing = [name for name in REQUIRED_RESULT_FILES if not (RESULTS_DIR / name).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing result files required for deployment bundle:\n- "
            + "\n- ".join(missing)
        )

    for region in REGIONS:
        processed_path = PROCESSED_DIR / f"{region}_processed.csv"
        if not processed_path.exists():
            raise FileNotFoundError(f"Missing processed file: {processed_path}")


def build_daily_data() -> None:
    for region in REGIONS:
        processed_path = PROCESSED_DIR / f"{region}_processed.csv"
        df = pd.read_csv(processed_path, parse_dates=["datetime"])

        daily = (
            df.set_index("datetime")[["solar_irradiance", "wind_speed", "temperature", "humidity"]]
            .resample("D")
            .mean()
            .reset_index()
            .rename(columns={"datetime": "date"})
        )

        out_path = DEPLOY_DATA_DAILY / f"{region}_daily.csv"
        daily.to_csv(out_path, index=False)


def copy_results_data() -> None:
    for filename in REQUIRED_RESULT_FILES:
        src = RESULTS_DIR / filename
        dst = DEPLOY_DATA_RESULTS / filename
        shutil.copy2(src, dst)


def write_bundle_files() -> None:
    app_text = APP_TEMPLATE.replace(
        "__FORECAST_HORIZON_HOURS__", str(FORECAST_HORIZON_HOURS)
    ).replace(
        "__REGIONS__", repr(REGIONS)
    )
    (DEPLOY_DIR / "app.py").write_text(app_text, encoding="utf-8")
    (DEPLOY_DIR / "README.md").write_text(README_TEMPLATE, encoding="utf-8")
    (DEPLOY_DIR / "requirements.txt").write_text(REQUIREMENTS_TEXT, encoding="utf-8")


def write_manifest() -> None:
    metrics_path = DEPLOY_DATA_RESULTS / "all_metrics.json"
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    best_models = {}
    for region, region_data in metrics.items():
        best_models[region] = {}
        for target, target_data in region_data.items():
            best_name = max(target_data, key=lambda model_name: target_data[model_name]["R2"])
            best_models[region][target] = {
                "model": best_name,
                "metrics": target_data[best_name],
            }

    manifest = {
        "forecast_horizon_hours": FORECAST_HORIZON_HOURS,
        "regions": list(REGIONS.keys()),
        "best_models": best_models,
    }
    (DEPLOY_DIR / "model_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


def prepare_deploy_bundle() -> None:
    validate_input_artifacts()

    if DEPLOY_DIR.exists():
        shutil.rmtree(DEPLOY_DIR)

    DEPLOY_DATA_RESULTS.mkdir(parents=True, exist_ok=True)
    DEPLOY_DATA_DAILY.mkdir(parents=True, exist_ok=True)

    copy_results_data()
    build_daily_data()
    write_bundle_files()
    write_manifest()

    print("=" * 60)
    print("DEPLOY BUNDLE READY")
    print("=" * 60)
    print(f"Path: {DEPLOY_DIR}")
    print("Included:")
    print("- app.py")
    print("- requirements.txt (minimal)")
    print("- data/results/*.csv + all_metrics.json")
    print("- data/daily/*.csv")
    print("- model_manifest.json")
    print("- README.md")


if __name__ == "__main__":
    prepare_deploy_bundle()
