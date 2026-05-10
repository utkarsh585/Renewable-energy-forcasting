import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

FORECAST_HORIZON_HOURS = 1
REGIONS = {'rajasthan': (26.9124, 70.912), 'tamil_nadu': (8.2572, 77.566), 'odisha': (20.2961, 85.8245), 'maharashtra': (21.1458, 79.0882), 'himachal': (31.1048, 77.1734)}
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
