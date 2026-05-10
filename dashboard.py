"""
Streamlit dashboard for renewable energy forecasting outputs.
"""

import json
import os
from datetime import datetime, timedelta

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import FORECAST_HORIZON_HOURS, PROCESSED_DATA_DIR, REGIONS, RESULTS_DIR
from inference_service import predict_generation


MODEL_NAMES = ["Linear Regression", "Random Forest", "XGBoost", "LSTM"]
MODEL_COLORS = {
    "Linear Regression": "#58a6ff",
    "Random Forest": "#3fb950",
    "XGBoost": "#d29922",
    "LSTM": "#f778ba",
}
TARGETS = ["solar_power", "wind_power"]


def hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert hex color (#RRGGBB) to plotly-friendly rgba(r,g,b,a)."""
    h = hex_color.lstrip("#")
    if len(h) != 6:
        raise ValueError(f"Invalid hex color: {hex_color}")
    r = int(h[0:2], 16)
    g = int(h[2:4], 16)
    b = int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


st.set_page_config(
    page_title="Renewable Energy Forecast",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    * { font-family: 'Inter', sans-serif; }
    .stApp {
        background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0d1117 100%);
    }
    .metric-card {
        background: linear-gradient(145deg, #1a2332, #161b22);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 20px;
        margin: 8px 0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #58a6ff, #3fb950);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .header-glow {
        text-align: center;
        padding: 24px 0;
        background: linear-gradient(135deg, rgba(88,166,255,0.1), rgba(63,185,80,0.1));
        border-radius: 16px;
        border: 1px solid #30363d;
        margin-bottom: 24px;
    }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    f"""
<div class="header-glow">
    <h1 style="margin:0; font-size:2.1rem; background: linear-gradient(135deg, #58a6ff, #3fb950, #d29922);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
        Renewable Energy Forecasting Dashboard
    </h1>
    <p style="color:#8b949e; margin:10px 0 0;">
        Solar and Wind Predictions · NASA POWER · 5 Indian Regions · 2020-2024<br>
        Forecast horizon: +{FORECAST_HORIZON_HOURS} hour(s)
    </p>
</div>
""",
    unsafe_allow_html=True,
)


@st.cache_data
def load_metrics():
    path = os.path.join(RESULTS_DIR, "all_metrics.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


@st.cache_data
def load_manifest():
    path = os.path.join(RESULTS_DIR, "model_manifest.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


@st.cache_data
def load_predictions(region, target):
    path = os.path.join(RESULTS_DIR, f"{region}_{target}_predictions.csv")
    if os.path.exists(path):
        return pd.read_csv(path, parse_dates=["datetime"])
    return None


@st.cache_data
def load_processed(region):
    path = os.path.join(PROCESSED_DATA_DIR, f"{region}_processed.csv")
    if os.path.exists(path):
        return pd.read_csv(path, parse_dates=["datetime"])
    return None


def best_model(manifest, region, target, mode):
    if not manifest:
        return None
    try:
        return manifest["regions"][region][target][mode]["model"]
    except Exception:
        return None


def latest_prediction_row(region, target, model_name):
    df = load_predictions(region, target)
    if df is None or model_name not in df.columns:
        return None
    latest = df[["datetime", "actual", model_name]].dropna(subset=[model_name])
    if latest.empty:
        return None
    return latest.iloc[-1]


metrics = load_metrics()
manifest = load_manifest()

with st.sidebar:
    st.markdown("### Controls")
    region = st.selectbox(
        "Region",
        list(REGIONS.keys()),
        format_func=lambda x: x.replace("_", " ").title(),
    )
    target = st.selectbox(
        "Target",
        TARGETS,
        format_func=lambda x: x.replace("_", " ").title(),
    )

    bo = best_model(manifest, region, target, "best_overall")
    ba = best_model(manifest, region, target, "best_api")
    if bo or ba:
        st.markdown("---")
        st.markdown("### Best Models")
        if bo:
            st.caption(f"Overall ({target}): {bo}")
        if ba:
            st.caption(f"API-ready ({target}): {ba}")

    st.markdown("---")
    st.markdown("### Region Coordinates")
    for name, (lat, lon) in REGIONS.items():
        st.markdown(f"**{name.replace('_', ' ').title()}**")
        st.caption(f"{lat}N, {lon}E")


if metrics and region in metrics and target in metrics[region]:
    st.markdown("### Model Metrics")
    region_metrics = metrics[region][target]
    best_overall = best_model(manifest, region, target, "best_overall")
    cols = st.columns(4)

    for col, model_name in zip(cols, MODEL_NAMES):
        with col:
            m = region_metrics.get(model_name, {})
            r2 = m.get("R2", "N/A")
            mae = m.get("MAE", "N/A")
            rmse = m.get("RMSE", "N/A")
            crown = "BEST " if model_name == best_overall else ""
            st.markdown(
                f"""
<div class="metric-card">
    <div class="metric-label">{crown}{model_name}</div>
    <div class="metric-value">{r2 if r2 != 'N/A' else '-'}</div>
    <div style="color:#8b949e; font-size:0.8rem;">
        R2 Score<br>
        MAE: {mae} · RMSE: {rmse}
    </div>
</div>
""",
                unsafe_allow_html=True,
            )


st.markdown("### Actual vs Predicted")
pred_df = load_predictions(region, target)
if pred_df is not None:
    n_points = st.slider("Time window (hours from end)", 100, min(2000, len(pred_df)), 500)
    df_plot = pred_df.tail(n_points)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df_plot["datetime"],
            y=df_plot["actual"],
            name="Actual",
            mode="lines",
            line=dict(color="#8b949e", width=1),
            opacity=0.7,
        )
    )

    for model_name, color in MODEL_COLORS.items():
        if model_name in df_plot.columns:
            fig.add_trace(
                go.Scatter(
                    x=df_plot["datetime"],
                    y=df_plot[model_name],
                    name=model_name,
                    mode="lines",
                    line=dict(color=color, width=1.2),
                    opacity=0.85,
                )
            )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0d1117",
        plot_bgcolor="#161b22",
        height=450,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(gridcolor="#21262d"),
        yaxis=dict(gridcolor="#21262d", title=target.replace("_", " ").title()),
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No predictions available. Run the pipeline first.")


st.markdown("### Latest +1h Forecast (Best API Models)")
if manifest and region in manifest.get("regions", {}):
    c1, c2 = st.columns(2)
    for col, t in zip([c1, c2], TARGETS):
        with col:
            model_name = best_model(manifest, region, t, "best_api")
            row = latest_prediction_row(region, t, model_name) if model_name else None
            if row is None:
                st.info(f"No latest forecast for {t}.")
            else:
                st.metric(
                    label=f"{t.replace('_', ' ').title()} ({model_name})",
                    value=f"{float(row[model_name]):.2f}",
                    delta=f"Actual: {float(row['actual']):.2f}",
                )
                st.caption(f"Prediction timestamp: {row['datetime']}")
else:
    st.info("Model manifest not available yet. Run model selection/training first.")


st.markdown("### Predict for Selected Date-Time")
st.caption(
    "Choose the generation timestamp you want to forecast. "
    "The model uses weather features from one hour before the selected time."
)

default_dt = pd.Timestamp.now(tz="Asia/Kolkata").floor("h").tz_localize(None) + pd.Timedelta(hours=2)
c_dt1, c_dt2, c_dt3 = st.columns([1, 1, 1.2])
with c_dt1:
    selected_date = st.date_input("Generation Date", value=default_dt.date())
with c_dt2:
    selected_time = st.time_input("Generation Time", value=default_dt.time(), step=timedelta(hours=1))
with c_dt3:
    run_custom = st.button("Run Custom Forecast", use_container_width=True)

if run_custom:
    generation_dt = datetime.combine(selected_date, selected_time)
    with st.spinner("Generating forecast..."):
        try:
            custom_pred = predict_generation(region, generation_dt)
            st.success(
                f"Forecast ready for {custom_pred['generation_timestamp']} "
                f"({custom_pred['region'].replace('_', ' ').title()})"
            )
            st.caption(f"Weather source: {custom_pred['weather_source']}")
            c_pred1, c_pred2 = st.columns(2)
            with c_pred1:
                s = custom_pred["predictions"]["solar_power"]
                st.metric(
                    label=f"Solar Power ({s['model']})",
                    value=f"{s['prediction']:.2f}",
                    delta=f"R2: {s['metrics']['R2']}",
                )
            with c_pred2:
                w = custom_pred["predictions"]["wind_power"]
                st.metric(
                    label=f"Wind Power ({w['model']})",
                    value=f"{w['prediction']:.2f}",
                    delta=f"R2: {w['metrics']['R2']}",
                )
        except Exception as exc:
            st.error(
                "Could not generate custom forecast. "
                f"Reason: {exc}"
            )


st.markdown("### Model Comparison Across Regions")
if metrics:
    tab1, tab2 = st.tabs(["Bar Chart", "Heatmap"])

    with tab1:
        metric_choice = st.radio("Metric", ["R2", "MAE", "RMSE"], horizontal=True)
        chart_data = []
        for r in REGIONS:
            if r in metrics and target in metrics[r]:
                for model_name in MODEL_NAMES:
                    val = metrics[r][target].get(model_name, {}).get(metric_choice)
                    if val is not None:
                        chart_data.append(
                            {
                                "Region": r.replace("_", " ").title(),
                                "Model": model_name,
                                metric_choice: val,
                            }
                        )

        if chart_data:
            chart_df = pd.DataFrame(chart_data)
            fig_bar = px.bar(
                chart_df,
                x="Region",
                y=metric_choice,
                color="Model",
                barmode="group",
                color_discrete_map=MODEL_COLORS,
                template="plotly_dark",
            )
            fig_bar.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#161b22", height=400)
            st.plotly_chart(fig_bar, use_container_width=True)

    with tab2:
        heatmap_data = {}
        for r in REGIONS:
            if r in metrics and target in metrics[r]:
                heatmap_data[r.replace("_", " ").title()] = {
                    model_name: (metrics[r][target].get(model_name, {}).get("R2", 0) or 0)
                    for model_name in MODEL_NAMES
                }
        if heatmap_data:
            heatmap_df = pd.DataFrame(heatmap_data).T
            fig_heat = px.imshow(
                heatmap_df.values,
                x=heatmap_df.columns.tolist(),
                y=heatmap_df.index.tolist(),
                color_continuous_scale="YlGn",
                text_auto=".3f",
                template="plotly_dark",
            )
            fig_heat.update_layout(paper_bgcolor="#0d1117", height=350)
            st.plotly_chart(fig_heat, use_container_width=True)


st.markdown("### Raw Data Explorer")
proc_df = load_processed(region)
if proc_df is not None:
    daily = proc_df.set_index("datetime").resample("D").mean()
    raw_cols = st.columns(2)

    panels = [
        ("solar_irradiance", "#f0883e", "Daily Solar Irradiance"),
        ("wind_speed", "#58a6ff", "Daily Wind Speed"),
        ("temperature", "#3fb950", "Daily Temperature"),
        ("humidity", "#d29922", "Daily Humidity"),
    ]

    for idx, (col_name, color, title) in enumerate(panels):
        with raw_cols[idx % 2]:
            fig_raw = go.Figure()
            fig_raw.add_trace(
                go.Scatter(
                    x=daily.index,
                    y=daily[col_name],
                    fill="tozeroy",
                    fillcolor=hex_to_rgba(color, 0.13),
                    line=dict(color=color, width=1),
                    name=col_name,
                )
            )
            fig_raw.update_layout(
                title=title,
                template="plotly_dark",
                paper_bgcolor="#0d1117",
                plot_bgcolor="#161b22",
                height=300,
                margin=dict(l=20, r=20, t=40, b=20),
            )
            st.plotly_chart(fig_raw, use_container_width=True)

st.markdown("---")
st.caption(
    "Renewable Energy Forecasting Pipeline · NASA POWER API · "
    "Models: Linear Regression, Random Forest, XGBoost, LSTM"
)
