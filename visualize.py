"""
Visualization Module
────────────────────
• Actual vs Predicted plots for each model/region/target
• Model comparison bar charts
• Region-wise performance heatmaps
• Feature importance plots
• Saves all plots to plots/
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import sys

from config import RESULTS_DIR, PLOTS_DIR, REGIONS

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Style configuration
plt.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor": "#161b22",
    "axes.edgecolor": "#30363d",
    "axes.labelcolor": "#c9d1d9",
    "text.color": "#c9d1d9",
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "grid.color": "#21262d",
    "figure.dpi": 120,
    "font.size": 10,
})

COLORS = {
    "Linear Regression": "#58a6ff",
    "Random Forest": "#3fb950",
    "XGBoost": "#d29922",
    "LSTM": "#f778ba",
    "actual": "#8b949e",
}

TARGETS = ["solar_power", "wind_power"]
MODEL_NAMES = ["Linear Regression", "Random Forest", "XGBoost", "LSTM"]


def plot_actual_vs_predicted(region, target, n_points=500):
    """Plot actual vs predicted for all models (zoomed view)."""
    pred_path = os.path.join(RESULTS_DIR, f"{region}_{target}_predictions.csv")
    if not os.path.exists(pred_path):
        return

    df = pd.read_csv(pred_path, parse_dates=["datetime"])

    # Take a representative slice for clarity
    df_plot = df.tail(n_points).reset_index(drop=True)

    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    fig.suptitle(
        f"{region.replace('_', ' ').title()} — {target.replace('_', ' ').title()} "
        f"(Last {n_points} hours)",
        fontsize=16, fontweight="bold", color="#ffffff"
    )

    for idx, model_name in enumerate(MODEL_NAMES):
        ax = axes[idx // 2][idx % 2]
        if model_name not in df_plot.columns:
            ax.text(0.5, 0.5, "Not Available", ha="center", va="center",
                    transform=ax.transAxes, fontsize=14, color="#8b949e")
            ax.set_title(model_name, fontsize=12, color=COLORS.get(model_name, "#fff"))
            continue

        ax.plot(df_plot["datetime"], df_plot["actual"],
                color=COLORS["actual"], alpha=0.6, linewidth=0.8, label="Actual")
        ax.plot(df_plot["datetime"], df_plot[model_name],
                color=COLORS[model_name], alpha=0.85, linewidth=0.8, label=model_name)
        ax.fill_between(df_plot["datetime"], df_plot["actual"], df_plot[model_name],
                        alpha=0.1, color=COLORS[model_name])
        ax.set_title(model_name, fontsize=12, color=COLORS[model_name])
        ax.legend(fontsize=8, loc="upper right")
        ax.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, f"{region}_{target}_actual_vs_pred.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out}")


def plot_model_comparison():
    """Bar charts comparing model performance across regions."""
    metrics_path = os.path.join(RESULTS_DIR, "all_metrics.json")
    if not os.path.exists(metrics_path):
        print("  ⚠ No metrics file found, skipping comparison plot.")
        return

    with open(metrics_path) as f:
        all_metrics = json.load(f)

    for target in TARGETS:
        for metric_name in ["MAE", "RMSE", "R2"]:
            fig, ax = plt.subplots(figsize=(14, 6))

            region_names = []
            model_values = {m: [] for m in MODEL_NAMES}

            for region in REGIONS:
                if region not in all_metrics:
                    continue
                if target not in all_metrics[region]:
                    continue

                region_names.append(region.replace("_", " ").title())
                for model in MODEL_NAMES:
                    val = all_metrics[region][target].get(model, {}).get(metric_name)
                    model_values[model].append(val if val is not None else 0)

            x = np.arange(len(region_names))
            width = 0.18

            for i, model in enumerate(MODEL_NAMES):
                offset = (i - 1.5) * width
                bars = ax.bar(x + offset, model_values[model], width,
                              label=model, color=COLORS[model], alpha=0.85,
                              edgecolor="#30363d", linewidth=0.5)
                # Value labels
                for bar, val in zip(bars, model_values[model]):
                    if val != 0:
                        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                                f"{val:.2f}", ha="center", va="bottom",
                                fontsize=7, color="#c9d1d9")

            ax.set_xlabel("Region", fontsize=12)
            ax.set_ylabel(metric_name, fontsize=12)
            ax.set_title(
                f"{metric_name} Comparison — {target.replace('_', ' ').title()}",
                fontsize=14, fontweight="bold", color="#ffffff"
            )
            ax.set_xticks(x)
            ax.set_xticklabels(region_names, fontsize=10)
            ax.legend(fontsize=9, framealpha=0.3)
            ax.grid(axis="y", alpha=0.3)

            plt.tight_layout()
            out = os.path.join(PLOTS_DIR, f"comparison_{target}_{metric_name}.png")
            fig.savefig(out, bbox_inches="tight")
            plt.close(fig)
            print(f"  ✓ {out}")


def plot_performance_heatmap():
    """Heatmap of R² scores across regions and models."""
    metrics_path = os.path.join(RESULTS_DIR, "all_metrics.json")
    if not os.path.exists(metrics_path):
        return

    with open(metrics_path) as f:
        all_metrics = json.load(f)

    for target in TARGETS:
        data = {}
        for region in REGIONS:
            if region in all_metrics and target in all_metrics[region]:
                data[region.replace("_", " ").title()] = {
                    model: all_metrics[region][target].get(model, {}).get("R2", 0)
                    for model in MODEL_NAMES
                }

        if not data:
            continue

        df_heatmap = pd.DataFrame(data).T

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(
            df_heatmap, annot=True, fmt=".3f", cmap="YlGn",
            linewidths=0.5, linecolor="#30363d",
            cbar_kws={"label": "R² Score"}, ax=ax
        )
        ax.set_title(
            f"R² Score Heatmap — {target.replace('_', ' ').title()}",
            fontsize=14, fontweight="bold", color="#ffffff"
        )
        ax.tick_params(colors="#c9d1d9")

        plt.tight_layout()
        out = os.path.join(PLOTS_DIR, f"heatmap_r2_{target}.png")
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ {out}")


def plot_time_series_overview():
    """Overview of raw data patterns per region."""
    for region in REGIONS:
        path = os.path.join("processed_data", f"{region}_processed.csv")
        if not os.path.exists(path):
            continue

        df = pd.read_csv(path, parse_dates=["datetime"])

        fig, axes = plt.subplots(2, 2, figsize=(18, 10))
        fig.suptitle(
            f"{region.replace('_', ' ').title()} — Weather Data Overview",
            fontsize=16, fontweight="bold", color="#ffffff"
        )

        cols = ["solar_irradiance", "wind_speed", "temperature", "humidity"]
        colors_ts = ["#f0883e", "#58a6ff", "#3fb950", "#d29922"]

        for idx, (col, color) in enumerate(zip(cols, colors_ts)):
            ax = axes[idx // 2][idx % 2]
            # Downsample for visualization (daily mean)
            daily = df.set_index("datetime")[col].resample("D").mean()
            ax.plot(daily.index, daily.values, color=color, alpha=0.8, linewidth=0.7)
            ax.fill_between(daily.index, daily.values, alpha=0.15, color=color)
            ax.set_title(col.replace("_", " ").title(), fontsize=12, color=color)
            ax.grid(alpha=0.2)

        plt.tight_layout()
        out = os.path.join(PLOTS_DIR, f"{region}_data_overview.png")
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ {out}")


def run_visualization():
    """Generate all plots."""
    print("=" * 60)
    print("  STAGE 4: VISUALIZATION")
    print("=" * 60)

    print("\n▸ Time-series overviews...")
    plot_time_series_overview()

    print("\n▸ Actual vs Predicted plots...")
    for region in REGIONS:
        for target in TARGETS:
            plot_actual_vs_predicted(region, target)

    print("\n▸ Model comparison charts...")
    plot_model_comparison()

    print("\n▸ Performance heatmaps...")
    plot_performance_heatmap()

    print("\n✅ All plots saved to plots/\n")


if __name__ == "__main__":
    run_visualization()
