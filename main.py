"""
Main Pipeline Runner
────────────────────
Executes the full renewable energy forecasting pipeline end-to-end:
1. Data Preprocessing
2. Feature Engineering
3. Model Training & Evaluation
4. Visualization
"""

import time
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


def main():
    start = time.time()

    print("╔" + "═" * 58 + "╗")
    print("║  RENEWABLE ENERGY FORECASTING PIPELINE                   ║")
    print("║  Solar & Wind Power · 5 Indian Regions · 2020–2024       ║")
    print("╚" + "═" * 58 + "╝\n")

    # Stage 0: Data Acquisition
    from download_data import run_data_download
    run_data_download()

    # Stage 1: Preprocessing
    from preprocess import run_preprocessing
    run_preprocessing()

    # Stage 2: Feature Engineering
    from feature_engineering import run_feature_engineering
    run_feature_engineering()

    # Stage 3: Model Training
    from models import train_all_models
    results = train_all_models()

    # Stage 4: Visualization
    from visualize import run_visualization
    run_visualization()

    elapsed = time.time() - start
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)

    print("╔" + "═" * 58 + "╗")
    print(f"║  ✅ PIPELINE COMPLETE — {minutes}m {seconds}s" + " " * (34 - len(f"{minutes}m {seconds}s")) + "║")
    print("║                                                          ║")
    print("║  Outputs:                                                ║")
    print("║    📁 processed_data/  — Cleaned & featured datasets     ║")
    print("║    📁 models/          — Trained model files             ║")
    print("║    📁 results/         — Predictions & metrics JSON      ║")
    print("║    📁 plots/           — Visualization images            ║")
    print("║                                                          ║")
    print("║  🚀 Run dashboard:  streamlit run dashboard.py           ║")
    print("╚" + "═" * 58 + "╝")

    # Print summary table
    print("\n── RESULTS SUMMARY ──\n")
    print(f"{'Region':<15} {'Target':<15} {'Model':<20} {'MAE':>8} {'RMSE':>8} {'R²':>8}")
    print("─" * 74)

    for region, region_data in results.items():
        for target, target_data in region_data.items():
            for model, metrics in target_data.items():
                mae = metrics.get("MAE", "N/A")
                rmse = metrics.get("RMSE", "N/A")
                r2 = metrics.get("R2", "N/A")
                print(f"{region:<15} {target:<15} {model:<20} {str(mae):>8} {str(rmse):>8} {str(r2):>8}")
        print()


if __name__ == "__main__":
    main()
