"""
Automated refresh runner for scheduled jobs.

Runs:
1) Data download
2) Preprocessing
3) Feature engineering
4) Model training/evaluation (+ manifest)
5) Visualization
6) Optional deploy-bundle regeneration
"""

from __future__ import annotations

import argparse
import os

from download_data import run_data_download
from preprocess import run_preprocessing
from feature_engineering import run_feature_engineering
from models import train_all_models
from visualize import run_visualization


def run_refresh(skip_download: bool, skip_lstm: bool, prepare_deploy_bundle: bool) -> None:
    if skip_lstm:
        os.environ["SKIP_LSTM"] = "1"

    print("=" * 60)
    print("AUTOMATED PIPELINE REFRESH")
    print("=" * 60)
    print(f"skip_download={skip_download}")
    print(f"skip_lstm={skip_lstm}")
    print(f"prepare_deploy_bundle={prepare_deploy_bundle}")

    if not skip_download:
        run_data_download()
    run_preprocessing()
    run_feature_engineering()
    train_all_models()
    run_visualization()

    if prepare_deploy_bundle:
        from prepare_deploy_bundle import prepare_deploy_bundle as build_bundle

        build_bundle()

    print("\n✅ Refresh complete.\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run periodic refresh for forecasts.")
    parser.add_argument("--skip-download", action="store_true", help="Skip NASA POWER download stage")
    parser.add_argument("--skip-lstm", action="store_true", help="Skip LSTM training for faster automation runs")
    parser.add_argument(
        "--prepare-deploy-bundle",
        action="store_true",
        help="Regenerate deploy_streamlit_free bundle",
    )
    args = parser.parse_args()

    run_refresh(
        skip_download=args.skip_download,
        skip_lstm=args.skip_lstm,
        prepare_deploy_bundle=args.prepare_deploy_bundle,
    )


if __name__ == "__main__":
    main()

