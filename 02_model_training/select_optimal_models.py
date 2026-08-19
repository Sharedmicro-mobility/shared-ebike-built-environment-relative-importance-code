from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select the best city-specific model from four model-output files.")
    parser.add_argument("--metrics-dir", type=Path, required=True, help="Directory containing *_city_metrics.csv files.")
    parser.add_argument("--output-file", type=Path, required=True, help="CSV file for selected optimal models.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = sorted(args.metrics_dir.glob("*_city_metrics.csv"))
    if not files:
        raise FileNotFoundError(f"No *_city_metrics.csv files found in {args.metrics_dir}")

    frames: List[pd.DataFrame] = [pd.read_csv(path) for path in files]
    data = pd.concat(frames, ignore_index=True)
    required = {"city", "model", "test_R2", "test_RMSE", "best_params", "model_path"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Metrics files are missing required columns: {sorted(missing)}")

    selected = (
        data.sort_values(["city", "test_R2", "test_RMSE"], ascending=[True, False, True])
        .groupby("city", as_index=False)
        .first()
    )
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(args.output_file, index=False)
    print(f"Saved selected city-specific models to {args.output_file}")


if __name__ == "__main__":
    main()
