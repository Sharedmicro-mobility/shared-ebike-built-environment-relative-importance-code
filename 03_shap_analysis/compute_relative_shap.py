from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
import shap

sys.path.append(str(Path(__file__).resolve().parents[1] / "02_model_training"))
from shared_model_utils import FEATURE_COLUMNS, read_city_table, relative_mean_abs_shap


def compute_city_shap(model, X: pd.DataFrame) -> np.ndarray:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    shap_values = np.asarray(shap_values)
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 0]
    return shap_values


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute city-level relative SHAP values.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing processed city CSV files.")
    parser.add_argument("--selected-models", type=Path, required=True, help="CSV produced by select_optimal_models.py.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--file-pattern", default="*.csv")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected = pd.read_csv(args.selected_models)
    required = {"city", "model", "model_path"}
    missing = required - set(selected.columns)
    if missing:
        raise ValueError(f"{args.selected_models} is missing required columns: {sorted(missing)}")
    selected_by_city: Dict[str, pd.Series] = {row["city"]: row for _, row in selected.iterrows()}

    rows: List[Dict] = []
    for path in sorted(args.input_dir.glob(args.file_pattern)):
        city, X, _ = read_city_table(path)
        if city not in selected_by_city:
            raise ValueError(f"No selected model record found for {city}.")

        record = selected_by_city[city]
        model_path = Path(str(record["model_path"]))
        model = joblib.load(model_path)
        shap_values = compute_city_shap(model, X)
        rel = relative_mean_abs_shap(shap_values)

        row = {"city": city, "model": record["model"]}
        row.update({feature: float(value) for feature, value in zip(FEATURE_COLUMNS, rel)})
        rows.append(row)

    pd.DataFrame(rows).to_csv(args.output_dir / "relative_shap_values.csv", index=False)
    print(f"Saved relative SHAP values to {args.output_dir}")


if __name__ == "__main__":
    main()
