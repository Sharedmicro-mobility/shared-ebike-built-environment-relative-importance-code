from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr

sys.path.append(str(Path(__file__).resolve().parents[1] / "02_model_training"))
from shared_model_utils import FEATURE_COLUMNS, read_city_table, relative_mean_abs_shap


def classify_direction(rho: float, p_value: float, min_abs_rho: float = 0.10) -> str:
    if not np.isfinite(rho) or not np.isfinite(p_value) or p_value >= 0.05:
        return "mixed/non-monotonic"
    if rho >= min_abs_rho:
        return "positive"
    if rho <= -min_abs_rho:
        return "negative"
    return "weak directional"


def p_stars(p_value: float) -> str:
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


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
    parser = argparse.ArgumentParser(description="Compute relative SHAP values and SHAP-dependence direction diagnostics.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing processed city CSV files.")
    parser.add_argument("--selected-models", type=Path, required=True, help="CSV produced by select_optimal_models.py.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--file-pattern", default="*.csv")
    parser.add_argument("--min-direction-rho", type=float, default=0.10)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected = pd.read_csv(args.selected_models)
    selected_by_city: Dict[str, pd.Series] = {row["city"]: row for _, row in selected.iterrows()}

    rel_rows: List[Dict] = []
    dir_rows: List[Dict] = []

    for path in sorted(args.input_dir.glob(args.file_pattern)):
        city, X, _ = read_city_table(path)
        if city not in selected_by_city:
            raise ValueError(f"No selected model record found for {city}.")

        model_path = Path(str(selected_by_city[city]["model_path"]))
        model = joblib.load(model_path)
        shap_values = compute_city_shap(model, X)
        rel = relative_mean_abs_shap(shap_values)

        rel_row = {"city": city, "model": selected_by_city[city]["model"]}
        rel_row.update({feature: float(value) for feature, value in zip(FEATURE_COLUMNS, rel)})
        rel_rows.append(rel_row)

        for i, feature in enumerate(FEATURE_COLUMNS):
            rho, p_value = spearmanr(X[feature], shap_values[:, i], nan_policy="omit")
            dir_rows.append(
                {
                    "city": city,
                    "model": selected_by_city[city]["model"],
                    "feature": feature,
                    "spearman_rho": float(rho) if np.isfinite(rho) else np.nan,
                    "p_value": float(p_value) if np.isfinite(p_value) else np.nan,
                    "significance": p_stars(float(p_value)) if np.isfinite(p_value) else "",
                    "direction_class": classify_direction(float(rho), float(p_value), args.min_direction_rho),
                }
            )

    rel_df = pd.DataFrame(rel_rows)
    dir_df = pd.DataFrame(dir_rows)
    summary = (
        dir_df.groupby(["feature", "direction_class"]).size().unstack(fill_value=0)
        .reindex(FEATURE_COLUMNS)
        .reset_index()
    )

    rel_df.to_csv(args.output_dir / "relative_shap_values.csv", index=False)
    dir_df.to_csv(args.output_dir / "shap_directional_spearman.csv", index=False)
    summary.to_csv(args.output_dir / "shap_directional_summary.csv", index=False)
    print(f"Saved SHAP outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
