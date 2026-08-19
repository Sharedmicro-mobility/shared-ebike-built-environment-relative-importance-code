from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.base import clone

sys.path.append(str(Path(__file__).resolve().parents[1] / "02_model_training"))
from shared_model_utils import FEATURE_COLUMNS, RANDOM_STATE, TARGET_COLUMN, stratified_cv_splits
from train_gbdt import make_model as make_gbdt
from train_lightgbm import make_model as make_lightgbm
from train_random_forest import make_model as make_rf
from train_xgboost import make_model as make_xgboost


def model_from_name(model_name: str, params: Dict):
    if model_name == "RF":
        return make_rf(params)
    if model_name == "GBDT":
        return make_gbdt(params)
    if model_name == "XGBoost":
        return make_xgboost(params)
    if model_name == "LightGBM":
        return make_lightgbm(params)
    raise ValueError(f"Unsupported model name: {model_name}")


def read_city_geodata(path: Path) -> tuple[str, gpd.GeoDataFrame]:
    city = path.stem
    data = gpd.read_file(path)
    missing = [c for c in FEATURE_COLUMNS + [TARGET_COLUMN, "geometry"] if c not in data.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    data = data[FEATURE_COLUMNS + [TARGET_COLUMN, "geometry"]].copy()
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=[TARGET_COLUMN, "geometry"])
    data[FEATURE_COLUMNS] = data[FEATURE_COLUMNS].fillna(0)
    data = data[data[TARGET_COLUMN] > 7].copy()
    if len(data) < 100:
        raise ValueError(f"{city} has fewer than 100 effective grids after SER > 7 filtering.")
    return city, data


def out_of_fold_residuals(model, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
    residuals = pd.Series(index=y.index, dtype=float)
    for train_idx, valid_idx in stratified_cv_splits(X, y, random_state=RANDOM_STATE):
        fitted = clone(model)
        fitted.fit(X.iloc[train_idx], y.iloc[train_idx])
        predictions = fitted.predict(X.iloc[valid_idx])
        residuals.iloc[valid_idx] = y.iloc[valid_idx].to_numpy() - predictions
    if residuals.isna().any():
        raise RuntimeError("Out-of-fold residual calculation produced missing values.")
    return residuals.to_numpy()


def moran_from_values(values: np.ndarray, geodata: gpd.GeoDataFrame, permutations: int):
    try:
        from esda.moran import Moran
        from libpysal.weights import KNN
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Residual spatial autocorrelation diagnostics require the optional "
            "spatial-analysis dependencies 'esda' and 'libpysal'. Install the "
            "public-code environment with: pip install -r requirements.txt"
        ) from exc

    centroids = geodata.geometry.centroid
    coords = np.column_stack([centroids.x.to_numpy(), centroids.y.to_numpy()])
    weights = KNN.from_array(coords, k=8)
    weights.transform = "R"
    return Moran(values, weights, permutations=permutations)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate Global Moran's I for observed ridership and out-of-fold residuals."
    )
    parser.add_argument("--grid-dir", type=Path, required=True, help="Directory containing one processed grid geospatial file per city.")
    parser.add_argument("--selected-models", type=Path, required=True, help="CSV produced by select_optimal_models.py.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--file-pattern", default="*.gpkg")
    parser.add_argument("--permutations", type=int, default=999)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected = pd.read_csv(args.selected_models)
    selected_by_city = {row["city"]: row for _, row in selected.iterrows()}

    rows: List[Dict] = []
    for path in sorted(args.grid_dir.glob(args.file_pattern)):
        city, data = read_city_geodata(path)
        if city not in selected_by_city:
            raise ValueError(f"No selected model record found for {city}.")

        record = selected_by_city[city]
        params = json.loads(record["best_params"])
        model = model_from_name(record["model"], params)

        X = data[FEATURE_COLUMNS].copy()
        y = data[TARGET_COLUMN].copy()
        residuals = out_of_fold_residuals(model, X, y)

        observed_moran = moran_from_values(y.to_numpy(), data, args.permutations)
        residual_moran = moran_from_values(residuals, data, args.permutations)
        rows.append(
            {
                "city": city,
                "model": record["model"],
                "n_grids": len(data),
                "observed_moran_i": float(observed_moran.I),
                "observed_p_sim": float(observed_moran.p_sim),
                "residual_moran_i": float(residual_moran.I),
                "residual_p_sim": float(residual_moran.p_sim),
                "residual_mean": float(np.mean(residuals)),
                "residual_sd": float(np.std(residuals, ddof=1)),
            }
        )

    pd.DataFrame(rows).to_csv(args.output_dir / "residual_spatial_autocorrelation.csv", index=False)
    print(f"Saved spatial autocorrelation diagnostics to {args.output_dir}")


if __name__ == "__main__":
    main()
