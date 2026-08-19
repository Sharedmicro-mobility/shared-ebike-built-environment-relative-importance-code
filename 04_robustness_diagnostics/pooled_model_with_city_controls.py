from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import joblib
import numpy as np
import optuna
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "02_model_training"))
from shared_model_utils import (
    CITY_COLUMN,
    FEATURE_COLUMNS,
    RANDOM_STATE,
    TARGET_COLUMN,
    NoImprovementStopper,
    regression_metrics,
    split_train_test,
)
from train_gbdt import make_model as make_gbdt
from train_gbdt import suggest_params as suggest_gbdt
from train_lightgbm import make_model as make_lightgbm
from train_lightgbm import suggest_params as suggest_lightgbm
from train_random_forest import make_model as make_rf
from train_random_forest import suggest_params as suggest_rf
from train_xgboost import make_model as make_xgboost
from train_xgboost import suggest_params as suggest_xgboost


MODEL_SPECS: Dict[str, Tuple[Callable[[Dict], object], Callable[[optuna.Trial], Dict], int, Optional[int]]] = {
    "RF": (make_rf, suggest_rf, 80, None),
    "GBDT": (make_gbdt, suggest_gbdt, 500, 50),
    "XGBoost": (make_xgboost, suggest_xgboost, 500, 50),
    "LightGBM": (make_lightgbm, suggest_lightgbm, 500, 50),
}


def read_city_files(input_dir: Path, file_pattern: str) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path in sorted(input_dir.glob(file_pattern)):
        city = path.stem
        data = pd.read_csv(path)
        missing = [c for c in FEATURE_COLUMNS + [TARGET_COLUMN] if c not in data.columns]
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}")
        data = data[FEATURE_COLUMNS + [TARGET_COLUMN]].copy()
        data[CITY_COLUMN] = city
        frames.append(data)
    if not frames:
        raise FileNotFoundError(f"No input files matched {input_dir / file_pattern}")
    return pd.concat(frames, ignore_index=True)


def read_combined_file(path: Path, city_column: str) -> pd.DataFrame:
    data = pd.read_csv(path)
    missing = [c for c in FEATURE_COLUMNS + [TARGET_COLUMN, city_column] if c not in data.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    out = data[FEATURE_COLUMNS + [TARGET_COLUMN, city_column]].copy()
    if city_column != CITY_COLUMN:
        out = out.rename(columns={city_column: CITY_COLUMN})
    return out


def clean_pooled_data(data: pd.DataFrame) -> pd.DataFrame:
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=[TARGET_COLUMN, CITY_COLUMN]).copy()
    data[FEATURE_COLUMNS] = data[FEATURE_COLUMNS].fillna(0)
    data = data[data[TARGET_COLUMN] > 7].copy()

    counts = data.groupby(CITY_COLUMN).size()
    too_small = counts[counts < 100]
    if not too_small.empty:
        names = ", ".join(too_small.index.astype(str))
        raise ValueError(
            "The manuscript retained only cities with at least 100 effective grids. "
            f"The following cities do not meet this criterion after SER > 7 filtering: {names}"
        )
    return data


def make_design_matrix(data: pd.DataFrame) -> pd.DataFrame:
    city_dummies = pd.get_dummies(data[CITY_COLUMN], prefix="city", drop_first=True, dtype=float)
    return pd.concat([data[FEATURE_COLUMNS].reset_index(drop=True), city_dummies.reset_index(drop=True)], axis=1)


def mean_cv_rmse_for_pooled(
    model_factory: Callable[[Dict], object],
    params: Dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> float:
    from shared_model_utils import rmse, stratified_cv_splits

    scores: List[float] = []
    for train_idx, valid_idx in stratified_cv_splits(X_train, y_train):
        model = model_factory(params)
        model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
        scores.append(rmse(y_train.iloc[valid_idx], model.predict(X_train.iloc[valid_idx])))
    return float(np.mean(scores))


def train_one_model(
    model_name: str,
    model_factory: Callable[[Dict], object],
    suggest_params: Callable[[optuna.Trial], Dict],
    n_trials: int,
    early_stop_patience: Optional[int],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> Dict:
    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial)
        return mean_cv_rmse_for_pooled(model_factory, params, X_train, y_train)

    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    callbacks = [NoImprovementStopper(early_stop_patience)] if early_stop_patience else None
    study.optimize(objective, n_trials=n_trials, callbacks=callbacks, show_progress_bar=False)

    best_params = dict(study.best_params)
    model = model_factory(best_params)
    model.fit(X_train, y_train)
    train = regression_metrics(model, X_train, y_train)
    test = regression_metrics(model, X_test, y_test)

    return {
        "model_name": model_name,
        "model": model,
        "best_params": best_params,
        "cv_rmse": float(study.best_value),
        "trials_run": len(study.trials),
        "train_R2": train["R2"],
        "test_R2": test["R2"],
        "train_RMSE": train["RMSE"],
        "test_RMSE": test["RMSE"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train pooled robustness models with city dummy controls."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input-dir", type=Path, help="Directory containing one processed grid CSV per city.")
    input_group.add_argument("--combined-csv", type=Path, help="Single processed CSV containing all cities.")
    parser.add_argument("--file-pattern", default="*.csv")
    parser.add_argument("--city-column", default=CITY_COLUMN)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.input_dir:
        data = read_city_files(args.input_dir, args.file_pattern)
    else:
        data = read_combined_file(args.combined_csv, args.city_column)
    data = clean_pooled_data(data)

    X = make_design_matrix(data)
    y = data[TARGET_COLUMN].reset_index(drop=True)
    X_train, X_test, y_train, y_test = split_train_test(X, y, RANDOM_STATE)

    rows: List[Dict] = []
    best_result: Dict | None = None
    for model_name, (factory, suggest, n_trials, patience) in MODEL_SPECS.items():
        result = train_one_model(
            model_name,
            factory,
            suggest,
            n_trials,
            patience,
            X_train,
            X_test,
            y_train,
            y_test,
        )
        rows.append(
            {
                "model": model_name,
                "n_total": len(X),
                "n_train": len(X_train),
                "n_test": len(X_test),
                "cv_rmse": result["cv_rmse"],
                "train_R2": result["train_R2"],
                "test_R2": result["test_R2"],
                "train_RMSE": result["train_RMSE"],
                "test_RMSE": result["test_RMSE"],
                "trials_run": result["trials_run"],
                "best_params": json.dumps(result["best_params"], ensure_ascii=False),
            }
        )
        if best_result is None or (
            result["test_R2"],
            -result["test_RMSE"],
        ) > (
            best_result["test_R2"],
            -best_result["test_RMSE"],
        ):
            best_result = result

    metrics = pd.DataFrame(rows)
    metrics.to_csv(args.output_dir / "pooled_model_metrics.csv", index=False)

    assert best_result is not None
    joblib.dump(best_result["model"], args.output_dir / "best_pooled_model.joblib")
    pd.DataFrame(
        [
            {
                "selected_model": best_result["model_name"],
                "test_R2": best_result["test_R2"],
                "test_RMSE": best_result["test_RMSE"],
                "best_params": json.dumps(best_result["best_params"], ensure_ascii=False),
            }
        ]
    ).to_csv(args.output_dir / "selected_pooled_model.csv", index=False)
    print(f"Saved pooled robustness outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
