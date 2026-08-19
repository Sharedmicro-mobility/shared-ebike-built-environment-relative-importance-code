from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import optuna
import xgboost as xgb

from shared_model_utils import ModelRunConfig, RANDOM_STATE, run_city_model_training


MODEL_NAME = "XGBoost"
N_TRIALS = 500
EARLY_STOPPING_PATIENCE = 50


def suggest_params(trial: optuna.Trial) -> Dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 10, 1500),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.300, log=True),
        "subsample": trial.suggest_float("subsample", 0.500, 1.000),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.500, 1.000),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 10.0, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.1, 0.5),
    }


def make_model(params: Dict) -> xgb.XGBRegressor:
    model_params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }
    model_params.update(params)
    return xgb.XGBRegressor(**model_params)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train city-specific XGBoost models.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing one processed grid CSV per city.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for fitted models and metrics.")
    parser.add_argument("--file-pattern", default="*.csv")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_city_model_training(
        ModelRunConfig(args.input_dir, args.output_dir, args.file_pattern),
        MODEL_NAME,
        make_model,
        suggest_params,
        n_trials=N_TRIALS,
        early_stop_patience=EARLY_STOPPING_PATIENCE,
    )
