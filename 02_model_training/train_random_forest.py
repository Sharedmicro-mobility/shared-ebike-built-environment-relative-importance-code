from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import optuna
from sklearn.ensemble import RandomForestRegressor

from shared_model_utils import ModelRunConfig, RANDOM_STATE, run_city_model_training


MODEL_NAME = "RF"
N_TRIALS = 80


def suggest_params(trial: optuna.Trial) -> Dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 10, 1500),
        "max_depth": trial.suggest_int("max_depth", 2, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
    }


def make_model(params: Dict) -> RandomForestRegressor:
    model_params = {
        "criterion": "squared_error",
        "bootstrap": True,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }
    model_params.update(params)
    return RandomForestRegressor(**model_params)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train city-specific Random Forest models.")
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
    )
