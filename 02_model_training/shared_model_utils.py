from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import StratifiedKFold, train_test_split


FEATURE_COLUMNS = ["PD", "LUM", "CBD", "LEC", "REC", "RC", "EC", "TSC", "CLL", "IC"]
TARGET_COLUMN = "SER"
CITY_COLUMN = "city"

RANDOM_STATE = 42
TEST_SIZE = 0.10
N_STRATA = 5
CV_FOLDS = 10


@dataclass(frozen=True)
class ModelRunConfig:
    input_dir: Path
    output_dir: Path
    file_pattern: str = "*.csv"
    n_jobs: int = -1
    random_state: int = RANDOM_STATE


class NoImprovementStopper:
    """Stop an Optuna study after a fixed number of non-improving trials."""

    def __init__(self, patience: int) -> None:
        self.patience = patience
        self.best_value: Optional[float] = None
        self.best_trial_number = 0

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if trial.value is None:
            return
        if self.best_value is None or trial.value < self.best_value:
            self.best_value = float(trial.value)
            self.best_trial_number = trial.number
        elif trial.number - self.best_trial_number >= self.patience:
            study.stop()


def infer_city_name(path: Path, data: pd.DataFrame) -> str:
    if CITY_COLUMN in data.columns:
        values = data[CITY_COLUMN].dropna().astype(str).unique()
        if len(values) == 1:
            return str(values[0])
        if len(values) > 1:
            raise ValueError(
                f"{path} contains multiple city values. The public workflow expects "
                "one processed grid-level file per city."
            )
    return path.stem


def read_city_table(path: Path) -> Tuple[str, pd.DataFrame, pd.Series]:
    data = pd.read_csv(path)
    city = infer_city_name(path, data)
    missing = [c for c in FEATURE_COLUMNS + [TARGET_COLUMN] if c not in data.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    data = data[FEATURE_COLUMNS + [TARGET_COLUMN]].copy()
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=[TARGET_COLUMN])
    data[FEATURE_COLUMNS] = data[FEATURE_COLUMNS].fillna(0)
    data = data[data[TARGET_COLUMN] > 7].copy()
    if len(data) < 100:
        raise ValueError(
            f"{city} has {len(data)} effective grids after SER > 7 filtering; "
            "the manuscript retained cities with at least 100 effective grids."
        )
    return city, data[FEATURE_COLUMNS], data[TARGET_COLUMN]


def make_target_strata(y: pd.Series, n_strata: int = N_STRATA) -> pd.Series:
    strata = pd.qcut(y, q=n_strata, labels=False, duplicates="drop")
    strata = pd.Series(strata, index=y.index)
    if strata.nunique(dropna=True) < 2:
        raise ValueError("Target quintile stratification failed: fewer than two strata.")
    return strata


def split_train_test(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    strata = make_target_strata(y)
    if strata.value_counts().min() < 2:
        raise ValueError("Each target stratum must contain at least two observations.")
    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=random_state,
        stratify=strata,
    )


def stratified_cv_splits(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = RANDOM_STATE,
) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
    strata = make_target_strata(y_train)
    min_count = int(strata.value_counts().min())
    if min_count < CV_FOLDS:
        raise ValueError(
            f"10-fold stratified CV requires at least {CV_FOLDS} samples per "
            f"target stratum; minimum stratum count is {min_count}."
        )
    splitter = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=random_state)
    return splitter.split(X_train, strata)


def rmse(y_true: pd.Series, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def regression_metrics(model, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
    pred = model.predict(X)
    return {"R2": float(r2_score(y, pred)), "RMSE": rmse(y, pred)}


def mean_cv_rmse(model_factory: Callable[[Dict], object], params: Dict, X: pd.DataFrame, y: pd.Series) -> float:
    scores: List[float] = []
    for train_idx, valid_idx in stratified_cv_splits(X, y):
        X_fit, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_fit, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
        model = model_factory(params)
        model.fit(X_fit, y_fit)
        scores.append(rmse(y_valid, model.predict(X_valid)))
    return float(np.mean(scores))


def tune_with_optuna(
    objective: Callable[[optuna.Trial], float],
    n_trials: int,
    random_state: int = RANDOM_STATE,
    early_stop_patience: Optional[int] = None,
) -> Tuple[Dict, float, int]:
    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    callbacks = [NoImprovementStopper(early_stop_patience)] if early_stop_patience else None
    study.optimize(objective, n_trials=n_trials, callbacks=callbacks, show_progress_bar=False)
    return dict(study.best_params), float(study.best_value), len(study.trials)


def relative_mean_abs_shap(shap_values: np.ndarray) -> np.ndarray:
    values = np.asarray(shap_values)
    if values.ndim == 3:
        values = values[:, :, 0]
    abs_mean = np.abs(values).mean(axis=0)
    total = float(abs_mean.sum())
    if total <= 0 or not np.isfinite(total):
        return np.zeros_like(abs_mean, dtype=float)
    return abs_mean / total


def save_model(model: object, output_dir: Path, model_name: str, city: str) -> str:
    model_dir = output_dir / "fitted_models" / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / f"{city}.joblib"
    joblib.dump(model, path)
    return str(path.as_posix())


def run_city_model_training(
    config: ModelRunConfig,
    model_name: str,
    model_factory: Callable[[Dict], object],
    suggest_params: Callable[[optuna.Trial], Dict],
    n_trials: int,
    early_stop_patience: Optional[int] = None,
) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict] = []

    for path in sorted(config.input_dir.glob(config.file_pattern)):
        city, X, y = read_city_table(path)
        X_train, X_test, y_train, y_test = split_train_test(X, y, config.random_state)

        def objective(trial: optuna.Trial) -> float:
            params = suggest_params(trial)
            return mean_cv_rmse(model_factory, params, X_train, y_train)

        best_params, best_cv_rmse, trials_run = tune_with_optuna(
            objective,
            n_trials=n_trials,
            random_state=config.random_state,
            early_stop_patience=early_stop_patience,
        )
        model = model_factory(best_params)
        model.fit(X_train, y_train)

        train = regression_metrics(model, X_train, y_train)
        test = regression_metrics(model, X_test, y_test)
        model_path = save_model(model, config.output_dir, model_name, city)

        rows.append(
            {
                "city": city,
                "model": model_name,
                "n_total": len(X),
                "n_train": len(X_train),
                "n_test": len(X_test),
                "cv_rmse": best_cv_rmse,
                "train_R2": train["R2"],
                "test_R2": test["R2"],
                "train_RMSE": train["RMSE"],
                "test_RMSE": test["RMSE"],
                "trials_run": trials_run,
                "best_params": json.dumps(best_params, ensure_ascii=False),
                "model_path": model_path,
            }
        )

    pd.DataFrame(rows).to_csv(config.output_dir / f"{model_name}_city_metrics.csv", index=False)
