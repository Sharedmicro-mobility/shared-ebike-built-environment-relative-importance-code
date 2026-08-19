from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Set

import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr

sys.path.append(str(Path(__file__).resolve().parents[1] / "02_model_training"))
from shared_model_utils import (
    FEATURE_COLUMNS,
    regression_metrics,
    relative_mean_abs_shap,
    read_city_table,
    split_train_test,
)
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


def set_random_state(params: Dict, seed: int) -> Dict:
    out = dict(params)
    out["random_state"] = seed
    return out


def relative_shap(model, X: pd.DataFrame) -> np.ndarray:
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(X)
    if isinstance(values, list):
        values = values[0]
    return relative_mean_abs_shap(np.asarray(values))


def rank_profile(values: Sequence[float]) -> np.ndarray:
    series = pd.Series(values, index=FEATURE_COLUMNS)
    return series.rank(method="min", ascending=False).loc[FEATURE_COLUMNS].to_numpy(dtype=float)


def top_set(values_or_ranks: Sequence[float], is_rank: bool = False) -> Set[str]:
    series = pd.Series(values_or_ranks, index=FEATURE_COLUMNS)
    if is_rank:
        return set(series.sort_values(ascending=True).head(3).index)
    return set(series.sort_values(ascending=False).head(3).index)


def jaccard(a: Set[str], b: Set[str]) -> float:
    return len(a & b) / len(a | b)


def main() -> None:
    parser = argparse.ArgumentParser(description="Repeated-refitting SHAP ranking-stability diagnostics.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--selected-models", type=Path, required=True)
    parser.add_argument("--baseline-shap", type=Path, required=True, help="CSV with city and the ten baseline relative SHAP columns.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--file-pattern", default="*.csv")
    parser.add_argument("--n-refits", type=int, default=30)
    parser.add_argument("--first-seed", type=int, default=1001)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected = pd.read_csv(args.selected_models)
    selected_by_city = {row["city"]: row for _, row in selected.iterrows()}
    baseline = pd.read_csv(args.baseline_shap).set_index("city")

    city_rows: List[Dict] = []
    run_metric_rows: List[Dict] = []
    run_shap_rows: List[Dict] = []
    variable_rows: List[Dict] = []
    value_rows: List[Dict] = []

    for path in sorted(args.input_dir.glob(args.file_pattern)):
        city, X, y = read_city_table(path)
        if city not in selected_by_city:
            raise ValueError(f"No selected model record found for {city}.")
        if city not in baseline.index:
            raise ValueError(f"No baseline SHAP record found for {city}.")

        record = selected_by_city[city]
        model_name = record["model"]
        base_params = json.loads(record["best_params"])
        baseline_values = baseline.loc[city, FEATURE_COLUMNS].astype(float).to_numpy()
        baseline_rank = rank_profile(baseline_values)

        refit_values = []
        refit_ranks = []
        rank_profiles = [baseline_rank]
        top_profiles = [top_set(baseline_rank, is_rank=True)]

        for run in range(args.n_refits):
            seed = args.first_seed + run
            params = set_random_state(base_params, seed)
            X_train, X_test, y_train, y_test = split_train_test(X, y, random_state=seed)
            model = model_from_name(model_name, params)
            model.fit(X_train, y_train)
            train_metrics = regression_metrics(model, X_train, y_train)
            test_metrics = regression_metrics(model, X_test, y_test)
            shap_rel = relative_shap(model, X)
            ranks = rank_profile(shap_rel)

            refit_values.append(shap_rel)
            refit_ranks.append(ranks)
            rank_profiles.append(ranks)
            top_profiles.append(top_set(ranks, is_rank=True))

            metric_row = {
                "city": city,
                "model": model_name,
                "run": run + 1,
                "seed": seed,
                "train_R2": train_metrics["R2"],
                "test_R2": test_metrics["R2"],
                "train_RMSE": train_metrics["RMSE"],
                "test_RMSE": test_metrics["RMSE"],
            }
            run_metric_rows.append(metric_row)
            shap_row = {"city": city, "model": model_name, "run": run + 1, "seed": seed}
            shap_row.update({f: float(v) for f, v in zip(FEATURE_COLUMNS, shap_rel)})
            run_shap_rows.append(shap_row)

        pairwise_rhos = []
        top_jaccards = []
        for i, j in itertools.combinations(range(len(rank_profiles)), 2):
            pairwise_rhos.append(float(spearmanr(rank_profiles[i], rank_profiles[j]).correlation))
            top_jaccards.append(jaccard(top_profiles[i], top_profiles[j]))

        baseline_top = top_profiles[0]
        top3_agreement = [len(baseline_top & top_profiles[i]) / 3.0 for i in range(1, len(top_profiles))]

        city_rows.append(
            {
                "city": city,
                "model": model_name,
                "median_pairwise_rank_correlation": float(np.median(pairwise_rhos)),
                "mean_pairwise_rank_correlation": float(np.mean(pairwise_rhos)),
                "mean_top3_agreement_with_baseline": float(np.mean(top3_agreement)),
                "mean_pairwise_top3_jaccard": float(np.mean(top_jaccards)),
            }
        )

        refit_values_arr = np.vstack(refit_values)
        refit_ranks_arr = np.vstack(refit_ranks)
        for k, feature in enumerate(FEATURE_COLUMNS):
            variable_rows.append(
                {
                    "city": city,
                    "feature": feature,
                    "baseline_rank": float(baseline_rank[k]),
                    "mean_refit_rank": float(refit_ranks_arr[:, k].mean()),
                    "median_refit_rank": float(np.median(refit_ranks_arr[:, k])),
                }
            )
            value_rows.append(
                {
                    "city": city,
                    "feature": feature,
                    "baseline_relative_shap": float(baseline_values[k]),
                    "mean_refit_relative_shap": float(refit_values_arr[:, k].mean()),
                }
            )

    pd.DataFrame(city_rows).to_csv(args.output_dir / "city_level_ranking_stability.csv", index=False)
    pd.DataFrame(variable_rows).to_csv(args.output_dir / "variable_rank_comparison.csv", index=False)
    pd.DataFrame(value_rows).to_csv(args.output_dir / "relative_shap_value_comparison.csv", index=False)
    pd.DataFrame(run_metric_rows).to_csv(args.output_dir / "repeated_refit_run_metrics.csv", index=False)
    pd.DataFrame(run_shap_rows).to_csv(args.output_dir / "repeated_refit_relative_shap_profiles.csv", index=False)
    print(f"Saved repeated-refit SHAP stability outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
