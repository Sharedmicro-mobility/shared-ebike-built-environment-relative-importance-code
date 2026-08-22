from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

sys.path.append(str(Path(__file__).resolve().parents[1] / "02_model_training"))
from shared_model_utils import FEATURE_COLUMNS, RANDOM_STATE, read_city_table
from train_gbdt import make_model as make_gbdt
from train_lightgbm import make_model as make_lightgbm
from train_random_forest import make_model as make_rf
from train_xgboost import make_model as make_xgboost


@dataclass
class AleResult:
    x_values: np.ndarray
    ale_values: np.ndarray
    local_effects: np.ndarray
    interval_counts: np.ndarray
    edges: np.ndarray
    effective_bins: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute first-order ALE curves and direction classes for selected city-specific models."
    )
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing processed city CSV files.")
    parser.add_argument("--selected-models", type=Path, required=True, help="CSV produced by select_optimal_models.py.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--file-pattern", default="*.csv")
    parser.add_argument("--n-bins", type=int, default=10, help="Number of quantile intervals used for ALE.")
    parser.add_argument("--min-effective-bins", type=int, default=3)
    parser.add_argument("--contrast-threshold", type=float, default=0.05)
    parser.add_argument("--consistency-threshold", type=float, default=0.70)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    parser.add_argument("--save-plots", action="store_true", help="Save one ALE curve image for each city-variable pair.")
    parser.add_argument("--plot-dir", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def model_from_name(model_name: str, params: Dict[str, Any]):
    if model_name == "RF":
        return make_rf(params)
    if model_name == "GBDT":
        return make_gbdt(params)
    if model_name == "XGBoost":
        return make_xgboost(params)
    if model_name == "LightGBM":
        return make_lightgbm(params)
    raise ValueError(f"Unsupported model name: {model_name}")


def params_with_random_state(params: Dict[str, Any], seed: int) -> Dict[str, Any]:
    out = dict(params)
    out["random_state"] = seed
    return out


def make_quantile_edges(values: np.ndarray, n_bins: int) -> np.ndarray:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return np.array([], dtype=float)
    if np.nanmin(finite) == np.nanmax(finite):
        return np.array([float(np.nanmin(finite))], dtype=float)
    edges = np.quantile(finite, np.linspace(0.0, 1.0, n_bins + 1))
    return np.unique(edges).astype(float)


def compute_first_order_ale(model, X: pd.DataFrame, feature: str, n_bins: int) -> AleResult:
    feature_values = X[feature].to_numpy(dtype=float)
    edges = make_quantile_edges(feature_values, n_bins=n_bins)
    if len(edges) < 2:
        return AleResult(
            x_values=np.asarray([], dtype=float),
            ale_values=np.asarray([], dtype=float),
            local_effects=np.asarray([], dtype=float),
            interval_counts=np.asarray([], dtype=int),
            edges=edges,
            effective_bins=0,
        )

    local_effects: List[float] = []
    interval_counts: List[int] = []
    for idx in range(1, len(edges)):
        lower = edges[idx - 1]
        upper = edges[idx]
        if idx == 1:
            mask = (feature_values >= lower) & (feature_values <= upper)
        else:
            mask = (feature_values > lower) & (feature_values <= upper)
        X_interval = X.loc[mask].copy()
        interval_counts.append(int(len(X_interval)))
        if X_interval.empty:
            local_effects.append(0.0)
            continue

        X_low = X_interval.copy()
        X_high = X_interval.copy()
        X_low[feature] = lower
        X_high[feature] = upper
        diff = np.asarray(model.predict(X_high), dtype=float) - np.asarray(model.predict(X_low), dtype=float)
        local_effects.append(float(np.nanmean(diff)))

    local = np.asarray(local_effects, dtype=float)
    counts = np.asarray(interval_counts, dtype=int)
    cumulative = np.concatenate([[0.0], np.cumsum(local)])
    weights = counts / counts.sum() if counts.sum() > 0 else np.ones_like(counts, dtype=float) / len(counts)
    centre = float(np.sum(cumulative[1:] * weights))
    ale_at_edges = cumulative - centre
    x_values = (edges[:-1] + edges[1:]) / 2.0
    ale_values = ale_at_edges[1:]
    return AleResult(
        x_values=x_values,
        ale_values=ale_values,
        local_effects=local,
        interval_counts=counts,
        edges=edges,
        effective_bins=int(np.sum(counts > 0)),
    )


def ale_at_quantile(result: AleResult, q_value: float) -> float:
    if result.x_values.size == 0:
        return float("nan")
    return float(np.interp(q_value, result.x_values, result.ale_values))


def classify_ale_direction(
    result: AleResult,
    feature_values: np.ndarray,
    predicted_values: np.ndarray,
    min_effective_bins: int,
    contrast_threshold: float,
    consistency_threshold: float,
) -> Dict[str, Any]:
    pred_spread = float(np.quantile(predicted_values, 0.90) - np.quantile(predicted_values, 0.10))
    q10 = float(np.quantile(feature_values, 0.10))
    q90 = float(np.quantile(feature_values, 0.90))

    if result.effective_bins < min_effective_bins or pred_spread <= 0 or not np.isfinite(pred_spread):
        return {
            "direction": "Non-monotonic",
            "ale_contrast": float("nan"),
            "standardised_ale_contrast": float("nan"),
            "directional_consistency": float("nan"),
            "q10": q10,
            "q90": q90,
            "predicted_p10_p90_spread": pred_spread,
        }

    low_ale = ale_at_quantile(result, q10)
    high_ale = ale_at_quantile(result, q90)
    contrast = high_ale - low_ale
    standardised = contrast / pred_spread

    tolerance = max(1e-12, 0.001 * pred_spread)
    local = result.local_effects[np.isfinite(result.local_effects)]
    local = local[np.abs(local) > tolerance]
    if local.size == 0:
        consistency = 0.0
    elif standardised > 0:
        consistency = float(np.mean(local > 0))
    elif standardised < 0:
        consistency = float(np.mean(local < 0))
    else:
        consistency = 0.0

    if standardised >= contrast_threshold and consistency >= consistency_threshold:
        direction = "Positive"
    elif standardised <= -contrast_threshold and consistency >= consistency_threshold:
        direction = "Negative"
    else:
        direction = "Non-monotonic"

    return {
        "direction": direction,
        "ale_contrast": float(contrast),
        "standardised_ale_contrast": float(standardised),
        "directional_consistency": float(consistency),
        "q10": q10,
        "q90": q90,
        "predicted_p10_p90_spread": pred_spread,
    }


def safe_name(text: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in text)


def plot_ale(
    city: str,
    feature: str,
    result: AleResult,
    direction: str,
    standardised_contrast: float,
    output_path: Path,
    dpi: int,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
        }
    )
    fig, ax = plt.subplots(figsize=(2.05, 1.55))
    if result.x_values.size > 0:
        colour = {"Positive": "#1B7F3A", "Negative": "#B02A2A"}.get(direction, "#4A4A4A")
        ax.plot(result.x_values, result.ale_values, color=colour, linewidth=1.2)
        ax.scatter(result.x_values, result.ale_values, color=colour, s=8, zorder=3)
    ax.axhline(0, color="#808080", linewidth=0.6, linestyle="--")
    ax.set_title(f"{city} - {feature}", fontsize=7.5, pad=2)
    ax.set_xlabel(feature, fontsize=6.5, labelpad=1)
    ax.set_ylabel("ALE", fontsize=6.5, labelpad=1)
    ax.tick_params(labelsize=5.8, pad=1)
    ax.grid(True, color="#E6E6E6", linewidth=0.4)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    if np.isfinite(standardised_contrast):
        label = f"{direction}\nD*={standardised_contrast:.2f}"
    else:
        label = direction
    ax.text(
        0.02,
        0.98,
        label,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=5.5,
        bbox={"facecolor": "white", "edgecolor": "#BFBFBF", "linewidth": 0.3, "pad": 1.5},
    )
    fig.tight_layout(pad=0.35)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def analyse_city(
    city: str,
    model_name: str,
    best_params: Dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    args: argparse.Namespace,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    model = model_from_name(model_name, params_with_random_state(best_params, args.random_state))
    model.fit(X, y)
    predicted = np.asarray(model.predict(X), dtype=float)

    detail_rows: List[Dict[str, Any]] = []
    curve_rows: List[Dict[str, Any]] = []
    for feature in FEATURE_COLUMNS:
        result = compute_first_order_ale(model, X, feature=feature, n_bins=args.n_bins)
        classification = classify_ale_direction(
            result=result,
            feature_values=X[feature].to_numpy(dtype=float),
            predicted_values=predicted,
            min_effective_bins=args.min_effective_bins,
            contrast_threshold=args.contrast_threshold,
            consistency_threshold=args.consistency_threshold,
        )
        plot_path = ""
        if args.save_plots:
            plot_dir = args.plot_dir if args.plot_dir is not None else args.output_dir / "ale_plots"
            path = plot_dir / f"{safe_name(city)}_{feature}_ALE.png"
            plot_ale(
                city=city,
                feature=feature,
                result=result,
                direction=classification["direction"],
                standardised_contrast=classification["standardised_ale_contrast"],
                output_path=path,
                dpi=args.dpi,
            )
            plot_path = str(path.as_posix())

        detail_rows.append(
            {
                "city": city,
                "model": model_name,
                "feature": feature,
                "direction": classification["direction"],
                "ale_contrast_Q90_minus_Q10": classification["ale_contrast"],
                "standardised_ale_contrast": classification["standardised_ale_contrast"],
                "directional_consistency": classification["directional_consistency"],
                "effective_bins": result.effective_bins,
                "q10": classification["q10"],
                "q90": classification["q90"],
                "predicted_p10_p90_spread": classification["predicted_p10_p90_spread"],
                "n_city_samples": int(len(X)),
                "plot_path": plot_path,
            }
        )
        for x_value, ale_value, count in zip(result.x_values, result.ale_values, result.interval_counts):
            curve_rows.append(
                {
                    "city": city,
                    "feature": feature,
                    "x_value": float(x_value),
                    "ale_value": float(ale_value),
                    "interval_count": int(count),
                }
            )

    city_summary = {
        "city": city,
        "model": model_name,
        "n_city_samples": int(len(X)),
        "in_sample_R2_for_ALE_model": float(r2_score(y, predicted)),
        "in_sample_RMSE_for_ALE_model": float(math.sqrt(mean_squared_error(y, predicted))),
    }
    return detail_rows, curve_rows, city_summary


def build_variable_summary(detail_df: pd.DataFrame) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    for feature in FEATURE_COLUMNS:
        group = detail_df[detail_df["feature"].eq(feature)]
        counts = group["direction"].value_counts()
        records.append(
            {
                "feature": feature,
                "n_cities": int(group["city"].nunique()),
                "positive_city_count": int(counts.get("Positive", 0)),
                "negative_city_count": int(counts.get("Negative", 0)),
                "non_monotonic_city_count": int(counts.get("Non-monotonic", 0)),
                "mean_standardised_ale_contrast": float(group["standardised_ale_contrast"].mean()),
                "median_standardised_ale_contrast": float(group["standardised_ale_contrast"].median()),
                "mean_directional_consistency": float(group["directional_consistency"].mean()),
            }
        )
    return pd.DataFrame(records)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    selected = pd.read_csv(args.selected_models)
    required = {"city", "model", "best_params"}
    missing = required - set(selected.columns)
    if missing:
        raise ValueError(f"{args.selected_models} is missing required columns: {sorted(missing)}")
    selected_by_city: Dict[str, pd.Series] = {row["city"]: row for _, row in selected.iterrows()}

    detail_rows: List[Dict[str, Any]] = []
    curve_rows: List[Dict[str, Any]] = []
    city_summary_rows: List[Dict[str, Any]] = []

    for path in sorted(args.input_dir.glob(args.file_pattern)):
        city, X, y = read_city_table(path)
        if city not in selected_by_city:
            raise ValueError(f"No selected model record found for {city}.")
        record = selected_by_city[city]
        best_params = json.loads(record["best_params"])
        city_rows, city_curve_rows, city_summary = analyse_city(
            city=city,
            model_name=str(record["model"]),
            best_params=best_params,
            X=X,
            y=y,
            args=args,
        )
        detail_rows.extend(city_rows)
        curve_rows.extend(city_curve_rows)
        city_summary_rows.append(city_summary)

    detail_df = pd.DataFrame(detail_rows)
    curve_df = pd.DataFrame(curve_rows)
    city_summary_df = pd.DataFrame(city_summary_rows)
    variable_summary_df = build_variable_summary(detail_df)
    settings_df = pd.DataFrame(
        [
            {"setting": "n_bins", "value": args.n_bins},
            {"setting": "min_effective_bins", "value": args.min_effective_bins},
            {"setting": "contrast_threshold", "value": args.contrast_threshold},
            {"setting": "consistency_threshold", "value": args.consistency_threshold},
            {"setting": "random_state", "value": args.random_state},
            {
                "setting": "classification_rule",
                "value": (
                    "Positive/Negative require the standardised Q90-Q10 ALE contrast to exceed "
                    "the contrast threshold in the corresponding direction and directional "
                    "consistency to meet the consistency threshold; otherwise Non-monotonic."
                ),
            },
        ]
    )

    detail_df.to_csv(args.output_dir / "ale_city_variable_direction.csv", index=False)
    variable_summary_df.to_csv(args.output_dir / "ale_variable_direction_summary.csv", index=False)
    curve_df.to_csv(args.output_dir / "ale_curve_points.csv", index=False)
    city_summary_df.to_csv(args.output_dir / "ale_model_fit_summary.csv", index=False)
    settings_df.to_csv(args.output_dir / "ale_settings.csv", index=False)
    print(f"Saved ALE direction outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
