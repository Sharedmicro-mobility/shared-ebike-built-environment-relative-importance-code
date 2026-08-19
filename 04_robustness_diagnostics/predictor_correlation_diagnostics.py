from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "02_model_training"))
from shared_model_utils import FEATURE_COLUMNS, read_city_table


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute city-level Spearman correlations among built-environment predictors.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--file-pattern", default="*.csv")
    parser.add_argument("--strong-threshold", type=float, default=0.70)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: List[Dict] = []
    strong_rows: List[Dict] = []

    for path in sorted(args.input_dir.glob(args.file_pattern)):
        city, X, _ = read_city_table(path)
        corr = X[FEATURE_COLUMNS].corr(method="spearman")
        for v1, v2 in itertools.combinations(FEATURE_COLUMNS, 2):
            rho = float(corr.loc[v1, v2])
            row = {
                "city": city,
                "variable_1": v1,
                "variable_2": v2,
                "spearman_rho": rho,
                "abs_spearman_rho": abs(rho),
                "valid_observations": int(X[[v1, v2]].dropna().shape[0]),
                "correlation_direction": "positive" if rho >= 0 else "negative",
            }
            all_rows.append(row)
            if np.isfinite(rho) and abs(rho) >= args.strong_threshold:
                strong_rows.append(row)

    pd.DataFrame(all_rows).to_csv(args.output_dir / "all_predictor_spearman_pairs.csv", index=False)
    pd.DataFrame(strong_rows).to_csv(args.output_dir / "strong_predictor_spearman_pairs.csv", index=False)
    print(f"Saved predictor-correlation diagnostics to {args.output_dir}")


if __name__ == "__main__":
    main()
