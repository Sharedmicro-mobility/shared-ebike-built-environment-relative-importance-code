from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = ["PD", "LUM", "CBD", "LEC", "REC", "RC", "EC", "TSC", "CLL", "IC"]
RANDOM_STATE = 42
PATTERN_LABELS: Dict[str, str] = {
    "PD": "population density",
    "LUM": "land-use mix",
    "CBD": "distance to CBD",
    "LEC": "leisure POI count",
    "REC": "retail POI count",
    "RC": "residential POI count",
    "EC": "employment POI count",
    "TSC": "public transport stop count",
    "CLL": "cycling road length",
    "IC": "intersection count",
}


def read_shap_profiles(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    missing = [c for c in ["city"] + FEATURE_COLUMNS if c not in data.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    data = data[["city"] + FEATURE_COLUMNS].copy()
    data[FEATURE_COLUMNS] = data[FEATURE_COLUMNS].astype(float)
    row_sums = data[FEATURE_COLUMNS].sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-3):
        raise ValueError("Each city's relative SHAP values should sum to 1.")
    return data


def pca_transform(values: pd.DataFrame) -> tuple[StandardScaler, PCA, np.ndarray]:
    scaler = StandardScaler()
    scaled = scaler.fit_transform(values)
    pca = PCA(n_components=0.90, random_state=RANDOM_STATE)
    scores = pca.fit_transform(scaled)
    return scaler, pca, scores


def evaluate_kmeans(scores: np.ndarray, min_k: int = 2, max_k: int = 10) -> pd.DataFrame:
    rows: List[Dict] = []
    max_k = min(max_k, scores.shape[0] - 1)
    for k in range(min_k, max_k + 1):
        model = KMeans(n_clusters=k, init="k-means++", n_init=50, random_state=RANDOM_STATE)
        labels = model.fit_predict(scores)
        rows.append(
            {
                "k": k,
                "silhouette": float(silhouette_score(scores, labels)),
                "within_cluster_sum_of_squares": float(model.inertia_),
            }
        )
    return pd.DataFrame(rows)


def choose_k(k_table: pd.DataFrame, requested_k: int | None) -> int:
    if requested_k is not None:
        if requested_k not in set(k_table["k"]):
            raise ValueError(f"Requested k={requested_k} is outside the evaluated range.")
        return requested_k
    best = k_table.sort_values(["silhouette", "k"], ascending=[False, True]).iloc[0]
    return int(best["k"])


def plot_pca_variance(pca: PCA, output_dir: Path) -> None:
    components = np.arange(1, len(pca.explained_variance_ratio_) + 1)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(components, pca.explained_variance_ratio_, color="#7c8db5", label="Individual")
    ax.plot(components, cumulative, marker="o", color="#b23a48", label="Cumulative")
    ax.axhline(0.90, color="#2a9d8f", linestyle="--", linewidth=1)
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Explained variance ratio")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "pca_explained_variance.png", dpi=400)
    plt.close(fig)


def plot_k_diagnostics(k_table: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(k_table["k"], k_table["silhouette"], marker="o", color="#355c7d")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Average silhouette coefficient")
    ax.set_xticks(k_table["k"])
    fig.tight_layout()
    fig.savefig(output_dir / "kmeans_silhouette_scores.png", dpi=400)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(k_table["k"], k_table["within_cluster_sum_of_squares"], marker="o", color="#c06c84")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Within-cluster sum of squares")
    ax.set_xticks(k_table["k"])
    fig.tight_layout()
    fig.savefig(output_dir / "kmeans_within_cluster_sum_of_squares.png", dpi=400)
    plt.close(fig)


def plot_centroid_radar(centroids: pd.DataFrame, output_dir: Path) -> None:
    labels = [PATTERN_LABELS[c] for c in FEATURE_COLUMNS]
    angles = np.linspace(0, 2 * np.pi, len(FEATURE_COLUMNS), endpoint=False).tolist()
    angles += angles[:1]

    for _, row in centroids.iterrows():
        values = row[FEATURE_COLUMNS].to_numpy(dtype=float).tolist()
        values += values[:1]
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111, polar=True)
        ax.plot(angles, values, color="#355c7d", linewidth=2)
        ax.fill(angles, values, color="#355c7d", alpha=0.20)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylim(0, max(0.30, float(np.nanmax(centroids[FEATURE_COLUMNS].to_numpy())) * 1.15))
        ax.set_title(f"Cluster {int(row['cluster'])}", pad=18)
        fig.tight_layout()
        fig.savefig(output_dir / f"cluster_{int(row['cluster'])}_centroid_radar.png", dpi=400)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="PCA and K-Means++ clustering of city-level relative SHAP profiles.")
    parser.add_argument("--relative-shap", type=Path, required=True, help="CSV with city and ten normalised SHAP columns.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-clusters", type=int, default=None, help="Optional fixed k. If omitted, k with highest silhouette is selected.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    profiles = read_shap_profiles(args.relative_shap)
    _, pca, scores = pca_transform(profiles[FEATURE_COLUMNS])
    k_table = evaluate_kmeans(scores)
    selected_k = choose_k(k_table, args.n_clusters)

    final_model = KMeans(n_clusters=selected_k, init="k-means++", n_init=50, random_state=RANDOM_STATE)
    labels = final_model.fit_predict(scores) + 1

    assignments = profiles[["city"]].copy()
    assignments["cluster"] = labels
    score_columns = [f"PC{i + 1}" for i in range(scores.shape[1])]
    pca_scores = pd.concat([profiles[["city"]], pd.DataFrame(scores, columns=score_columns)], axis=1)
    pca_scores["cluster"] = labels

    centroids = profiles.assign(cluster=labels).groupby("cluster", as_index=False)[FEATURE_COLUMNS].mean()
    loadings = pd.DataFrame(
        pca.components_.T,
        index=FEATURE_COLUMNS,
        columns=score_columns,
    ).reset_index(names="feature")
    explained = pd.DataFrame(
        {
            "component": score_columns,
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_explained_variance": np.cumsum(pca.explained_variance_ratio_),
        }
    )

    assignments.to_csv(args.output_dir / "city_cluster_assignments.csv", index=False)
    centroids.to_csv(args.output_dir / "cluster_centroids_relative_shap.csv", index=False)
    pca_scores.to_csv(args.output_dir / "pca_scores.csv", index=False)
    loadings.to_csv(args.output_dir / "pca_loadings.csv", index=False)
    explained.to_csv(args.output_dir / "pca_explained_variance.csv", index=False)
    k_table.to_csv(args.output_dir / "kmeans_k_diagnostics.csv", index=False)

    plot_pca_variance(pca, args.output_dir)
    plot_k_diagnostics(k_table, args.output_dir)
    plot_centroid_radar(centroids, args.output_dir)

    print(f"Retained {scores.shape[1]} principal components and selected k={selected_k}.")
    print(f"Saved PCA and clustering outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
