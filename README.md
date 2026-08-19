# Built-Environment Predictive Importance for Shared E-Bike Ridership

This repository contains the analysis code for the study of built-environment relative-importance patterns in predicting realised HelloBike shared e-bike ridership across 36 Chinese cities.

The code is organised to reproduce the main computational workflow described in the manuscript: trip cleaning, 500 m hexagonal grid construction, built-environment variable calculation, city-specific ensemble model training, SHAP interpretation, robustness diagnostics, and PCA/K-Means++ clustering.

## Data Availability and Restrictions

The repository contains code only. It does not contain raw HelloBike records, local file paths, credentials, or proprietary input data. The raw HelloBike trip records are not publicly shared because access is restricted by the data-use agreement. Non-HelloBike spatial datasets should be obtained from their original providers subject to their terms of use. These include POI data, OpenStreetMap-derived cycling network data, WorldPop population data, and administrative boundary data.

## Repository Structure

```text
reproducibility_code/
  README.md
  requirements.txt
  01_variable_construction/
    01_build_hex_grid_and_ridership.py
    02_construct_built_environment_variables.py
  02_model_training/
    shared_model_utils.py
    train_random_forest.py
    train_gbdt.py
    train_xgboost.py
    train_lightgbm.py
    select_optimal_models.py
  03_shap_analysis/
    compute_relative_shap_and_direction.py
  04_robustness_diagnostics/
    predictor_correlation_diagnostics.py
    repeated_refit_shap_stability.py
    pooled_model_with_city_controls.py
    residual_spatial_autocorrelation.py
  05_pca_clustering/
    pca_kmeans_clustering.py
```

## Environment

The code was written for Python 3.10 or later. Install the required packages with:

```bash
pip install -r requirements.txt
```

Main dependencies include `pandas`, `geopandas`, `rasterio`, `scikit-learn`, `optuna`, `shap`, `xgboost`, `lightgbm`, `scipy`, `libpysal`, and `esda`.

## Required Processed Columns

The modelling scripts expect one processed grid-level CSV per city, with at least the following columns. If a `city` column is provided, it must contain a single city name within each file; otherwise, the city name is inferred from the file name.

| Column | Meaning |
|---|---|
| `city` | City name or city identifier |
| `grid_id` | Grid-cell identifier |
| `SER` | Weekly shared e-bike ridership, defined as trip origins plus trip destinations within the grid |
| `PD` | Population density |
| `LUM` | Land-use mix |
| `CBD` | Distance to the central business district |
| `LEC` | Leisure POI count |
| `REC` | Retail POI count |
| `RC` | Residential POI count |
| `EC` | Employment POI count |
| `TSC` | Public transport stop count |
| `CLL` | Cycling road length |
| `IC` | Intersection count |

## Workflow

### 1. Build Hexagonal Grids and Ridership

Use `01_variable_construction/01_build_hex_grid_and_ridership.py` to clean trip records, construct 500 m hexagonal grids, and aggregate trip origins and destinations.

Example:

```bash
python 01_variable_construction/01_build_hex_grid_and_ridership.py \
  --city example_city \
  --trip-csv path/to/trips.csv \
  --boundary-file path/to/boundary.gpkg \
  --output-grid outputs/grids/example_city.gpkg
```
Records are removed if trip distance is below 50 m or above 15 km, duration is below 1 minute or above 120 minutes, or origin/destination points fall outside the city boundary.

### 2. Construct Built-Environment Variables

Use `01_variable_construction/02_construct_built_environment_variables.py` to calculate the ten built-environment variables for each grid.

Example:

```bash
python 01_variable_construction/02_construct_built_environment_variables.py \
  --grid-file outputs/grids/example_city.gpkg \
  --poi-file path/to/poi.csv \
  --cbd-file path/to/cbd.gpkg \
  --roads-file path/to/cycling_roads.gpkg \
  --nodes-file path/to/cycling_nodes.gpkg \
  --population-raster path/to/worldpop.tif \
  --output-csv outputs/processed/example_city.csv \
  --output-gpkg outputs/processed/example_city.gpkg
```

### 3. Train City-Specific Models

Four tree-based ensemble models are trained independently for each city:

- Random Forest
- Gradient Boosted Decision Trees
- XGBoost
- LightGBM

Examples:

```bash
python 02_model_training/train_random_forest.py \
  --input-dir outputs/processed \
  --output-dir outputs/model_training

python 02_model_training/train_gbdt.py \
  --input-dir outputs/processed \
  --output-dir outputs/model_training

python 02_model_training/train_xgboost.py \
  --input-dir outputs/processed \
  --output-dir outputs/model_training

python 02_model_training/train_lightgbm.py \
  --input-dir outputs/processed \
  --output-dir outputs/model_training
```

All models use a 90% training and 10% testing split, stratified by target-value quantile groups. The quantile groups are used only for stratified sampling. Hyperparameter tuning uses Optuna with 10-fold stratified cross-validation and mean RMSE as the optimisation objective. Random Forest uses 80 Optuna trials. GBDT, XGBoost, and LightGBM use up to 500 trials with a 50-trial no-improvement stopping rule.

Select the best city-specific model:

```bash
python 02_model_training/select_optimal_models.py \
  --metrics-dir outputs/model_training \
  --output-file outputs/model_training/selected_city_models.csv
```

The selected model is the model with the highest test-set `R2`, with test-set `RMSE` used as a secondary criterion.

### 4. Compute SHAP Values and Directional Diagnostics

Use `03_shap_analysis/compute_relative_shap_and_direction.py` to compute normalised global SHAP values and SHAP-dependence direction diagnostics.

```bash
python 03_shap_analysis/compute_relative_shap_and_direction.py \
  --input-dir outputs/processed \
  --selected-models outputs/model_training/selected_city_models.csv \
  --output-dir outputs/shap
```

Relative SHAP values are calculated by normalising the mean absolute SHAP values across the ten built-environment variables within each city, so that the ten values sum to 1.

For directionality, the script calculates Spearman's rank correlation between each predictor's observed grid-level values and its corresponding SHAP values. A variable is classified as positive or negative only when the correlation is statistically significant and `|rho_s| >= 0.10`.

### 5. Run Robustness Diagnostics

Predictor-correlation diagnostics:

```bash
python 04_robustness_diagnostics/predictor_correlation_diagnostics.py \
  --input-dir outputs/processed \
  --output-dir outputs/robustness
```

Repeated-refitting SHAP stability diagnostics:

```bash
python 04_robustness_diagnostics/repeated_refit_shap_stability.py \
  --input-dir outputs/processed \
  --selected-models outputs/model_training/selected_city_models.csv \
  --baseline-shap outputs/shap/relative_shap_values.csv \
  --output-dir outputs/robustness/repeated_refit
```

Pooled model with city dummy controls:

```bash
python 04_robustness_diagnostics/pooled_model_with_city_controls.py \
  --input-dir outputs/processed \
  --output-dir outputs/robustness/pooled_model
```

Residual spatial autocorrelation diagnostics:

```bash
python 04_robustness_diagnostics/residual_spatial_autocorrelation.py \
  --grid-dir outputs/processed_geodata \
  --selected-models outputs/model_training/selected_city_models.csv \
  --output-dir outputs/robustness/spatial_autocorrelation
```

The residual spatial autocorrelation script requires geospatial grid files with geometry. It calculates Global Moran's I for observed ridership and for 10-fold out-of-fold residuals from the selected city-specific model. Spatial weights are based on eight nearest neighbours.

### 6. Identify Cross-City Relative-Importance Patterns

Use `05_pca_clustering/pca_kmeans_clustering.py` to perform PCA and K-Means++ clustering on the city-level normalised SHAP profiles.

```bash
python 05_pca_clustering/pca_kmeans_clustering.py \
  --relative-shap outputs/shap/relative_shap_values.csv \
  --output-dir outputs/clustering
```

The script standardises the ten-dimensional city-level SHAP vectors, applies PCA, retains principal components explaining at least 90% of variance, and then clusters cities using K-Means++ on the PCA-transformed vectors. It exports city assignments, cluster centroids, PCA scores, PCA loadings, explained variance, silhouette diagnostics, within-cluster sum of squares, and 400 dpi figures.

## Notes on Reproducibility

- The default random seed is 42 for the main model training and clustering steps.
- Repeated-refitting diagnostics use 30 random seeds by default, beginning at 1001.
- The public code keeps model type and tuned hyperparameters fixed during repeated refitting. This is intended to isolate SHAP stability under the selected city-specific model specification, rather than introducing a new model-selection process in each repetition.
- The pooled model includes city dummy variables to control for unobserved city-level heterogeneity. When pooled SHAP values are compared with city-specific relative SHAP values, the normalisation denominator should include only the ten built-environment variables.

## Citation

If you use this code, please cite the associated manuscript after publication.
