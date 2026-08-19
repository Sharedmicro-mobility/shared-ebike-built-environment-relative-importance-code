from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.features import shapes
from rasterio.mask import mask
from shapely.geometry import mapping, shape
from shapely.ops import transform


POI_MAJOR_CATEGORIES = [
    "Transport facilities",
    "Leisure amenities",
    "Commercial and industrial facilities",
    "Healthcare facilities",
    "Residential facilities",
    "Tourist attractions",
    "Automotive services",
    "Daily life services",
    "Educational and cultural institutions",
    "Retail facilities",
    "Sports facilities",
    "Accommodation facilities",
    "Financial institutions",
    "Food and beverage services",
]

FEATURE_COLUMNS = ["PD", "LUM", "CBD", "LEC", "REC", "RC", "EC", "TSC", "CLL", "IC"]


def read_points(path: Path, lon_col: str, lat_col: str, crs: str) -> gpd.GeoDataFrame:
    if path.suffix.lower() in {".csv", ".txt"}:
        data = pd.read_csv(path)
        if lon_col not in data.columns or lat_col not in data.columns:
            raise ValueError(f"{path} must contain {lon_col} and {lat_col}.")
        return gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(data[lon_col], data[lat_col]), crs=crs)
    return gpd.read_file(path)


def safe_entropy(counts: Iterable[float]) -> float:
    arr = np.asarray(list(counts), dtype=float)
    total = arr.sum()
    if total <= 0:
        return 0.0
    p = arr[arr > 0] / total
    return float(-(p * np.log(p)).sum())


def add_poi_metrics(
    grids: gpd.GeoDataFrame,
    poi: gpd.GeoDataFrame,
    major_col: str,
    sub_col: Optional[str],
    transit_pattern: str,
) -> gpd.GeoDataFrame:
    poi = poi.to_crs(grids.crs)
    joined = gpd.sjoin(
        poi[[major_col] + ([sub_col] if sub_col and sub_col in poi.columns else []) + ["geometry"]],
        grids[["grid_id", "geometry"]],
        how="inner",
        predicate="within",
    )

    category_counts = (
        joined.groupby(["grid_id", major_col]).size().unstack(fill_value=0)
        if not joined.empty else pd.DataFrame(index=grids["grid_id"])
    )
    for category in POI_MAJOR_CATEGORIES:
        if category not in category_counts.columns:
            category_counts[category] = 0
    category_counts = category_counts[POI_MAJOR_CATEGORIES]

    grids = grids.copy()
    grids["LUM"] = grids["grid_id"].map(category_counts.apply(safe_entropy, axis=1)).fillna(0.0)
    grids["LEC"] = grids["grid_id"].map(category_counts["Leisure amenities"]).fillna(0).astype(int)
    grids["REC"] = grids["grid_id"].map(category_counts["Retail facilities"]).fillna(0).astype(int)
    grids["RC"] = grids["grid_id"].map(category_counts["Residential facilities"]).fillna(0).astype(int)
    grids["EC"] = grids["grid_id"].map(category_counts["Commercial and industrial facilities"]).fillna(0).astype(int)

    transit = joined[joined[major_col].eq("Transport facilities")].copy()
    if sub_col and sub_col in transit.columns:
        pattern = re.compile(transit_pattern, flags=re.IGNORECASE)
        transit = transit[transit[sub_col].astype(str).str.contains(pattern, na=False)]
    grids["TSC"] = grids["grid_id"].map(transit.groupby("grid_id").size()).fillna(0).astype(int)
    return grids


def add_cbd_distance(grids: gpd.GeoDataFrame, cbd: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    cbd = cbd.to_crs(grids.crs)
    cbd_union = cbd.geometry.union_all()
    grids = grids.copy()
    grids["CBD"] = grids.geometry.centroid.distance(cbd_union)
    return grids


def add_road_metrics(
    grids: gpd.GeoDataFrame,
    roads: gpd.GeoDataFrame,
    nodes: gpd.GeoDataFrame,
    min_street_count: int = 0,
) -> gpd.GeoDataFrame:
    grids = grids.copy()
    roads = roads.to_crs(grids.crs)
    nodes = nodes.to_crs(grids.crs)

    road_intersections = gpd.overlay(roads[["geometry"]], grids[["grid_id", "geometry"]], how="intersection")
    grids["CLL"] = grids["grid_id"].map(road_intersections.groupby("grid_id").geometry.length.sum()).fillna(0.0)

    street_col = next((c for c in ["street_count", "street_cou"] if c in nodes.columns), None)
    if street_col and min_street_count > 0:
        nodes = nodes[nodes[street_col].fillna(0).astype(float) >= min_street_count]
    node_join = gpd.sjoin(nodes[["geometry"]], grids[["grid_id", "geometry"]], how="inner", predicate="within")
    grids["IC"] = grids["grid_id"].map(node_join.groupby("grid_id").size()).fillna(0).astype(int)
    return grids


def area_weighted_population_for_grid(
    raster: rasterio.io.DatasetReader,
    grid_geom_projected,
    grid_crs,
) -> float:
    grid_in_raster_crs = gpd.GeoSeries([grid_geom_projected], crs=grid_crs).to_crs(raster.crs).iloc[0]
    out_image, out_transform = mask(raster, [mapping(grid_in_raster_crs)], crop=True, filled=False)
    values = out_image[0]
    if np.ma.is_masked(values):
        valid_mask = ~values.mask
        arr = values.filled(np.nan)
    else:
        valid_mask = np.isfinite(values)
        arr = values

    if raster.nodata is not None:
        valid_mask &= arr != raster.nodata

    transformer = Transformer.from_crs(raster.crs, grid_crs, always_xy=True)
    to_grid_crs = lambda x, y, z=None: transformer.transform(x, y)

    total = 0.0
    for geom, value in shapes(arr.astype("float64"), mask=valid_mask, transform=out_transform):
        if not np.isfinite(value) or value <= 0:
            continue
        cell = transform(to_grid_crs, shape(geom))
        if cell.is_empty or cell.area <= 0:
            continue
        overlap_area = cell.intersection(grid_geom_projected).area
        if overlap_area > 0:
            total += float(value) * (overlap_area / cell.area)
    return total


def add_population_density(grids: gpd.GeoDataFrame, raster_path: Path) -> gpd.GeoDataFrame:
    grids = grids.copy()
    population = []
    with rasterio.open(raster_path) as raster:
        for geom in grids.geometry:
            population.append(area_weighted_population_for_grid(raster, geom, grids.crs))
    grids["population"] = population
    grids["PD"] = grids["population"] / grids.geometry.area
    return grids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Construct the ten built-environment variables used in the manuscript.")
    parser.add_argument("--grid-file", type=Path, required=True, help="GPKG produced by 01_build_hex_grid_and_ridership.py.")
    parser.add_argument("--poi-file", type=Path, required=True)
    parser.add_argument("--cbd-file", type=Path, required=True)
    parser.add_argument("--roads-file", type=Path, required=True)
    parser.add_argument("--nodes-file", type=Path, required=True)
    parser.add_argument("--population-raster", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-gpkg", type=Path)
    parser.add_argument("--point-crs", default="EPSG:4326")
    parser.add_argument("--lon-col", default="longitude")
    parser.add_argument("--lat-col", default="latitude")
    parser.add_argument("--poi-major-col", default="major_category")
    parser.add_argument("--poi-sub-col", default="sub_category")
    parser.add_argument("--transit-subcategory-pattern", default="bus|metro|subway")
    parser.add_argument("--min-street-count", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    grids = gpd.read_file(args.grid_file)
    if "grid_id" not in grids.columns or "SER" not in grids.columns:
        raise ValueError("Grid file must contain grid_id and SER columns.")

    poi = read_points(args.poi_file, args.lon_col, args.lat_col, args.point_crs)
    cbd = read_points(args.cbd_file, args.lon_col, args.lat_col, args.point_crs)
    roads = gpd.read_file(args.roads_file)
    nodes = gpd.read_file(args.nodes_file)

    grids = add_poi_metrics(grids, poi, args.poi_major_col, args.poi_sub_col, args.transit_subcategory_pattern)
    grids = add_cbd_distance(grids, cbd)
    grids = add_road_metrics(grids, roads, nodes, args.min_street_count)
    grids = add_population_density(grids, args.population_raster)

    output_cols = ["city", "grid_id", "SER"] + FEATURE_COLUMNS
    missing = [c for c in output_cols if c not in grids.columns]
    if missing:
        raise ValueError(f"Output is missing columns: {missing}")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    grids[output_cols].to_csv(args.output_csv, index=False)
    if args.output_gpkg:
        args.output_gpkg.parent.mkdir(parents=True, exist_ok=True)
        grids[output_cols + ["geometry"]].to_file(args.output_gpkg, driver="GPKG")
    print(f"Saved built-environment variables to {args.output_csv}")


if __name__ == "__main__":
    main()
