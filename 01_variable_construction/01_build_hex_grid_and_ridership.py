from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Polygon


def regular_hexagon(cx: float, cy: float, radius: float) -> Polygon:
    angles = np.deg2rad(np.arange(0, 360, 60))
    return Polygon([(cx + radius * math.cos(a), cy + radius * math.sin(a)) for a in angles])


def make_hex_grid(boundary: gpd.GeoDataFrame, radius: float, crs: str) -> gpd.GeoDataFrame:
    boundary = boundary.to_crs(crs)
    union = boundary.geometry.union_all()
    minx, miny, maxx, maxy = union.bounds
    dx = math.sqrt(3) * radius
    dy = 1.5 * radius

    hexes = []
    y = miny - radius
    row = 0
    while y <= maxy + radius:
        x_offset = 0.5 * dx if row % 2 else 0.0
        x = minx - dx + x_offset
        while x <= maxx + dx:
            poly = regular_hexagon(x, y, radius)
            if poly.intersects(union):
                hexes.append(poly.intersection(union))
            x += dx
        y += dy
        row += 1

    grids = gpd.GeoDataFrame({"grid_id": range(1, len(hexes) + 1)}, geometry=hexes, crs=crs)
    grids = grids[~grids.geometry.is_empty].copy()
    grids["grid_area_m2"] = grids.geometry.area
    grids["centroid_x"] = grids.geometry.centroid.x
    grids["centroid_y"] = grids.geometry.centroid.y
    return grids


def clean_trips(args: argparse.Namespace) -> pd.DataFrame:
    trips = pd.read_csv(args.trip_csv)
    required = [
        args.start_lon,
        args.start_lat,
        args.end_lon,
        args.end_lat,
        args.distance_col,
        args.duration_col,
    ]
    missing = [c for c in required if c not in trips.columns]
    if missing:
        raise ValueError(f"Trip table is missing required columns: {missing}")

    trips = trips.dropna(subset=required).copy()
    trips = trips[
        (trips[args.distance_col] >= 50)
        & (trips[args.distance_col] <= 15000)
        & (trips[args.duration_col] >= 1)
        & (trips[args.duration_col] <= 120)
    ].copy()
    return trips


def points_from_clean_trips(
    trips: pd.DataFrame,
    lon_col: str,
    lat_col: str,
    source_crs: str,
    target_crs: str,
) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        trips,
        geometry=gpd.points_from_xy(trips[lon_col], trips[lat_col]),
        crs=source_crs,
    ).to_crs(target_crs)


def count_points_in_grids(points: gpd.GeoDataFrame, grids: gpd.GeoDataFrame) -> pd.Series:
    joined = gpd.sjoin(points[["geometry"]], grids[["grid_id", "geometry"]], how="inner", predicate="within")
    return joined.groupby("grid_id").size()


def filter_trips_to_boundary(
    trips: pd.DataFrame,
    boundary: gpd.GeoDataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    boundary_wgs84 = boundary.to_crs(args.input_crs)
    start = points_from_clean_trips(trips, args.start_lon, args.start_lat, args.input_crs, args.input_crs)
    end = points_from_clean_trips(trips, args.end_lon, args.end_lat, args.input_crs, args.input_crs)
    inside_start = gpd.sjoin(start[["geometry"]], boundary_wgs84[["geometry"]], how="inner", predicate="within").index
    inside_end = gpd.sjoin(end[["geometry"]], boundary_wgs84[["geometry"]], how="inner", predicate="within").index
    keep = trips.index.intersection(inside_start).intersection(inside_end)
    return trips.loc[keep].copy()


def build_city_grid_ridership(args: argparse.Namespace) -> Tuple[gpd.GeoDataFrame, pd.DataFrame]:
    boundary = gpd.read_file(args.boundary_file)
    trips = clean_trips(args)
    trips = filter_trips_to_boundary(trips, boundary, args)

    if len(trips) < args.min_city_trips:
        raise ValueError(
            f"{args.city} has {len(trips)} cleaned trips; the manuscript retained "
            f"cities with at least {args.min_city_trips} trips."
        )

    grids = make_hex_grid(boundary, args.hex_radius_m, args.projected_crs)
    starts = points_from_clean_trips(trips, args.start_lon, args.start_lat, args.input_crs, args.projected_crs)
    ends = points_from_clean_trips(trips, args.end_lon, args.end_lat, args.input_crs, args.projected_crs)

    grids["origin_count"] = grids["grid_id"].map(count_points_in_grids(starts, grids)).fillna(0).astype(int)
    grids["destination_count"] = grids["grid_id"].map(count_points_in_grids(ends, grids)).fillna(0).astype(int)
    grids["SER"] = grids["origin_count"] + grids["destination_count"]
    valid = grids[grids["SER"] > args.min_grid_rides].copy()
    if len(valid) < args.min_effective_grids:
        raise ValueError(
            f"{args.city} has {len(valid)} effective grids after SER > {args.min_grid_rides}; "
            f"the manuscript retained cities with at least {args.min_effective_grids} effective grids."
        )
    valid["city"] = args.city
    return valid, trips


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build 500 m hexagonal grids and weekly realised HelloBike ridership.")
    parser.add_argument("--city", required=True)
    parser.add_argument("--trip-csv", type=Path, required=True)
    parser.add_argument("--boundary-file", type=Path, required=True)
    parser.add_argument("--output-grid", type=Path, required=True)
    parser.add_argument("--output-cleaned-trip-summary", type=Path)
    parser.add_argument("--input-crs", default="EPSG:4326")
    parser.add_argument("--projected-crs", default="EPSG:3857")
    parser.add_argument("--hex-radius-m", type=float, default=500.0)
    parser.add_argument("--min-city-trips", type=int, default=20000)
    parser.add_argument("--min-grid-rides", type=int, default=7)
    parser.add_argument("--min-effective-grids", type=int, default=100)
    parser.add_argument("--start-lon", default="start_lng")
    parser.add_argument("--start-lat", default="start_lat")
    parser.add_argument("--end-lon", default="end_lng")
    parser.add_argument("--end-lat", default="end_lat")
    parser.add_argument("--distance-col", default="ride_distance")
    parser.add_argument("--duration-col", default="ride_time")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    grids, trips = build_city_grid_ridership(args)
    args.output_grid.parent.mkdir(parents=True, exist_ok=True)
    grids.to_file(args.output_grid, driver="GPKG")
    if args.output_cleaned_trip_summary:
        args.output_cleaned_trip_summary.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [{
                "city": args.city,
                "cleaned_trip_count": len(trips),
                "effective_grid_count": len(grids),
                "total_SER": int(grids["SER"].sum()),
            }]
        ).to_csv(args.output_cleaned_trip_summary, index=False)
    print(f"Saved valid grids and SER counts to {args.output_grid}")


if __name__ == "__main__":
    main()
