from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Literal, Optional, Sequence, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd

from gigaspatial.config import config
from gigaspatial.processing.geo import buffer_geodataframe, calculate_distance


SourceFilter = Optional[Literal["google", "microsoft"]]

LOGGER = config.get_logger("GoogleMSBuildingsEngine")


@dataclass(frozen=True)
class BuildingCountsResult:
    """Result of counting buildings per zone."""

    counts: pd.Series


@dataclass(frozen=True)
class NearestBuildingsResult:
    """Result of nearest-building search for POIs."""

    distances_m: pd.Series


class GoogleMSBuildingsEngine:
    """
    Shared building-processing engine used by multiple view generators.

    This module intentionally contains the "heavy" logic (partitioning, job creation,
    per-tile scans, and accumulation) so generator classes can remain focused on
    view orchestration and UX.
    """

    # -----------------------------
    # Shared S2 job creation
    # -----------------------------
    @staticmethod
    def _s2_grid_gdf_from_building_files(
        building_files: Sequence[Path],
    ) -> gpd.GeoDataFrame:
        """Build an S2 grid GeoDataFrame with a `filepath` column from S2-tile filenames."""
        from gigaspatial.grid.s2 import S2Cells

        cells = {int(p.stem): p for p in building_files}
        s2_grid = S2Cells.from_cells(cells.keys())
        grid_gdf = s2_grid.to_geodataframe()
        grid_gdf["filepath"] = grid_gdf.cell_id.map(cells)
        return grid_gdf

    @classmethod
    def create_partitioned_jobs_for_zones(
        cls,
        zones_gdf: gpd.GeoDataFrame,
        building_files: Sequence[Path],
        *,
        predicate: Literal["intersects"] = "intersects",
    ) -> List[Tuple[Path, np.ndarray]]:
        """
        Create job list for partitioned building data by finding intersecting S2 cells.

        Returns list of (filepath, zone_id_array) tuples.
        """
        grid_gdf = cls._s2_grid_gdf_from_building_files(building_files)

        zone_to_cell_map = gpd.sjoin(
            zones_gdf,
            grid_gdf,
            how="inner",
            predicate=predicate,
        )

        if len(zone_to_cell_map) == 0:
            return []

        jobs: List[Tuple[Path, np.ndarray]] = []
        for filepath, group in zone_to_cell_map.groupby("filepath"):
            jobs.append((filepath, group.zone_id.values))
        return jobs

    @classmethod
    def create_partitioned_jobs_for_pois(
        cls,
        pois_gdf: gpd.GeoDataFrame,
        building_files: Sequence[Path],
        *,
        search_radius_m: float,
        predicate: Literal["intersects"] = "intersects",
    ) -> List[Tuple[Path, np.ndarray]]:
        """
        Create job list for partitioned building data by buffering POIs and finding intersecting S2 cells.

        Returns list of (filepath, poi_id_array) tuples.
        """
        grid_gdf = cls._s2_grid_gdf_from_building_files(building_files)

        buffered_pois = buffer_geodataframe(
            pois_gdf,
            buffer_distance_meters=search_radius_m,
        )

        poi_to_cell_map = gpd.sjoin(
            buffered_pois,
            grid_gdf,
            how="inner",
            predicate=predicate,
        )

        if len(poi_to_cell_map) == 0:
            return []

        jobs: List[Tuple[Path, np.ndarray]] = []
        for filepath, group in poi_to_cell_map.groupby("filepath"):
            jobs.append((filepath, group.poi_id.values))
        return jobs

    # -----------------------------
    # Zonal: count buildings
    # -----------------------------
    @classmethod
    def count_buildings_in_zones(
        cls,
        *,
        handler,
        building_files: Sequence[Path],
        zones_gdf: gpd.GeoDataFrame,
        source_filter: SourceFilter,
        logger=None,
    ) -> BuildingCountsResult:
        """
        Count buildings intersecting each zone.

        Expects:
        - `zones_gdf` contains columns: `zone_id`, `geometry`
        - `building_files` is a sequence of paths (single-file or S2-partitioned)
        """
        from shapely.strtree import STRtree

        logger = logger or LOGGER

        global_counts = pd.Series(0, index=zones_gdf.zone_id, dtype=int)

        def _iter_jobs() -> Iterable[Tuple[Path, np.ndarray, bool]]:
            if len(building_files) == 1:
                yield (building_files[0], zones_gdf.zone_id.values, True)
                return
            jobs = cls.create_partitioned_jobs_for_zones(zones_gdf, building_files)
            for fp, zone_ids in jobs:
                yield (fp, zone_ids, False)

        jobs_list = list(_iter_jobs())
        if len(building_files) > 1 and len(jobs_list) == 0:
            # Partitioned case but no intersecting tiles
            return BuildingCountsResult(counts=global_counts)

        logger.info(f"Processing {len(jobs_list)} building file(s)...")

        for filepath, zone_ids, is_single_file in jobs_list:
            try:
                columns_to_read = (
                    ["geometry"] if source_filter is None else ["bf_source", "geometry"]
                )
                buildings = handler.reader.load(filepath, columns=columns_to_read)

                if source_filter is not None and len(buildings) > 0:
                    buildings = buildings.loc[buildings["bf_source"] == source_filter]

                if len(buildings) == 0:
                    logger.debug(f"No buildings in {filepath.name} after filtering")
                    continue

                subset_zones = zones_gdf.loc[zones_gdf.zone_id.isin(zone_ids)].copy()
                subset_zones = subset_zones.reset_index(drop=True)

                tree = STRtree(buildings.geometry)
                zone_idxs, _ = tree.query(subset_zones.geometry, predicate="intersects")
                building_counts = np.bincount(zone_idxs, minlength=len(subset_zones))

                zone_id_array = subset_zones.zone_id.values
                if is_single_file:
                    global_counts.loc[zone_id_array] = building_counts
                else:
                    global_counts.loc[zone_id_array] += building_counts

                updated_zones = int((building_counts > 0).sum())
                logger.info(
                    f"Processed {filepath.name} - {updated_zones}/{len(subset_zones)} zones have buildings"
                )
            except Exception as e:
                logger.error(f"Failed to process {filepath.name}: {str(e)}")

        return BuildingCountsResult(counts=global_counts)

    # -----------------------------
    # POI: nearest building distance
    # -----------------------------
    @classmethod
    def nearest_buildings_to_pois(
        cls,
        *,
        handler,
        building_files: Sequence[Path],
        pois_gdf: gpd.GeoDataFrame,
        source_filter: SourceFilter,
        search_radius_m: float,
        logger=None,
    ) -> NearestBuildingsResult:
        """
        Find the nearest building distance (meters) per POI (haversine).

        Expects `pois_gdf` contains columns: `poi_id`, `geometry` (EPSG:4326).
        """
        from scipy.spatial import cKDTree

        logger = logger or LOGGER

        global_min_dists = pd.Series(np.inf, index=pois_gdf.poi_id, dtype=float)

        if len(building_files) == 1:
            jobs: List[Tuple[Path, np.ndarray, bool]] = [
                (building_files[0], pois_gdf.poi_id.values, True)
            ]
        else:
            jobs_list = cls.create_partitioned_jobs_for_pois(
                pois_gdf,
                building_files,
                search_radius_m=search_radius_m,
            )
            jobs = [(fp, poi_ids, False) for fp, poi_ids in jobs_list]

        if len(building_files) > 1 and len(jobs) == 0:
            return NearestBuildingsResult(distances_m=global_min_dists)

        logger.info(f"Processing {len(jobs)} building file(s)...")

        for filepath, poi_ids, is_single_file in jobs:
            try:
                columns_to_read = (
                    ["geometry"] if source_filter is None else ["bf_source", "geometry"]
                )
                buildings = handler.reader.load(filepath, columns=columns_to_read)

                if source_filter is not None and len(buildings) > 0:
                    buildings = buildings.loc[buildings["bf_source"] == source_filter]

                if len(buildings) == 0:
                    logger.debug(f"No buildings in {filepath.name} after filtering")
                    continue

                subset_pois = pois_gdf.loc[pois_gdf.poi_id.isin(poi_ids)]
                if len(subset_pois) == 0:
                    continue

                poi_coords = np.vstack(
                    (subset_pois.geometry.x, subset_pois.geometry.y)
                ).T

                b_centroids = buildings.geometry.centroid
                b_coords = np.vstack((b_centroids.x, b_centroids.y)).T
                if len(b_coords) == 0:
                    continue

                tree = cKDTree(b_coords)
                _, building_idxs = tree.query(poi_coords, k=1)

                nearest_b_coords = b_coords[building_idxs]
                distances_meters = calculate_distance(
                    lat1=poi_coords[:, 1],
                    lon1=poi_coords[:, 0],
                    lat2=nearest_b_coords[:, 1],
                    lon2=nearest_b_coords[:, 0],
                )

                if is_single_file:
                    global_min_dists.loc[poi_ids] = distances_meters
                else:
                    current_bests = global_min_dists.loc[poi_ids].values
                    improvement_mask = distances_meters < current_bests
                    if improvement_mask.any():
                        improved_poi_ids = poi_ids[improvement_mask]
                        improved_dists = distances_meters[improvement_mask]
                        global_min_dists.loc[improved_poi_ids] = improved_dists

                logger.info(f"Processed {filepath.name} - updated {len(poi_ids)} POIs")
            except Exception as e:
                logger.error(f"Failed to process {filepath.name}: {str(e)}")

        # Replace inf with NaN at the callsite (so callers can choose how to represent missing)
        return NearestBuildingsResult(distances_m=global_min_dists)
