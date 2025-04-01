from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Literal

import geopandas as gpd
import pandas as pd

from gigaspatial.core.io.data_store import DataStore
from gigaspatial.processing.geo import convert_to_geodataframe, buffer_geodataframe
from gigaspatial.processing.tif_processor import (
    TifProcessor,
    sample_multiple_tifs_by_polygons,
)
from gigaspatial.handlers.ghsl import GHSLDataConfig
from gigaspatial.generators.poi.base import PoiViewGeneratorConfig, PoiViewGenerator


class GhslBuiltSurfacePoiViewGenerator(PoiViewGenerator):
    """Generate POI views from GHSL Built Surface."""

    def __init__(
        self,
        data_config: Optional[GHSLDataConfig] = None,
        generator_config: Optional[PoiViewGeneratorConfig] = None,
        data_store: Optional[DataStore] = None,
    ):
        super().__init__(generator_config=generator_config, data_store=data_store)
        self.data_config = data_config or GHSLDataConfig(
            product="GHS_BUILT_S", year=2020, resolution=100, coord_system=4326
        )

    def _pre_load_hook(self, source_data_path, **kwargs) -> Any:
        """Pre-processing before loading data files."""

        # Convert single path to list for consistent handling
        if isinstance(source_data_path, (Path, str)):
            source_data_path = [source_data_path]

        # change source suffix, .zip, to .tif and paths to string
        source_data_path = [
            str(file_path.with_suffix(".tif")) for file_path in source_data_path
        ]

        # Validate all paths exist
        for file_path in source_data_path:
            if not self.data_store.file_exists(file_path):
                raise RuntimeError(
                    f"Source raster does not exist in the data store: {file_path}"
                )

        self.logger.info(
            f"Pre-loading validation complete for {len(source_data_path)} files"
        )
        return source_data_path

    def _post_load_hook(self, data, **kwargs) -> Any:
        """Post-processing after loading data files."""
        if not data:
            self.logger.warning("No data was loaded from the source files")
            return data

        self.logger.info(
            f"Post-load processing complete. {len(data)} valid TifProcessors."
        )
        return data

    def resolve_source_paths(
        self,
        poi_data: Union[pd.DataFrame, gpd.GeoDataFrame],
        explicit_paths: Optional[Union[Path, str, List[Union[str, Path]]]] = None,
        **kwargs,
    ) -> List[Union[str, Path]]:
        """
        For GHSL Built Surface rasters, resolve source data paths based on POI data geography.

        Returns:
            List of paths to relevant GHSL BUILT S tile rasters
        """
        if explicit_paths is not None:
            if isinstance(explicit_paths, (str, Path)):
                return [explicit_paths]
            return list(explicit_paths)

        # Convert to GeoDataFrame if needed
        if isinstance(poi_data, pd.DataFrame):
            poi_data = convert_to_geodataframe(poi_data)

        # Find intersecting tiles
        intersection_tiles = self.data_config.get_intersecting_tiles(
            geometry=poi_data, crs=poi_data.crs
        )

        if not intersection_tiles:
            self.logger.warning("There are no matching GHSL tiles for the POI data")
            return []

        # Generate paths for each intersecting tile
        source_data_paths = [
            self.data_config.get_tile_path(tile_id=tile) for tile in intersection_tiles
        ]

        self.logger.info(f"Resolved {len(source_data_paths)} tile paths for POI data")
        return source_data_paths

    def load_data(
        self, source_data_path: Union[Path, List[Path]], **kwargs
    ) -> List[TifProcessor]:
        """
        Load GHSL Built Surface rasters into TifProcessors from paths.

        Args:
            source_data_path: Path(s) to the source data
            **kwargs: Additional loading parameters

        Returns:
            List of TifProcessors with built surface data
        """

        processed_paths = self._pre_load_hook(source_data_path, **kwargs)

        tif_processors = [
            TifProcessor(data_path, self.data_store, mode="single")
            for data_path in processed_paths
        ]

        return self._post_load_hook(tif_processors)

    def map_to_poi(
        self,
        processed_data: List[TifProcessor],
        poi_data: Union[pd.DataFrame, gpd.GeoDataFrame],
        map_radius_meters: float,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Map from TifProcessors to POI data.

        Args:
            processed_data: TifProcessors
            poi_data: POI data to map to
            **kwargs: Additional mapping parameters

        Returns:
            DataFrame with POI data and built surface information
        """

        # Convert to GeoDataFrame if needed
        if not isinstance(poi_data, gpd.GeoDataFrame):
            gdf_points = convert_to_geodataframe(poi_data)
        else:
            gdf_points = poi_data

        gdf_points = gdf_points.to_crs(self.data_config.crs)

        polygon_list = buffer_geodataframe(
            gdf_points, buffer_distance_meters=map_radius_meters, cap_style="round"
        ).geometry

        sampled_values = sample_multiple_tifs_by_polygons(
            tif_processors=processed_data, polygon_list=polygon_list, stat="sum"
        )

        poi_data["built_surface_m2"] = sampled_values

        return poi_data

    def generate_view(
        self,
        poi_data: Union[pd.DataFrame, gpd.GeoDataFrame],
        source_data_path: Optional[Union[Path, List[Path]]] = None,
        map_radius_meters: float = 150,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Generate POI view from GHSL Built Surface.

        Returns:
            Enhanced POI data with Built Surface information
        """
        self.logger.info("Generating GHSL Built Surface POI view")

        return self.generate_poi_view(
            poi_data=poi_data,
            source_data_path=source_data_path,
            map_radius_meters=map_radius_meters,
            **kwargs,
        )
