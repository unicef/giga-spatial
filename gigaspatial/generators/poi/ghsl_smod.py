from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Literal

import geopandas as gpd
import pandas as pd

from gigaspatial.core.io.data_store import DataStore
from gigaspatial.processing.geo import (
    convert_to_geodataframe,
)
from gigaspatial.processing.tif_processor import (
    TifProcessor,
    sample_multiple_tifs_by_coordinates,
)
from gigaspatial.handlers.ghsl import GHSLDataConfig
from gigaspatial.generators.poi.base import PoiViewGeneratorConfig, PoiViewGenerator


class GhslSmodPoiViewGenerator(PoiViewGenerator):
    """Generate POI views from GHSL Settlement Model (SMOD)."""

    def __init__(
        self,
        data_config: Optional[GHSLDataConfig] = None,
        generator_config: Optional[PoiViewGeneratorConfig] = None,
        data_store: Optional[DataStore] = None,
    ):
        super().__init__(generator_config=generator_config, data_store=data_store)
        self.data_config = data_config or GHSLDataConfig(
            product="GHS_SMOD", year=2020, resolution=1000, coord_system=54009
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
        For GHSL SMOD rasters, resolve source data paths based on POI data geography.

        Returns:
            List of paths to relevant GHSL SMOD tile rasters
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
        Load GHSL SMOD rasters into TifProcessors from paths.

        Args:
            source_data_path: Path(s) to the source data
            **kwargs: Additional loading parameters

        Returns:
            List of TifProcessors with settlement model data
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
        **kwargs,
    ) -> pd.DataFrame:
        """
        Map from TifProcessors to POI data.

        Args:
            processed_data: TifProcessors
            poi_data: POI data to map to
            **kwargs: Additional mapping parameters

        Returns:
            DataFrame with POI data and SMOD classification information
        """

        # Convert to GeoDataFrame if needed
        if not isinstance(poi_data, gpd.GeoDataFrame):
            gdf_points = convert_to_geodataframe(poi_data)
        else:
            gdf_points = poi_data

        gdf_points = gdf_points.to_crs(self.data_config.crs)

        coord_list = [
            (x, y) for x, y in zip(gdf_points["geometry"].x, gdf_points["geometry"].y)
        ]

        sampled_values = sample_multiple_tifs_by_coordinates(
            tif_processors=processed_data, coordinate_list=coord_list
        )

        poi_data["smod_class"] = sampled_values

        return poi_data

    def generate_view(
        self,
        poi_data: Union[pd.DataFrame, gpd.GeoDataFrame],
        source_data_path: Optional[Union[Path, List[Path]]] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Generate POI view from GHSL Settlement Model.

        Returns:
            Enhanced POI data with SMOD classification
        """
        self.logger.info("Generating GHSL Settlement Model POI view")

        return self.generate_poi_view(
            poi_data=poi_data,
            source_data_path=source_data_path,
            **kwargs,
        )
