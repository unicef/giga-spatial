from pathlib import Path
from typing import Dict, List, Optional, Union, Any

import geopandas as gpd
import pandas as pd
from scipy.spatial import cKDTree

from gigaspatial.core.io.data_store import DataStore
from gigaspatial.core.io.readers import read_dataset
from gigaspatial.processing.geo import (
    convert_to_geodataframe,
    calculate_distance,
    detect_coordinate_columns,
)
from gigaspatial.handlers.google_open_buildings import (
    GoogleOpenBuildingsConfig,
    GoogleOpenBuildingsDownloader,
)
from gigaspatial.generators.poi.base import PoiViewGeneratorConfig, PoiViewGenerator


class GoogleBuildingsPoiViewGenerator(PoiViewGenerator):
    """Generate POI views from Google Open Buildings data."""

    def __init__(
        self,
        data_config: Optional[GoogleOpenBuildingsConfig] = None,
        generator_config: Optional[PoiViewGeneratorConfig] = None,
        data_store: Optional[DataStore] = None,
    ):
        super().__init__(generator_config=generator_config, data_store=data_store)
        self.data_config = data_config or GoogleOpenBuildingsConfig()
        self.handler = GoogleOpenBuildingsDownloader(
            config=self.data_config, data_store=self.data_store
        )

    def _pre_load_hook(self, source_data_path, **kwargs) -> Any:
        """Pre-processing before loading data files."""

        # Convert single path to list for consistent handling
        if isinstance(source_data_path, (Path, str)):
            source_data_path = [source_data_path]

        source_data_path = [str(file_path) for file_path in source_data_path]

        # Validate all paths exist
        for file_path in source_data_path:
            if not self.handler.data_store.file_exists(file_path):
                raise RuntimeError(
                    f"Source buildings file does not exist in the data store: {file_path}"
                )

        self.logger.info(
            f"Pre-loading validation complete for {len(source_data_path)} files"
        )
        return source_data_path

    def _post_load_hook(self, data, **kwargs) -> Any:
        """Post-processing after loading data files."""
        if data.empty:
            self.logger.warning("No data was loaded from the source files")
            return data

        self.logger.info(
            f"Post-load processing complete. {len(data)} valid building records."
        )
        return data

    def resolve_source_paths(
        self,
        poi_data: Union[pd.DataFrame, gpd.GeoDataFrame],
        explicit_paths: Optional[Union[Path, str, List[Union[str, Path]]]] = None,
        **kwargs,
    ) -> List[Union[str, Path]]:
        """
        For Google Open Buildings, resolve source data paths based on POI data geography. This determines which tile files
        intersect with the POI data's geographic extent.
        """
        if explicit_paths is not None:
            if isinstance(explicit_paths, (str, Path)):
                return [explicit_paths]
            return list(explicit_paths)

        # Convert to GeoDataFrame if needed
        if not isinstance(poi_data, gpd.GeoDataFrame):
            poi_data = convert_to_geodataframe(poi_data)

        # Find intersecting tiles
        intersection_tiles = self.handler._get_intersecting_tiles(poi_data)

        if intersection_tiles.empty:
            self.logger.warning(
                "There are no matching Google buildings tiles for the POI data"
            )
            return []

        # Generate paths for each intersecting tile
        source_data_paths = [
            self.data_config.get_tile_path(tile_id=tile, data_type="points")
            for tile in intersection_tiles.tile_id
        ]

        self.logger.info(f"Resolved {len(source_data_paths)} tile paths for POI data")
        return source_data_paths

    def load_data(
        self, source_data_path: Union[Path, List[Path]], **kwargs
    ) -> pd.DataFrame:
        """
        Load building data from paths.

        Args:
            source_data_path: Path(s) to the source data
            **kwargs: Additional loading parameters

        Returns:
            DataFrame containing building data
        """

        processed_paths = self._pre_load_hook(source_data_path, **kwargs)

        all_data = []
        for file_path in processed_paths:
            all_data.append(read_dataset(self.handler.data_store, file_path))

        if not all_data:
            return pd.DataFrame()

        # Concatenate all tile data
        result = pd.concat(all_data, ignore_index=True)

        return self._post_load_hook(result)

    def map_to_poi(
        self, processed_data: pd.DataFrame, poi_data: pd.DataFrame, **kwargs
    ) -> pd.DataFrame:
        """
        Map processed building data to POI data.

        Args:
            processed_data: Processed building data as GeoDataFrame
            poi_data: POI data to map to
            **kwargs: Additional mapping parameters

        Returns:
            DataFrame with POI data and nearest building information
        """

        tree = cKDTree(processed_data[["latitude", "longitude"]])

        if "latitude" not in poi_data:
            poi_lat_col, poi_lon_col = detect_coordinate_columns(poi_data)
            df_points = poi_data.rename(
                columns={poi_lat_col: "latitude", poi_lon_col: "longitude"}
            )
        else:
            df_points = poi_data.copy()

        _, idx = tree.query(df_points[["latitude", "longitude"]], k=1)

        df_nearest_buildings = processed_data.iloc[idx]

        dist = calculate_distance(
            lat1=df_points.latitude,
            lon1=df_points.longitude,
            lat2=df_nearest_buildings.latitude,
            lon2=df_nearest_buildings.longitude,
        )

        poi_data["nearest_google_building_id"] = (
            df_nearest_buildings.full_plus_code.to_numpy()
        )
        poi_data["nearest_google_building_distance"] = dist

        return poi_data

    def generate_view(
        self,
        poi_data: Union[pd.DataFrame, gpd.GeoDataFrame],
        source_data_path: Optional[Union[Path, List[Path]]] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Generate POI view for Google Open Buildings.

        Returns:
            Enhanced POI data with nearest building information
        """
        self.logger.info("Generating Google Open Buildings POI view")

        return self.generate_poi_view(
            poi_data=poi_data,
            source_data_path=source_data_path,
            **kwargs,
        )
