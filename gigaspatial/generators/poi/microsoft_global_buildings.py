from pathlib import Path
from typing import List, Optional, Union, Any

import geopandas as gpd
import pandas as pd
from scipy.spatial import cKDTree
from shapely.geometry import shape

from gigaspatial.core.io.data_store import DataStore
from gigaspatial.core.io.readers import read_gzipped_json_or_csv
from gigaspatial.processing.geo import (
    calculate_distance,
    detect_coordinate_columns,
)
from gigaspatial.handlers.microsoft_global_buildings import MSBuildingsConfig
from gigaspatial.generators.poi.base import PoiViewGeneratorConfig, PoiViewGenerator


class MSBuildingsPoiViewGenerator(PoiViewGenerator):
    """Generate POI views from Microsoft Global Buildings data."""

    def __init__(
        self,
        data_config: Optional[MSBuildingsConfig] = None,
        generator_config: Optional[PoiViewGeneratorConfig] = None,
        data_store: Optional[DataStore] = None,
    ):
        super().__init__(generator_config=generator_config, data_store=data_store)
        self.data_config = data_config or MSBuildingsConfig(data_store=self.data_store)

    def _pre_load_hook(self, source_data_path, **kwargs) -> Any:
        """Pre-processing before loading data files."""

        # Convert single path to list for consistent handling
        if isinstance(source_data_path, (Path, str)):
            source_data_path = [source_data_path]

        source_data_path = [str(file_path) for file_path in source_data_path]

        # Validate all paths exist
        for file_path in source_data_path:
            if not self.data_store.file_exists(file_path):
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
        For Microsoft Buildings, resolve source data paths based on POI data geography.

        Returns:
            List of paths to relevant Microsoft Buildings tile files
        """
        # Return explicit paths if provided
        if explicit_paths is not None:
            if isinstance(explicit_paths, (str, Path)):
                return [explicit_paths]
            return list(explicit_paths)

        if "latitude" not in poi_data:
            poi_lat_col, poi_lon_col = detect_coordinate_columns(poi_data)
        else:
            poi_lat_col, poi_lon_col = ("latitude", "longitude")

        points = poi_data[[poi_lat_col, poi_lon_col]].to_numpy()

        # Find intersecting tiles
        tiles = self.data_config.get_tiles_for_points(points)

        if tiles.empty:
            self.logger.warning(
                "There are no matching Microsoft Buildings tiles for the POI data"
            )
            return []

        # Generate paths for each intersecting tile
        source_data_paths = [
            self.data_config.get_tile_path(
                quadkey=tile["quadkey"],
                location=tile["country"] if tile["country"] else tile["location"],
            )
            for _, tile in tiles.iterrows()
        ]

        self.logger.info(f"Resolved {len(source_data_paths)} tile paths for POI data")
        return source_data_paths

    def load_data(
        self, source_data_path: Union[Path, List[Path]], **kwargs
    ) -> gpd.GeoDataFrame:
        """
        Load building data from Microsoft Buildings dataset.

        Args:
            source_data_path: Path(s) to the source data files
            **kwargs: Additional loading parameters

        Returns:
            DataFrame containing building data
        """

        def read_ms_dataset(data_store: DataStore, file_path: str):
            df = read_gzipped_json_or_csv(file_path=file_path, data_store=data_store)
            df["geometry"] = df["geometry"].apply(shape)
            return gpd.GeoDataFrame(df, crs=4326)

        processed_paths = self._pre_load_hook(source_data_path, **kwargs)

        all_data = []
        for file_path in processed_paths:
            all_data.append(read_ms_dataset(self.data_store, file_path))

        if not all_data:
            return pd.DataFrame()

        # Concatenate all tile data
        result = pd.concat(all_data, ignore_index=True)

        return self._post_load_hook(result)

    def process_data(self, data: gpd.GeoDataFrame, **kwargs) -> pd.DataFrame:
        return data.get_coordinates()

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

        tree = cKDTree(processed_data[["y", "x"]])

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
            lat2=df_nearest_buildings.y,
            lon2=df_nearest_buildings.x,
        )

        poi_data["nearest_ms_building_id"] = df_nearest_buildings.get(
            "building_id", None
        )
        poi_data["nearest_ms_building_distance"] = dist

        return poi_data

    def generate_view(
        self,
        poi_data: Union[pd.DataFrame, gpd.GeoDataFrame],
        source_data_path: Optional[Union[Path, List[Path]]] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Generate POI view for Microsoft Global Buildings.

        Returns:
            Enhanced POI data with nearest building information
        """
        self.logger.info("Generating MicrosoftBuildings POI view")

        return self.generate_poi_view(
            poi_data=poi_data,
            source_data_path=source_data_path,
            **kwargs,
        )
