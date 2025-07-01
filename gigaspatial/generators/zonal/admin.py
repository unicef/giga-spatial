from typing import Optional, Union
from pathlib import Path

import logging

from gigaspatial.core.io.data_store import DataStore
from gigaspatial.handlers.boundaries import AdminBoundaries
from gigaspatial.generators.zonal.base import (
    ZonalViewGeneratorConfig,
    T,
)
from gigaspatial.generators.zonal.geometry import GeometryBasedZonalViewGenerator


class AdminBoundariesViewGenerator(GeometryBasedZonalViewGenerator[T]):
    """
    Generates zonal views using administrative boundaries as the zones.

    This class specializes in creating zonal views where the zones are defined by
    administrative boundaries (e.g., countries, states, districts) at a specified
    administrative level. It extends the `GeometryBasedZonalViewGenerator` and
    leverages the `AdminBoundaries` handler to load the necessary geographical data.

    The administrative boundaries serve as the base geometries to which other
    geospatial data (points, polygons, rasters) can be mapped and aggregated.

    Attributes:
        country (str): The name or code of the country for which to load administrative boundaries.
        admin_level (int): The administrative level to load (e.g., 0 for country, 1 for states/provinces).
        admin_path (Union[str, Path], optional): Optional path to a local GeoJSON/Shapefile
            containing the administrative boundaries. If provided, this local file will be
            used instead of downloading.
        config (Optional[ZonalViewGeneratorConfig]): Configuration for the zonal view generation process.
        data_store (Optional[DataStore]): A DataStore instance for accessing data.
        logger (Optional[logging.Logger]): A logger instance for logging messages.
    """

    def __init__(
        self,
        country: str,
        admin_level: int,
        data_store: Optional[DataStore] = None,
        admin_path: Optional[Union[str, Path]] = None,
        config: Optional[ZonalViewGeneratorConfig] = None,
        logger: logging.Logger = None,
    ):
        """
        Initializes the AdminBoundariesViewGenerator.

        Args:
            country (str): The name or code of the country (e.g., "USA", "Germany").
            admin_level (int): The administrative level to load (e.g., 0 for country, 1 for states, 2 for districts).
            admin_path (Union[str, Path], optional): Path to a local administrative boundaries file (GeoJSON, Shapefile).
                                                     If provided, overrides default data loading.
            config (Optional[ZonalViewGeneratorConfig]): Configuration for the zonal view generator.
                                                         If None, a default config will be used.
            data_store (Optional[DataStore]): Data storage interface. If None, LocalDataStore is used.
            logger (Optional[logging.Logger]): Custom logger instance. If None, a default logger is used.
        """

        super().__init__(
            zone_data=self._init_zone_data(
                country, admin_level, data_store, admin_path
            ),
            zone_id_column="id",
            config=config,
            data_store=data_store,
            logger=logger,
        )
        self.logger.info(
            f"Initialized AdminBoundariesViewGenerator for {country} (level {admin_level})"
        )

    def _init_zone_data(
        self,
        country,
        admin_level,
        data_store: Optional[DataStore] = None,
        admin_path: Optional[Union[str, Path]] = None,
    ):
        gdf_boundaries = AdminBoundaries.create(
            country, admin_level, data_store, admin_path
        ).to_geodataframe()
        return gdf_boundaries
