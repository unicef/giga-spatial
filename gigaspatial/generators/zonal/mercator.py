from typing import List, Optional, Union, Tuple, Iterable
from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry

import geopandas as gpd
import logging

from gigaspatial.core.io.data_store import DataStore
from gigaspatial.grid.mercator_tiles import MercatorTiles, CountryMercatorTiles
from gigaspatial.generators.zonal.base import (
    ZonalViewGeneratorConfig,
    T,
)
from gigaspatial.generators.zonal.geometry import GeometryBasedZonalViewGenerator


class MercatorViewGenerator(GeometryBasedZonalViewGenerator[T]):
    """
    Generates zonal views using Mercator tiles as the zones.

    This class specializes in creating zonal views where the zones are defined by
    Mercator tiles. It extends the `GeometryBasedZonalViewGenerator` and leverages
    the `MercatorTiles` and `CountryMercatorTiles` classes to generate tiles based on
    various input sources.

    The primary input source defines the geographical area of interest. This can be
    a country, a specific geometry, a set of points, or even a list of predefined
    quadkeys. The `zoom_level` determines the granularity of the Mercator tiles.

    Attributes:
        source (Union[str, BaseGeometry, gpd.GeoDataFrame, List[Union[Point, Tuple[float, float]]], List[str]]):
            Specifies the geographic area or specific tiles to use. Can be:
            - A country name (str): Uses `CountryMercatorTiles` to generate tiles covering the country.
            - A Shapely geometry (BaseGeometry):  Uses `MercatorTiles.from_spatial` to create tiles intersecting the geometry.
            - A GeoDataFrame (gpd.GeoDataFrame): Uses `MercatorTiles.from_spatial` to create tiles intersecting the geometries.
            - A list of points (List[Union[Point, Tuple[float, float]]]):  Uses `MercatorTiles.from_spatial` to create tiles containing the points.
            - A list of quadkeys (List[str]): Uses `MercatorTiles.from_quadkeys` to use the specified tiles directly.
        zoom_level (int): The zoom level of the Mercator tiles. Higher zoom levels result in smaller, more detailed tiles.
        predicate (str):  The spatial predicate used when filtering tiles based on a spatial source (e.g., "intersects", "contains"). Defaults to "intersects".
        config (Optional[ZonalViewGeneratorConfig]): Configuration for the zonal view generation process.
        data_store (Optional[DataStore]):  A DataStore instance for accessing data.
        logger (Optional[logging.Logger]):  A logger instance for logging.

    Methods:
        _init_zone_data(source, zoom_level, predicate):  Initializes the Mercator tile GeoDataFrame based on the input source.
        # Inherits other methods from GeometryBasedZonalViewGenerator, such as:
        # map_ghsl(), map_google_buildings(), map_ms_buildings(), aggregate_data(), save_view()

    Example:
        # Create a MercatorViewGenerator for tiles covering Germany at zoom level 6
        generator = MercatorViewGenerator(source="Germany", zoom_level=6)

        # Create a MercatorViewGenerator for tiles intersecting a specific polygon
        polygon = ... # Define a Shapely Polygon
        generator = MercatorViewGenerator(source=polygon, zoom_level=8)

        # Create a MercatorViewGenerator from a list of quadkeys
        quadkeys = ["0020023131023032", "0020023131023033"]
        generator = MercatorViewGenerator(source=quadkeys, zoom_level=12)
    """

    def __init__(
        self,
        source: Union[
            str,  # country
            BaseGeometry,  # shapely geom
            gpd.GeoDataFrame,
            List[Union[Point, Tuple[float, float]]],  # points
            List[str],  # quadkeys
        ],
        zoom_level: int,
        predicate="intersects",
        config: Optional[ZonalViewGeneratorConfig] = None,
        data_store: Optional[DataStore] = None,
        logger: logging.Logger = None,
    ):

        super().__init__(
            zone_data=self._init_zone_data(source, zoom_level, predicate, data_store),
            zone_id_column="quadkey",
            config=config,
            data_store=data_store,
            logger=logger,
        )
        self.logger.info(f"Initialized MercatorViewGenerator")

    def _init_zone_data(self, source, zoom_level, predicate, data_store=None):
        if isinstance(source, str):
            tiles = CountryMercatorTiles.create(
                country=source, zoom_level=zoom_level, data_store=data_store
            )
        elif isinstance(source, (BaseGeometry, Iterable)):
            if isinstance(source, Iterable) and all(
                isinstance(qk, str) for qk in source
            ):
                tiles = MercatorTiles.from_quadkeys(source)
            else:
                tiles = MercatorTiles.from_spatial(
                    source=source, zoom_level=zoom_level, predicate=predicate
                )
        else:
            raise TypeError(
                f"Unsupported source type for MercatorViewGenerator. 'source' must be "
                f"a country name (str), a Shapely geometry, a GeoDataFrame, "
                f"a list of quadkeys (str), or a list of (lon, lat) tuples/Shapely Point objects. "
                f"Received type: {type(source)}."
            )

        return tiles.to_geodataframe()
