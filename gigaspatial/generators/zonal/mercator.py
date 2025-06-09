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
    """Mid-level class for zonal view generation based on geometries with identifiers.

    This class serves as an intermediate between the abstract ZonalViewGenerator and specific
    implementations like MercatorViewGenerator or H3ViewGenerator. It handles the common case
    where zones are defined by a mapping between zone identifiers and geometries, either
    provided as a dictionary or as a GeoDataFrame.

    The class extends the base functionality with methods for mapping common geospatial
    datasets including GHSL (Global Human Settlement Layer), Google Open Buildings,
    and Microsoft Global Buildings data.

    Attributes:
        zone_dict (Dict[T, Polygon]): Mapping of zone identifiers to geometries.
        zone_id_column (str): Name of the column containing zone identifiers.
        zone_data_crs (str): Coordinate reference system of the zone data.
        _zone_gdf (gpd.GeoDataFrame): Cached GeoDataFrame representation of zones.
        data_store (DataStore): For accessing input data.
        generator_config (ZonalViewGeneratorConfig): Configuration for view generation.
        logger: Logger instance for this class.
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
            zone_data=self._init_zone_data(source, zoom_level, predicate),
            zone_id_column="quadkey",
            config=config,
            data_store=data_store,
            logger=logger,
        )

    def _init_zone_data(self, source, zoom_level, predicate):
        if isinstance(source, str):
            tiles = CountryMercatorTiles.create(country=source, zoom_level=zoom_level)
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
            raise ValueError("sadadasfasfkasmf")

        return tiles.to_geodataframe()
