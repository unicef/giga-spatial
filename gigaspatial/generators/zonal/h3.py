from typing import List, Optional, Union, Tuple, Iterable
from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry

import geopandas as gpd
import logging

from gigaspatial.core.io.data_store import DataStore
from gigaspatial.grid.h3 import H3Hexagons, CountryH3Hexagons
from gigaspatial.generators.zonal.base import (
    ZonalViewGeneratorConfig,
    T,
)
from gigaspatial.generators.zonal.geometry import GeometryBasedZonalViewGenerator


class H3ViewGenerator(GeometryBasedZonalViewGenerator[T]):
    """
    Generates zonal views using H3 hexagons as the zones.

    Mirrors `MercatorViewGenerator`/`S2ViewGenerator` but uses H3 cells
    (resolutions 0-15). The input `source` defines the area/cells and
    `resolution` determines the granularity.

    Supported sources:
    - Country string → `CountryH3Hexagons.create`
    - Shapely geometry or GeoDataFrame → `H3Hexagons.from_spatial`
    - List of points (shapely Points or (lon, lat) tuples) → `from_spatial`
    - List of H3 indexes (strings) → `H3Hexagons.from_hexagons`
    """

    def __init__(
        self,
        source: Union[
            str,  # country
            BaseGeometry,  # shapely geom
            gpd.GeoDataFrame,
            List[Union[Point, Tuple[float, float]]],  # points
            List[str],  # h3 indexes
        ],
        resolution: int,
        contain: str = "overlap",
        config: Optional[ZonalViewGeneratorConfig] = None,
        data_store: Optional[DataStore] = None,
        logger: logging.Logger = None,
    ):

        super().__init__(
            zone_data=self._init_zone_data(
                source, resolution, contain=contain, data_store=data_store
            ),
            zone_id_column="h3",
            config=config,
            data_store=data_store,
            logger=logger,
        )
        self.logger.info("Initialized H3ViewGenerator")

    def _init_zone_data(
        self,
        source,
        resolution: int,
        contain: str = "overlap",
        data_store: Optional[DataStore] = None,
    ):
        if isinstance(source, str):
            hexes = CountryH3Hexagons.create(
                country=source,
                resolution=resolution,
                contain=contain,
                data_store=data_store,
            )
            self._country = source
        elif isinstance(source, (BaseGeometry, gpd.GeoDataFrame, Iterable)):
            if isinstance(source, Iterable) and all(isinstance(h, str) for h in source):
                hexes = H3Hexagons.from_hexagons(list(source))
            else:
                hexes = H3Hexagons.from_spatial(
                    source=source, resolution=resolution, contain=contain
                )
        else:
            raise TypeError(
                "Unsupported source type for H3ViewGenerator. 'source' must be "
                "a country name (str), a Shapely geometry, a GeoDataFrame, "
                "a list of H3 indexes (str), or a list of (lon, lat) tuples/Shapely Point objects. "
                f"Received type: {type(source)}."
            )

        return hexes.to_geodataframe()

    def map_wp_pop(
        self,
        country=None,
        resolution=1000,
        predicate="intersects",
        output_column="population",
        **kwargs,
    ):
        if hasattr(self, "_country") and country is None:
            country = self._country

        return super().map_wp_pop(
            country, resolution, predicate, output_column, **kwargs
        )
