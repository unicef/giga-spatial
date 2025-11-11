from typing import List, Optional, Union, Tuple, Iterable
from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry

import geopandas as gpd
import logging

from gigaspatial.core.io.data_store import DataStore
from gigaspatial.grid.s2 import S2Cells, CountryS2Cells
from gigaspatial.generators.zonal.base import (
    ZonalViewGeneratorConfig,
    T,
)
from gigaspatial.generators.zonal.geometry import GeometryBasedZonalViewGenerator


class S2ViewGenerator(GeometryBasedZonalViewGenerator[T]):
    """
    Generates zonal views using Google S2 cells as the zones.

    This mirrors the `MercatorViewGenerator` but uses S2 cells (levels 0-30)
    as zone units. The primary input source defines the geographic area of
    interest and the `level` determines the granularity of S2 cells.

    Attributes:
        source (Union[str, BaseGeometry, gpd.GeoDataFrame, List[Union[Point, Tuple[float, float]]], List[Union[int, str]]]):
            Specifies the geographic area or specific cells to use. Can be:
            - A country name (str): Uses `CountryS2Cells` to generate cells covering the country.
            - A Shapely geometry (BaseGeometry): Uses `S2Cells.from_spatial` to create cells intersecting the geometry.
            - A GeoDataFrame (gpd.GeoDataFrame): Uses `S2Cells.from_spatial` to create cells intersecting geometries or from points.
            - A list of points (List[Union[Point, Tuple[float, float]]]): Uses `S2Cells.from_points` via `from_spatial`.
            - A list of cell identifiers (List[Union[int, str]]): Uses `S2Cells.from_cells` (accepts integer IDs or token strings).
        level (int): The S2 level (0-30). Higher levels produce smaller cells.
        config (Optional[ZonalViewGeneratorConfig]): Configuration for generation.
        data_store (Optional[DataStore]): Optional data store instance.
        logger (Optional[logging.Logger]): Optional logger.
    """

    def __init__(
        self,
        source: Union[
            str,  # country
            BaseGeometry,  # shapely geom
            gpd.GeoDataFrame,
            List[Union[Point, Tuple[float, float]]],  # points
            List[Union[int, str]],  # cell ids or tokens
        ],
        level: int,
        config: Optional[ZonalViewGeneratorConfig] = None,
        data_store: Optional[DataStore] = None,
        logger: logging.Logger = None,
        max_cells: int = 1000,
    ):

        super().__init__(
            zone_data=self._init_zone_data(
                source, level, data_store=data_store, max_cells=max_cells
            ),
            zone_id_column="cell_token",
            config=config,
            data_store=data_store,
            logger=logger,
        )
        self.logger.info("Initialized S2ViewGenerator")

    def _init_zone_data(
        self,
        source,
        level: int,
        data_store: Optional[DataStore] = None,
        max_cells: int = 1000,
    ):
        if isinstance(source, str):
            cells = CountryS2Cells.create(
                country=source, level=level, data_store=data_store, max_cells=max_cells
            )
            self._country = source
        elif isinstance(source, (BaseGeometry, gpd.GeoDataFrame, Iterable)):
            # If it's an explicit cells list of ids/tokens
            if isinstance(source, Iterable) and all(
                isinstance(c, (int, str)) for c in source
            ):
                cells = S2Cells.from_cells(list(source))
            else:
                # Spatial extraction from geometry/points/gdf
                cells = S2Cells.from_spatial(
                    source=source, level=level, max_cells=max_cells
                )
        else:
            raise TypeError(
                "Unsupported source type for S2ViewGenerator. 'source' must be "
                "a country name (str), a Shapely geometry, a GeoDataFrame, "
                "a list of S2 cell ids/tokens (int/str), or a list of (lon, lat) tuples/Shapely Point objects. "
                f"Received type: {type(source)}."
            )

        return cells.to_geodataframe()

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
