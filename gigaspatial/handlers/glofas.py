"""
GloFAS global river flood hazard maps (Copernicus EMS, JRC), version 2.1.2.

Publisher: Copernicus Emergency Management Service, Joint Research Centre.
Licence: CC BY 4.0.
Landing page: https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/CEMS-GLOFAS/flood_hazard/

A data unit is one GeoTIFF: one source tile, one layer, and, for the depth
layers, one return period. The tile grid is the published
``tile_extents.geojson``. The base turns a country or a geometry into the
search geometry; this config keeps the tiles that geometry intersects.

Layers are ``depth``, ``depth_reclass``, ``permanent_water``, and
``spurious_depth``. One load uses one return period, default 100. The mask
layers have no return period, and supplying one is an error. The reader
returns a ``TifProcessor`` per stored tile.
"""

from __future__ import annotations

import dataclasses
import io
from pathlib import Path
from typing import ClassVar, List, Literal, Optional, Tuple, get_args

import geopandas as gpd
import requests
from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass
from shapely.geometry.base import BaseGeometry

from gigaspatial.config import config as global_config
from gigaspatial.handlers.base import (
    BaseHandler,
    BaseHandlerConfig,
    BaseHandlerDownloader,
    BaseHandlerReader,
    DownloadOptions,
    RasterReadOptions,
    ResolveOptions,
)
from gigaspatial.handlers.base.fetch import stream_to_store, tiles_intersecting
from gigaspatial.processing.tif_processor import TifProcessor


BASE_URL = "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/CEMS-GLOFAS/flood_hazard"
TILES_URL = f"{BASE_URL}/tile_extents.geojson"
RETURN_PERIODS: Tuple[int, ...] = (10, 20, 50, 75, 100, 200, 500)
DEFAULT_RETURN_PERIOD = 100

LayerName = Literal["depth", "depth_reclass", "permanent_water", "spurious_depth"]
LAYERS: Tuple[str, ...] = get_args(LayerName)
DEPTH_LAYERS = frozenset({"depth", "depth_reclass"})
MASK_LAYERS = frozenset({"permanent_water", "spurious_depth"})


def _filename(tile_id: int, tile_name: str, layer: str, return_period: Optional[int]) -> str:
    if layer == "depth":
        return f"ID{tile_id}_{tile_name}_RP{return_period}_depth.tif"
    if layer == "depth_reclass":
        return f"ID{tile_id}_{tile_name}_RP{return_period}_depth_reclass.tif"
    if layer == "permanent_water":
        return f"ID{tile_id}_{tile_name}_permanent_water.tif"
    return f"ID{tile_id}_{tile_name}_spurious_depth_areas.tif"


def _remote_url(tile_id: int, tile_name: str, layer: str, return_period: Optional[int]) -> str:
    filename = _filename(tile_id, tile_name, layer, return_period)
    if layer in DEPTH_LAYERS:
        return f"{BASE_URL}/RP{return_period}/{filename}"
    if layer == "permanent_water":
        return f"{BASE_URL}/Permanent_WaterBodies/{filename}"
    return f"{BASE_URL}/Spurious_Depths/{filename}"


@dataclasses.dataclass(frozen=True)
class GlofasResolveOptions(ResolveOptions):
    """
    Resolve options for one GloFAS load.

    Attributes:
        layer: Which map to fetch. Defaults to raw water depth.
        return_period: Return period in years for a depth layer. Omitted means
            100. Supplying it for a mask layer is an error.
    """

    layer: LayerName = "depth"
    return_period: Optional[int] = None

    def __post_init__(self):
        if self.layer not in LAYERS:
            raise ValueError(f"layer must be one of {LAYERS}, got {self.layer!r}")
        if self.layer in MASK_LAYERS:
            if self.return_period is not None:
                raise ValueError(
                    f"{self.layer} has no return period; "
                    "do not pass return_period"
                )
            return
        period = DEFAULT_RETURN_PERIOD if self.return_period is None else self.return_period
        if period not in RETURN_PERIODS:
            raise ValueError(
                f"return_period must be one of {RETURN_PERIODS}, got {self.return_period!r}"
            )
        object.__setattr__(self, "return_period", period)


@dataclass(config=ConfigDict(arbitrary_types_allowed=True, extra="forbid"))
class GlofasConfig(BaseHandlerConfig):
    """
    Configuration for GloFAS flood hazard tiles.

    Attributes:
        base_path: Directory of the stored files.
        n_workers: Parallel downloads. The base default is the CPU count.
    """

    options_type = GlofasResolveOptions

    base_path: Path = Field(default_factory=lambda: global_config.get_path("glofas", "bronze"))
    tiles_gdf: Optional[gpd.GeoDataFrame] = Field(default=None, init=False)

    def get_relevant_data_units_by_key(
        self, search_key: BaseGeometry, options: GlofasResolveOptions
    ) -> List[dict]:
        """
        Return one unit per tile that intersects the search geometry.

        The search geometry comes from the base: a country boundary, or the
        geometry passed as the source. Each unit carries the tile, the layer,
        the return period when the layer has one, and the URL.
        """
        hits = tiles_intersecting(self._tiles(), search_key, self.crs)
        units = []
        for record in hits[["id", "name"]].to_dict("records"):
            tile_id = int(record["id"])
            tile_name = str(record["name"])
            units.append(
                {
                    "tile_id": tile_id,
                    "tile_name": tile_name,
                    "layer": options.layer,
                    "return_period": options.return_period,
                    "url": _remote_url(
                        tile_id, tile_name, options.layer, options.return_period
                    ),
                }
            )
        return units

    def get_data_unit_path(self, unit: dict, options=None) -> Path:
        """
        Return the stored GeoTIFF of one unit.

        The path includes the layer. Depth layers also include the return period.
        """
        filename = _filename(
            unit["tile_id"], unit["tile_name"], unit["layer"], unit["return_period"]
        )
        directory = self.base_path / unit["layer"]
        if unit["layer"] in DEPTH_LAYERS:
            directory = directory / f"RP{unit['return_period']}"
        return directory / filename

    def _tiles(self) -> gpd.GeoDataFrame:
        """Load the published tile grid, from the data store when it is cached."""
        if self.tiles_gdf is not None:
            return self.tiles_gdf

        from gigaspatial.core.io.readers import read_dataset
        from gigaspatial.core.io.writers import write_dataset

        cache = self.base_path / "cache" / "tile_extents.geojson"
        if self.data_store.file_exists(str(cache)):
            try:
                grid = read_dataset(cache, data_store=self.data_store)
                self.tiles_gdf = _as_wgs84(grid)
                self.logger.info("Loaded GloFAS tile grid from cache")
                return self.tiles_gdf
            except ValueError:
                self.logger.warning(
                    "Cached GloFAS tile grid is unreadable, downloading it again"
                )

        response = requests.get(TILES_URL, timeout=60)
        response.raise_for_status()
        grid = _as_wgs84(gpd.read_file(io.BytesIO(response.content)))
        missing = {"id", "name"} - set(grid.columns)
        if missing:
            raise ValueError(f"GloFAS tile grid is missing columns: {sorted(missing)}")
        write_dataset(grid, self.data_store, str(cache))
        self.tiles_gdf = grid
        self.logger.info("GloFAS tile grid cached")
        return self.tiles_gdf


class GlofasDownloader(BaseHandlerDownloader):
    """Downloads one GloFAS tile into the data store."""

    options_type = DownloadOptions
    config_type: ClassVar[type] = GlofasConfig

    def download_data_unit(self, unit: dict, options: DownloadOptions) -> Path:
        """
        Download one tile.

        Args:
            unit: A tile from the config. The URL is on the unit.
            options: ``show_progress`` controls the byte progress bar.

        Returns:
            The stored file path.

        Raises:
            requests.RequestException: If the download fails.
        """
        return stream_to_store(
            unit["url"],
            self.config.get_data_unit_path(unit),
            self.data_store,
            show_progress=options.show_progress,
        )


class GlofasReader(BaseHandlerReader):
    """Loads stored GloFAS tiles as ``TifProcessor`` objects."""

    options_type = RasterReadOptions
    config_type: ClassVar[type] = GlofasConfig

    def load_from_paths(
        self, source_data_path: List[str], options: RasterReadOptions
    ) -> TifProcessor | List[TifProcessor]:
        """
        Load stored tiles.

        Args:
            source_data_path: Stored GeoTIFF paths.
            options: ``merge_rasters`` merges every path into one processor.

        Returns:
            One ``TifProcessor``, or a list of them when several tiles are
            loaded separately.
        """
        return self._load_raster_data(
            source_data_path, merge_rasters=options.merge_rasters
        )

    def crop_to_geometry(self, data, geometry, predicate="intersects", **kwargs):
        """Crop one tile, or each tile when the source hits several."""
        if isinstance(data, list):
            return [
                super().crop_to_geometry(item, geometry, predicate=predicate, **kwargs)
                for item in data
            ]
        return super().crop_to_geometry(data, geometry, predicate=predicate, **kwargs)


class GlofasHandler(BaseHandler):
    """
    Handler for GloFAS flood hazard maps.

    A geometry or a country selects the intersecting tiles. One load fetches
    one layer and, for depth layers, one return period:

        GlofasHandler().load_data(geometry)
        GlofasHandler().load_data("KEN", layer="depth_reclass", return_period=50)
        GlofasHandler().load_data(geometry, layer="permanent_water")
    """

    def create_config(self, data_store, logger, **kwargs) -> GlofasConfig:
        return GlofasConfig(data_store=data_store, logger=logger, **kwargs)

    def create_downloader(self, config, data_store, logger, **kwargs) -> GlofasDownloader:
        return GlofasDownloader(config=config, data_store=data_store, logger=logger)

    def create_reader(self, config, data_store, logger, **kwargs) -> GlofasReader:
        return GlofasReader(config=config, data_store=data_store, logger=logger)


def _as_wgs84(grid: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if grid.crs is None:
        return grid.set_crs("EPSG:4326")
    return grid
