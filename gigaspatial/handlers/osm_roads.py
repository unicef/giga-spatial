"""
OpenStreetMap road networks from Geofabrik extracts.

``OSMRoadsHandler`` stores Geofabrik's ``.osm.pbf`` extracts in a data store
and builds a ``SpatialNetwork`` from them, for areas up to whole countries.
For cities, districts, and areas around schools, ``OSMRoadsFetcher`` queries
Overpass instead and returns the same network.

The design is in ``docs/design/osm-roads-handler.md``, with ADRs 0006 (storage)
and 0007 (choosing extracts). Needs ``pip install 'giga-spatial[roads]'``.
"""

import dataclasses
import logging
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, Tuple, Union

from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass

from gigaspatial.config import config as global_config
from gigaspatial.core.io.data_store import DataStore
from gigaspatial.handlers.base import (
    BaseHandler,
    BaseHandlerConfig,
    BaseHandlerDownloader,
    BaseHandlerReader,
    DownloadOptions,
    ReadOptions,
    ResolveOptions,
)

GEOFABRIK_URL = "https://download.geofabrik.de"


@dataclasses.dataclass(frozen=True)
class OSMRoadsResolveOptions(ResolveOptions):
    """
    Options that choose the extracts for a source.

    Attributes:
        regions: For a country source, use the country's sub-region extracts
            instead of its country extract.
        extracts: Geofabrik extract IDs to use, overriding the choice.
        date: Use the extracts stored on this date (``YYYY-MM-DD``), instead
            of the newest stored ones.
        refresh: Download today's extracts even when stored ones exist.
        max_age_days: Download again when the stored extract is older than
            this many days. Never, by default.
        buffer: Meters around point and line sources.
        area: How points become an area: ``"hull"``, ``"bbox"``, or ``"buffers"``.
    """

    regions: bool = False
    extracts: Optional[Tuple[str, ...]] = None
    date: Optional[str] = None
    refresh: bool = False
    max_age_days: Optional[int] = None
    buffer: float = 2000.0
    area: Literal["hull", "bbox", "buffers"] = "hull"


@dataclasses.dataclass(frozen=True)
class OSMRoadsReadOptions(ReadOptions):
    """
    Options for building and loading road networks.

    Attributes:
        network_type: osmnx network type, such as ``"drive_service"``,
            ``"drive"``, ``"walk"``, ``"bike"``, ``"all_public"``, or ``"all"``.
        tags: Allowed values per OSM key, such as
            ``{"highway": ["primary", "secondary"]}``, replacing ``network_type``.
        simplify: Join edges between junctions, as osmnx does.
        retain_all: Keep every disconnected part, not only the largest.
        truncate_by_edge: Keep edges that leave the area at one end.
        node_index: pyosmium node location index, such as ``"flex_mem"`` or
            ``"sparse_file_array"``. Chosen from the extract's size otherwise.
    """

    network_type: str = "drive_service"
    tags: Optional[Dict[str, Sequence[str]]] = None
    simplify: bool = True
    retain_all: bool = False
    truncate_by_edge: bool = False
    node_index: Optional[str] = None


@dataclass(config=ConfigDict(arbitrary_types_allowed=True, extra="forbid"))
class OSMRoadsConfig(BaseHandlerConfig):
    """
    Configuration for Geofabrik extracts.

    Attributes:
        base_path: Root of the stored extracts, laid out by date and then by
            Geofabrik's own folders (ADR 0006).
        index_url: Geofabrik's index of extracts, with their polygons.
    """

    base_path: Path = Field(default=global_config.get_path("osm", "bronze") / "geofabrik")
    index_url: str = f"{GEOFABRIK_URL}/index-v1.json"

    options_type = OSMRoadsResolveOptions

    def get_relevant_data_units_by_key(self, search_key, options: OSMRoadsResolveOptions) -> List[dict]:
        """
        The Geofabrik extracts for a search key (ADR 0007).

        Args:
            search_key: The source's search geometry.
            options: The extract choice options.

        Returns:
            Data units ``{"id", "url", "date"}``, one per extract.
        """
        raise NotImplementedError("OSMRoadsConfig.get_relevant_data_units_by_key is not implemented yet")


class OSMRoadsDownloader(BaseHandlerDownloader):
    """Downloads Geofabrik extracts into the data store, checked against Geofabrik's checksums."""

    options_type = DownloadOptions
    config_type = OSMRoadsConfig

    def download_data_unit(self, unit: dict, options: DownloadOptions) -> Path:
        """
        Download one extract.

        Args:
            unit: ``{"id", "url", "date"}``.
            options: The download options.

        Returns:
            The stored ``.osm.pbf`` path.
        """
        raise NotImplementedError("OSMRoadsDownloader.download_data_unit is not implemented yet")


class OSMRoadsReader(BaseHandlerReader):
    """Builds or loads the road network of each stored extract and combines them for a source."""

    options_type = OSMRoadsReadOptions
    config_type = OSMRoadsConfig

    def load_from_paths(self, source_data_path: List[Union[str, Path]], options: OSMRoadsReadOptions):
        """
        Load the road networks of stored extracts.

        Args:
            source_data_path: Stored ``.osm.pbf`` paths, or saved network folders.
            options: The network options.

        Returns:
            A ``SpatialNetwork``.
        """
        raise NotImplementedError("OSMRoadsReader.load_from_paths is not implemented yet")


class OSMRoadsHandler(BaseHandler):
    """
    OpenStreetMap road networks from Geofabrik extracts.

    Stores each extract once and returns a ``SpatialNetwork`` for a country,
    a geometry, a frame, or points.
    """

    def create_config(self, data_store: DataStore, logger: logging.Logger, **kwargs) -> OSMRoadsConfig:
        return OSMRoadsConfig(data_store=data_store, logger=logger, **kwargs)

    def create_downloader(
        self, config: OSMRoadsConfig, data_store: DataStore, logger: logging.Logger, **kwargs
    ) -> OSMRoadsDownloader:
        return OSMRoadsDownloader(config=config, data_store=data_store, logger=logger)

    def create_reader(
        self, config: OSMRoadsConfig, data_store: DataStore, logger: logging.Logger, **kwargs
    ) -> OSMRoadsReader:
        return OSMRoadsReader(config=config, data_store=data_store, logger=logger)
