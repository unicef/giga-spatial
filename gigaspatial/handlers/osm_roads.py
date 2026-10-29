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
import datetime
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import pycountry
import shapely
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
from gigaspatial.handlers.base.config import source_geometry
from gigaspatial.handlers.base.fetch import stream_to_store
from gigaspatial.handlers.base.source import SourceKind, classify_source

GEOFABRIK_URL = "https://download.geofabrik.de"

# Extracts holding more than one country, which a country source must be cropped from.
# Several carry one ISO code or none in Geofabrik's index; tested against a stored copy of it.
MULTI_COUNTRY_EXTRACTS = frozenset(
    {
        "alps",
        "american-oceania",
        "britain-and-ireland",
        "dach",
        "gcc-states",
        "haiti-and-domrep",
        "ireland-and-northern-ireland",
        "israel-and-palestine",
        "malaysia-singapore-brunei",
        "sea",
        "senegal-and-gambia",
        "south-africa-and-lesotho",
    }
)
# ISO codes the index gets wrong (October 2026): these territories carry Vanuatu's or the Marshall Islands' code.
CODE_FIXES = {
    "ile-de-clipperton": (),
    "pitcairn-islands": ("PN",),
    "polynesie-francaise": ("PF",),
    "tokelau": ("TK",),
    "wallis-et-futuna": ("WF",),
}


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

    def __post_init__(self):
        # Resolve options key the unit cache, so a list of extracts becomes a tuple.
        if self.extracts is not None and not isinstance(self.extracts, tuple):
            object.__setattr__(self, "extracts", tuple(self.extracts))


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

    def get_search_key(self, source, options: OSMRoadsResolveOptions) -> Dict[str, Any]:
        """
        A country's ISO code, or the area roads are wanted for.

        Args:
            source: Country name or code, geometry, frame, or points.
            options: ``buffer`` and ``area`` shape the area of point and line sources.

        Returns:
            ``{"country": "RW"}`` for a country source, ``{"area": polygon}`` otherwise,
            with the area built exactly as ``OSMRoadsFetcher`` builds it.

        Raises:
            ValueError: If the country is unknown or the source has no area.
        """
        if classify_source(source) is SourceKind.COUNTRY:
            try:
                return {"country": pycountry.countries.lookup(source).alpha_2}
            except LookupError as error:
                raise ValueError(f"Unknown country: {source}") from error
        from gigaspatial.fetchers.osm_roads import query_area

        return {"area": query_area(source, buffer=options.buffer, area=options.area, logger=self.logger)}

    def get_relevant_data_units_by_key(self, search_key: Dict[str, Any], options: OSMRoadsResolveOptions) -> List[dict]:
        """
        The Geofabrik extracts for a search key (ADR 0007).

        A country source uses its own extract, or with ``regions`` its
        sub-region extracts, and otherwise the extracts covering its boundary.
        Any other source uses the smallest extracts that together cover its
        area. Continents are used only when named in ``extracts``.

        Args:
            search_key: ``{"country": code}`` or ``{"area": polygon}``.
            options: The extract choice options.

        Returns:
            Data units ``{"id", "url", "path", "matches_country"}``, one per
            extract. ``matches_country`` is true when the extract is the
            country's own file or one of its regions, so it needs no crop.

        Raises:
            ValueError: If a named extract is unknown, or no extract covers the area.
        """
        index = self.geofabrik_index(options)
        country = search_key.get("country")
        own = index.own_extract(country) if country else None
        regions = index.regions(country, own) if country else []
        if options.extracts:
            chosen = index.require(options.extracts)
        elif country and options.regions and regions:
            chosen = regions
        elif own is not None:
            if options.regions:
                self.logger.info(f"{country} has no sub-region extracts; using its country extract")
            chosen = [own]
        else:
            area = search_key.get("area")
            if area is None:
                area = source_geometry(country, "EPSG:4326")
            chosen = index.cover(area)
        matching = set(regions) | ({own} if own else set())
        self.logger.info(f"Geofabrik extracts: {', '.join(chosen)}")
        return [index.unit(extract, matches_country=extract in matching) for extract in chosen]

    def geofabrik_index(self, options: Optional[OSMRoadsResolveOptions] = None) -> "GeofabrikIndex":
        """
        Geofabrik's index of extracts, from the data store, downloading it when missing or stale.

        The index is stored at ``base_path / "index-v1.json"``, with the time
        it was downloaded beside it, and kept in memory after the first read.
        ``refresh`` downloads it again; ``max_age_days`` does when it is older.
        """
        options = options or OSMRoadsResolveOptions()
        path = self.base_path / "index-v1.json"
        stamp = self.base_path / "index-v1.json.downloaded"
        stale = options.refresh or not self.data_store.file_exists(str(path))
        if not stale and options.max_age_days is not None:
            downloaded = _read_stamp(self.data_store, stamp)
            age = datetime.datetime.now(datetime.timezone.utc) - downloaded if downloaded else None
            stale = age is None or age > datetime.timedelta(days=options.max_age_days)
        cached = self.__dict__.get("_geofabrik_index")
        if stale:
            self.logger.info(f"Downloading Geofabrik's index from {self.index_url}")
            stream_to_store(self.index_url, path, self.data_store)
            now = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
            self.data_store.write_file(str(stamp), now)
            cached = None
        if cached is None:
            cached = GeofabrikIndex(json.loads(self.data_store.read_file(str(path))))
            object.__setattr__(self, "_geofabrik_index", cached)
        return cached


class GeofabrikIndex:
    """
    Geofabrik's extracts, with their polygons and ISO codes (corrected by ``CODE_FIXES``).

    A continent is a root extract without an ISO code, such as ``africa``;
    Russia and Antarctica are roots with a code and count as countries.
    """

    def __init__(self, collection: Dict[str, Any]):
        features = collection["features"]
        properties = [f["properties"] for f in features]
        self.ids: List[str] = [p["id"] for p in properties]
        self._position = {extract: i for i, extract in enumerate(self.ids)}
        self.parents = [p.get("parent") for p in properties]
        self.codes = [CODE_FIXES.get(p["id"], tuple(p.get("iso3166-1:alpha2") or ())) for p in properties]
        self.subdivisions = [tuple(p.get("iso3166-2") or ()) for p in properties]
        self.urls = [p["urls"]["pbf"] for p in properties]
        self.geometries = np.array(
            [shapely.geometry.shape(f["geometry"]) if f.get("geometry") else None for f in features], dtype=object
        )
        self.continents = np.array([parent is None and not codes for parent, codes in zip(self.parents, self.codes)])

    def own_extract(self, country: str) -> Optional[str]:
        """The country's own extract: the only one carrying just its code, and not holding other countries."""
        own = [
            extract
            for extract, codes in zip(self.ids, self.codes)
            if codes == (country,) and extract not in MULTI_COUNTRY_EXTRACTS
        ]
        return own[0] if len(own) == 1 else None

    def regions(self, country: str, own: Optional[str]) -> List[str]:
        """The sub-region extracts of a country's own extract.

        These are its children, or for a country whose regions are filed
        elsewhere (the US states sit under ``north-america``), the extracts
        coded with its ISO 3166-2 subdivisions.
        """
        if own is None:
            return []
        children = [extract for extract, parent in zip(self.ids, self.parents) if parent == own]
        if children:
            return children
        return [
            extract
            for extract, subdivisions in zip(self.ids, self.subdivisions)
            if subdivisions and all(code.startswith(f"{country}-") for code in subdivisions)
        ]

    def require(self, extracts: Iterable[str]) -> List[str]:
        """The named extracts, which must exist."""
        extracts = list(extracts)
        unknown = [extract for extract in extracts if extract not in self._position]
        if unknown:
            raise ValueError(f"Unknown Geofabrik extracts: {unknown}")
        return extracts

    def cover(self, area: shapely.Geometry) -> List[str]:
        """The smallest extracts that together cover an area, continents left out.

        Extracts are taken from the smallest up while part of the area is
        still uncovered; any extract the others already cover is then dropped.
        """
        present = np.array([g is not None for g in self.geometries])
        if not present.any():
            raise ValueError("This copy of Geofabrik's index has no polygons")
        candidates = np.flatnonzero(present & ~self.continents)
        touching = candidates[shapely.intersects(self.geometries[candidates], area)]
        order = touching[np.argsort(shapely.area(self.geometries[touching]), kind="stable")]
        chosen: List[int] = []
        uncovered = area
        for i in order:
            if uncovered.is_empty or uncovered.area <= area.area * 1e-9:
                break
            if self.geometries[i].intersects(uncovered):
                chosen.append(int(i))
                uncovered = uncovered.difference(self.geometries[i])
        for i in sorted(chosen, key=lambda j: -self.geometries[j].area):
            others = [self.geometries[j] for j in chosen if j != i]
            if others and area.intersection(self.geometries[i]).difference(shapely.union_all(others)).area <= area.area * 1e-9:
                chosen.remove(i)
        if not chosen:
            raise ValueError(
                "No country or regional Geofabrik extract covers the area; "
                'name a continent with extracts=["africa"] to use one'
            )
        return [self.ids[i] for i in chosen]

    def unit(self, extract: str, matches_country: bool) -> Dict[str, Any]:
        url = self.urls[self._position[extract]]
        return {"id": extract, "url": url, "path": _extract_path(url), "matches_country": matches_country}


def _extract_path(url: str) -> str:
    """An extract's path in Geofabrik's folders, such as ``africa/rwanda``."""
    from urllib.parse import urlparse

    path = urlparse(url).path.lstrip("/")
    return path[: -len("-latest.osm.pbf")] if path.endswith("-latest.osm.pbf") else path


def _read_stamp(data_store: DataStore, path: Path) -> Optional[datetime.datetime]:
    if not data_store.file_exists(str(path)):
        return None
    text = data_store.read_file(str(path), encoding="utf-8").strip()
    try:
        return datetime.datetime.fromisoformat(text)
    except ValueError:
        return None


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
