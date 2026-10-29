"""
OpenStreetMap road networks from Geofabrik extracts.

``OSMRoadsHandler`` stores Geofabrik's ``.osm.pbf`` extracts in a data store
and builds a ``SpatialNetwork`` from them, for areas up to whole countries.
For cities, districts, and areas around schools, ``OSMRoadsFetcher`` queries
Overpass instead and returns the same network.

The design is in ``docs/design/osm-roads-handler.md``, with ADRs 0006 (storage)
and 0007 (choosing extracts). Needs ``pip install 'giga-spatial[roads]'``.
"""

import contextlib
import dataclasses
import datetime
import json
import logging
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import pycountry
import requests
import shapely
from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass

from gigaspatial.config import config as global_config
from gigaspatial.core.io.data_store import DataStore
from gigaspatial.core.io.local_data_store import LocalDataStore
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
from gigaspatial.handlers.base.fetch import TIMEOUT, stream_to_store
from gigaspatial.handlers.base.source import SourceKind, classify_source
from gigaspatial.handlers.osm_pbf import build_road_network, import_osmium
from gigaspatial.processing.network import SpatialNetwork

GEOFABRIK_URL = "https://download.geofabrik.de"
LATEST = "-latest.osm.pbf"
EXTRACT_SUFFIX = ".osm.pbf"
# An extract's header block sits in its first bytes, so its timestamp is read without the rest.
HEADER_BYTES = 64 * 1024

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
        date: Use the extracts of this date (``YYYY-MM-DD``), stored or
            downloaded, instead of the newest stored ones. Geofabrik keeps
            daily files for about a week, then those of the 1st of each
            month and of January 1st.
        refresh: Use Geofabrik's latest extracts even when older ones are
            stored, and download Geofabrik's index again.
        max_age_days: Use Geofabrik's latest extracts when the newest stored
            one is older than this many days. Never, by default.
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
        if self.date is not None:
            try:
                date = self.date if isinstance(self.date, datetime.date) else datetime.date.fromisoformat(self.date)
            except (TypeError, ValueError) as error:
                raise ValueError(f"date must be YYYY-MM-DD, got {self.date!r}") from error
            object.__setattr__(self, "date", date.isoformat())
            if self.refresh or self.max_age_days is not None:
                raise ValueError("date= pins the extracts' date; refresh and max_age_days choose the newest")


@dataclasses.dataclass(frozen=True)
class OSMRoadsReadOptions(ReadOptions):
    """
    Options for building and loading road networks.

    Attributes:
        network_type: osmnx network type, such as ``"drive_service"``,
            ``"drive"``, ``"walk"``, ``"bike"``, ``"all_public"``, or ``"all"``.
        tags: Allowed values per OSM key, such as
            ``{"highway": ["primary", "secondary"]}``, replacing ``network_type``.
            A way must match every key; ``None`` allows any value of a key.
        simplify: Join edges between junctions, as osmnx does.
        retain_all: Keep every disconnected part, not only the largest.
        truncate_by_edge: Keep edges that leave the area at one end.
        node_index: pyosmium node location index, such as ``"flex_mem"`` or
            ``"sparse_file_array"``. Chosen from the extract's size otherwise.
    """

    network_type: str = "drive_service"
    tags: Optional[Dict[str, Optional[Sequence[str]]]] = None
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
            Data units ``{"id", "date", "url", "path", "matches_country"}``,
            one per extract. ``date`` is the extract's date (see
            ``choose_date``) and ``url`` its dated file. ``matches_country``
            is true when the extract is the country's own file or one of its
            regions, so it needs no crop.

        Raises:
            ValueError: If a named extract is unknown, or no extract covers the area.
            requests.RequestException: If Geofabrik is needed for the latest
                date and cannot be reached.
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
        units = []
        for extract in chosen:
            unit = index.unit(extract, matches_country=extract in matching)
            date = self.choose_date(unit, options)
            units.append({**unit, "date": date, "url": _dated_url(unit["url"], date)})
        self.logger.info("Geofabrik extracts: " + ", ".join(f"{u['id']} ({u['date']})" for u in units))
        return units

    def get_data_unit_path(self, unit: dict, options: Optional[OSMRoadsResolveOptions] = None) -> Path:
        """An extract's stored file: ``base_path / date / path.osm.pbf``, as in ``2026-10-27/africa/rwanda.osm.pbf``."""
        return self.base_path / unit["date"] / f"{unit['path']}{EXTRACT_SUFFIX}"

    def extract_of_path(self, path: Union[str, Path]) -> Optional[Dict[str, str]]:
        """
        The extract a stored file holds, read from its place in the layout.

        Returns:
            ``{"id", "date", "url", "path"}``, as in the file's data unit, or
            None for a file outside ``base_path / date /``.
        """
        try:
            relative = Path(path).relative_to(self.base_path)
        except ValueError:
            return None
        date, *rest = relative.parts
        if not rest or not _ISO_DATE.fullmatch(date) or not relative.name.endswith(EXTRACT_SUFFIX):
            return None
        extract_path = "/".join(rest)[: -len(EXTRACT_SUFFIX)]
        return {
            "id": extract_path.rsplit("/", 1)[-1],
            "date": date,
            "url": _dated_url(f"{GEOFABRIK_URL}/{extract_path}{LATEST}", date),
            "path": extract_path,
        }

    def choose_date(self, unit: dict, options: OSMRoadsResolveOptions) -> str:
        """
        The date of the extract to use, as ``YYYY-MM-DD``.

        ``date=`` when given; otherwise the newest stored date, unless
        ``refresh`` is set or that date is more than ``max_age_days`` old, in
        which case Geofabrik's latest. Geofabrik is asked only then; if it
        cannot be reached when ``max_age_days`` asked, the stored extract is
        used with a warning.
        """
        if options.date:
            return options.date
        stored = self.stored_dates(unit["path"])
        newest = stored[-1] if stored else None
        if newest is None or options.refresh:
            return self.latest_date(unit["url"])
        if options.max_age_days is None or _age_days(newest) <= options.max_age_days:
            return newest
        try:
            return self.latest_date(unit["url"])
        except requests.RequestException as error:
            self.logger.warning(f"Cannot reach Geofabrik ({error}); using {unit['id']} of {newest}")
            return newest

    def stored_dates(self, path: str) -> List[str]:
        """The dates an extract is stored for, oldest first; ``path`` is its Geofabrik path, such as ``africa/rwanda``."""
        if not self.data_store.is_dir(str(self.base_path)):
            return []
        dates = [Path(name).name for name in self.data_store.list_directories(str(self.base_path))]
        return sorted(
            date
            for date in dates
            if _ISO_DATE.fullmatch(date) and self.data_store.file_exists(str(self.base_path / date / f"{path}.osm.pbf"))
        )

    def latest_date(self, url: str) -> str:
        """
        The date of Geofabrik's latest file for an extract, from its ``-latest`` URL.

        ``-latest`` redirects to the dated file, except for large extracts
        that redirect to a mirror under the same name; their date is that of
        the extract's replication state.
        """
        response = requests.head(url, allow_redirects=False, timeout=TIMEOUT)
        response.raise_for_status()
        match = _DATED_NAME.search(response.headers.get("location", ""))
        if match:
            return _iso_date(match.group(1))
        state = requests.get(url[: -len(LATEST)] + "-updates/state.txt", timeout=TIMEOUT)
        state.raise_for_status()
        match = re.search(r"^timestamp=(\d{4}-\d{2}-\d{2})", state.text, re.MULTILINE)
        if not match:
            raise ValueError(f"Cannot tell the date of {url}: no dated redirect or replication timestamp")
        return match.group(1)

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


_ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
_DATED_NAME = re.compile(r"-(\d{6})\.osm\.pbf$")


def _extract_path(url: str) -> str:
    """An extract's path in Geofabrik's folders, such as ``africa/rwanda``."""
    from urllib.parse import urlparse

    path = urlparse(url).path.lstrip("/")
    return path[: -len(LATEST)] if path.endswith(LATEST) else path


def _dated_url(latest_url: str, date: str) -> str:
    """The dated file for a ``-latest`` URL: ``rwanda-latest.osm.pbf`` of 2026-10-27 is ``rwanda-261027.osm.pbf``."""
    stamp = datetime.date.fromisoformat(date).strftime("%y%m%d")
    return f"{latest_url[: -len(LATEST)]}-{stamp}.osm.pbf"


def _iso_date(yymmdd: str) -> str:
    return datetime.datetime.strptime(yymmdd, "%y%m%d").date().isoformat()


def _age_days(date: str) -> int:
    return (datetime.datetime.now(datetime.timezone.utc).date() - datetime.date.fromisoformat(date)).days


def replication_timestamp(path: Union[str, Path], data_store: Optional[DataStore] = None) -> Optional[datetime.datetime]:
    """
    The time an extract's data was taken from OpenStreetMap, from its header.

    Reads only the file's first bytes, so it works offline and on remote data
    stores. The UTC date of this timestamp is the extract's date.

    Args:
        path: A stored ``.osm.pbf`` file.
        data_store: The store holding it; a local one by default.

    Returns:
        The timestamp, in UTC, or None if the header has none.
    """
    osmium = import_osmium()
    with (data_store or LocalDataStore()).open(str(path), "rb") as file:
        head = file.read(HEADER_BYTES)
    reader = osmium.io.Reader(osmium.io.FileBuffer(head, "pbf"), osmium.osm.osm_entity_bits.NOTHING)
    try:
        stamp = reader.header().get("osmosis_replication_timestamp")
    finally:
        reader.close()
    if not stamp:
        return None
    return datetime.datetime.fromisoformat(stamp.replace("Z", "+00:00"))


@contextlib.contextmanager
def _local_file(data_store: DataStore, path: Union[str, Path]):
    """A local path for a stored file, which pyosmium needs: in place on a local store, else a temporary copy."""
    if isinstance(data_store, LocalDataStore):
        path = Path(path)
        yield path if path.is_absolute() else data_store.base_path / path
        return
    with tempfile.TemporaryDirectory(prefix="gigaspatial-osm-") as scratch:
        local = Path(scratch) / Path(path).name
        with data_store.open(str(path), "rb") as source, open(local, "wb") as target:
            shutil.copyfileobj(source, target, 16 * 1024 * 1024)
        yield local


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
        Download one extract's dated file, checked against Geofabrik's ``.md5``.

        Args:
            unit: A unit from ``OSMRoadsConfig.get_relevant_data_units``.
            options: The download options.

        Returns:
            The stored ``.osm.pbf`` path.

        Raises:
            FileNotFoundError: If Geofabrik no longer keeps the extract of that date.
            ValueError: If the download does not match its checksum; nothing is stored.
            requests.RequestException: If the download fails; nothing is stored.
        """
        path = self.config.get_data_unit_path(unit)
        url = unit["url"]
        try:
            checksum = requests.get(f"{url}.md5", timeout=TIMEOUT)
            checksum.raise_for_status()
            self.logger.info(f"Downloading {unit['id']} of {unit['date']} from {url}")
            stream_to_store(
                url,
                path,
                self.data_store,
                show_progress=options.show_progress,
                desc=f"{unit['id']} {unit['date']}",
                md5=checksum.text.split()[0],
            )
        except requests.HTTPError as error:
            if error.response is not None and error.response.status_code == 404:
                raise FileNotFoundError(
                    f"Geofabrik no longer keeps {unit['id']} of {unit['date']} ({url}). Daily files are "
                    "kept for about a week, then those of the 1st of each month and of January 1st."
                ) from error
            raise
        return path


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

    def build_network(
        self, path: Union[str, Path], options: Optional[OSMRoadsReadOptions] = None
    ) -> SpatialNetwork:
        """
        The road network of one stored extract, as osmnx builds it (``osm_pbf.build_road_network``).

        The network covers the whole extract, with every connected part
        (ADR 0006); cropping and keeping the largest part happen at load.

        Args:
            path: A stored ``.osm.pbf`` file.
            options: ``network_type``, ``tags``, ``simplify``, and ``node_index``.

        Returns:
            A directed network whose metadata records the extract's ID, date,
            URL, and replication timestamp.
        """
        options = options or OSMRoadsReadOptions()
        with _local_file(self.data_store, path) as local:
            return build_road_network(
                local,
                network_type=options.network_type,
                tags=options.tags,
                simplify=options.simplify,
                node_index=options.node_index,
                extract=self.config.extract_of_path(path),
                logger=self.logger,
            )


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
