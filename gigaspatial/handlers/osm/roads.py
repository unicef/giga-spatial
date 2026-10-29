"""
OpenStreetMap road networks from Geofabrik extracts.

``OSMRoadsHandler`` stores Geofabrik's ``.osm.pbf`` extracts in a data store
and builds a ``SpatialNetwork`` from them, for areas up to whole countries.
For cities, districts, and areas around schools, ``OSMRoadsFetcher`` queries
Overpass instead and returns the same network.

Choosing extracts is in ``geofabrik``, and building a network from one in
``pbf``. The design is in ``docs/design/osm-roads-handler.md``, with ADRs
0006 (storage) and 0007 (choosing extracts). Needs
``pip install 'giga-spatial[roads]'``.
"""

import contextlib
import dataclasses
import datetime
import json
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import pycountry
import requests
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
from gigaspatial.handlers.osm.geofabrik import (
    EXTRACT_SUFFIX,
    GEOFABRIK_URL,
    ISO_DATE,
    LATEST,
    GeofabrikIndex,
    dated_url,
    latest_date,
)
from gigaspatial.handlers.osm.pbf import build_road_network
from gigaspatial.processing.network import SpatialNetwork


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
            units.append({**unit, "date": date, "url": dated_url(unit["url"], date)})
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
        if not rest or not ISO_DATE.fullmatch(date) or not relative.name.endswith(EXTRACT_SUFFIX):
            return None
        extract_path = "/".join(rest)[: -len(EXTRACT_SUFFIX)]
        return {
            "id": extract_path.rsplit("/", 1)[-1],
            "date": date,
            "url": dated_url(f"{GEOFABRIK_URL}/{extract_path}{LATEST}", date),
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
            return latest_date(unit["url"])
        if options.max_age_days is None or _age_days(newest) <= options.max_age_days:
            return newest
        try:
            return latest_date(unit["url"])
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
            if ISO_DATE.fullmatch(date)
            and self.data_store.file_exists(str(self.base_path / date / f"{path}{EXTRACT_SUFFIX}"))
        )

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


def _age_days(date: str) -> int:
    return (datetime.datetime.now(datetime.timezone.utc).date() - datetime.date.fromisoformat(date)).days


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
        The road network of one stored extract, as osmnx builds it (``pbf.build_road_network``).

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
