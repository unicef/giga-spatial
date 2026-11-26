"""
Tropical cyclone tracks for the ECMWF forecasting systems.

The essential portal https://essential.ecmwf.int publishes one BUFR4 file
per storm for a forecast cycle. The WMO heading centre selects the system:
ECMF is the IFS deterministic track, ECEP is the IFS ENS track. AIFS Single
and AIFS ENS tracks are the open-data BUFR files (``type=tf``) at
https://data.ecmwf.int/forecasts when the essential listing has no file for
that system. Open data is also the fallback for an IFS system whose listing
is empty.

Tracks are kept as a GeoJSON of positions in the data store. When ecCodes
can read the BUFR template, every ensemble member and forecast step is
stored. Otherwise a per-storm essential file still contributes the position
written into its file name, which is the position ECMWF puts in the name
and not the full trajectory.

Horizons of the track product:

- IFS deterministic and IFS ENS: 360 h at 00 and 12 UTC, 144 h at 06 and 18 UTC.
- AIFS Single and AIFS ENS: 360 h at every cycle.

Licence of the open-data copy: Creative Commons CC-BY-4.0.
"""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from typing import List, Optional

import geopandas as gpd
import pandas as pd
import requests
from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass
from shapely.geometry import Point

from gigaspatial.config import config as global_config
from gigaspatial.core.io.readers import read_dataset
from gigaspatial.fetchers.ecmwf.catalogue import (
    ATTRIBUTION,
    DISS_CENTRES,
    ESSENTIAL_ROOT,
    OPEN_DATA_ROOT,
    expand_systems,
    parse_date,
    parse_time,
    track_url,
)
from gigaspatial.handlers.base import (
    BaseHandler,
    BaseHandlerConfig,
    BaseHandlerDownloader,
    BaseHandlerReader,
    DownloadOptions,
    ReadOptions,
    ResolveOptions,
)

_HEADING = re.compile(r"^A_[A-Z]{4}\d{2}([A-Z]{4})\d{6}_")
_LISTED = re.compile(
    r"A_[A-Z0-9]+_C_[A-Z]+_\d+_tropical_cyclone_track_[^\s\"'<>]+?_bufr4\.bin"
)
_TRACK_NAME = re.compile(
    r"tropical_cyclone_track_(?P<storm>.+?)_"
    r"(?P<lon>-?\d+(?:p\d+)?deg[EW])_(?P<lat>-?\d+(?:p\d+)?deg[NS])_bufr4",
    re.IGNORECASE,
)
_COORD = re.compile(r"(-?)(\d+)(?:p(\d+))?deg([NSEW])", re.IGNORECASE)

TRACK_COLUMNS = (
    "storm_id",
    "member",
    "step",
    "latitude",
    "longitude",
    "pressure",
    "maximum_wind",
    "system",
    "source",
)


def http_get(url: str, headers=None, timeout: float = 60):
    """GET a URL. A 404 returns None so a missing cycle can fall back."""
    response = requests.get(url, headers=headers or {}, timeout=timeout)
    try:
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.content
    finally:
        response.close()


def filenames_from_listing(body: str) -> List[str]:
    """Return tropical-cyclone track file names mentioned in a listing."""
    return sorted(set(_LISTED.findall(body)))


def centre_system(filename: str) -> Optional[str]:
    """Return the forecasting system encoded by a DISS file heading."""
    match = _HEADING.match(Path(filename).name)
    if match is None:
        return None
    return DISS_CENTRES.get(match.group(1))


def parse_diss_coordinate(token: str) -> float:
    """Read a coordinate token such as ``-108p5degW`` or ``10p8degN``."""
    match = _COORD.fullmatch(token)
    if match is None:
        raise ValueError(f"Cannot read a coordinate from {token!r}.")
    value = float(match.group(2) + (("." + match.group(3)) if match.group(3) else ""))
    if match.group(1) == "-":
        value = -value
    if match.group(4).upper() in {"S", "W"}:
        value = -abs(value)
    else:
        value = abs(value)
    return value


def parse_track_filename(filename: str) -> Optional[dict]:
    """
    Read the storm id and the position ECMWF writes into a DISS file name.

    Returns None when the name is an open-data track file with no position.
    """
    match = _TRACK_NAME.search(Path(filename).name)
    if match is None:
        return None
    return {
        "storm_id": match.group("storm"),
        "longitude": parse_diss_coordinate(match.group("lon")),
        "latitude": parse_diss_coordinate(match.group("lat")),
    }


def essential_listing_url(when, hour: int, root: str = ESSENTIAL_ROOT) -> str:
    """Return the essential-portal directory for one cycle."""
    stamp = f"{when:%Y%m%d}{hour:02d}0000"
    return f"{root.rstrip('/')}/file/{stamp}/"


def decode_bufr(content: bytes) -> Optional[pd.DataFrame]:
    """
    Read tropical-cyclone positions from a BUFR4 message.

    Returns None when ecCodes is not installed or the bytes are not a track
    message this decoder understands. Callers then keep the file-name position.
    """
    if not content.startswith(b"BUFR"):
        return None
    try:
        from eccodes import (  # type: ignore
            CodesInternalError,
            codes_bufr_new_from_file,
            codes_get,
            codes_get_array,
            codes_release,
            codes_set,
        )
    except ImportError:
        return None
    handle = tempfile.NamedTemporaryFile(suffix=".bufr", delete=False)
    path = handle.name
    try:
        handle.write(content)
        handle.close()
        return _decode_bufr_path(
            path,
            codes_bufr_new_from_file,
            codes_set,
            codes_get,
            codes_get_array,
            codes_release,
            CodesInternalError,
        )
    except Exception:
        return None
    finally:
        os.unlink(path)


def _first(values, index: int):
    if len(values) == 1:
        return values[0]
    if index < len(values):
        return values[index]
    return values[0]


def _missing(value) -> bool:
    if value is None:
        return True
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    if number != number:
        return True
    return abs(number) > 1.0e10 or int(number) == 2147483647


def _decode_bufr_path(path, new_from_file, codes_set, codes_get, codes_get_array, codes_release, internal_error):
    rows = []
    with open(path, "rb") as handle:
        while True:
            message = new_from_file(handle)
            if message is None:
                break
            try:
                codes_set(message, "unpack", 1)
                rows.extend(
                    _message_rows(message, codes_get, codes_get_array, internal_error)
                )
            finally:
                codes_release(message)
    if not rows:
        return None
    return pd.DataFrame(rows)


def _message_rows(message, codes_get, codes_get_array, internal_error) -> list:
    try:
        storm_id = str(codes_get(message, "stormIdentifier")).strip()
    except Exception:
        storm_id = ""
    try:
        long_name = str(codes_get(message, "longStormName")).strip()
    except Exception:
        long_name = ""
    if long_name:
        storm_id = long_name
    members = list(codes_get_array(message, "ensembleMemberNumber"))
    if not members:
        members = [None]
    periods = 0
    while True:
        periods += 1
        try:
            codes_get_array(message, f"#{periods}#timePeriod")
        except internal_error:
            break
        except Exception:
            break
    rows = []
    analysis = _analysis(message, codes_get, codes_get_array)
    if analysis is not None:
        for index, member in enumerate(members):
            latitude, longitude, pressure, wind_lat, wind_lon, wind = (
                _first(item, index) for item in analysis
            )
            if _missing(latitude) or _missing(longitude):
                continue
            rows.append(
                _row(storm_id, member, 0, latitude, longitude, pressure, wind)
            )
    for period in range(1, periods):
        rank_centre = period * 2 + 2
        rank_wind = period * 2 + 3
        try:
            step_values = codes_get_array(message, f"#{period}#timePeriod")
            centre_lat = codes_get_array(message, f"#{rank_centre}#latitude")
            centre_lon = codes_get_array(message, f"#{rank_centre}#longitude")
            pressure = codes_get_array(
                message, f"#{period + 1}#pressureReducedToMeanSeaLevel"
            )
            wind = codes_get_array(message, f"#{period + 1}#windSpeedAt10M")
        except Exception:
            continue
        step = next((int(value) for value in step_values if not _missing(value)), None)
        if step is None:
            continue
        for index, member in enumerate(members):
            latitude = _first(centre_lat, index)
            longitude = _first(centre_lon, index)
            if _missing(latitude) or _missing(longitude):
                continue
            rows.append(
                _row(
                    storm_id,
                    member,
                    step,
                    latitude,
                    longitude,
                    _first(pressure, index),
                    _first(wind, index),
                )
            )
    return rows


def _analysis(message, codes_get, codes_get_array):
    try:
        latitude = codes_get_array(message, "#2#latitude")
        longitude = codes_get_array(message, "#2#longitude")
        pressure = codes_get_array(message, "#1#pressureReducedToMeanSeaLevel")
        wind_lat = codes_get_array(message, "#3#latitude")
        wind_lon = codes_get_array(message, "#3#longitude")
        wind = codes_get_array(message, "#1#windSpeedAt10M")
    except Exception:
        return None
    return latitude, longitude, pressure, wind_lat, wind_lon, wind


def _row(storm_id, member, step, latitude, longitude, pressure, wind) -> dict:
    member_value = None if member is None or _missing(member) else int(member)
    return {
        "storm_id": storm_id,
        "member": member_value,
        "step": int(step),
        "latitude": float(latitude),
        "longitude": float(longitude),
        "pressure": None if _missing(pressure) else float(pressure),
        "maximum_wind": None if _missing(wind) else float(wind),
    }


def download_track_payloads(when, hour: int, system: str, essential_root: str, open_data_root: str):
    """
    Download the BUFR files for one system.

    Essential-portal files whose heading matches the system come first. When
    that listing has none, the open-data track file is used.

    Returns:
        ``(name, content, origin)`` triples.

    Raises:
        FileNotFoundError: If neither source has a track file.
    """
    listing_url = essential_listing_url(when, hour, essential_root)
    listing = http_get(listing_url, headers={"Accept": "text/html"}, timeout=60)
    names = []
    if listing is not None:
        names = [
            name
            for name in filenames_from_listing(listing.decode("utf-8", errors="replace"))
            if centre_system(name) == system
        ]
    stamp = f"{when:%Y%m%d}{hour:02d}0000"
    payloads = []
    for name in names:
        url = f"{essential_root.rstrip('/')}/file/{stamp}/{name}"
        body = http_get(url, timeout=120)
        if body:
            payloads.append((name, body, "essential"))
    if payloads:
        return payloads
    url = track_url(when, hour, system, root=open_data_root)
    body = http_get(url, timeout=120)
    if not body:
        raise FileNotFoundError(
            f"No {system} tropical cyclone tracks for {when.isoformat()} {hour:02d} UTC."
        )
    return [(url.rsplit("/", 1)[-1], body, "open-data")]


def points_from_payloads(payloads, system: str) -> pd.DataFrame:
    """
    Build a track table from downloaded files.

    A decoded BUFR message supplies the trajectory. A file that cannot be
    decoded keeps the storm id and position in its essential-portal name.
    """
    frames = []
    for name, content, origin in payloads:
        decoded = decode_bufr(content)
        named = parse_track_filename(name) or {}
        if decoded is not None and len(decoded):
            frame = decoded.copy()
            if named.get("storm_id") and (
                "storm_id" not in frame or frame["storm_id"].replace("", pd.NA).isna().all()
            ):
                frame["storm_id"] = named["storm_id"]
            frame["system"] = system
            frame["source"] = origin
            frames.append(frame)
            continue
        if "latitude" not in named:
            continue
        frames.append(
            pd.DataFrame(
                [
                    {
                        "storm_id": named["storm_id"],
                        "member": None,
                        "step": 0,
                        "latitude": named["latitude"],
                        "longitude": named["longitude"],
                        "pressure": None,
                        "maximum_wind": None,
                        "system": system,
                        "source": origin,
                    }
                ]
            )
        )
    if not frames:
        return pd.DataFrame(columns=TRACK_COLUMNS)
    return pd.concat(frames, ignore_index=True)


@dataclass(config=ConfigDict(arbitrary_types_allowed=True, extra="forbid"))
class EcmwfTropicalCycloneConfig(BaseHandlerConfig):
    """
    One forecast cycle of ECMWF tropical-cyclone tracks.

    Attributes:
        base_path: Directory of the stored track tables.
        date: Run date, ``YYYY-MM-DD`` or ``YYYYMMDD``.
        time: Cycle hour: 0, 6, 12, or 18.
        system: ``all`` or one of ``ifs``, ``ifs-ens``, ``aifs-single``,
            ``aifs-ens``. ``all`` stores each system as its own file.
        essential_root: Listing root for the essential portal.
        open_data_root: Open-data root used when a system has no essential file.
    """

    options_type = ResolveOptions
    units_match_country = False

    base_path: Path = Field(
        default_factory=lambda: global_config.get_path("ecmwf", "bronze")
        / "tropical_cyclone"
    )
    date: str = ""
    time: int = 0
    system: str = "all"
    essential_root: str = ESSENTIAL_ROOT
    open_data_root: str = OPEN_DATA_ROOT

    def __post_init__(self):
        super().__post_init__()
        if not self.date:
            raise ValueError("EcmwfTropicalCycloneHandler needs a run date.")
        self.date = parse_date(self.date).isoformat()
        self.time = parse_time(self.time)
        expand_systems(self.system)

    def get_search_key(self, source, options: ResolveOptions):
        """Tracks are global files. The source crops them and does not choose them."""
        return None

    def get_relevant_data_units_by_key(self, search_key, options: ResolveOptions) -> list:
        """Return one unit per configured forecasting system."""
        return [
            {"system": name, "date": self.date, "time": self.time}
            for name in expand_systems(self.system)
        ]

    def get_data_unit_path(self, unit: dict, options=None) -> Path:
        """Return the GeoJSON path of one system's tracks."""
        hour = f"{int(unit['time']):02d}"
        return self.base_path / unit["date"] / hour / unit["system"] / "tracks.geojson"


class EcmwfTropicalCycloneDownloader(BaseHandlerDownloader):
    """Download one cycle of tropical-cyclone tracks into the data store."""

    options_type = DownloadOptions
    config_type = EcmwfTropicalCycloneConfig

    def download_data_unit(self, unit: dict, options: DownloadOptions) -> List[Path]:
        """
        Download the tracks for one system and store them as GeoJSON.

        Essential-portal files for that system's centre are preferred. When
        the listing has none, the open-data ``type=tf`` BUFR is downloaded.

        Raises:
            FileNotFoundError: If neither source has a track file. A cycle
                with no active cyclone is published that way.
            RuntimeError: If a file was downloaded and no position could be read.
        """
        when = parse_date(unit["date"])
        hour = parse_time(unit["time"])
        system = unit["system"]
        payloads = download_track_payloads(
            when,
            hour,
            system,
            self.config.essential_root,
            self.config.open_data_root,
        )
        frame = points_from_payloads(payloads, system)
        located = frame.dropna(subset=["latitude", "longitude"])
        if located.empty:
            raise RuntimeError(
                f"No positions could be read for {system} on {unit['date']} "
                f"{hour:02d} UTC. Install ecCodes to decode a combined BUFR trajectory."
            )
        frame = located.reset_index(drop=True)
        geometry = [Point(lon, lat) for lon, lat in zip(frame.longitude, frame.latitude)]
        table = gpd.GeoDataFrame(frame, geometry=geometry, crs="EPSG:4326")
        table.attrs["attribution"] = ATTRIBUTION
        path = self.config.get_data_unit_path(unit)
        self.data_store.write_file(str(path), table.to_json())
        return [path]


class EcmwfTropicalCycloneReader(BaseHandlerReader):
    """Load stored tropical-cyclone track GeoJSON."""

    options_type = ReadOptions
    config_type = EcmwfTropicalCycloneConfig

    def load_from_paths(self, source_data_path: List[str], options: ReadOptions):
        """
        Load one or more stored track tables.

        Returns:
            A GeoDataFrame in EPSG:4326 with one row per storm position.
        """
        frames = []
        for path in source_data_path:
            frame = read_dataset(path, data_store=self.data_store)
            if frame is None or len(frame) == 0:
                continue
            if getattr(frame, "crs", None) is None and "geometry" in getattr(frame, "columns", []):
                frame = gpd.GeoDataFrame(frame, geometry="geometry", crs="EPSG:4326")
            frames.append(frame)
        if not frames:
            return gpd.GeoDataFrame(columns=list(TRACK_COLUMNS), geometry=[], crs="EPSG:4326")
        combined = pd.concat(frames, ignore_index=True)
        if "geometry" not in combined.columns:
            return combined
        return gpd.GeoDataFrame(combined, geometry="geometry", crs="EPSG:4326")


class EcmwfTropicalCycloneHandler(BaseHandler):
    """
    Handler for ECMWF tropical-cyclone tracks.

    One instance is one forecast cycle. ``system="all"`` stores IFS
    deterministic, IFS ENS, AIFS Single, and AIFS ENS as separate files.
    ``load_data(source, crop_to_source=True)`` keeps the positions inside
    that source.

    Pass the cycle as ``date`` and ``time``:

    ```python
    from gigaspatial import EcmwfTropicalCycloneHandler

    tracks = EcmwfTropicalCycloneHandler(date="2026-10-15", time=0, system="ifs-ens")
    frame = tracks.load_data(area, crop_to_source=True)
    ```
    """

    def create_config(self, data_store, logger, **kwargs) -> EcmwfTropicalCycloneConfig:
        return EcmwfTropicalCycloneConfig(data_store=data_store, logger=logger, **kwargs)

    def create_downloader(self, config, data_store, logger) -> EcmwfTropicalCycloneDownloader:
        return EcmwfTropicalCycloneDownloader(
            config=config, data_store=data_store, logger=logger
        )

    def create_reader(self, config, data_store, logger) -> EcmwfTropicalCycloneReader:
        return EcmwfTropicalCycloneReader(
            config=config, data_store=data_store, logger=logger
        )
