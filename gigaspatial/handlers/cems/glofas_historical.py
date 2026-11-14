"""
GloFAS historical reanalysis at country level (CEMS Early Warning Data Store).

Publisher: Copernicus Emergency Management Service, ECMWF.
Dataset: ``cems-glofas-historical``.
Licence: CEMS-FLOODS. The EWDS account must accept that licence before a
download will run.
Landing page: https://ewds.climate.copernicus.eu/datasets/cems-glofas-historical

The operational system version is ``version_4_0`` (0.05 degree native grid,
daily, 1979 to near real time). ``version_5_0`` is pre-operational. Variable
names, the timespan of each variable, and which days exist are read from the
dataset's catalogue constraints. A day is not requested unless that
constraint lists it.

A data unit is one variable for one rectangular extract of a country. The
country's administrative boundary, including islands and other parts, decides
the extract. Cells stay on the native grid: the request asks EWDS for a
sub-region and does not set a target grid. Consolidated days are kept once
downloaded. Intermediate days are requested again when the catalogue's
updated timestamp changes, because that product is revised until the
consolidated month is published. A run with nothing new logs that the
country is already up to date.

The reader returns an ``xarray.Dataset`` for one variable and one area, or a
dict when several variables or areas were requested. Time stays daily.
"""

from __future__ import annotations

import dataclasses
import io
import json
import math
import os
import tempfile
import threading
import time
import zipfile
from datetime import date, datetime, timezone
from pathlib import Path
from typing import ClassVar, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pycountry
import requests
from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass
from shapely.geometry import box as shapely_box
from shapely.geometry.base import BaseGeometry

from gigaspatial.config import config as global_config
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
from gigaspatial.handlers.base.fetch import stream_to_store
from gigaspatial.handlers.base.source import SourceKind, classify_source


DATASET_ID = "cems-glofas-historical"
DEFAULT_API_URL = "https://ewds.climate.copernicus.eu/api"
DATASET_PAGE = "https://ewds.climate.copernicus.eu/datasets/cems-glofas-historical"
LICENCE = "CEMS-FLOODS"
HYDROLOGICAL_MODEL = "lisflood"
SYSTEM_VERSIONS = ("version_4_0", "version_5_0")
# Published request limit for this dataset is 500 fields. One field is one day
# of one variable. Stay under that limit.
MAX_FIELDS_PER_REQUEST = 366
# Version 4.0 is the operational 0.05 degree grid. The form does not publish a
# separate spacing for version 5.0, so only version 4.0 is checked against a
# number. Both versions must still be a regular latitude/longitude grid.
NATIVE_RESOLUTION_DEGREES = {"version_4_0": 0.05}
# One native cell, plus the EWDS rule that a point on the area boundary is
# dropped. The form's coordinate step is 0.001 degree.
AREA_MARGIN_DEGREES = 0.051
CLUSTER_GAP_DEGREES = 2.0
LON_MIN, LON_MAX = -180.0, 180.0
LAT_MIN, LAT_MAX = -60.0, 90.0
TRANSIENT_STATUS = {429, 500, 502, 503, 504}
JOB_DONE = {"successful"}
JOB_FAILED = {"failed", "rejected", "dismissed"}

# Names confirmed from the EWDS form and constraints on 2026-11-14.
# ``dis24`` and the other GRIB short names are not the API variable names.
VARIABLES: Tuple[str, ...] = (
    "average_river_discharge_in_the_last_24_hours",
    "runoff_water_equivalent",
    "snow_depth_water_equivalent",
    "soil_wetness_index",
)
VARIABLE_LABELS = {
    "average_river_discharge_in_the_last_24_hours": "Average river discharge in the last 24 hours",
    "runoff_water_equivalent": "Runoff water equivalent (surface plus subsurface)",
    "snow_depth_water_equivalent": "Snow depth water equivalent",
    "soil_wetness_index": "Soil wetness index (root zone)",
}
DOCUMENTED_UNITS = {
    "average_river_discharge_in_the_last_24_hours": "m3 s-1",
    "runoff_water_equivalent": "kg m-2",
    "snow_depth_water_equivalent": "kg m-2",
    "soil_wetness_index": "1",
}
GRIB_SHORT_NAMES = {
    "average_river_discharge_in_the_last_24_hours": "dis24",
    "runoff_water_equivalent": "rowe",
    "snow_depth_water_equivalent": "sd",
    "soil_wetness_index": "swir",
}
# Static maps published for version 4.0. No version 5.0 file is on the
# auxiliary-data page. They are not part of the time-series request.
AUXILIARY = {
    "upstream_area": {
        "label": "Upstream area",
        "units": "m2",
        "versions": frozenset({"version_4_0"}),
        "url": (
            "https://confluence.ecmwf.int/download/attachments/242067380/"
            "uparea_glofas_v4_0.nc"
        ),
    },
    "elevation": {
        "label": "Elevation",
        "units": "m",
        "versions": frozenset({"version_4_0"}),
        "url": (
            "https://confluence.ecmwf.int/download/attachments/242067380/"
            "elevation_glofas_v4_0.nc"
        ),
    },
}
TIME_NAMES = ("time", "valid_time")
LAT_NAMES = ("latitude", "lat")
LON_NAMES = ("longitude", "lon")


@dataclasses.dataclass(frozen=True)
class GlofasHistoricalResolveOptions(ResolveOptions):
    """
    Resolve options for one GloFAS historical load.

    Attributes:
        variable: One EWDS variable name. Omit it to load all four time-series
            variables. ``upstream_area`` and ``elevation`` select a static map.
        include_auxiliary: Also fetch upstream area and elevation when the
            selected system version has a published file. Off by default.
    """

    variable: Optional[str] = None
    include_auxiliary: bool = False

    def __post_init__(self):
        known = set(VARIABLES) | set(AUXILIARY)
        if self.variable is not None and self.variable not in known:
            raise ValueError(
                f"variable must be one of {tuple(sorted(known))}, got {self.variable!r}. "
                "Use the EWDS name, for example "
                "'average_river_discharge_in_the_last_24_hours', not 'dis24'."
            )


class Availability:
    """Days the catalogue currently publishes for one variable."""

    def __init__(self, timespan: str, preferred: Dict[date, str]):
        self.timespan = timespan
        self.preferred = preferred

    @property
    def latest(self) -> Optional[date]:
        if not self.preferred:
            return None
        return max(self.preferred)


class HistoricalCatalogue:
    """Constraints and update time for ``cems-glofas-historical``."""

    def __init__(self, collection: dict, constraints: list):
        self.updated = collection.get("updated")
        self.doi = collection.get("sci:doi")
        interval = collection["extent"]["temporal"]["interval"][0]
        self.temporal_start = _parse_date(interval[0])
        self.temporal_end = _parse_date(interval[1])
        self._constraints = constraints
        self._cache: Dict[Tuple[str, str], Availability] = {}

    def availability(self, system_version: str, variable: str) -> Availability:
        key = (system_version, variable)
        if key not in self._cache:
            self._cache[key] = _availability(self._constraints, system_version, variable)
        return self._cache[key]


def fetch_catalogue(api_url: str = DEFAULT_API_URL, session=None) -> HistoricalCatalogue:
    """Read the public catalogue. This call does not need an API key."""
    session = session or requests.Session()
    root = api_url.rstrip("/")
    collection_url = f"{root}/catalogue/v1/collections/{DATASET_ID}"
    collection = _request(session, "GET", collection_url, timeout=60).json()
    constraints_url = next(
        link["href"] for link in collection["links"] if link.get("rel") == "constraints"
    )
    constraints = _request(session, "GET", constraints_url, timeout=120).json()
    return HistoricalCatalogue(collection, constraints)


def dates_to_fetch(
    stored: Dict[str, str],
    preferred: Dict[date, str],
    *,
    catalogue_updated: Optional[str],
    stored_catalogue_updated: Optional[str],
) -> List[Tuple[date, str]]:
    """
    Days that still need a request.

    A stored day is kept when it already came from the preferred product.
    Consolidated replaces intermediate. When ``catalogue_updated`` has changed,
    days that are still only published as intermediate are requested again.
    """
    have = {}
    for key, product in stored.items():
        have[date.fromisoformat(key)] = product
    refresh_tail = catalogue_updated != stored_catalogue_updated
    pending = []
    for day, product in sorted(preferred.items()):
        current = have.get(day)
        revised_tail = refresh_tail and product == "intermediate" and current == "intermediate"
        if current != product or revised_tail:
            pending.append((day, product))
    return pending


def group_dates(
    pending: Sequence[Tuple[date, str]], limit: int = MAX_FIELDS_PER_REQUEST
) -> List[dict]:
    """
    Group days into requests whose month/day lists are valid together.

    A request lists months and days as two independent lists, so every month
    in the request must need exactly the same days. The number of fields
    (months times days) stays under the dataset's published limit.
    """
    by_month: Dict[Tuple[str, int, int], List[int]] = {}
    for day, product in pending:
        by_month.setdefault((product, day.year, day.month), []).append(day.day)
    rectangles: Dict[Tuple[str, int, frozenset], List[int]] = {}
    for (product, year, month), days in by_month.items():
        rectangles.setdefault((product, year, frozenset(days)), []).append(month)

    groups = []
    order = sorted(rectangles, key=lambda item: (item[0], item[1], sorted(item[2])))
    for product, year, doms in order:
        months = sorted(rectangles[(product, year, doms)])
        day_numbers = sorted(doms)
        width = max(len(day_numbers), 1)
        capacity = max(1, limit // width)
        for start in range(0, len(months), capacity):
            chunk_months = months[start : start + capacity]
            dates = [
                date(year, month, day_number)
                for month in chunk_months
                for day_number in day_numbers
            ]
            groups.append(
                {
                    "product_type": product,
                    "year": year,
                    "months": chunk_months,
                    "days": day_numbers,
                    "dates": dates,
                }
            )
    return groups


def extraction_areas(
    geometry: BaseGeometry, margin: float = AREA_MARGIN_DEGREES
) -> List[Tuple[float, float, float, float]]:
    """
    Rectangles that cover every part of a country, as ``(north, west, south, east)``.

    Nearby parts share one rectangle. Parts that sit far apart, including
    islands and the two sides of the antimeridian, stay separate so the
    request does not span the empty ocean between them. Each rectangle is
    expanded by one grid cell. EWDS returns cells inside the rectangle, not
    cells that only touch its edge.
    """
    pieces = []
    west_hemisphere = shapely_box(-180, -90, 0, 90)
    east_hemisphere = shapely_box(0, -90, 180, 90)
    for part in _polygons(geometry):
        minx, _, maxx, _ = part.bounds
        if maxx - minx > 180:
            for hemisphere in (west_hemisphere, east_hemisphere):
                piece = part.intersection(hemisphere)
                pieces.extend(_polygons(piece))
        else:
            pieces.append(part)
    if not pieces:
        raise ValueError("The country geometry is empty.")

    clusters = _cluster_bounds([piece.bounds for piece in pieces])
    areas = []
    for minx, miny, maxx, maxy in clusters:
        north, west, south, east = _snap_outward(
            min(LAT_MAX, maxy + margin),
            max(LON_MIN, minx - margin),
            max(LAT_MIN, miny - margin),
            min(LON_MAX, maxx + margin),
        )
        if east <= west or north <= south:
            continue
        areas.append((north, west, south, east))
    if not areas:
        raise ValueError("The country does not intersect the GloFAS domain (90N-60S).")
    areas.sort(key=lambda area: (area[1], area[3], area[2], area[0]))
    return areas


def merge_cubes(existing, incoming):
    """Append days. A repeated timestamp keeps the incoming value."""
    incoming = _orient(incoming)
    if existing is None:
        return incoming
    existing = _orient(existing)
    if not _same_grid(existing, incoming):
        raise ValueError(
            "The new file is not on the same native grid as the stored file. "
            "It was not resampled or written."
        )
    combined = _xarray().concat([existing, incoming], dim="time")
    stamps = list(pd.to_datetime(combined["time"].values))
    last = {}
    for index, stamp in enumerate(stamps):
        last[pd.Timestamp(stamp).to_datetime64()] = index
    order = [last[stamp] for stamp in sorted(last)]
    return combined.isel(time=order)


def validate_cube(dataset, *, system_version: str, requested: Sequence[date], area: Tuple[float, float, float, float]):
    """
    Check a downloaded cube before it is merged.

    Returns the grid step in degrees and the requested days that were absent.
    """
    if "time" not in dataset.dims or "latitude" not in dataset.dims or "longitude" not in dataset.dims:
        raise ValueError("Downloaded file is missing time, latitude, or longitude.")
    extra = [
        name
        for name, size in dataset.sizes.items()
        if name not in ("time", "latitude", "longitude") and size > 1
    ]
    if extra:
        raise ValueError(f"Unexpected dimension {extra[0]} of size {dataset.sizes[extra[0]]}.")

    stamps = [pd.Timestamp(value).date() for value in dataset["time"].values]
    if not stamps:
        raise ValueError("Downloaded file has no timestamps.")
    if len(stamps) != len(set(stamps)):
        raise ValueError("Downloaded file has duplicate timestamps.")
    unexpected = sorted(set(stamps) - set(requested))
    if unexpected:
        raise ValueError(f"Downloaded file has timestamps that were not requested: {unexpected[:5]}.")

    latitude = np.asarray(dataset["latitude"].values, dtype=float)
    longitude = np.asarray(dataset["longitude"].values, dtype=float)
    lat_step = _grid_step(latitude)
    lon_step = _grid_step(longitude)
    if lat_step is not None and lon_step is not None and abs(lat_step - lon_step) > 1e-3:
        raise ValueError("Latitude and longitude spacing differ.")
    step = lat_step if lat_step is not None else lon_step
    expected = NATIVE_RESOLUTION_DEGREES.get(system_version)
    if expected is not None and step is not None and abs(step - expected) > 1e-3:
        raise ValueError(
            f"Spatial step is {step} degrees. Version {system_version} is published at {expected} degrees. "
            "The file was not resampled."
        )
    north, west, south, east = area
    pad = (step or expected or 0.05) + 0.02
    if latitude.size and (float(latitude.max()) > north + pad or float(latitude.min()) < south - pad):
        raise ValueError("Latitude extent is outside the requested country area.")
    if longitude.size and (float(longitude.max()) > east + pad or float(longitude.min()) < west - pad):
        raise ValueError("Longitude extent is outside the requested country area.")
    span = max(east - west, north - south)
    if step is None and span > 0.2:
        raise ValueError("The country area covers many cells but the file has a single cell.")
    missing = sorted(set(requested) - set(stamps))
    return step, missing


def standardize_cube(dataset, variable: str):
    """Rename coordinates to ``time``, ``latitude``, and ``longitude`` and keep one data variable."""
    xr = _xarray()
    rename = {}
    time_name = _coordinate_name(dataset, TIME_NAMES)
    lat_name = _coordinate_name(dataset, LAT_NAMES)
    lon_name = _coordinate_name(dataset, LON_NAMES)
    if time_name != "time":
        rename[time_name] = "time"
    if lat_name != "latitude":
        rename[lat_name] = "latitude"
    if lon_name != "longitude":
        rename[lon_name] = "longitude"
    if rename:
        dataset = dataset.rename(rename)
    for name, size in list(dataset.sizes.items()):
        if name not in ("time", "latitude", "longitude") and size == 1:
            dataset = dataset.squeeze(name, drop=True)
    data_names = [
        name
        for name, var in dataset.data_vars.items()
        if {"time", "latitude", "longitude"} <= set(var.dims)
    ]
    if len(data_names) != 1:
        raise ValueError(f"Expected one data variable, found {data_names}.")
    source_name = data_names[0]
    dataset = dataset[[source_name]]
    if source_name != variable:
        dataset = dataset.rename({source_name: variable})
    units = dataset[variable].attrs.get("units") or DOCUMENTED_UNITS.get(variable, "")
    dataset[variable].attrs["units"] = units
    times = pd.to_datetime(dataset["time"].values, utc=True).floor("D").tz_localize(None)
    dataset = dataset.assign_coords(time=times.values)
    return dataset


@dataclass(config=ConfigDict(arbitrary_types_allowed=True, extra="forbid"))
class GlofasHistoricalConfig(BaseHandlerConfig):
    """
    Configuration for country-level GloFAS historical files.

    Attributes:
        base_path: Directory of the stored files, under ``bronze/cems/glofas_historical``.
        system_version: ``version_4_0`` is operational. ``version_5_0`` is pre-operational.
        api_url: EWDS API root.
        api_key: Personal access token. Defaults to ``EWDS_API_KEY``.
        request_timeout_seconds: How long to wait for one EWDS job.
    """

    options_type = GlofasHistoricalResolveOptions
    units_match_country = True

    base_path: Path = Field(
        default_factory=lambda: global_config.get_path("cems", "bronze") / "glofas_historical"
    )
    system_version: str = "version_4_0"
    api_url: str = Field(default_factory=lambda: global_config.EWDS_API_URL or DEFAULT_API_URL)
    api_key: str = Field(default_factory=lambda: global_config.EWDS_API_KEY)
    request_timeout_seconds: int = 3600

    def __post_init__(self):
        super().__post_init__()
        if self.system_version not in SYSTEM_VERSIONS:
            raise ValueError(
                f"system_version must be one of {SYSTEM_VERSIONS}, got {self.system_version!r}."
            )
        self.api_url = self.api_url.rstrip("/")
        self._catalogue = None
        self._catalogue_lock = threading.Lock()

    def catalogue(self) -> HistoricalCatalogue:
        """The catalogue for this run. Fetched once."""
        if self._catalogue is not None:
            return self._catalogue
        with self._catalogue_lock:
            if self._catalogue is None:
                self._catalogue = fetch_catalogue(self.api_url)
        return self._catalogue

    def get_search_key(self, source, options: GlofasHistoricalResolveOptions) -> str:
        """Return the ISO3 code. The files are stored one country at a time."""
        if classify_source(source) is not SourceKind.COUNTRY:
            raise ValueError(
                "GloFAS historical is stored by country. "
                "Pass a country name or ISO code, for example 'RWA'."
            )
        try:
            return pycountry.countries.lookup(source).alpha_3
        except LookupError as exc:
            raise ValueError(f"Unknown country: {source!r}.") from exc

    def get_relevant_data_units_by_key(
        self, search_key: str, options: GlofasHistoricalResolveOptions
    ) -> List[dict]:
        """One unit per variable and per rectangle that covers the country."""
        geometry = source_geometry(search_key, crs=self.crs)
        areas = extraction_areas(geometry)
        country = pycountry.countries.get(alpha_3=search_key)
        country_name = country.name if country is not None else search_key
        variables = self._series_variables(options)
        units = []
        for variable in variables:
            for area in areas:
                units.append(
                    self._unit(
                        search_key,
                        country_name,
                        variable,
                        area,
                        kind="series",
                    )
                )
        for name in self._auxiliary_variables(options):
            spec = AUXILIARY[name]
            if self.system_version not in spec["versions"]:
                self.logger.warning(
                    "No %s file is published for %s. The time series is unchanged.",
                    name,
                    self.system_version,
                )
                continue
            for area in areas:
                units.append(
                    self._unit(search_key, country_name, name, area, kind="auxiliary")
                )
        if not units:
            raise ValueError(f"No GloFAS historical variables were selected for {search_key}.")
        return units

    def get_data_unit_path(self, unit: dict, options=None) -> Path:
        """Metadata file for one variable and one country rectangle."""
        return self._unit_dir(unit) / "metadata.json"

    def get_stored_paths(self, unit: dict, options=None) -> List[Path]:
        """Metadata plus every NetCDF written for the unit."""
        directory = self._unit_dir(unit)
        paths = [directory / "metadata.json"]
        info = _read_json(self.data_store, directory / "metadata.json")
        if info:
            for name in info.get("files", []):
                paths.append(directory / name)
        return paths

    def is_unit_available(self, unit: dict, paths) -> bool:
        """True when the stored days already match the catalogue."""
        if unit["kind"] == "auxiliary":
            return self.data_store.file_exists(str(self._unit_dir(unit) / "static.nc"))
        stored, meta = self._stored_series(unit)
        availability = self.catalogue().availability(unit["system_version"], unit["variable"])
        pending = dates_to_fetch(
            stored,
            availability.preferred,
            catalogue_updated=self.catalogue().updated,
            stored_catalogue_updated=meta.get("catalogue_updated"),
        )
        if pending or not stored:
            return False
        self.logger.info(
            "%s %s is already up to date. Latest observation %s. Last download %s.",
            unit["country_iso3"],
            unit["variable"],
            meta.get("source_latest_observation"),
            meta.get("last_successful_download_utc"),
        )
        return True

    def _series_variables(self, options: GlofasHistoricalResolveOptions) -> Tuple[str, ...]:
        if options.variable in AUXILIARY:
            return ()
        if options.variable is None:
            return VARIABLES
        return (options.variable,)

    def _auxiliary_variables(self, options: GlofasHistoricalResolveOptions) -> Tuple[str, ...]:
        if options.variable in AUXILIARY:
            return (options.variable,)
        if options.include_auxiliary and options.variable not in AUXILIARY:
            return tuple(AUXILIARY)
        return ()

    def _unit(self, iso3: str, country_name: str, variable: str, area, kind: str) -> dict:
        north, west, south, east = area
        return {
            "kind": kind,
            "country_iso3": iso3,
            "country_name": country_name,
            "system_version": self.system_version,
            "variable": variable,
            "area": area,
            "area_id": _area_id(north, west, south, east),
        }

    def _unit_dir(self, unit: dict) -> Path:
        return (
            self.base_path
            / unit["system_version"]
            / unit["country_iso3"]
            / unit["variable"]
            / unit["area_id"]
        )

    def _year_path(self, unit: dict, year: int) -> Path:
        return self._unit_dir(unit) / f"{year}.nc"

    def _source_auxiliary_path(self, variable: str) -> Path:
        return self.base_path / self.system_version / "_source" / f"{variable}.nc"

    def _stored_series(self, unit: dict) -> Tuple[Dict[str, str], dict]:
        meta = _read_json(self.data_store, self._unit_dir(unit) / "metadata.json") or {}
        stored = {}
        for key, product in (meta.get("dates") or {}).items():
            if self.data_store.file_exists(str(self._year_path(unit, int(key[:4])))):
                stored[key] = product
        if stored:
            return stored, meta
        recovered = {}
        for path in _nc_files(self.data_store, self._unit_dir(unit)):
            if path.name == "static.nc":
                continue
            try:
                dataset = _open_path(self.data_store, path)
            except (OSError, ValueError):
                continue
            try:
                recovered.update(json.loads(dataset.attrs.get("gigaspatial_dates") or "{}"))
            finally:
                dataset.close()
        return recovered, meta


class GlofasHistoricalDownloader(BaseHandlerDownloader):
    """Downloads GloFAS historical days that are not stored yet."""

    options_type = DownloadOptions
    config_type: ClassVar[type] = GlofasHistoricalConfig

    def download_data_unit(self, unit: dict, options: DownloadOptions) -> List[Path]:
        """
        Sync one variable for one country rectangle.

        Raises:
            RuntimeError: The EWDS request failed, or ``EWDS_API_KEY`` is missing.
            ValueError: The file is not a regular daily grid at the native spacing.
        """
        if unit["kind"] == "auxiliary":
            self._download_auxiliary(unit, options)
        else:
            self._sync_series(unit, options)
        return self.config.get_stored_paths(unit)

    def _sync_series(self, unit: dict, options: DownloadOptions) -> None:
        catalogue = self.config.catalogue()
        availability = catalogue.availability(unit["system_version"], unit["variable"])
        stored, meta = self.config._stored_series(unit)
        pending = dates_to_fetch(
            stored,
            availability.preferred,
            catalogue_updated=catalogue.updated,
            stored_catalogue_updated=meta.get("catalogue_updated"),
        )
        if not pending:
            self._write_checked(unit, meta, catalogue, availability, stored, downloaded=False)
            self.logger.info(
                "%s %s is already up to date. Latest observation %s. Last download %s.",
                unit["country_iso3"],
                unit["variable"],
                meta.get("source_latest_observation") or _iso(availability.latest),
                meta.get("last_successful_download_utc"),
            )
            return
        self._require_key()
        groups = group_dates(pending)
        self.logger.info(
            "%s %s: retrieving %d day(s) in %d request(s). Source latest %s.",
            unit["country_iso3"],
            unit["variable"],
            len(pending),
            len(groups),
            _iso(availability.latest),
        )
        for group in groups:
            returned = self._fetch_group(unit, availability.timespan, group, options, meta)
            for day in returned:
                stored[day.isoformat()] = group["product_type"]
            self._write_progress(unit, meta, catalogue, availability, stored)
        written = self._write_checked(
            unit, meta, catalogue, availability, stored, downloaded=True
        )
        self.logger.info(
            "%s %s stored through %s. Downloaded at %s. Source latest %s.",
            unit["country_iso3"],
            unit["variable"],
            written.get("latest_stored_observation"),
            written.get("last_successful_download_utc"),
            written.get("source_latest_observation"),
        )

    def _fetch_group(self, unit, timespan: str, group: dict, options: DownloadOptions, meta: dict) -> List[date]:
        request = {
            "system_version": [unit["system_version"]],
            "hydrological_model": [HYDROLOGICAL_MODEL],
            "product_type": [group["product_type"]],
            "variable": [unit["variable"]],
            "timespan": [timespan],
            "year": [f"{group['year']:04d}"],
            "month": [f"{month:02d}" for month in group["months"]],
            "day": [f"{day:02d}" for day in group["days"]],
            "area": list(unit["area"]),
            "data_format": "netcdf",
            "download_format": "zip",
        }
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "download.bin"
            self._retrieve(request, dest, options)
            dataset = standardize_cube(_open_download(dest), unit["variable"])
            try:
                step, missing = validate_cube(
                    dataset,
                    system_version=unit["system_version"],
                    requested=group["dates"],
                    area=unit["area"],
                )
                if missing:
                    self.logger.warning(
                        "%s %s: EWDS omitted %d of %d requested day(s) for %s %s. "
                        "Those days stay pending.",
                        unit["country_iso3"],
                        unit["variable"],
                        len(missing),
                        len(group["dates"]),
                        group["product_type"],
                        group["year"],
                    )
                returned = [day for day in group["dates"] if day not in missing]
                if not returned:
                    raise RuntimeError(
                        f"EWDS returned none of the {len(group['dates'])} requested days "
                        f"for {unit['variable']}."
                    )
                meta["units"] = str(
                    dataset[unit["variable"]].attrs.get("units")
                    or DOCUMENTED_UNITS.get(unit["variable"], "")
                )
                if step is not None:
                    meta["spatial_resolution_degrees"] = round(float(step), 6)
                self._merge_year(unit, group, dataset, returned)
            finally:
                dataset.close()
        return returned

    def _merge_year(self, unit, group, incoming, returned: Sequence[date]) -> None:
        path = self.config._year_path(unit, group["year"])
        existing = None
        merged = None
        try:
            if self.data_store.file_exists(str(path)):
                existing = _open_path(self.data_store, path)
            merged = merge_cubes(existing, incoming)
            products = {}
            if existing is not None:
                products.update(json.loads(existing.attrs.get("gigaspatial_dates") or "{}"))
            for day in returned:
                products[day.isoformat()] = group["product_type"]
            merged.attrs["gigaspatial_dates"] = json.dumps(products, sort_keys=True)
            _commit_dataset(self.data_store, path, merged)
        finally:
            if existing is not None:
                existing.close()
            if merged is not None and merged is not incoming:
                merged.close()

    def _download_auxiliary(self, unit: dict, options: DownloadOptions) -> None:
        spec = AUXILIARY[unit["variable"]]
        source = self.config._source_auxiliary_path(unit["variable"])
        if not self.data_store.file_exists(str(source)):
            self.logger.info("Downloading GloFAS %s for %s.", unit["variable"], unit["system_version"])
            stream_to_store(
                spec["url"],
                source,
                self.data_store,
                show_progress=options.show_progress,
            )
        dataset = _orient(_rename_static(_open_path(self.data_store, source)))
        subset = None
        try:
            subset = _subset_grid(dataset, unit["area"]).load()
            _reject_irregular(subset, unit["system_version"])
            target = self.config._unit_dir(unit) / "static.nc"
            _commit_dataset(self.data_store, target, subset)
        finally:
            dataset.close()
            if subset is not None:
                subset.close()
        now = _utc_now()
        _write_json(
            self.data_store,
            self.config.get_data_unit_path(unit),
            {
                "country_iso3": unit["country_iso3"],
                "country_name": unit["country_name"],
                "dataset": DATASET_ID,
                "system_version": unit["system_version"],
                "variable": unit["variable"],
                "variable_label": spec["label"],
                "units": spec["units"],
                "temporal_resolution": "static",
                "spatial_resolution_degrees": NATIVE_RESOLUTION_DEGREES.get(unit["system_version"]),
                "last_successful_download_utc": now,
                "last_successful_check_utc": now,
                "latest_stored_observation": None,
                "source_latest_observation": None,
                "files": ["static.nc"],
                "source_url": spec["url"],
                "licence": LICENCE,
                "area": list(unit["area"]),
            },
        )

    def _retrieve(self, request: dict, dest: Path, options: DownloadOptions) -> None:
        """Submit one EWDS job and store its file at ``dest``."""
        session = requests.Session()
        session.headers["PRIVATE-TOKEN"] = self.config.api_key
        root = self.config.api_url
        response = _request(
            session,
            "POST",
            f"{root}/retrieve/v1/processes/{DATASET_ID}/execution",
            json={"inputs": request},
            timeout=120,
        )
        if response.status_code >= 400:
            raise RuntimeError(_provider_message(response, self.config.api_key))
        job = response.json()
        job_id = job["jobID"]
        status = job.get("status")
        deadline = time.monotonic() + self.config.request_timeout_seconds
        pause = 2.0
        while status not in JOB_DONE | JOB_FAILED:
            if time.monotonic() > deadline:
                raise TimeoutError(f"EWDS job {job_id} is still {status}.")
            time.sleep(pause)
            pause = min(pause * 1.5, 30.0)
            status_response = _request(
                session, "GET", f"{root}/retrieve/v1/jobs/{job_id}", timeout=60
            )
            if status_response.status_code >= 400:
                raise RuntimeError(_provider_message(status_response, self.config.api_key))
            job = status_response.json()
            status = job.get("status")
        if status not in JOB_DONE:
            message = job.get("message") or ""
            raise RuntimeError(f"EWDS job {job_id} {status}. {message}".strip())
        results = _request(
            session, "GET", f"{root}/retrieve/v1/jobs/{job_id}/results", timeout=60
        )
        if results.status_code >= 400:
            raise RuntimeError(_provider_message(results, self.config.api_key))
        href = _asset_href(results.json())
        if not href:
            raise RuntimeError(f"EWDS job {job_id} finished without a download URL.")
        stream_to_store(
            href,
            dest,
            self.data_store,
            session=session,
            show_progress=options.show_progress,
            timeout=300,
        )

    def _require_key(self) -> None:
        if not self.config.api_key:
            raise RuntimeError(
                "EWDS_API_KEY is not set. Create a token at "
                f"{DATASET_PAGE} and accept the {LICENCE} licence, "
                "then set EWDS_API_KEY."
            )

    def _write_progress(self, unit, meta, catalogue, availability, stored) -> None:
        """Record days already on disk so an interrupted run can resume."""
        meta = _base_metadata(unit, meta, catalogue, availability, stored, self.config.api_url)
        meta["files"] = _year_file_names(stored)
        _write_json(self.data_store, self.config.get_data_unit_path(unit), meta)

    def _write_checked(self, unit, meta, catalogue, availability, stored, downloaded: bool) -> None:
        now = _utc_now()
        meta = _base_metadata(unit, meta, catalogue, availability, stored, self.config.api_url)
        meta["files"] = _year_file_names(stored)
        meta["last_successful_check_utc"] = now
        meta["catalogue_updated"] = catalogue.updated
        if downloaded:
            meta["last_successful_download_utc"] = now
        elif "last_successful_download_utc" not in meta:
            meta["last_successful_download_utc"] = None
        _write_json(self.data_store, self.config.get_data_unit_path(unit), meta)
        return meta


class GlofasHistoricalReader(BaseHandlerReader):
    """Loads stored GloFAS historical NetCDF files."""

    options_type = ReadOptions
    config_type: ClassVar[type] = GlofasHistoricalConfig

    def load_from_paths(self, source_data_path: List[str], options: ReadOptions):
        """
        Load stored files.

        One variable and one area returns a Dataset. Several variables return
        a dict keyed by variable. Several areas return a dict keyed by area id.
        """
        groups: Dict[str, Dict[str, List]] = {}
        for raw in source_data_path:
            path = Path(raw)
            if path.suffix != ".nc":
                continue
            dataset = _open_path(self.data_store, path)
            try:
                if path.name == "static.nc":
                    loaded = _rename_static(dataset).load()
                else:
                    variable = path.parent.parent.name
                    loaded = standardize_cube(dataset, variable).load()
            finally:
                dataset.close()
            variable = path.parent.parent.name
            area_id = path.parent.name
            groups.setdefault(variable, {}).setdefault(area_id, []).append(loaded)
        if not groups:
            raise ValueError("No stored GloFAS historical files were found.")
        assembled = {}
        for variable, areas in groups.items():
            loaded_areas = {
                area_id: _concat_loaded(parts) for area_id, parts in areas.items()
            }
            assembled[variable] = (
                next(iter(loaded_areas.values()))
                if len(loaded_areas) == 1
                else loaded_areas
            )
        if len(assembled) == 1:
            return next(iter(assembled.values()))
        return assembled


class GlofasHistoricalHandler(BaseHandler):
    """
    Handler for GloFAS historical reanalysis at country level.

    The operational dataset is version 4.0. One call checks the catalogue and
    downloads only days that are missing or still published as intermediate
    after a catalogue update:

        GlofasHistoricalHandler().load_data("RWA")
        GlofasHistoricalHandler().load_data(
            "RWA", variable="average_river_discharge_in_the_last_24_hours"
        )

    Set ``EWDS_API_KEY`` and accept the CEMS-FLOODS licence on the dataset
    page before the first download. ``include_auxiliary=True`` also stores
    upstream area and elevation for version 4.0.
    """

    def create_config(self, data_store, logger, **kwargs) -> GlofasHistoricalConfig:
        kwargs.setdefault("n_workers", 1)
        return GlofasHistoricalConfig(data_store=data_store, logger=logger, **kwargs)

    def create_downloader(self, config, data_store, logger, **kwargs) -> GlofasHistoricalDownloader:
        return GlofasHistoricalDownloader(config=config, data_store=data_store, logger=logger)

    def create_reader(self, config, data_store, logger, **kwargs) -> GlofasHistoricalReader:
        return GlofasHistoricalReader(config=config, data_store=data_store, logger=logger)


def _availability(constraints: list, system_version: str, variable: str) -> Availability:
    timespans = set()
    consolidated = set()
    intermediate = set()
    for row in constraints:
        if system_version not in row.get("system_version", []):
            continue
        if variable not in row.get("variable", []):
            continue
        timespans.update(row.get("timespan") or [])
        products = set(row.get("product_type") or [])
        for year in row.get("year") or []:
            for month in row.get("month") or []:
                for day in row.get("day") or []:
                    try:
                        stamp = date(int(year), int(month), int(day))
                    except ValueError:
                        continue
                    if "consolidated" in products:
                        consolidated.add(stamp)
                    if "intermediate" in products:
                        intermediate.add(stamp)
    if not timespans:
        raise ValueError(
            f"{variable} is not in the {system_version} constraints for {DATASET_ID}. "
            "The EWDS variable name may have changed."
        )
    if len(timespans) != 1:
        raise ValueError(
            f"{variable} has more than one timespan in the constraints: {sorted(timespans)}."
        )
    preferred = {}
    for stamp in consolidated | intermediate:
        preferred[stamp] = "consolidated" if stamp in consolidated else "intermediate"
    if not preferred:
        raise ValueError(f"{variable} has no dates in the {system_version} constraints.")
    return Availability(next(iter(timespans)), preferred)


def _polygons(geometry: BaseGeometry) -> List[BaseGeometry]:
    if geometry is None or geometry.is_empty:
        return []
    kind = geometry.geom_type
    if kind == "Polygon":
        return [geometry]
    if kind in ("MultiPolygon", "GeometryCollection"):
        parts = []
        for child in geometry.geoms:
            parts.extend(_polygons(child))
        return parts
    return [geometry]


def _cluster_bounds(bounds: Sequence[Tuple[float, float, float, float]]):
    clusters = list(bounds)
    changed = True
    while changed:
        changed = False
        merged = []
        for item in clusters:
            hit = None
            for index, other in enumerate(merged):
                if _bounds_gap(item, other) <= CLUSTER_GAP_DEGREES:
                    hit = index
                    break
            if hit is None:
                merged.append(item)
            else:
                merged[hit] = _union_bounds(merged[hit], item)
                changed = True
        clusters = merged
    return clusters


def _bounds_gap(left, right) -> float:
    return max(
        _axis_gap(left[0], left[2], right[0], right[2]),
        _axis_gap(left[1], left[3], right[1], right[3]),
    )


def _axis_gap(a_min, a_max, b_min, b_max) -> float:
    if a_max < b_min:
        return b_min - a_max
    if b_max < a_min:
        return a_min - b_max
    return 0.0


def _union_bounds(left, right):
    return (
        min(left[0], right[0]),
        min(left[1], right[1]),
        max(left[2], right[2]),
        max(left[3], right[3]),
    )


def _snap_outward(north: float, west: float, south: float, east: float):
    step = 0.001

    def ceil(value: float) -> float:
        return math.ceil(round(value / step, 6)) * step

    def floor(value: float) -> float:
        return math.floor(round(value / step, 6)) * step

    north = min(LAT_MAX, ceil(north))
    south = max(LAT_MIN, floor(south))
    west = max(LON_MIN, floor(west))
    east = min(LON_MAX, ceil(east))
    return tuple(round(value, 3) for value in (north, west, south, east))


def _area_id(north: float, west: float, south: float, east: float) -> str:
    def part(prefix: str, value: float) -> str:
        text = f"{value:.3f}".replace("-", "m").replace(".", "p")
        return f"{prefix}{text}"

    return "_".join(
        (part("n", north), part("w", west), part("s", south), part("e", east))
    )


def _xarray():
    try:
        import xarray as xr
    except ImportError as exc:
        raise ImportError(
            "Reading GloFAS historical NetCDF needs the xarray and netCDF4 packages."
        ) from exc
    return xr


def _open_bytes(raw: bytes):
    """Open a NetCDF payload. The netCDF4 engine needs a real file path."""
    handle = tempfile.NamedTemporaryFile(suffix=".nc", delete=False)
    handle.write(raw)
    handle.close()
    temp_path = handle.name
    dataset = None
    try:
        dataset = _xarray().open_dataset(temp_path, engine="netcdf4")
        loaded = dataset.load()
    finally:
        if dataset is not None:
            dataset.close()
            del dataset
        os.unlink(temp_path)
    return loaded


def _dataset_bytes(dataset) -> bytes:
    handle = tempfile.NamedTemporaryFile(suffix=".nc", delete=False)
    handle.close()
    temp_path = handle.name
    try:
        dataset.to_netcdf(temp_path, engine="netcdf4")
        return Path(temp_path).read_bytes()
    finally:
        os.unlink(temp_path)


def _open_path(store, path: Path):
    return _open_bytes(store.read_file(str(path)))


def _open_download(path: Path):
    raw = path.read_bytes()
    if raw[:2] == b"PK":
        archive = zipfile.ZipFile(io.BytesIO(raw))
        members = [name for name in archive.namelist() if name.lower().endswith(".nc")]
        if not members:
            raise ValueError("The EWDS download is a zip with no NetCDF file.")
        datasets = [_open_bytes(archive.read(name)) for name in members]
        if len(datasets) == 1:
            return datasets[0]
        return _xarray().merge(datasets)
    return _open_bytes(raw)


def _commit_dataset(store, path: Path, dataset) -> None:
    payload = _dataset_bytes(dataset)
    partial = path.with_name(path.name + ".partial")
    store.write_file(str(partial), payload)
    if isinstance(store, LocalDataStore):
        source = store._resolve_path(partial)
        target = store._resolve_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(source, target)
        return
    store.write_file(str(path), payload)
    if store.file_exists(str(partial)):
        store.remove(str(partial))


def _orient(dataset):
    if dataset["latitude"].size > 1 and float(dataset["latitude"].values[0]) > float(
        dataset["latitude"].values[-1]
    ):
        dataset = dataset.sortby("latitude")
    if dataset["longitude"].size > 1 and float(dataset["longitude"].values[0]) > float(
        dataset["longitude"].values[-1]
    ):
        dataset = dataset.sortby("longitude")
    return dataset


def _same_grid(left, right) -> bool:
    return np.allclose(left["latitude"].values, right["latitude"].values) and np.allclose(
        left["longitude"].values, right["longitude"].values
    )


def _coordinate_name(dataset, names: Sequence[str]) -> str:
    for name in names:
        if name in dataset.coords or name in dataset.dims or name in dataset.variables:
            return name
    raise ValueError(f"No coordinate named {names} in the downloaded file.")


def _grid_step(values: np.ndarray) -> Optional[float]:
    if values.size < 2:
        return None
    diffs = np.diff(np.sort(values.astype(float)))
    if float(diffs.max() - diffs.min()) > 1e-4:
        raise ValueError("The downloaded grid spacing is not regular.")
    return float(np.median(diffs))


def _rename_static(dataset):
    rename = {}
    lat_name = _coordinate_name(dataset, LAT_NAMES)
    lon_name = _coordinate_name(dataset, LON_NAMES)
    if lat_name != "latitude":
        rename[lat_name] = "latitude"
    if lon_name != "longitude":
        rename[lon_name] = "longitude"
    if rename:
        dataset = dataset.rename(rename)
    return dataset


def _subset_grid(dataset, area):
    """Select native cells inside the rectangle. This does not interpolate."""
    north, west, south, east = area
    latitude = np.asarray(dataset["latitude"].values, dtype=float)
    longitude = np.asarray(dataset["longitude"].values, dtype=float)
    lat_index = np.where((latitude <= north + 1e-6) & (latitude >= south - 1e-6))[0]
    lon_index = np.where((longitude <= east + 1e-6) & (longitude >= west - 1e-6))[0]
    if lat_index.size == 0 or lon_index.size == 0:
        raise ValueError("The auxiliary grid does not cover the country area.")
    return dataset.isel(latitude=lat_index, longitude=lon_index)


def _reject_irregular(dataset, system_version: str) -> None:
    lat_step = _grid_step(np.asarray(dataset["latitude"].values, dtype=float))
    lon_step = _grid_step(np.asarray(dataset["longitude"].values, dtype=float))
    if lat_step is not None and lon_step is not None and abs(lat_step - lon_step) > 1e-3:
        raise ValueError("Auxiliary latitude and longitude spacing differ.")
    expected = NATIVE_RESOLUTION_DEGREES.get(system_version)
    step = lat_step if lat_step is not None else lon_step
    if expected is not None and step is not None and abs(step - expected) > 1e-3:
        raise ValueError(
            f"Auxiliary step is {step} degrees, not the native {expected} degree grid."
        )


def _concat_loaded(parts: List):
    if len(parts) == 1 or "time" not in parts[0].dims:
        return parts[0]
    combined = parts[0]
    for part in parts[1:]:
        combined = merge_cubes(combined, part)
    return combined


def _nc_files(store, directory: Path) -> List[Path]:
    names: List[str] = []
    try:
        names = list(store.list_files(str(directory)))
    except (FileNotFoundError, OSError, ValueError):
        names = []
    found = []
    for name in names:
        path = Path(name)
        if path.suffix == ".nc":
            found.append(path if path.is_absolute() else directory / path.name)
    if found:
        return sorted(found)
    if directory.is_dir():
        return sorted(path for path in directory.glob("*.nc") if path.is_file())
    return []


def _read_json(store, path: Path) -> Optional[dict]:
    if not store.file_exists(str(path)):
        return None
    raw = store.read_file(str(path), encoding="utf-8")
    return json.loads(raw)


def _write_json(store, path: Path, payload: dict) -> None:
    store.write_file(str(path), json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _base_metadata(unit, meta, catalogue, availability, stored, api_url: str) -> dict:
    stored_days = sorted(stored)
    latest_stored = stored_days[-1] if stored_days else None
    units = meta.get("units") or DOCUMENTED_UNITS.get(unit["variable"], "")
    step = meta.get("spatial_resolution_degrees")
    if step is None:
        step = NATIVE_RESOLUTION_DEGREES.get(unit["system_version"])
    payload = {
        "country_iso3": unit["country_iso3"],
        "country_name": unit["country_name"],
        "dataset": DATASET_ID,
        "doi": catalogue.doi,
        "system_version": unit["system_version"],
        "hydrological_model": HYDROLOGICAL_MODEL,
        "variable": unit["variable"],
        "variable_label": VARIABLE_LABELS.get(unit["variable"], unit["variable"]),
        "grib_short_name": GRIB_SHORT_NAMES.get(unit["variable"]),
        "units": units,
        "timespan": availability.timespan,
        "spatial_resolution_degrees": step,
        "temporal_resolution": "daily",
        "catalogue_temporal_start": _iso(catalogue.temporal_start),
        "catalogue_temporal_end": _iso(catalogue.temporal_end),
        "historical_coverage_start": stored_days[0] if stored_days else None,
        "historical_coverage_end": latest_stored,
        "latest_stored_observation": latest_stored,
        "source_latest_observation": _iso(availability.latest),
        "last_successful_download_utc": meta.get("last_successful_download_utc"),
        "last_successful_check_utc": meta.get("last_successful_check_utc"),
        "catalogue_updated": meta.get("catalogue_updated"),
        "dates": dict(sorted(stored.items())),
        "area": list(unit["area"]),
        "licence": LICENCE,
        "retrieval": {
            "dataset": DATASET_ID,
            "api_url": api_url,
            "dataset_page": DATASET_PAGE,
            "area_order": ["north", "west", "south", "east"],
        },
    }
    return payload


def _year_file_names(stored: Dict[str, str]) -> List[str]:
    years = sorted({key[:4] for key in stored})
    return [f"{year}.nc" for year in years]


def _request(session, method: str, url: str, **kwargs):
    delays = (1, 2, 4)
    last_error = None
    for attempt, delay in enumerate((*delays, None)):
        try:
            response = session.request(method, url, **kwargs)
        except requests.RequestException as exc:
            last_error = exc
            if delay is None:
                raise
            time.sleep(delay)
            continue
        if response.status_code in TRANSIENT_STATUS and delay is not None:
            time.sleep(delay)
            continue
        if response.status_code >= 400 and response.status_code not in TRANSIENT_STATUS:
            return response
        if response.status_code < 400:
            return response
        last_error = requests.HTTPError(f"{response.status_code} from {url}")
    if last_error:
        raise last_error
    raise requests.HTTPError(f"Request failed: {url}")


def _provider_message(response, api_key: str) -> str:
    text = response.text[:500].replace(api_key, "***") if api_key else response.text[:500]
    return f"EWDS request failed with HTTP {response.status_code}. {text}"


def _asset_href(payload) -> Optional[str]:
    if isinstance(payload, dict):
        asset = payload.get("asset")
        if isinstance(asset, dict):
            value = asset.get("value", asset)
            if isinstance(value, dict) and value.get("href"):
                return value["href"]
        if payload.get("href") and payload.get("rel") in (None, "data", "results"):
            return payload["href"]
        for value in payload.values():
            found = _asset_href(value)
            if found:
                return found
    if isinstance(payload, list):
        for item in payload:
            found = _asset_href(item)
            if found:
                return found
    return None


def _parse_date(value: str) -> date:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).date()


def _iso(value: Optional[date]) -> Optional[str]:
    if value is None:
        return None
    return value.isoformat()


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
