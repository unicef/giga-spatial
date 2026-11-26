"""
GloFAS historical reanalysis at country level (CEMS Early Warning Data Store).

Publisher: Copernicus Emergency Management Service, ECMWF.
Dataset: ``cems-glofas-historical``.
Licence: CEMS-FLOODS. The EWDS account must accept that licence before a
query will run. The static auxiliary maps are separate NetCDF files and do
not use the EWDS key.
Landing page: https://ewds.climate.copernicus.eu/datasets/cems-glofas-historical

``GlofasHistoricalFetcher`` queries that API and returns the result. It does
not write the dataset into a data store. Every call asks EWDS again.

The operational system version is ``version_4_0`` (0.05 degree native grid,
daily, 1979 to near real time). ``version_5_0`` is pre-operational. Variable
names, the timespan of each variable, and which days exist are read from the
dataset's catalogue constraints. A day is not requested unless that
constraint lists it.

The country's administrative boundary, including islands and other parts,
selects the native grid cells. The request asks EWDS for a sub-region and
does not set a target grid. Values are not resampled. Consolidated days are
preferred over intermediate days when the catalogue lists both. Omitting
``start`` and ``end`` queries every day the catalogue publishes.
"""

from __future__ import annotations

import dataclasses
import io
import logging
import math
import os
import tempfile
import threading
import time
import zipfile
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pycountry
import requests
from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass
from shapely.geometry import box as shapely_box
from shapely.geometry.base import BaseGeometry

from gigaspatial.config import config as global_config
from gigaspatial.handlers.base.config import source_geometry
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
class GlofasHistoricalResolveOptions:
    """
    Options for one GloFAS historical query.

    Attributes:
        variable: One EWDS variable name. Omit it to query all four time-series
            variables. ``upstream_area`` and ``elevation`` select a static map.
        include_auxiliary: Also query upstream area and elevation when the
            selected system version has a published file. Off by default.
        start: First day to query, inclusive. Omit it to start at the first
            day the catalogue publishes.
        end: Last day to query, inclusive. Omit it to stop at the latest day
            the catalogue publishes.
    """

    variable: Optional[str] = None
    include_auxiliary: bool = False
    start: Optional[date] = None
    end: Optional[date] = None

    def __post_init__(self):
        known = set(VARIABLES) | set(AUXILIARY)
        if self.variable is not None and self.variable not in known:
            raise ValueError(
                f"variable must be one of {tuple(sorted(known))}, got {self.variable!r}. "
                "Use the EWDS name, for example "
                "'average_river_discharge_in_the_last_24_hours', not 'dis24'."
            )
        object.__setattr__(self, "start", _coerce_date(self.start, "start"))
        object.__setattr__(self, "end", _coerce_date(self.end, "end"))
        if self.start is not None and self.end is not None and self.start > self.end:
            raise ValueError("start is after end.")


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


def selected_days(
    preferred: Dict[date, str],
    start: Optional[date],
    end: Optional[date],
) -> List[Tuple[date, str]]:
    """Catalogue days inside the inclusive window. Consolidated wins over intermediate."""
    chosen = []
    for day, product in sorted(preferred.items()):
        if start is not None and day < start:
            continue
        if end is not None and day > end:
            continue
        chosen.append((day, product))
    return chosen


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
            "The new file is not on the same native grid as the days already queried. "
            "It was not resampled."
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
    Check a queried cube.

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
class GlofasHistoricalFetcher:
    """
    Query GloFAS historical reanalysis for one country.

    The dataset is served by the EWDS retrieve API. ``fetch`` returns the
    time series, and any auxiliary maps requested with it, in memory. Nothing
    is written to a data store. A later call queries EWDS again.

    ::

        GlofasHistoricalFetcher().fetch("RWA", start="2020-01-01", end="2020-01-31")
        GlofasHistoricalFetcher().fetch(
            "RWA",
            variable="average_river_discharge_in_the_last_24_hours",
            start="2020-01-01",
            end="2020-01-02",
        )
        GlofasHistoricalFetcher().fetch("RWA", variable="upstream_area")
        GlofasHistoricalFetcher().fetch("RWA", include_auxiliary=True, start="2020-01-01", end="2020-01-02")

    Set ``EWDS_API_KEY`` and accept the CEMS-FLOODS licence before a time-series
    query. Auxiliary maps do not use that key. Omitting ``start`` and ``end``
    queries every day the catalogue publishes.
    """

    system_version: str = "version_4_0"
    api_url: str = Field(default_factory=lambda: global_config.EWDS_API_URL or DEFAULT_API_URL)
    api_key: str = Field(default_factory=lambda: global_config.EWDS_API_KEY, repr=False)
    request_timeout_seconds: int = 3600
    logger: Optional[logging.Logger] = Field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.system_version not in SYSTEM_VERSIONS:
            raise ValueError(
                f"system_version must be one of {SYSTEM_VERSIONS}, got {self.system_version!r}."
            )
        self.api_url = self.api_url.rstrip("/")
        if self.logger is None:
            self.logger = global_config.get_logger(self.__class__.__name__)
        self._catalogue = None
        self._catalogue_lock = threading.Lock()

    def fetch(
        self,
        country: str,
        *,
        variable: Optional[str] = None,
        include_auxiliary: bool = False,
        start=None,
        end=None,
    ):
        """
        Query one country and return the reanalysis.

        One series returns a Dataset. Several variables or areas return nested
        dicts, collapsing any level that has a single entry.

        Raises:
            RuntimeError: The EWDS request failed, or ``EWDS_API_KEY`` is missing.
            ValueError: The country, variable, window, or grid is invalid.
        """
        query = GlofasHistoricalResolveOptions(
            variable=variable,
            include_auxiliary=include_auxiliary,
            start=start,
            end=end,
        )
        iso3, _country_name = self._country(country)
        areas = extraction_areas(source_geometry(iso3, crs="EPSG:4326"))
        groups: Dict[str, Dict[str, List]] = {}
        if query.variable not in AUXILIARY:
            self._query_series(iso3, areas, query, groups)
        self._query_auxiliary(areas, query, groups)
        if not groups:
            raise ValueError(f"No GloFAS historical variables were selected for {iso3}.")
        return _assemble(groups)

    def catalogue(self) -> HistoricalCatalogue:
        """The catalogue for this query. Fetched once. This call does not need an API key."""
        if self._catalogue is not None:
            return self._catalogue
        with self._catalogue_lock:
            if self._catalogue is None:
                self._catalogue = fetch_catalogue(self.api_url)
        return self._catalogue

    def _country(self, country: str) -> Tuple[str, str]:
        if classify_source(country) is not SourceKind.COUNTRY:
            raise ValueError(
                "GloFAS historical is queried by country. "
                "Pass a country name or ISO code, for example 'RWA'."
            )
        try:
            record = pycountry.countries.lookup(country)
        except LookupError as exc:
            raise ValueError(f"Unknown country: {country!r}.") from exc
        return record.alpha_3, record.name

    def _query_series(self, iso3, areas, query, groups) -> None:
        requested = False
        for variable in _series_variables(query):
            try:
                availability = self.catalogue().availability(self.system_version, variable)
            except ValueError as exc:
                if query.variable == variable:
                    raise
                self.logger.warning("%s", exc)
                continue
            pending = selected_days(availability.preferred, query.start, query.end)
            if not pending:
                message = (
                    f"No {variable} days are published for {self.system_version} "
                    f"between {query.start} and {query.end}."
                )
                if query.variable == variable:
                    raise ValueError(message)
                self.logger.warning("%s", message)
                continue
            self._require_key()
            requested = True
            self.logger.info(
                "%s %s: retrieving %d day(s) in %d request(s). Source latest %s.",
                iso3,
                variable,
                len(pending),
                len(group_dates(pending)) * len(areas),
                _iso(availability.latest),
            )
            for area in areas:
                for group in group_dates(pending):
                    dataset = self._fetch_group(iso3, variable, availability.timespan, area, group)
                    groups.setdefault(variable, {}).setdefault(_area_id(*area), []).append(dataset)
        if _series_variables(query) and not requested and not groups:
            return

    def _query_auxiliary(self, areas, query, groups) -> None:
        for name in _auxiliary_names(query):
            spec = AUXILIARY[name]
            if self.system_version not in spec["versions"]:
                self.logger.warning(
                    "No %s file is published for %s. The time series is unchanged.",
                    name,
                    self.system_version,
                )
                continue
            for area in areas:
                dataset = self._fetch_auxiliary(name, area)
                groups.setdefault(name, {}).setdefault(_area_id(*area), []).append(dataset)

    def _fetch_group(self, iso3, variable, timespan, area, group) -> object:
        request = {
            "system_version": [self.system_version],
            "hydrological_model": [HYDROLOGICAL_MODEL],
            "product_type": [group["product_type"]],
            "variable": [variable],
            "timespan": [timespan],
            "year": [f"{group['year']:04d}"],
            "month": [f"{month:02d}" for month in group["months"]],
            "day": [f"{day:02d}" for day in group["days"]],
            "area": list(area),
            "data_format": "netcdf",
            "download_format": "zip",
        }
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "download.bin"
            self._retrieve(request, dest)
            dataset = standardize_cube(_open_download(dest), variable)
        step, missing = validate_cube(
            dataset,
            system_version=self.system_version,
            requested=group["dates"],
            area=area,
        )
        if missing:
            self.logger.warning(
                "%s %s: EWDS omitted %d of %d requested day(s) for %s %s.",
                iso3,
                variable,
                len(missing),
                len(group["dates"]),
                group["product_type"],
                group["year"],
            )
        returned = [day for day in group["dates"] if day not in missing]
        if not returned:
            raise RuntimeError(
                f"EWDS returned none of the {len(group['dates'])} requested days for {variable}."
            )
        if missing:
            keep = [np.datetime64(day) for day in returned]
            dataset = dataset.sel(time=keep)
        dataset.attrs["dataset"] = DATASET_ID
        dataset.attrs["system_version"] = self.system_version
        dataset.attrs["country_iso3"] = iso3
        dataset.attrs["spatial_resolution_degrees"] = (
            round(float(step), 6) if step is not None else NATIVE_RESOLUTION_DEGREES.get(self.system_version)
        )
        return dataset

    def _fetch_auxiliary(self, name: str, area):
        spec = AUXILIARY[name]
        self.logger.info("Downloading GloFAS %s for %s.", name, self.system_version)
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "static.nc"
            self._download_to_path(spec["url"], dest)
            dataset = _orient(_rename_static(_open_download(dest)))
        subset = _subset_grid(dataset, area).load()
        _reject_irregular(subset, self.system_version)
        subset.attrs["dataset"] = DATASET_ID
        subset.attrs["system_version"] = self.system_version
        subset.attrs["variable"] = name
        return subset

    def _retrieve(self, request: dict, dest: Path) -> None:
        """Submit one EWDS job and write its file at ``dest``."""
        session = requests.Session()
        session.headers["PRIVATE-TOKEN"] = self.api_key
        root = self.api_url
        response = _request(
            session,
            "POST",
            f"{root}/retrieve/v1/processes/{DATASET_ID}/execution",
            json={"inputs": request},
            timeout=120,
        )
        if response.status_code >= 400:
            raise RuntimeError(_provider_message(response, self.api_key))
        job = response.json()
        job_id = job["jobID"]
        status = job.get("status")
        deadline = time.monotonic() + self.request_timeout_seconds
        pause = 2.0
        while status not in JOB_DONE | JOB_FAILED:
            if time.monotonic() > deadline:
                raise TimeoutError(f"EWDS job {job_id} is still {status}.")
            time.sleep(pause)
            pause = min(pause * 1.5, 30.0)
            status_response = _request(session, "GET", f"{root}/retrieve/v1/jobs/{job_id}", timeout=60)
            if status_response.status_code >= 400:
                raise RuntimeError(_provider_message(status_response, self.api_key))
            job = status_response.json()
            status = job.get("status")
        if status not in JOB_DONE:
            message = job.get("message") or ""
            raise RuntimeError(f"EWDS job {job_id} {status}. {message}".strip())
        results = _request(session, "GET", f"{root}/retrieve/v1/jobs/{job_id}/results", timeout=60)
        if results.status_code >= 400:
            raise RuntimeError(_provider_message(results, self.api_key))
        href = _asset_href(results.json())
        if not href:
            raise RuntimeError(f"EWDS job {job_id} finished without a download URL.")
        self._download_to_path(href, dest, session=session)

    def _download_to_path(self, url: str, dest: Path, session=None) -> None:
        session = session or requests.Session()
        delays = (1, 2, 4)
        last_error = None
        for delay in (*delays, None):
            try:
                response = _request(session, "GET", url, timeout=300)
                if response.status_code >= 400:
                    raise requests.HTTPError(f"GET returned {response.status_code}.")
                dest.write_bytes(response.content)
                return
            except requests.RequestException as exc:
                last_error = exc
                if delay is None:
                    raise
                time.sleep(delay)
        if last_error:
            raise last_error

    def _require_key(self) -> None:
        if not self.api_key:
            raise RuntimeError(
                "EWDS_API_KEY is not set. Create a token at "
                f"{DATASET_PAGE} and accept the {LICENCE} licence, "
                "then set EWDS_API_KEY."
            )


def _series_variables(query: GlofasHistoricalResolveOptions) -> Tuple[str, ...]:
    if query.variable in AUXILIARY:
        return ()
    if query.variable is None:
        return VARIABLES
    return (query.variable,)


def _auxiliary_names(query: GlofasHistoricalResolveOptions) -> Tuple[str, ...]:
    if query.variable in AUXILIARY:
        return (query.variable,)
    if query.include_auxiliary:
        return tuple(AUXILIARY)
    return ()


def _assemble(groups: Dict[str, Dict[str, List]]):
    assembled = {}
    for variable, areas in groups.items():
        loaded_areas = {area_id: _concat_loaded(parts) for area_id, parts in areas.items()}
        assembled[variable] = (
            next(iter(loaded_areas.values())) if len(loaded_areas) == 1 else loaded_areas
        )
    if len(assembled) == 1:
        return next(iter(assembled.values()))
    return assembled


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

    return "_".join((part("n", north), part("w", west), part("s", south), part("e", east)))


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


def _coerce_date(value, name: str) -> Optional[date]:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        return date.fromisoformat(value[:10])
    raise TypeError(f"{name} must be a date or YYYY-MM-DD string.")


def _iso(value: Optional[date]) -> Optional[str]:
    if value is None:
        return None
    return value.isoformat()
