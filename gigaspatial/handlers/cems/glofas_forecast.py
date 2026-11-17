"""
GloFAS operational forecasts at country level (CEMS Early Warning Data Store).

Publisher: Copernicus Emergency Management Service, ECMWF.
Dataset: ``cems-glofas-forecast``.
Licence: CEMS-FLOODS. The EWDS account must accept that licence before a
forecast download will run. The static auxiliary maps are separate NetCDF
files on the CEMS auxiliary-data page and do not use the EWDS key.
Landing page: https://ewds.climate.copernicus.eu/datasets/cems-glofas-forecast
Auxiliary page: https://confluence.ecmwf.int/spaces/CEMS/pages/242067380/Auxiliary+Data

The country's administrative boundary, including islands and other parts,
selects the native grid cells. The request asks EWDS for a sub-region and
does not set a target grid. Values are not resampled, interpolated, or
reprojected. Variable names, coordinates, ensemble members, forecast steps,
and units are kept as the file provides them.

A forecast run is one variable, one product (control or perturbed ensemble),
one hydrological model, and one reference date. Lead times and the dates
each variable actually has are read from the catalogue constraints. They are
not assumed to match across variables. A stored run is kept when it already
contains the catalogue lead times. The latest reference date, and any run
that is still short of those lead times, is requested again when the
catalogue update time changes. A repeated run with nothing new does not
download again.

The v4.0 auxiliary maps (0.05 degree, EPSG:4326) apply to operational
forecasts from 2023-07-26, including GloFAS v4.1, which reuses those files.
They are stored apart from the forecast runs and can be loaded on their own.
They are not paired with ``version_2_1``, ``version_3_1``, or operational
runs before 2023-07-26, which are the 0.1 degree grids.

On 2026-11-16 the public constraints showed:

- ``operational`` / ``htessel_lisflood`` / river discharge: 2019-11-05 to 2021-05-25
- ``operational`` / ``lisflood`` / river discharge: 2021-05-26 onward
- snow depth water equivalent and soil wetness index: 2024-02-28 onward
- runoff water equivalent: 2024-02-29 onward
- ``version_2_1``: river discharge, 2021-05-26 to 2022-09-13, 0.1 degree
- ``version_3_1``: river discharge, 2023-07-26 to 2024-09-25, documented as 0.1 degree
- lead times 24 through 720 hours, step 24, for every published run

Those dates are checked against the constraints at runtime.
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


DATASET_ID = "cems-glofas-forecast"
DEFAULT_API_URL = "https://ewds.climate.copernicus.eu/api"
DATASET_PAGE = "https://ewds.climate.copernicus.eu/datasets/cems-glofas-forecast"
AUXILIARY_PAGE = "https://confluence.ecmwf.int/spaces/CEMS/pages/242067380/Auxiliary+Data"
LICENCE = "CEMS-FLOODS"
ATTACHMENT_ROOT = "https://confluence.ecmwf.int/download/attachments/242067380/"
SYSTEM_VERSIONS = ("operational", "version_2_1", "version_3_1")
HYDROLOGICAL_MODELS = ("lisflood", "htessel_lisflood")
PRODUCT_TYPES = ("control_forecast", "ensemble_perturbed_forecasts")
DATA_FORMATS = ("netcdf", "grib2")
# GloFAS v4.0 was released on this date at 0.05 degree. v4.1 (2024-02-28)
# keeps that grid and these auxiliary files.
V4_GRID_FROM = date(2023, 7, 26)
V4_RESOLUTION_DEGREES = 0.05
LEGACY_RESOLUTION_DEGREES = 0.1
AUXILIARY_VERSION = "4.0"
AREA_MARGIN_DEGREES = 0.051
CLUSTER_GAP_DEGREES = 2.0
LON_MIN, LON_MAX = -180.0, 180.0
LAT_MIN, LAT_MAX = -60.0, 90.0
TRANSIENT_STATUS = {429, 500, 502, 503, 504}
JOB_DONE = {"successful"}
JOB_FAILED = {"failed", "rejected", "dismissed"}
ENSEMBLE_DIMS = ("number", "realization", "ensemble", "ensemble_member")
STEP_DIMS = ("step", "leadtime", "lead_time", "forecast_period")
REFERENCE_DIMS = ("forecast_reference_time", "time")
LAT_NAMES = ("latitude", "lat")
LON_NAMES = ("longitude", "lon")

# EWDS form names confirmed from the catalogue on 2026-11-16.
# GRIB short names are not the request names.
VARIABLES: Tuple[str, ...] = (
    "river_discharge_in_the_last_24_hours",
    "runoff_water_equivalent",
    "snow_depth_water_equivalent",
    "soil_wetness_index",
)
VARIABLE_LABELS = {
    "river_discharge_in_the_last_24_hours": "River discharge in the last 24 hours",
    "runoff_water_equivalent": "Runoff water equivalent (surface plus subsurface)",
    "snow_depth_water_equivalent": "Snow depth water equivalent",
    "soil_wetness_index": "Soil wetness index (root zone)",
}
DOCUMENTED_UNITS = {
    "river_discharge_in_the_last_24_hours": "m3 s-1",
    "runoff_water_equivalent": "kg m-2",
    "snow_depth_water_equivalent": "kg m-2",
    "soil_wetness_index": "1",
}
GRIB_SHORT_NAMES = {
    "river_discharge_in_the_last_24_hours": "dis24",
    "runoff_water_equivalent": "rowe",
    "snow_depth_water_equivalent": "sd",
    "soil_wetness_index": "swir",
}
RETURN_PERIODS = ("1.5", "2.0", "5.0", "10.0", "20.0", "50.0", "100.0", "200.0", "500.0")


def _attachment(filename: str) -> str:
    return ATTACHMENT_ROOT + filename


def _auxiliary_specs() -> Dict[str, dict]:
    specs = {
        "upstream_area": {
            "label": "Upstream area",
            "filename": "uparea_glofas_v4_0.nc",
            "documented_units": "m2",
            "categorical": False,
            "return_period_years": None,
            "distribution_parameter": None,
        },
        "elevation": {
            "label": "Elevation",
            "filename": "elevation_glofas_v4_0.nc",
            "documented_units": "m",
            "categorical": False,
            "return_period_years": None,
            "distribution_parameter": None,
        },
        "ldd": {
            "label": "Local drain direction",
            "filename": "ldd_glofas_v4_0.nc",
            "documented_units": None,
            "categorical": True,
            "return_period_years": None,
            "distribution_parameter": None,
        },
    }
    for period in RETURN_PERIODS:
        specs[f"flood_threshold_rl_{period}"] = {
            "label": f"Flood threshold, {period}-year return period",
            "filename": f"flood_threshold_glofas_v4_rl_{period}.nc",
            "documented_units": None,
            "categorical": False,
            "return_period_years": float(period),
            "distribution_parameter": None,
        }
    for parameter in ("mu", "sigma"):
        specs[f"flood_threshold_{parameter}"] = {
            "label": f"Flood threshold distribution parameter {parameter}",
            "filename": f"flood_threshold_glofas_v4_{parameter}.nc",
            "documented_units": None,
            "categorical": False,
            "return_period_years": None,
            "distribution_parameter": parameter,
        }
    for spec in specs.values():
        spec["url"] = _attachment(spec["filename"])
        spec["version"] = AUXILIARY_VERSION
        spec["applies_from"] = V4_GRID_FROM.isoformat()
    return specs


AUXILIARY = _auxiliary_specs()


@dataclasses.dataclass(frozen=True)
class GlofasForecastResolveOptions(ResolveOptions):
    """
    Resolve options for one GloFAS forecast load.

    Attributes:
        variable: One EWDS variable, or one auxiliary name such as ``ldd``
            or ``flood_threshold_rl_10.0``. Omit it to load every forecast
            variable published for the selected system version.
        product_type: ``control_forecast`` or ``ensemble_perturbed_forecasts``.
            Omit it to load both.
        hydrological_model: ``lisflood`` or ``htessel_lisflood``. Omit it to
            follow the model the catalogue lists for each reference date.
        include_auxiliary: Also store the v4.0 auxiliary maps when they apply
            to this system version. Off by default. A named auxiliary variable
            loads that map on its own.
        reference_start: First forecast reference date to sync, inclusive.
        reference_end: Last forecast reference date to sync, inclusive.
            Omit both to sync every run the catalogue publishes. That archive
            is large.
    """

    variable: Optional[str] = None
    product_type: Optional[str] = None
    hydrological_model: Optional[str] = None
    include_auxiliary: bool = False
    reference_start: Optional[date] = None
    reference_end: Optional[date] = None

    def __post_init__(self):
        object.__setattr__(self, "reference_start", _coerce_date(self.reference_start, "reference_start"))
        object.__setattr__(self, "reference_end", _coerce_date(self.reference_end, "reference_end"))
        known = set(VARIABLES) | set(AUXILIARY)
        if self.variable is not None and self.variable not in known:
            raise ValueError(
                f"variable must be one of {tuple(sorted(known))}, got {self.variable!r}. "
                "Use the EWDS name, for example "
                "'river_discharge_in_the_last_24_hours', not 'dis24'."
            )
        if self.product_type is not None and self.product_type not in PRODUCT_TYPES:
            raise ValueError(
                f"product_type must be one of {PRODUCT_TYPES}, got {self.product_type!r}."
            )
        if self.hydrological_model is not None and self.hydrological_model not in HYDROLOGICAL_MODELS:
            raise ValueError(
                f"hydrological_model must be one of {HYDROLOGICAL_MODELS}, "
                f"got {self.hydrological_model!r}."
            )
        if (
            self.reference_start is not None
            and self.reference_end is not None
            and self.reference_start > self.reference_end
        ):
            raise ValueError("reference_start is after reference_end.")


class ForecastAvailability:
    """Reference dates and lead times the catalogue publishes for one variable."""

    def __init__(self, runs: Dict[Tuple[str, str, date], Tuple[int, ...]]):
        self.runs = runs

    def days(self, model: str, product: str) -> Dict[date, Tuple[int, ...]]:
        selected = {
            day: leads
            for (row_model, row_product, day), leads in self.runs.items()
            if row_model == model and row_product == product
        }
        return dict(sorted(selected.items()))

    def groups(self) -> List[Tuple[str, str]]:
        found = {(model, product) for model, product, _day in self.runs}
        return sorted(found)


class ForecastCatalogue:
    """Constraints and update time for ``cems-glofas-forecast``."""

    def __init__(self, collection: dict, constraints: list):
        self.updated = collection.get("updated")
        self.doi = collection.get("sci:doi")
        interval = collection["extent"]["temporal"]["interval"][0]
        self.temporal_start = _parse_date(interval[0])
        self.temporal_end = _parse_date(interval[1])
        self._constraints = constraints
        self._cache: Dict[Tuple[str, str], ForecastAvailability] = {}

    def availability(self, system_version: str, variable: str) -> ForecastAvailability:
        key = (system_version, variable)
        if key not in self._cache:
            self._cache[key] = _availability(self._constraints, system_version, variable)
        return self._cache[key]


def fetch_catalogue(api_url: str = DEFAULT_API_URL, session=None) -> ForecastCatalogue:
    """Read the public catalogue. This call does not need an API key."""
    session = session or requests.Session()
    root = api_url.rstrip("/")
    collection_url = f"{root}/catalogue/v1/collections/{DATASET_ID}"
    collection = _request(session, "GET", collection_url, timeout=60).json()
    constraints_url = next(
        link["href"] for link in collection["links"] if link.get("rel") == "constraints"
    )
    constraints = _request(session, "GET", constraints_url, timeout=120).json()
    return ForecastCatalogue(collection, constraints)


def runs_to_fetch(
    stored: Dict[str, dict],
    expected: Dict[date, Tuple[int, ...]],
    *,
    catalogue_updated: Optional[str],
    latest: Optional[date],
) -> List[date]:
    """
    Reference dates that still need a request.

    A missing run is requested. A run already fetched against this catalogue
    update is kept, including a run that was published with fewer lead times
    than the constraint lists. When the catalogue update time changes, a short
    run is requested again, and so is the latest reference date. Older runs
    that already have every listed lead time stay stored.
    """
    pending = []
    for day, leads in sorted(expected.items()):
        current = stored.get(day.isoformat())
        if current is None:
            pending.append(day)
            continue
        have = {int(hour) for hour in current.get("lead_hours") or []}
        complete = set(leads).issubset(have)
        stale = current.get("catalogue_updated") != catalogue_updated
        open_tail = bool(current.get("open")) or day == latest
        if stale and (not complete or open_tail):
            pending.append(day)
    return pending


def expected_resolution(system_version: str, reference: date) -> float:
    """
    Native grid step, in degrees, for one published forecast run.

    Operational runs on and after 2023-07-26 are the GloFAS v4 0.05 degree
    grid. Operational runs before that date, ``version_2_1``, and
    ``version_3_1`` are the 0.1 degree grids documented for v3.1 and older.
    """
    if system_version == "operational" and reference >= V4_GRID_FROM:
        return V4_RESOLUTION_DEGREES
    if system_version in SYSTEM_VERSIONS:
        return LEGACY_RESOLUTION_DEGREES
    raise ValueError(f"No published grid step for system version {system_version!r}.")


def v4_auxiliary_applies(system_version: str, reference: Optional[date] = None) -> bool:
    """
    True when the v4.0 auxiliary maps are the grid for this forecast.

    The maps apply to operational runs on and after 2023-07-26. A missing
    reference date means the operational system in general, not a legacy
    version.
    """
    if system_version != "operational":
        return False
    if reference is not None and reference < V4_GRID_FROM:
        return False
    return True


def extraction_areas(
    geometry: BaseGeometry, margin: float = AREA_MARGIN_DEGREES
) -> List[Tuple[float, float, float, float]]:
    """
    Rectangles that cover every part of a country, as ``(north, west, south, east)``.

    Nearby parts share one rectangle. Islands and the two sides of the
    antimeridian stay separate. Each rectangle is expanded by one grid cell.
    EWDS returns cells inside the rectangle, not cells that only touch its edge.
    """
    pieces = []
    west_hemisphere = shapely_box(-180, -90, 0, 90)
    east_hemisphere = shapely_box(0, -90, 180, 90)
    for part in _polygons(geometry):
        minx, _miny, maxx, _maxy = part.bounds
        if maxx - minx > 180:
            for hemisphere in (west_hemisphere, east_hemisphere):
                pieces.extend(_polygons(part.intersection(hemisphere)))
        else:
            pieces.append(part)
    if not pieces:
        raise ValueError("The country geometry is empty.")

    areas = []
    for minx, miny, maxx, maxy in _cluster_bounds([piece.bounds for piece in pieces]):
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


def subset_native(dataset, area: Tuple[float, float, float, float]):
    """
    Select native cells inside ``area``.

    Cell coordinates and dtypes are unchanged. Nothing is interpolated.
    """
    lat_name, lon_name = _lat_lon_names(dataset)
    north, west, south, east = area
    latitude = np.asarray(dataset[lat_name].values, dtype=float)
    longitude = np.asarray(dataset[lon_name].values, dtype=float)
    lat_index = np.where((latitude <= north + 1e-6) & (latitude >= south - 1e-6))[0]
    lon_index = np.where((longitude <= east + 1e-6) & (longitude >= west - 1e-6))[0]
    if lat_index.size == 0 or lon_index.size == 0:
        raise ValueError("The grid does not cover the country area.")
    subset = dataset.isel({lat_name: lat_index, lon_name: lon_index})
    for name in dataset.data_vars:
        if subset[name].dtype != dataset[name].dtype:
            raise ValueError(
                f"{name} changed from {dataset[name].dtype} to {subset[name].dtype} "
                "while selecting cells."
            )
    return subset


def decode_payload(raw: bytes):
    """Open a NetCDF-4 or GRIB2 payload, including a zip of either."""
    if raw[:2] == b"PK":
        return _decode_zip(raw)
    if raw[:4] == b"GRIB":
        return open_grib_bytes(raw)
    if raw[:3] == b"CDF" or raw[:4] == b"\x89HDF":
        return _open_netcdf_bytes(raw)
    raise ValueError("Downloaded file is not NetCDF-4 or GRIB2.")


def open_grib_bytes(raw: bytes):
    """Open a GRIB2 message with cfgrib and keep every hypercube."""
    try:
        import cfgrib
    except ImportError as exc:
        raise ImportError(
            "Reading GloFAS GRIB2 needs cfgrib. "
            "Install it with pip install 'giga-spatial[glofas]', "
            "or set data_format='netcdf'."
        ) from exc
    handle = tempfile.NamedTemporaryFile(suffix=".grib", delete=False)
    handle.write(raw)
    handle.close()
    path = handle.name
    opened = []
    try:
        opened = list(cfgrib.open_datasets(path))
        if not opened:
            raise ValueError("GRIB2 file has no datasets.")
        loaded = [dataset.load() for dataset in opened]
    finally:
        for dataset in opened:
            dataset.close()
        os.unlink(path)
    return _combine_datasets(loaded)


@dataclass(config=ConfigDict(arbitrary_types_allowed=True, extra="forbid"))
class GlofasForecastConfig(BaseHandlerConfig):
    """
    Configuration for country-level GloFAS forecast and auxiliary files.

    Attributes:
        base_path: Forecast files, under ``bronze/cems/glofas/forecast``.
        auxiliary_base_path: Static maps, under ``bronze/cems/glofas/auxiliary``.
        system_version: ``operational`` is the current system. ``version_2_1``
            and ``version_3_1`` are the legacy 0.1 degree archives.
        data_format: ``netcdf`` or ``grib2``. NetCDF-4 is marked experimental
            on the EWDS form. The handler refuses a file that drops the
            ensemble dimension from a perturbed forecast.
        api_url: EWDS API root.
        api_key: Personal access token. Defaults to ``EWDS_API_KEY``.
        request_timeout_seconds: How long to wait for one EWDS job.
    """

    options_type = GlofasForecastResolveOptions
    units_match_country = True

    base_path: Path = Field(
        default_factory=lambda: global_config.get_path("cems", "bronze") / "glofas" / "forecast"
    )
    auxiliary_base_path: Path = Field(
        default_factory=lambda: global_config.get_path("cems", "bronze") / "glofas" / "auxiliary"
    )
    system_version: str = "operational"
    data_format: str = "netcdf"
    api_url: str = Field(default_factory=lambda: global_config.EWDS_API_URL or DEFAULT_API_URL)
    api_key: str = Field(default_factory=lambda: global_config.EWDS_API_KEY, repr=False)
    request_timeout_seconds: int = 3600

    def __post_init__(self):
        super().__post_init__()
        if self.system_version not in SYSTEM_VERSIONS:
            raise ValueError(
                f"system_version must be one of {SYSTEM_VERSIONS}, got {self.system_version!r}."
            )
        if self.data_format not in DATA_FORMATS:
            raise ValueError(f"data_format must be one of {DATA_FORMATS}, got {self.data_format!r}.")
        self.api_url = self.api_url.rstrip("/")
        self._catalogue = None
        self._catalogue_lock = threading.Lock()
        self._fingerprints: Dict[str, dict] = {}

    def catalogue(self) -> ForecastCatalogue:
        """The catalogue for this run. Fetched once."""
        if self._catalogue is not None:
            return self._catalogue
        with self._catalogue_lock:
            if self._catalogue is None:
                self._catalogue = fetch_catalogue(self.api_url)
        return self._catalogue

    def get_search_key(self, source, options: GlofasForecastResolveOptions) -> str:
        """Return the ISO3 code. The files are stored one country at a time."""
        if classify_source(source) is not SourceKind.COUNTRY:
            raise ValueError(
                "GloFAS forecast is stored by country. "
                "Pass a country name or ISO code, for example 'RWA'."
            )
        try:
            return pycountry.countries.lookup(source).alpha_3
        except LookupError as exc:
            raise ValueError(f"Unknown country: {source!r}.") from exc

    def get_relevant_data_units_by_key(
        self, search_key: str, options: GlofasForecastResolveOptions
    ) -> List[dict]:
        """One unit per forecast series, and one per auxiliary map, for each rectangle."""
        geometry = source_geometry(search_key, crs=self.crs)
        areas = extraction_areas(geometry)
        country = pycountry.countries.get(alpha_3=search_key)
        country_name = country.name if country is not None else search_key
        units = []
        if options.variable not in AUXILIARY:
            units.extend(self._forecast_units(search_key, country_name, areas, options))
        units.extend(self._auxiliary_units(search_key, country_name, areas, options))
        if not units:
            raise ValueError(f"No GloFAS forecast variables were selected for {search_key}.")
        return units

    def get_data_unit_path(self, unit: dict, options=None) -> Path:
        """Metadata file for one series or one auxiliary map."""
        return self._unit_dir(unit) / "metadata.json"

    def get_stored_paths(self, unit: dict, options=None) -> List[Path]:
        """Metadata plus the NetCDF files for this unit."""
        directory = self._unit_dir(unit)
        paths = [directory / "metadata.json"]
        if unit["kind"] == "auxiliary":
            paths.append(directory / "static.nc")
            return paths
        start = _coerce_date(unit.get("reference_start"), "reference_start")
        end = _coerce_date(unit.get("reference_end"), "reference_end")
        names = set()
        info = _read_json(self.data_store, directory / "metadata.json") or {}
        for key in sorted(info.get("runs") or {}):
            day = date.fromisoformat(key)
            if _in_window(day, start, end):
                names.add(f"{key}.nc")
        for path in _nc_files(self.data_store, directory):
            if path.name == "static.nc":
                continue
            try:
                day = date.fromisoformat(path.stem)
            except ValueError:
                continue
            if _in_window(day, start, end):
                names.add(path.name)
        for name in sorted(names):
            candidate = directory / name
            if self.data_store.file_exists(str(candidate)):
                paths.append(candidate)
        return paths

    def is_unit_available(self, unit: dict, paths) -> bool:
        """True when the stored files already match the catalogue."""
        if unit["kind"] == "auxiliary":
            return self._auxiliary_current(unit)
        stored, meta = self._stored_runs(unit)
        expected, latest = self._expected_runs(unit)
        pending = runs_to_fetch(
            stored,
            expected,
            catalogue_updated=self.catalogue().updated,
            latest=latest,
        )
        if pending or not stored:
            return False
        self.logger.info(
            "%s %s %s is already up to date through %s. Last download %s.",
            unit["country_iso3"],
            unit["variable"],
            unit["product_type"],
            meta.get("source_latest_reference"),
            meta.get("last_successful_download_utc"),
        )
        return True

    def _forecast_units(self, iso3: str, country_name: str, areas, options) -> List[dict]:
        units = []
        for variable in self._series_variables(options):
            try:
                availability = self.catalogue().availability(self.system_version, variable)
            except ValueError as exc:
                if options.variable == variable:
                    raise
                self.logger.warning("%s", exc)
                continue
            for model, product in availability.groups():
                if options.hydrological_model not in (None, model):
                    continue
                if options.product_type not in (None, product):
                    continue
                expected, _latest = _window(
                    availability.days(model, product),
                    options.reference_start,
                    options.reference_end,
                    availability,
                    model,
                    product,
                )
                if not expected:
                    continue
                for area in areas:
                    units.append(
                        self._forecast_unit(
                            iso3, country_name, variable, model, product, area, options
                        )
                    )
        return units

    def _auxiliary_units(self, iso3: str, country_name: str, areas, options) -> List[dict]:
        names = self._auxiliary_names(options)
        if not names:
            return []
        if not v4_auxiliary_applies(self.system_version, options.reference_end if options.variable not in AUXILIARY else None):
            message = (
                f"GloFAS v{AUXILIARY_VERSION} auxiliary maps apply to operational forecasts "
                f"from {V4_GRID_FROM.isoformat()}. They are not paired with "
                f"{self.system_version}"
                + (
                    f" through {options.reference_end.isoformat()}"
                    if options.reference_end is not None and self.system_version == "operational"
                    else ""
                )
                + "."
            )
            if options.variable in AUXILIARY:
                raise ValueError(message)
            self.logger.warning("%s", message)
            return []
        units = []
        for name in names:
            for area in areas:
                units.append(self._auxiliary_unit(iso3, country_name, name, area))
        return units

    def _series_variables(self, options: GlofasForecastResolveOptions) -> Tuple[str, ...]:
        if options.variable in AUXILIARY:
            return ()
        if options.variable is None:
            return VARIABLES
        return (options.variable,)

    def _auxiliary_names(self, options: GlofasForecastResolveOptions) -> Tuple[str, ...]:
        if options.variable in AUXILIARY:
            return (options.variable,)
        if options.include_auxiliary and options.variable not in AUXILIARY:
            return tuple(AUXILIARY)
        return ()

    def _forecast_unit(self, iso3, country_name, variable, model, product, area, options) -> dict:
        north, west, south, east = area
        return {
            "kind": "forecast",
            "country_iso3": iso3,
            "country_name": country_name,
            "system_version": self.system_version,
            "hydrological_model": model,
            "product_type": product,
            "variable": variable,
            "area": area,
            "area_id": _area_id(north, west, south, east),
            "reference_start": _iso(options.reference_start),
            "reference_end": _iso(options.reference_end),
        }

    def _auxiliary_unit(self, iso3, country_name, name, area) -> dict:
        north, west, south, east = area
        return {
            "kind": "auxiliary",
            "country_iso3": iso3,
            "country_name": country_name,
            "system_version": self.system_version,
            "variable": name,
            "area": area,
            "area_id": _area_id(north, west, south, east),
        }

    def _unit_dir(self, unit: dict) -> Path:
        if unit["kind"] == "auxiliary":
            return (
                self.auxiliary_base_path
                / unit["country_iso3"]
                / unit["variable"]
                / unit["area_id"]
            )
        return (
            self.base_path
            / unit["country_iso3"]
            / unit["system_version"]
            / unit["hydrological_model"]
            / unit["variable"]
            / unit["product_type"]
            / unit["area_id"]
        )

    def _run_path(self, unit: dict, day: date) -> Path:
        return self._unit_dir(unit) / f"{day.isoformat()}.nc"

    def _source_auxiliary_path(self, name: str) -> Path:
        return (
            self.auxiliary_base_path
            / "_source"
            / f"v{AUXILIARY_VERSION.replace('.', '_')}"
            / AUXILIARY[name]["filename"]
        )

    def _expected_runs(self, unit: dict) -> Tuple[Dict[date, Tuple[int, ...]], Optional[date]]:
        availability = self.catalogue().availability(unit["system_version"], unit["variable"])
        return _window(
            availability.days(unit["hydrological_model"], unit["product_type"]),
            _coerce_date(unit.get("reference_start"), "reference_start"),
            _coerce_date(unit.get("reference_end"), "reference_end"),
            availability,
            unit["hydrological_model"],
            unit["product_type"],
        )

    def _stored_runs(self, unit: dict) -> Tuple[Dict[str, dict], dict]:
        meta = _read_json(self.data_store, self._unit_dir(unit) / "metadata.json") or {}
        stored = {}
        for key, info in (meta.get("runs") or {}).items():
            if self.data_store.file_exists(str(self._run_path(unit, date.fromisoformat(key)))):
                stored[key] = info
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
                key = dataset.attrs.get("gigaspatial_reference_date") or path.stem
                hours = json.loads(dataset.attrs.get("gigaspatial_lead_hours") or "[]")
                if hours:
                    recovered[str(key)] = {
                        "lead_hours": hours,
                        "open": dataset.attrs.get("gigaspatial_open") == "true",
                        "catalogue_updated": dataset.attrs.get("gigaspatial_catalogue_updated"),
                    }
            finally:
                dataset.close()
        return recovered, meta

    def _auxiliary_current(self, unit: dict) -> bool:
        target = self._unit_dir(unit) / "static.nc"
        if not self.data_store.file_exists(str(target)):
            return False
        meta = _read_json(self.data_store, self._unit_dir(unit) / "metadata.json") or {}
        try:
            remote = self.auxiliary_fingerprint(unit["variable"])
        except requests.RequestException as exc:
            self.logger.warning(
                "Could not check %s at the source (%s). The stored file is kept.",
                unit["variable"],
                exc.__class__.__name__,
            )
            return True
        return meta.get("source_fingerprint") == _fingerprint_token(remote)

    def auxiliary_fingerprint(self, name: str) -> dict:
        """HEAD the published file once per run and return its identity."""
        if name in self._fingerprints:
            return self._fingerprints[name]
        spec = AUXILIARY[name]
        response = _request(requests.Session(), "HEAD", spec["url"], timeout=60, allow_redirects=True)
        if response.status_code >= 400:
            raise requests.HTTPError(f"HEAD {spec['filename']} returned {response.status_code}.")
        content_type = response.headers.get("Content-Type", "")
        if "html" in content_type.lower():
            raise requests.HTTPError(f"{spec['filename']} returned HTML rather than NetCDF.")
        fingerprint = {
            "content_length": response.headers.get("Content-Length"),
            "last_modified": response.headers.get("Last-Modified"),
            "content_type": content_type,
            "url": spec["url"],
        }
        self._fingerprints[name] = fingerprint
        return fingerprint


class GlofasForecastDownloader(BaseHandlerDownloader):
    """Downloads GloFAS forecast runs and auxiliary maps that are not stored yet."""

    options_type = DownloadOptions
    config_type: ClassVar[type] = GlofasForecastConfig

    def download_data_unit(self, unit: dict, options: DownloadOptions) -> List[Path]:
        """
        Sync one forecast series, or one auxiliary map, for one country rectangle.

        Raises:
            RuntimeError: The EWDS request failed, or ``EWDS_API_KEY`` is missing.
            ValueError: The file drops ensemble members or lead times, or it is
                not on the native grid.
        """
        if unit["kind"] == "auxiliary":
            self._download_auxiliary(unit, options)
        else:
            self._sync_series(unit, options)
        return self.config.get_stored_paths(unit)

    def _sync_series(self, unit: dict, options: DownloadOptions) -> None:
        catalogue = self.config.catalogue()
        expected, latest = self.config._expected_runs(unit)
        stored, meta = self.config._stored_runs(unit)
        pending = runs_to_fetch(
            stored,
            expected,
            catalogue_updated=catalogue.updated,
            latest=latest,
        )
        if not pending:
            self._write_checked(unit, meta, catalogue, expected, latest, stored, downloaded=False)
            self.logger.info(
                "%s %s %s is already up to date through %s. Last download %s.",
                unit["country_iso3"],
                unit["variable"],
                unit["product_type"],
                _iso(latest),
                meta.get("last_successful_download_utc"),
            )
            return
        self._require_key()
        self.logger.info(
            "%s %s %s: retrieving %d reference date(s). Source latest %s.",
            unit["country_iso3"],
            unit["variable"],
            unit["product_type"],
            len(pending),
            _iso(latest),
        )
        for day in pending:
            self._fetch_day(unit, day, expected[day], latest, catalogue, stored, meta, options)
            self._write_progress(unit, meta, catalogue, expected, latest, stored)
        self._write_checked(unit, meta, catalogue, expected, latest, stored, downloaded=True)

    def _fetch_day(self, unit, day, leads, latest, catalogue, stored, meta, options) -> None:
        request = {
            "system_version": [unit["system_version"]],
            "hydrological_model": [unit["hydrological_model"]],
            "product_type": [unit["product_type"]],
            "variable": [unit["variable"]],
            "year": [f"{day.year:04d}"],
            "month": [f"{day.month:02d}"],
            "day": [f"{day.day:02d}"],
            "leadtime_hour": [str(hour) for hour in leads],
            "area": list(unit["area"]),
            "data_format": self.config.data_format,
            "download_format": "zip",
        }
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "download.bin"
            self._retrieve(request, dest, options)
            dataset = decode_payload(dest.read_bytes())
            try:
                present = _lead_hours(dataset)
                _require_ensemble(dataset, unit["product_type"])
                step = _validate_grid(
                    dataset,
                    system_version=unit["system_version"],
                    reference=day,
                    area=unit["area"],
                )
                previous = stored.get(day.isoformat())
                previous_hours = {int(hour) for hour in (previous or {}).get("lead_hours") or []}
                if previous_hours and not previous_hours.issubset(set(present)):
                    self.logger.warning(
                        "%s %s %s: EWDS returned lead times %s, fewer than the stored %s. "
                        "The stored file was kept.",
                        unit["country_iso3"],
                        unit["variable"],
                        day.isoformat(),
                        present,
                        sorted(previous_hours),
                    )
                    previous["catalogue_updated"] = catalogue.updated
                    previous["open"] = day == latest
                    return
                missing = sorted(set(leads) - set(present))
                if missing:
                    self.logger.warning(
                        "%s %s %s: EWDS omitted lead times %s. They stay pending until the catalogue changes.",
                        unit["country_iso3"],
                        unit["variable"],
                        day.isoformat(),
                        missing,
                    )
                _stamp_run(dataset, day, present, day == latest, catalogue.updated)
                if step is not None:
                    meta["spatial_resolution_degrees"] = round(float(step), 6)
                meta["units"] = _file_units(dataset) or DOCUMENTED_UNITS.get(unit["variable"], "")
                _commit_dataset(self.data_store, self.config._run_path(unit, day), dataset)
                stored[day.isoformat()] = {
                    "lead_hours": present,
                    "open": day == latest,
                    "catalogue_updated": catalogue.updated,
                }
            finally:
                dataset.close()

    def _download_auxiliary(self, unit: dict, options: DownloadOptions) -> None:
        spec = AUXILIARY[unit["variable"]]
        remote = self.config.auxiliary_fingerprint(unit["variable"])
        source = self.config._source_auxiliary_path(unit["variable"])
        token = _fingerprint_token(remote)
        stored = _read_json(self.data_store, source.with_suffix(".json")) or {}
        if not self.data_store.file_exists(str(source)) or stored.get("source_fingerprint") != token:
            self.logger.info(
                "Downloading GloFAS v%s %s (%s).",
                AUXILIARY_VERSION,
                unit["variable"],
                spec["filename"],
            )
            _stream_with_retry(
                spec["url"],
                source,
                self.data_store,
                show_progress=options.show_progress,
            )
            _write_json(
                self.data_store,
                source.with_suffix(".json"),
                {
                    "filename": spec["filename"],
                    "version": AUXILIARY_VERSION,
                    "source_fingerprint": token,
                    "source_url": spec["url"],
                    "applies_from": V4_GRID_FROM.isoformat(),
                },
            )
        dataset = _open_path(self.data_store, source)
        subset = None
        try:
            subset = subset_native(dataset, unit["area"]).load()
            _reject_irregular(subset, V4_RESOLUTION_DEGREES)
            if spec["categorical"]:
                _require_categorical(subset)
            target = self.config._unit_dir(unit) / "static.nc"
            _commit_dataset(self.data_store, target, subset)
            units = _file_units(subset) or spec["documented_units"]
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
                "dataset": "glofas-auxiliary",
                "variable": unit["variable"],
                "variable_label": spec["label"],
                "auxiliary_version": AUXILIARY_VERSION,
                "applies_from": V4_GRID_FROM.isoformat(),
                "applies_to": (
                    "Operational GloFAS forecasts from 2023-07-26 "
                    "(v4.0 and v4.1). Not version_2_1, version_3_1, "
                    "or earlier operational runs."
                ),
                "return_period_years": spec["return_period_years"],
                "distribution_parameter": spec["distribution_parameter"],
                "categorical": spec["categorical"],
                "units": units,
                "spatial_resolution_degrees": V4_RESOLUTION_DEGREES,
                "crs": "EPSG:4326",
                "temporal_resolution": "static",
                "source_filename": spec["filename"],
                "source_url": spec["url"],
                "source_page": AUXILIARY_PAGE,
                "source_fingerprint": token,
                "source_content_length": remote.get("content_length"),
                "source_last_modified": remote.get("last_modified"),
                "last_successful_download_utc": now,
                "last_successful_check_utc": now,
                "files": ["static.nc"],
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
            status_response = _request(session, "GET", f"{root}/retrieve/v1/jobs/{job_id}", timeout=60)
            if status_response.status_code >= 400:
                raise RuntimeError(_provider_message(status_response, self.config.api_key))
            job = status_response.json()
            status = job.get("status")
        if status not in JOB_DONE:
            message = job.get("message") or ""
            raise RuntimeError(f"EWDS job {job_id} {status}. {message}".strip())
        results = _request(session, "GET", f"{root}/retrieve/v1/jobs/{job_id}/results", timeout=60)
        if results.status_code >= 400:
            raise RuntimeError(_provider_message(results, self.config.api_key))
        href = _asset_href(results.json())
        if not href:
            raise RuntimeError(f"EWDS job {job_id} finished without a download URL.")
        _stream_with_retry(
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

    def _write_progress(self, unit, meta, catalogue, expected, latest, stored) -> None:
        meta = _base_metadata(unit, meta, catalogue, expected, latest, stored, self.config)
        _write_json(self.data_store, self.config.get_data_unit_path(unit), meta)

    def _write_checked(self, unit, meta, catalogue, expected, latest, stored, downloaded: bool) -> dict:
        now = _utc_now()
        meta = _base_metadata(unit, meta, catalogue, expected, latest, stored, self.config)
        meta["last_successful_check_utc"] = now
        meta["catalogue_updated"] = catalogue.updated
        if downloaded:
            meta["last_successful_download_utc"] = now
        elif "last_successful_download_utc" not in meta:
            meta["last_successful_download_utc"] = None
        _write_json(self.data_store, self.config.get_data_unit_path(unit), meta)
        return meta


class GlofasForecastReader(BaseHandlerReader):
    """Loads stored GloFAS forecast and auxiliary NetCDF files."""

    options_type = ReadOptions
    config_type: ClassVar[type] = GlofasForecastConfig

    def load_from_paths(self, source_data_path: List[str], options: ReadOptions):
        """
        Load stored files.

        One series returns a Dataset. Several variables, products, or areas
        return nested dicts, collapsing any level that has a single entry.
        Forecast runs are stacked on their reference-time dimension. Auxiliary
        maps stay separate from those runs.
        """
        groups: Dict[str, Dict[str, Dict[str, List]]] = {}
        for raw in source_data_path:
            path = Path(raw)
            if path.suffix != ".nc":
                continue
            dataset = _open_path(self.data_store, path)
            try:
                loaded = dataset.load()
            finally:
                dataset.close()
            if path.name == "static.nc":
                variable = path.parent.parent.name
                product = "static"
                area_id = path.parent.name
            else:
                area_id = path.parent.name
                product = path.parent.parent.name
                variable = path.parent.parent.parent.name
                loaded = _tag_reference(loaded, date.fromisoformat(path.stem))
            groups.setdefault(variable, {}).setdefault(product, {}).setdefault(area_id, []).append(loaded)
        if not groups:
            raise ValueError("No stored GloFAS forecast files were found.")
        assembled = {}
        for variable, products in groups.items():
            packed = {
                product: _collapse({area_id: _concat_runs(parts) for area_id, parts in areas.items()})
                for product, areas in products.items()
            }
            assembled[variable] = _collapse(packed)
        return _collapse(assembled)


class GlofasForecastHandler(BaseHandler):
    """
    Handler for GloFAS operational forecasts and v4.0 auxiliary maps.

    One call checks the catalogue and downloads forecast runs that are missing
    or that the latest catalogue update still revises::

        GlofasForecastHandler().load_data("RWA")
        GlofasForecastHandler().load_data(
            "RWA",
            variable="river_discharge_in_the_last_24_hours",
            product_type="control_forecast",
            reference_start="2026-11-15",
            reference_end="2026-11-15",
        )

    Auxiliary maps do not need a forecast download::

        GlofasForecastHandler().load_data("RWA", variable="ldd")
        GlofasForecastHandler().load_data("RWA", include_auxiliary=True)

    Set ``EWDS_API_KEY`` and accept the CEMS-FLOODS licence before the first
    forecast download. Omitting the reference dates syncs the whole published
    archive.
    """

    def create_config(self, data_store, logger, **kwargs) -> GlofasForecastConfig:
        kwargs.setdefault("n_workers", 1)
        return GlofasForecastConfig(data_store=data_store, logger=logger, **kwargs)

    def create_downloader(self, config, data_store, logger, **kwargs) -> GlofasForecastDownloader:
        return GlofasForecastDownloader(config=config, data_store=data_store, logger=logger)

    def create_reader(self, config, data_store, logger, **kwargs) -> GlofasForecastReader:
        return GlofasForecastReader(config=config, data_store=data_store, logger=logger)


def _availability(constraints: list, system_version: str, variable: str) -> ForecastAvailability:
    runs: Dict[Tuple[str, str, date], Tuple[int, ...]] = {}
    seen = False
    for row in constraints:
        if system_version not in (row.get("system_version") or []):
            continue
        if variable not in (row.get("variable") or []):
            continue
        seen = True
        leads = tuple(sorted({int(hour) for hour in row.get("leadtime_hour") or []}))
        if not leads:
            continue
        for model in row.get("hydrological_model") or []:
            for product in row.get("product_type") or []:
                for year in row.get("year") or []:
                    for month in row.get("month") or []:
                        for day_text in row.get("day") or []:
                            try:
                                stamp = date(int(year), int(month), int(day_text))
                            except ValueError:
                                continue
                            key = (model, product, stamp)
                            previous = runs.get(key)
                            if previous is None:
                                runs[key] = leads
                            elif previous != leads:
                                runs[key] = tuple(sorted(set(previous) | set(leads)))
    if not seen:
        raise ValueError(
            f"{variable} is not in the {system_version} constraints for {DATASET_ID}."
        )
    if not runs:
        raise ValueError(f"{variable} has no reference dates in the {system_version} constraints.")
    return ForecastAvailability(runs)


def _window(days, start, end, availability, model, product):
    selected = {day: leads for day, leads in days.items() if _in_window(day, start, end)}
    full = availability.days(model, product)
    latest = max(full) if full else None
    return selected, latest


def _in_window(day: date, start: Optional[date], end: Optional[date]) -> bool:
    if start is not None and day < start:
        return False
    if end is not None and day > end:
        return False
    return True


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


def _lead_hours(dataset) -> List[int]:
    for name in STEP_DIMS:
        if name in dataset.coords or name in dataset.dims or name in dataset.variables:
            values = np.asarray(dataset[name].values).reshape(-1)
            hours = [_to_hours(value) for value in values]
            if len(hours) != len(set(hours)):
                raise ValueError(f"Downloaded file repeats forecast lead times in {name!r}.")
            return sorted(hours)
    reference = _reference_values(dataset)
    if reference is not None and "valid_time" in dataset.coords:
        valid = np.asarray(dataset["valid_time"].values).reshape(-1)
        origin = np.asarray(reference).reshape(-1)[0]
        hours = sorted({_to_hours(value - origin) for value in valid})
        if hours:
            return hours
    raise ValueError(
        "Downloaded file has no forecast lead time. "
        f"Dimensions are {dict(dataset.sizes)}."
    )


def _to_hours(value) -> int:
    if isinstance(value, np.timedelta64):
        seconds = float(value / np.timedelta64(1, "s"))
        return int(round(seconds / 3600.0))
    try:
        stamp = pd.Timestamp(value)
        if isinstance(value, (pd.Timedelta, np.timedelta64)):
            return int(round(pd.Timedelta(value).total_seconds() / 3600.0))
        if not isinstance(value, (int, float, np.integer, np.floating)):
            if stamp is not pd.NaT:
                raise TypeError
    except (TypeError, ValueError):
        pass
    if isinstance(value, pd.Timedelta):
        return int(round(value.total_seconds() / 3600.0))
    number = float(value)
    if number > 720 and number % 3600 == 0:
        return int(number // 3600)
    return int(round(number))


def _reference_values(dataset):
    for name in REFERENCE_DIMS:
        if name in dataset.coords or name in dataset.variables:
            return dataset[name].values
    return None


def _require_ensemble(dataset, product: str) -> None:
    if product != "ensemble_perturbed_forecasts":
        return
    if any(name in dataset.dims and dataset.sizes[name] >= 1 for name in ENSEMBLE_DIMS):
        return
    raise ValueError(
        "Ensemble perturbed forecast has no ensemble dimension. "
        f"Dimensions are {dict(dataset.sizes)}. The file was not stored."
    )


def _validate_grid(dataset, *, system_version: str, reference: date, area) -> Optional[float]:
    lat_name, lon_name = _lat_lon_names(dataset)
    latitude = np.asarray(dataset[lat_name].values, dtype=float)
    longitude = np.asarray(dataset[lon_name].values, dtype=float)
    lat_step = _grid_step(latitude)
    lon_step = _grid_step(longitude)
    if lat_step is not None and lon_step is not None and abs(lat_step - lon_step) > 1e-3:
        raise ValueError("Latitude and longitude spacing differ.")
    step = lat_step if lat_step is not None else lon_step
    expected = expected_resolution(system_version, reference)
    if step is not None and abs(step - expected) > 1e-3:
        raise ValueError(
            f"Spatial step is {step} degrees. {system_version} on {reference.isoformat()} "
            f"is published at {expected} degrees. The file was not resampled."
        )
    north, west, south, east = area
    pad = (step or expected) + 0.02
    if latitude.size and (float(np.nanmax(latitude)) > north + pad or float(np.nanmin(latitude)) < south - pad):
        raise ValueError("Latitude extent is outside the requested country area.")
    if longitude.size and (float(np.nanmax(longitude)) > east + pad or float(np.nanmin(longitude)) < west - pad):
        raise ValueError("Longitude extent is outside the requested country area.")
    return step


def _reject_irregular(dataset, expected: float) -> None:
    lat_name, lon_name = _lat_lon_names(dataset)
    lat_step = _grid_step(np.asarray(dataset[lat_name].values, dtype=float))
    lon_step = _grid_step(np.asarray(dataset[lon_name].values, dtype=float))
    if lat_step is not None and lon_step is not None and abs(lat_step - lon_step) > 1e-3:
        raise ValueError("Auxiliary latitude and longitude spacing differ.")
    step = lat_step if lat_step is not None else lon_step
    if step is not None and abs(step - expected) > 1e-3:
        raise ValueError(f"Auxiliary step is {step} degrees, not the native {expected} degree grid.")


def _require_categorical(dataset) -> None:
    """Require every gridded field to keep an integer dtype.

    A CRS or grid-mapping variable, such as the scalar ``crs`` character
    field in the v4.0 local-drain-direction file, is not a drainage code.
    """
    lat_name, lon_name = _lat_lon_names(dataset)
    checked = False
    for name, var in dataset.data_vars.items():
        if lat_name not in var.dims or lon_name not in var.dims:
            continue
        checked = True
        if var.dtype.kind not in "iu":
            raise ValueError(
                f"{name} is {var.dtype}, a continuous dtype. "
                "Local drain direction is stored as its original integer codes."
            )
    if not checked:
        raise ValueError("Local drain direction has no gridded field.")


def _grid_step(values: np.ndarray) -> Optional[float]:
    finite = values[np.isfinite(values)]
    if finite.size < 2:
        return None
    diffs = np.abs(np.diff(np.sort(finite.astype(float))))
    if float(diffs.max() - diffs.min()) > 1e-4:
        raise ValueError("The grid spacing is not regular.")
    return float(np.median(diffs))


def _lat_lon_names(dataset) -> Tuple[str, str]:
    return _coordinate_name(dataset, LAT_NAMES), _coordinate_name(dataset, LON_NAMES)


def _coordinate_name(dataset, names: Sequence[str]) -> str:
    for name in names:
        if name in dataset.coords or name in dataset.dims or name in dataset.variables:
            return name
    raise ValueError(f"No coordinate named {names} in the file.")


def _stamp_run(dataset, day: date, leads: Sequence[int], open_tail: bool, catalogue_updated: Optional[str]) -> None:
    dataset.attrs["gigaspatial_reference_date"] = day.isoformat()
    dataset.attrs["gigaspatial_lead_hours"] = json.dumps(list(leads))
    dataset.attrs["gigaspatial_open"] = "true" if open_tail else "false"
    dataset.attrs["gigaspatial_catalogue_updated"] = catalogue_updated or ""


def _tag_reference(dataset, day: date):
    if any(name in dataset.dims for name in REFERENCE_DIMS):
        reference = next(name for name in REFERENCE_DIMS if name in dataset.dims)
        return _broadcast_run_coords(dataset, reference)
    stamp = np.datetime64(day.isoformat(), "ns")
    tagged = dataset.expand_dims(forecast_reference_time=[stamp])
    return _broadcast_run_coords(tagged, "forecast_reference_time")


def _broadcast_run_coords(dataset, reference: str):
    if dataset.sizes.get(reference, 0) != 1:
        return dataset
    updates = {}
    for name, coord in dataset.coords.items():
        if name == reference or name in dataset.dims or reference in coord.dims:
            continue
        updates[name] = coord.expand_dims({reference: np.asarray(dataset[reference].values)})
    if updates:
        dataset = dataset.assign_coords(updates)
    return dataset


def _concat_runs(parts: List):
    if len(parts) == 1:
        return parts[0]
    reference = next(name for name in REFERENCE_DIMS if name in parts[0].dims)
    combined = _xarray().concat(parts, dim=reference, coords="minimal", compat="override")
    stamps = list(pd.to_datetime(combined[reference].values))
    last = {}
    for index, stamp in enumerate(stamps):
        last[pd.Timestamp(stamp).to_datetime64()] = index
    order = [last[stamp] for stamp in sorted(last)]
    return combined.isel({reference: order})


def _collapse(mapping: dict):
    if len(mapping) == 1:
        return next(iter(mapping.values()))
    return mapping


def _combine_datasets(datasets: List):
    if len(datasets) == 1:
        return datasets[0]
    try:
        return _xarray().merge(datasets, compat="no_conflicts", join="exact")
    except (ValueError, TypeError) as exc:
        raise ValueError(
            "The download has several grids that cannot be combined without dropping values."
        ) from exc


def _decode_zip(raw: bytes):
    archive = zipfile.ZipFile(io.BytesIO(raw))
    members = [name for name in archive.namelist() if not name.endswith("/")]
    netcdf = [name for name in members if name.lower().endswith(".nc")]
    grib = [name for name in members if name.lower().endswith((".grib", ".grib2", ".grb", ".grb2"))]
    if netcdf and grib:
        raise ValueError("The EWDS zip contains both NetCDF and GRIB files.")
    if netcdf:
        return _combine_datasets([_open_netcdf_bytes(archive.read(name)) for name in netcdf])
    if grib:
        return _combine_datasets([open_grib_bytes(archive.read(name)) for name in grib])
    if len(members) == 1:
        return decode_payload(archive.read(members[0]))
    raise ValueError("The EWDS zip has no NetCDF or GRIB file.")


def _xarray():
    try:
        import xarray as xr
    except ImportError as exc:
        raise ImportError("Reading GloFAS NetCDF needs the xarray and netCDF4 packages.") from exc
    return xr


def _open_netcdf_bytes(raw: bytes):
    handle = tempfile.NamedTemporaryFile(suffix=".nc", delete=False)
    handle.write(raw)
    handle.close()
    path = handle.name
    dataset = None
    try:
        dataset = _xarray().open_dataset(path, engine="netcdf4", mask_and_scale=False)
        loaded = dataset.load()
    finally:
        if dataset is not None:
            dataset.close()
        os.unlink(path)
    return loaded


def _open_path(store, path: Path):
    return _open_netcdf_bytes(store.read_file(str(path)))


def _dataset_bytes(dataset) -> bytes:
    """Write NetCDF without letting xarray recode a variable's dtype or fill."""
    encoding = {}
    removed = []
    for name, var in dataset.variables.items():
        if var.dtype.kind not in "iufb":
            continue
        entry = {"dtype": var.dtype}
        for key in ("_FillValue", "missing_value"):
            if key in var.attrs:
                if "_FillValue" not in entry:
                    entry["_FillValue"] = var.attrs[key]
                removed.append((name, key, var.attrs.pop(key)))
        encoding[name] = entry
    handle = tempfile.NamedTemporaryFile(suffix=".nc", delete=False)
    handle.close()
    path = handle.name
    try:
        dataset.to_netcdf(path, engine="netcdf4", encoding=encoding)
        return Path(path).read_bytes()
    finally:
        for name, key, value in removed:
            dataset[name].attrs[key] = value
        os.unlink(path)


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


def _file_units(dataset) -> Optional[str]:
    found = []
    for var in dataset.data_vars.values():
        units = var.attrs.get("units")
        if units not in (None, ""):
            found.append(str(units))
    if not found:
        return None
    unique = list(dict.fromkeys(found))
    return unique[0] if len(unique) == 1 else ", ".join(unique)


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
    return json.loads(store.read_file(str(path), encoding="utf-8"))


def _write_json(store, path: Path, payload: dict) -> None:
    store.write_file(str(path), json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _base_metadata(unit, meta, catalogue, expected, latest, stored, config) -> dict:
    stored_days = sorted(stored)
    step = meta.get("spatial_resolution_degrees")
    if step is None and stored_days:
        step = expected_resolution(unit["system_version"], date.fromisoformat(stored_days[-1]))
    return {
        "country_iso3": unit["country_iso3"],
        "country_name": unit["country_name"],
        "dataset": DATASET_ID,
        "doi": catalogue.doi,
        "system_version": unit["system_version"],
        "hydrological_model": unit["hydrological_model"],
        "product_type": unit["product_type"],
        "variable": unit["variable"],
        "variable_label": VARIABLE_LABELS.get(unit["variable"], unit["variable"]),
        "grib_short_name": GRIB_SHORT_NAMES.get(unit["variable"]),
        "units": meta.get("units") or DOCUMENTED_UNITS.get(unit["variable"], ""),
        "spatial_resolution_degrees": step,
        "temporal_resolution": "forecast",
        "lead_time_hours": list(next(iter(expected.values()))) if expected else [],
        "catalogue_temporal_start": _iso(catalogue.temporal_start),
        "catalogue_temporal_end": _iso(catalogue.temporal_end),
        "catalogue_updated": meta.get("catalogue_updated"),
        "coverage_start": stored_days[0] if stored_days else None,
        "coverage_end": stored_days[-1] if stored_days else None,
        "source_latest_reference": _iso(latest),
        "last_successful_download_utc": meta.get("last_successful_download_utc"),
        "last_successful_check_utc": meta.get("last_successful_check_utc"),
        "runs": {key: stored[key] for key in stored_days},
        "files": [f"{key}.nc" for key in stored_days],
        "area": list(unit["area"]),
        "reference_start": unit.get("reference_start"),
        "reference_end": unit.get("reference_end"),
        "licence": LICENCE,
        "data_format": config.data_format,
        "retrieval": {
            "dataset": DATASET_ID,
            "api_url": config.api_url,
            "dataset_page": DATASET_PAGE,
            "area_order": ["north", "west", "south", "east"],
        },
    }


def _fingerprint_token(remote: dict) -> str:
    return f"{remote.get('content_length')}|{remote.get('last_modified')}"


def _stream_with_retry(url, path, store, **kwargs) -> None:
    delays = (1, 2, 4)
    last_error = None
    for delay in (*delays, None):
        try:
            stream_to_store(url, path, store, **kwargs)
            return
        except requests.RequestException as exc:
            last_error = exc
            if delay is None:
                raise
            time.sleep(delay)
    if last_error:
        raise last_error


def _request(session, method: str, url: str, **kwargs):
    delays = (1, 2, 4)
    last_error = None
    for delay in (*delays, None):
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
        if response.status_code < 400 or response.status_code not in TRANSIENT_STATUS:
            return response
        last_error = requests.HTTPError(f"{response.status_code} from {url}")
    if last_error:
        raise last_error
    raise requests.HTTPError(f"Request failed: {url}")


def _provider_message(response, api_key: str) -> str:
    text = response.text[:500]
    if api_key:
        text = text.replace(api_key, "***")
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


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
