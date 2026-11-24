"""
GloFAS operational forecasts at country level (CEMS Early Warning Data Store).

Publisher: Copernicus Emergency Management Service, ECMWF.
Dataset: ``cems-glofas-forecast``.
Licence: CEMS-FLOODS. The EWDS account must accept that licence before a
forecast query will run. The static auxiliary maps are separate NetCDF
files on the CEMS auxiliary-data page and do not use the EWDS key.
Landing page: https://ewds.climate.copernicus.eu/datasets/cems-glofas-forecast
Auxiliary page: https://confluence.ecmwf.int/spaces/CEMS/pages/242067380/Auxiliary+Data

``GlofasForecastFetcher`` queries that API and returns the result. It does
not write the dataset into a data store.

The country's administrative boundary, including islands and other parts,
selects the native grid cells. The request asks EWDS for a sub-region and
does not set a target grid. Values are not resampled, interpolated, or
reprojected. Variable names, coordinates, ensemble members, forecast steps,
and units are kept as the file provides them.

A forecast query is one variable, one product (control or perturbed ensemble),
one hydrological model, and the reference dates the catalogue publishes in
the requested window. Lead times and the dates each variable actually has
are read from the catalogue constraints. They are not assumed to match
across variables.

The v4.0 auxiliary maps (0.05 degree, EPSG:4326) apply to operational
forecasts from 2023-07-26, including GloFAS v4.1, which reuses those files.
They are not paired with ``version_2_1``, ``version_3_1``, or operational
runs before 2023-07-26, which are the 0.1 degree grids.

When a query contains both a v4 forecast and these maps, upstream area and
the GloFAS flood thresholds are copied onto the forecast's native cells.
Cell centres must already coincide. Nothing is interpolated. The flood
thresholds are the GloFAS v4.0 files on the CEMS auxiliary-data page. The
EFAS thresholds on that page belong to the European model and a different
grid, so they are not applied to GloFAS discharge. A forecast that exceeds
a threshold gains ``exceeded_return_period_years``: the highest listed
return period the discharge exceeds, or 0 when it stays below the 1.5-year
threshold. Missing discharge, the auxiliary fill value, and cells with no
upstream area stay missing.

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
import logging
import math
import os
import tempfile
import threading
import time
import zipfile
from datetime import date, datetime, timezone
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
# NetCDF fill used by the published auxiliary maps.
AUXILIARY_FILL = -3.402823e38
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
class GlofasForecastResolveOptions:
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
class GlofasForecastFetcher:
    """
    Query GloFAS operational forecasts for one country.

    The dataset is served by the EWDS retrieve API. ``fetch`` returns the
    forecast, and any auxiliary maps requested with it, in memory. Nothing
    is written to a data store.

    ::

        GlofasForecastFetcher().fetch("RWA")
        GlofasForecastFetcher().fetch(
            "RWA",
            variable="river_discharge_in_the_last_24_hours",
            product_type="control_forecast",
            reference_start="2026-11-15",
            reference_end="2026-11-15",
        )
        GlofasForecastFetcher().fetch("RWA", variable="upstream_area")
        GlofasForecastFetcher().fetch("RWA", include_auxiliary=True)

    Set ``EWDS_API_KEY`` and accept the CEMS-FLOODS licence before a forecast
    query. Auxiliary maps do not use that key. Omitting the reference dates
    queries every run the catalogue publishes.
    """

    system_version: str = "operational"
    data_format: str = "netcdf"
    api_url: str = Field(default_factory=lambda: global_config.EWDS_API_URL or DEFAULT_API_URL)
    api_key: str = Field(default_factory=lambda: global_config.EWDS_API_KEY, repr=False)
    request_timeout_seconds: int = 3600
    logger: Optional[logging.Logger] = Field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.system_version not in SYSTEM_VERSIONS:
            raise ValueError(
                f"system_version must be one of {SYSTEM_VERSIONS}, got {self.system_version!r}."
            )
        if self.data_format not in DATA_FORMATS:
            raise ValueError(f"data_format must be one of {DATA_FORMATS}, got {self.data_format!r}.")
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
        product_type: Optional[str] = None,
        hydrological_model: Optional[str] = None,
        include_auxiliary: bool = False,
        reference_start=None,
        reference_end=None,
    ):
        """
        Query one country and return the forecast.

        One series returns a Dataset. Several variables, products, or areas
        return nested dicts, collapsing any level that has a single entry.
        A v4 forecast queried together with auxiliary maps also receives
        those maps on its own cells, including ``exceeded_return_period_years``
        when river discharge and flood thresholds are both present.

        Raises:
            RuntimeError: The EWDS request failed, or ``EWDS_API_KEY`` is missing.
            ValueError: The country, variable, grid, or ensemble dimension is invalid.
        """
        query = GlofasForecastResolveOptions(
            variable=variable,
            product_type=product_type,
            hydrological_model=hydrological_model,
            include_auxiliary=include_auxiliary,
            reference_start=reference_start,
            reference_end=reference_end,
        )
        iso3, country_name = self._country(country)
        areas = extraction_areas(source_geometry(iso3, crs="EPSG:4326"))
        groups: Dict[str, Dict[str, Dict[str, List]]] = {}
        if query.variable not in AUXILIARY:
            self._query_forecasts(iso3, country_name, areas, query, groups)
        self._query_auxiliary(areas, query, groups)
        if not groups:
            raise ValueError(f"No GloFAS forecast variables were selected for {iso3}.")
        self._align_auxiliary(groups)
        return _assemble(groups)

    def catalogue(self) -> ForecastCatalogue:
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
                "GloFAS forecast is queried by country. "
                "Pass a country name or ISO code, for example 'RWA'."
            )
        try:
            record = pycountry.countries.lookup(country)
        except LookupError as exc:
            raise ValueError(f"Unknown country: {country!r}.") from exc
        return record.alpha_3, record.name

    def _query_forecasts(self, iso3, country_name, areas, query, groups) -> None:
        requested = False
        for variable in _series_variables(query):
            try:
                availability = self.catalogue().availability(self.system_version, variable)
            except ValueError as exc:
                if query.variable == variable:
                    raise
                self.logger.warning("%s", exc)
                continue
            for model, product in availability.groups():
                if query.hydrological_model not in (None, model):
                    continue
                if query.product_type not in (None, product):
                    continue
                expected, _latest = _window(
                    availability.days(model, product),
                    query.reference_start,
                    query.reference_end,
                    availability,
                    model,
                    product,
                )
                for area in areas:
                    for day, leads in expected.items():
                        requested = True
                        dataset = self._fetch_day(
                            iso3, variable, model, product, area, day, leads
                        )
                        _collect(groups, variable, product, _area_id(*area), dataset)
        if _series_variables(query) and not requested and query.variable not in AUXILIARY:
            if not groups:
                return

    def _query_auxiliary(self, areas, query, groups) -> None:
        names = _auxiliary_names(query)
        if not names:
            return
        if not v4_auxiliary_applies(
            self.system_version,
            query.reference_end if query.variable not in AUXILIARY else None,
        ):
            message = (
                f"GloFAS v{AUXILIARY_VERSION} auxiliary maps apply to operational forecasts "
                f"from {V4_GRID_FROM.isoformat()}. They are not paired with "
                f"{self.system_version}"
                + (
                    f" through {query.reference_end.isoformat()}"
                    if query.reference_end is not None and self.system_version == "operational"
                    else ""
                )
                + "."
            )
            if query.variable in AUXILIARY:
                raise ValueError(message)
            self.logger.warning("%s", message)
            return
        for name in names:
            for area in areas:
                dataset = self._fetch_auxiliary(name, area)
                _collect(groups, name, "static", _area_id(*area), dataset)

    def _fetch_day(self, iso3, variable, model, product, area, day, leads):
        self._require_key()
        request = {
            "system_version": [self.system_version],
            "hydrological_model": [model],
            "product_type": [product],
            "variable": [variable],
            "year": [f"{day.year:04d}"],
            "month": [f"{day.month:02d}"],
            "day": [f"{day.day:02d}"],
            "leadtime_hour": [str(hour) for hour in leads],
            "area": list(area),
            "data_format": self.data_format,
            "download_format": "zip",
        }
        self.logger.info(
            "%s %s %s: retrieving %s.",
            iso3,
            variable,
            product,
            day.isoformat(),
        )
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "download.bin"
            self._retrieve(request, dest)
            dataset = decode_payload(dest.read_bytes())
            try:
                present = _lead_hours(dataset)
                _require_ensemble(dataset, product)
                _validate_grid(
                    dataset,
                    system_version=self.system_version,
                    reference=day,
                    area=area,
                )
                missing = sorted(set(leads) - set(present))
                if missing:
                    self.logger.warning(
                        "%s %s %s: EWDS omitted lead times %s.",
                        iso3,
                        variable,
                        day.isoformat(),
                        missing,
                    )
                loaded = dataset.load()
            finally:
                dataset.close()
        return _tag_reference(loaded, day)

    def _fetch_auxiliary(self, name: str, area):
        spec = AUXILIARY[name]
        self.logger.info(
            "Retrieving GloFAS v%s %s (%s).",
            AUXILIARY_VERSION,
            name,
            spec["filename"],
        )
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / spec["filename"]
            self._download_to_path(spec["url"], dest)
            dataset = _open_netcdf_bytes(dest.read_bytes())
        try:
            subset = subset_native(dataset, area).load()
            _reject_irregular(subset, V4_RESOLUTION_DEGREES)
            if spec["categorical"]:
                _require_categorical(subset)
            return subset
        finally:
            dataset.close()

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

    def _align_auxiliary(self, groups: Dict[str, Dict[str, Dict[str, List]]]) -> None:
        """Copy v4 static maps onto forecast datasets that share their cells."""
        static_by_area: Dict[str, Dict[str, object]] = {}
        for variable, products in groups.items():
            for area_id, parts in products.get("static", {}).items():
                if len(parts) != 1:
                    raise ValueError(
                        f"{variable} for area {area_id} has {len(parts)} static files."
                    )
                static_by_area.setdefault(area_id, {})[variable] = parts[0]
        if not static_by_area:
            return
        for variable, products in groups.items():
            for product, areas in products.items():
                if product == "static":
                    continue
                for area_id, parts in areas.items():
                    maps = static_by_area.get(area_id)
                    if not maps:
                        raise ValueError(
                            "GloFAS auxiliary maps were retrieved, but none match "
                            f"forecast area {area_id}."
                        )
                    if not all(_on_v4_grid(part) for part in parts):
                        self.logger.warning(
                            "GloFAS v4.0 auxiliary maps were not aligned onto %s %s %s. "
                            "They apply to operational forecasts from %s on the 0.05 degree grid.",
                            variable,
                            product,
                            area_id,
                            V4_GRID_FROM.isoformat(),
                        )
                        continue
                    areas[area_id] = [align_static_maps(part, maps) for part in parts]


def _series_variables(query: GlofasForecastResolveOptions) -> Tuple[str, ...]:
    if query.variable in AUXILIARY:
        return ()
    if query.variable is None:
        return VARIABLES
    return (query.variable,)


def _auxiliary_names(query: GlofasForecastResolveOptions) -> Tuple[str, ...]:
    if query.variable in AUXILIARY:
        return (query.variable,)
    if query.include_auxiliary and query.variable not in AUXILIARY:
        return tuple(AUXILIARY)
    return ()


def _collect(groups, variable, product, area_id, dataset) -> None:
    groups.setdefault(variable, {}).setdefault(product, {}).setdefault(area_id, []).append(dataset)


def _assemble(groups):
    assembled = {}
    for variable, products in groups.items():
        packed = {
            product: _collapse({area_id: _concat_runs(parts) for area_id, parts in areas.items()})
            for product, areas in products.items()
        }
        assembled[variable] = _collapse(packed)
    return _collapse(assembled)




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


def align_static_maps(forecast, maps: Dict[str, object]):
    """
    Copy static maps onto the forecast's native cells.

    Each map must use the same 0.05 degree cell centres. A centre that is
    missing, or farther than half a cell away, raises ``ValueError``. Values
    are not interpolated. The published fill value becomes missing. River
    discharge then records the highest GloFAS flood-threshold return period
    it exceeds.
    """
    if not _on_v4_grid(forecast):
        raise ValueError(
            "GloFAS v4.0 auxiliary maps apply from "
            f"{V4_GRID_FROM.isoformat()} on the 0.05 degree grid."
        )
    aligned = forecast
    discharge_name = _discharge_variable(aligned)
    if discharge_name is not None:
        masked = _mask_missing(aligned[discharge_name])
        aligned = aligned.assign({discharge_name: masked})
    thresholds = []
    upstream = None
    for name, static in maps.items():
        piece = _reindex_native(static, aligned, name)
        if name in aligned.data_vars:
            raise ValueError(f"Forecast already has a variable named {name}.")
        aligned = aligned.assign({name: piece})
        if name.startswith("flood_threshold_rl_"):
            thresholds.append((float(name[len("flood_threshold_rl_"):]), aligned[name]))
        elif name == "upstream_area":
            upstream = aligned[name]
    if discharge_name is not None and thresholds:
        exceeded = _exceeded_return_period(aligned[discharge_name], thresholds, upstream)
        aligned = aligned.assign(exceeded_return_period_years=exceeded)
    aligned.attrs["auxiliary_alignment"] = "native_cell"
    aligned.attrs["auxiliary_version"] = AUXILIARY_VERSION
    return aligned


def _on_v4_grid(dataset) -> bool:
    dates = _reference_dates(dataset)
    if dates and min(dates) < V4_GRID_FROM:
        return False
    lat_name, _lon_name = _lat_lon_names(dataset)
    step = _grid_step(np.asarray(dataset[lat_name].values, dtype=float))
    return step is not None and abs(step - V4_RESOLUTION_DEGREES) <= 1e-4


def _reference_dates(dataset) -> List[date]:
    for name in REFERENCE_DIMS:
        if name not in dataset.coords and name not in dataset.dims and name not in dataset.variables:
            continue
        found = []
        for value in np.asarray(dataset[name].values).reshape(-1):
            try:
                found.append(pd.Timestamp(value).date())
            except (ValueError, TypeError):
                continue
        if found:
            return found
    raw = dataset.attrs.get("gigaspatial_reference_date")
    if raw:
        return [date.fromisoformat(str(raw)[:10])]
    return []


def _discharge_variable(dataset) -> Optional[str]:
    for name in ("dis24", "river_discharge_in_the_last_24_hours"):
        if name in dataset.data_vars:
            return name
    return None


def _reindex_native(static, forecast, name: str):
    """Select the static cells whose centres match the forecast."""
    flat, flon = _lat_lon_names(forecast)
    slat, slon = _lat_lon_names(static)
    forecast_lat = np.asarray(forecast[flat].values, dtype=float)
    forecast_lon = np.asarray(forecast[flon].values, dtype=float)
    step = _grid_step(forecast_lat) or V4_RESOLUTION_DEGREES
    lat_index = _native_index(forecast_lat, np.asarray(static[slat].values, dtype=float), step)
    lon_index = _native_index(forecast_lon, np.asarray(static[slon].values, dtype=float), step)
    variable = _gridded_variable(static)
    order = [slat, slon] + [dim for dim in variable.dims if dim not in (slat, slon)]
    values = np.asarray(variable.transpose(*order).values)
    selected = values[lat_index][:, lon_index, ...]
    selected = _mask_array(selected, dict(variable.attrs))
    piece = _xarray().DataArray(
        selected,
        dims=(flat, flon) + tuple(order[2:]),
        coords={flat: forecast[flat], flon: forecast[flon]},
        name=name,
        attrs=dict(variable.attrs),
    )
    piece.attrs["alignment"] = "native_cell"
    return piece


def _gridded_variable(dataset):
    lat_name, lon_name = _lat_lon_names(dataset)
    gridded = [
        variable
        for variable in dataset.data_vars.values()
        if lat_name in variable.dims and lon_name in variable.dims
    ]
    if len(gridded) != 1:
        raise ValueError(
            "An auxiliary map must contain one gridded field, "
            f"found {len(gridded)}."
        )
    return gridded[0]


def _native_index(target: np.ndarray, source: np.ndarray, step: float) -> np.ndarray:
    order = np.argsort(source)
    ordered = np.asarray(source, dtype=float)[order]
    if ordered.size == 0:
        raise ValueError("Auxiliary map has no coordinates.")
    position = np.searchsorted(ordered, target, side="left")
    position = np.clip(position, 0, ordered.size - 1)
    previous = np.clip(position - 1, 0, ordered.size - 1)
    choose_previous = np.abs(ordered[previous] - target) <= np.abs(ordered[position] - target)
    nearest = np.where(choose_previous, previous, position)
    delta = np.abs(ordered[nearest] - target)
    if np.any(delta > (step / 2) + 1e-6):
        raise ValueError(
            "Forecast and auxiliary cells do not share the GloFAS grid. "
            "Values were not interpolated."
        )
    return order[nearest]


def _mask_missing(values):
    if not np.issubdtype(values.dtype, np.floating):
        return values
    data = values.astype("float64")
    invalid = ~np.isfinite(data) | (data < -1e30)
    number = _fill_value(values.attrs)
    if number is not None:
        invalid = invalid | (np.abs(data - number) <= abs(number) * 1e-6)
    masked = data.where(~invalid)
    masked.attrs = dict(values.attrs)
    return masked


def _mask_array(values: np.ndarray, attrs: dict) -> np.ndarray:
    if not np.issubdtype(values.dtype, np.floating):
        return values
    data = values.astype("float64")
    invalid = ~np.isfinite(data) | (data < -1e30)
    number = _fill_value(attrs)
    if number is not None:
        invalid = invalid | (np.abs(data - number) <= abs(number) * 1e-6)
    return np.where(invalid, np.nan, data)


def _fill_value(attrs: dict) -> Optional[float]:
    fill = attrs.get("_FillValue", attrs.get("missing_value"))
    if fill is None:
        return None
    try:
        number = float(fill)
    except (TypeError, ValueError):
        return None
    if abs(number) <= 1e30:
        return None
    return number


def _exceeded_return_period(discharge, thresholds, upstream):
    """Highest return period, in years, exceeded at each forecast cell."""
    finite = np.isfinite(discharge)
    exceeded = _xarray().full_like(discharge, np.nan, dtype="float64")
    exceeded = exceeded.where(~finite, 0.0)
    for years, threshold in sorted(thresholds):
        above = finite & np.isfinite(threshold) & (discharge > threshold)
        exceeded = exceeded.where(~above, float(years))
    if upstream is not None:
        river = np.isfinite(upstream) & (upstream > 0)
        exceeded = exceeded.where(river)
    exceeded.name = "exceeded_return_period_years"
    exceeded.attrs["units"] = "years"
    exceeded.attrs["long_name"] = (
        "Highest GloFAS flood-threshold return period exceeded by the forecast discharge"
    )
    exceeded.attrs["below_lowest_threshold"] = 0.0
    return exceeded


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

