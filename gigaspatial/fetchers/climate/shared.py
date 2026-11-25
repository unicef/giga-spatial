"""Shared climate retrieval helpers.

ERA5 and ERA5-Land both use this module. It resolves an area, builds a CDS
area, opens the download, standardizes the dataset, and clips it to the
requested geometry. Dataset ids, variable lists, and request fields that
differ stay in the dataset modules.
"""

from __future__ import annotations

import calendar
import json
import math
import os
import re
import tempfile
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import requests
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry

from gigaspatial.handlers.base.source import SourceKind, classify_source

DEFAULT_API_URL = "https://cds.climate.copernicus.eu/api"
DEFAULT_MAX_BYTES = 2 * 1024 ** 3
VARIABLES_PER_REQUEST = 10
SHORT_NAMES = {
    "2t": "2m_temperature",
    "t2m": "2m_temperature",
    "d2m": "2m_dewpoint_temperature",
    "msl": "mean_sea_level_pressure",
    "sp": "surface_pressure",
    "tp": "total_precipitation",
    "sst": "sea_surface_temperature",
}
_SECRET = re.compile(r"(?i)(key|token|private-token)(=|:)\s*([^\s&]+)")


class ClimateError(RuntimeError):
    """A climate retrieval failed."""


class ClimateAuthError(ClimateError):
    """CDS credentials are missing."""


class ClimateRequestError(ClimateError):
    """CDS rejected a request. The variable is named when one request failed."""

    def __init__(self, message: str, *, variable: Optional[str] = None, retryable: bool = False):
        super().__init__(message)
        self.variable = variable
        self.retryable = retryable


class ClimateAvailabilityError(ClimateError):
    """The requested time is outside what CDS currently publishes."""


class ClimateLimitError(ClimateError):
    """The request is larger than the configured memory guard."""


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    dataset_id: str
    documentation_url: str
    doi: str
    start_year: int
    resolution: float
    variables: tuple
    product_types: tuple = ()
    default_product: Optional[str] = None

    def by_id(self):
        return {item.id: item for item in self.variables}


_WATER_ACCUMULATIONS = frozenset(
    {
        "convective_precipitation",
        "convective_snowfall",
        "evaporation",
        "evaporation_from_bare_soil",
        "evaporation_from_open_water_surfaces_excluding_oceans",
        "evaporation_from_the_top_of_canopy",
        "evaporation_from_vegetation_transpiration",
        "large_scale_precipitation",
        "large_scale_snowfall",
        "potential_evaporation",
        "runoff",
        "snow_evaporation",
        "snowfall",
        "snowmelt",
        "sub_surface_runoff",
        "surface_runoff",
        "total_evaporation",
        "total_precipitation",
    }
)

_RADIATION_ACCUMULATIONS = frozenset(
    {
        "clear_sky_direct_solar_radiation_at_surface",
        "downward_uv_radiation_at_the_surface",
        "surface_latent_heat_flux",
        "surface_net_solar_radiation",
        "surface_net_solar_radiation_clear_sky",
        "surface_net_thermal_radiation",
        "surface_net_thermal_radiation_clear_sky",
        "surface_sensible_heat_flux",
        "surface_solar_radiation_downward_clear_sky",
        "surface_solar_radiation_downwards",
        "surface_thermal_radiation_downward_clear_sky",
        "surface_thermal_radiation_downwards",
        "toa_incident_solar_radiation",
        "top_net_solar_radiation",
        "top_net_solar_radiation_clear_sky",
        "top_net_thermal_radiation",
        "top_net_thermal_radiation_clear_sky",
        "total_sky_direct_solar_radiation_at_surface",
    }
)

_STRESS_ACCUMULATIONS = frozenset(
    {
        "eastward_gravity_wave_surface_stress",
        "eastward_turbulent_surface_stress",
        "northward_gravity_wave_surface_stress",
        "northward_turbulent_surface_stress",
    }
)

_DISSIPATION_ACCUMULATIONS = frozenset(
    {"boundary_layer_dissipation", "gravity_wave_dissipation"}
)

_HEIGHTS = frozenset(
    {
        "boundary_layer_height",
        "cloud_base_height",
        "duct_base_height",
        "lake_depth",
        "lake_ice_depth",
        "lake_mix_layer_depth",
        "lake_total_depth",
        "maximum_individual_wave_height",
        "model_bathymetry",
        "snow_depth",
        "snow_depth_water_equivalent",
        "standard_deviation_of_filtered_subgrid_orography",
        "standard_deviation_of_orography",
        "trapping_layer_base_height",
        "trapping_layer_top_height",
        "zero_degree_level",
    }
)


@dataclass(frozen=True)
class Variable:
    """One CDS variable, in the dataset's own name."""

    id: str
    description: str
    units: Optional[str]
    accumulated: bool
    time_invariant: bool
    groups: tuple[str, ...]


def infer_units(variable_id: str) -> tuple[Optional[str], bool]:
    """
    Return ``(units, accumulated)`` for a CDS variable name.

    Units are the SI units ECMWF publishes for that parameter. Names that are
    not in those tables return ``(None, False)`` rather than a guessed unit.
    Accumulated fields are hourly totals as CDS publishes them (metres of
    water, joules per square metre, or newton-seconds per square metre).
    """

    name = variable_id
    if name in {"precipitation_type", "soil_type", "type_of_high_vegetation", "type_of_low_vegetation"}:
        return "code", False
    if name in _WATER_ACCUMULATIONS:
        return "m", True
    if name in _RADIATION_ACCUMULATIONS:
        return "J m**-2", True
    if name in _STRESS_ACCUMULATIONS:
        return "N m**-2 s", True
    if name in _DISSIPATION_ACCUMULATIONS:
        return "J m**-2", True
    if name in _HEIGHTS or name.startswith("significant_height") or name.startswith("significant_wave_height"):
        return "m", False
    if "temperature" in name and "vertical_integral" not in name:
        return "K", False
    if name.endswith("_pressure") or name == "pressure":
        return "Pa", False
    if name in {"geopotential"}:
        return "m**2 s**-2", False
    if name in {"snow_density", "air_density_over_the_oceans"}:
        return "kg m**-3", False
    if name.startswith("volumetric_soil_water"):
        return "m**3 m**-3", False
    if name.startswith("leaf_area_index"):
        return "m**2 m**-2", False
    if name.startswith("total_column_"):
        return "kg m**-2", False
    if name in {"convective_available_potential_energy", "convective_inhibition"}:
        return "J kg**-1", False
    if name in {"k_index", "total_totals_index"}:
        return "K", False
    if name in {"skin_reservoir_content"}:
        return "m", False
    if name in {"forecast_surface_roughness"}:
        return "m", False
    if "direction" in name or name.startswith("angle_"):
        return "degrees", False
    if "wind_speed" in name or name.endswith("_velocity") or "component_of_wind" in name or "stokes_drift" in name or "wind_gust" in name:
        return "m s**-1", False
    if "period" in name:
        return "s", False
    if name.startswith("mean_") and name.endswith("_rate"):
        return "kg m**-2 s**-1", False
    if name.endswith("_rate"):
        return "kg m**-2 s**-1", False
    if name.startswith("mean_") and "flux" in name:
        return "W m**-2", False
    if name.startswith("mean_") and "stress" in name:
        return "N m**-2", False
    if name.startswith("mean_") and "dissipation" in name:
        return "W m**-2", False
    if name.startswith("instantaneous_") and "moisture_flux" in name:
        return "kg m**-2 s**-1", False
    if name.startswith("instantaneous_") and "flux" in name:
        return "W m**-2", False
    if name.startswith("instantaneous_") and "stress" in name:
        return "N m**-2", False
    if any(token in name for token in ("albedo", "cover", "mask", "fraction")):
        return "1", False
    if name in {
        "anisotropy_of_sub_gridscale_orography",
        "benjamin_feir_index",
        "charnock",
        "coefficient_of_drag_with_waves",
        "forecast_logarithm_of_surface_roughness_for_heat",
        "lake_shape_factor",
        "normalized_energy_flux_into_ocean",
        "normalized_energy_flux_into_waves",
        "normalized_stress_into_ocean",
        "slope_of_sub_gridscale_orography",
        "wave_spectral_directional_width",
        "wave_spectral_directional_width_for_swell",
        "wave_spectral_directional_width_for_wind_waves",
        "wave_spectral_kurtosis",
        "wave_spectral_peakedness",
        "wave_spectral_skewness",
        "mean_vertical_gradient_of_refractivity_inside_trapping_layer",
        "minimum_vertical_gradient_of_refractivity_inside_trapping_layer",
    }:
        return "1", False
    return None, False


def variables_from_rows(rows) -> tuple:
    items = []
    for name, description, groups in rows:
        units, accumulated = infer_units(name)
        group_names = tuple(groups)
        items.append(Variable(name, description, units, accumulated, "Invariant" in group_names, group_names))
    return tuple(items)


def select_variables(spec: DatasetSpec, requested) -> tuple:
    """One CDS name, several names, or ``"all"``. Unknown names raise."""

    if requested is None:
        raise ValueError("variables is required. Pass a CDS name, a list of names, or 'all'.")
    if isinstance(requested, str):
        requested = (requested,)
    else:
        requested = tuple(requested)
    if not requested:
        raise ValueError("variables is required. Pass a CDS name, a list of names, or 'all'.")
    if requested == ("all",) or requested == ("__all__",):
        return spec.variables
    known = spec.by_id()
    missing = [name for name in requested if name not in known]
    if missing:
        raise ValueError(
            "unknown CDS variable for "
            f"{spec.dataset_id}: {', '.join(missing)}. "
            "Names are the dataset's own CDS parameter names."
        )
    seen = []
    for name in requested:
        if name not in seen:
            seen.append(name)
    return tuple(known[name] for name in seen)


def resolve_area(area):
    """Return ``(geometry, country_id, country_name)`` for a supported area."""

    if isinstance(area, str):
        stripped = area.strip()
        if stripped.startswith("{") or stripped.startswith("["):
            area = json.loads(stripped)
    if isinstance(area, dict) and area.get("type"):
        geometry = shape(area)
        return geometry, None, None
    kind = classify_source(area)
    if kind is SourceKind.COUNTRY:
        from gigaspatial.handlers.boundaries import AdminBoundaries

        table = AdminBoundaries.create(country_code=area)
        entity = table.entities[0]
        return table.to_geoms()[0], entity.country_iso or entity.boundary_id, getattr(entity, "name_en", None) or entity.name
    from gigaspatial.handlers.base.config import source_geometry

    return source_geometry(area), None, None


def countries_intersecting(area, boundaries) -> list:
    """ISO3 codes in ``boundaries`` whose geometry intersects ``area``.

    ``boundaries`` is a GeoDataFrame or an AdminBoundaries table with a
    country code column (``country_iso``, ``iso_a3``, or ``boundary_id``).
    A polygon that crosses a border returns every intersecting country and
    is not collapsed to one ISO3 code. A country code with no table returns
    that country's own code.
    """

    geometry, country_id, _name = resolve_area(area)
    if boundaries is None:
        return [country_id] if country_id else []
    if hasattr(boundaries, "columns") and "geometry" in getattr(boundaries, "columns", ()):
        frame = boundaries
    else:
        frame = getattr(boundaries, "dataframe", None)
    if frame is None and hasattr(boundaries, "entities"):
        import geopandas as gpd

        records = []
        for entity, geom in zip(boundaries.entities, boundaries.to_geoms()):
            records.append({"country_iso": entity.country_iso or entity.boundary_id, "geometry": geom})
        frame = gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")
    code_column = next(
        (name for name in ("country_iso", "iso_a3", "ISO_A3", "boundary_id", "iso3") if name in frame.columns),
        None,
    )
    if code_column is None:
        raise ValueError("boundaries need a country_iso, iso_a3, or boundary_id column")
    hits = frame.loc[frame.geometry.intersects(geometry), code_column]
    return sorted({str(code) for code in hits if code})


def _snap(bounds, resolution: float):
    minx, miny, maxx, maxy = bounds
    west = math.floor(minx / resolution) * resolution
    south = math.floor(miny / resolution) * resolution
    east = math.ceil(maxx / resolution) * resolution
    north = math.ceil(maxy / resolution) * resolution
    if east <= west:
        east = west + resolution
    if north <= south:
        north = south + resolution
    west, east = max(-180.0, west), min(180.0, east)
    south, north = max(-90.0, south), min(90.0, north)
    return tuple(round(value, 6) for value in (west, south, east, north))


def _parts(geometry: BaseGeometry):
    if geometry.is_empty:
        raise ValueError("area geometry is empty")
    if geometry.geom_type == "Polygon":
        return [geometry]
    if geometry.geom_type == "MultiPolygon":
        return [part for part in geometry.geoms if not part.is_empty]
    if geometry.geom_type == "GeometryCollection":
        parts = []
        for part in geometry.geoms:
            parts.extend(_parts(part))
        return parts
    return [geometry]


def request_areas(geometry: BaseGeometry, resolution: float) -> tuple:
    """CDS areas as ``(north, west, south, east)``. Distant parts stay separate."""

    boxes = [_snap(part.bounds, resolution) for part in _parts(geometry)]
    merged = []
    for west, south, east, north in sorted(boxes):
        placed = False
        for item in merged:
            close = west <= item[2] + resolution and east >= item[0] - resolution
            lat_gap = max(item[1] - north, south - item[3])
            if close and lat_gap <= resolution:
                item[0] = min(item[0], west)
                item[1] = min(item[1], south)
                item[2] = max(item[2], east)
                item[3] = max(item[3], north)
                placed = True
                break
        if not placed:
            merged.append([west, south, east, north])
    return tuple((north, west, south, east) for west, south, east, north in merged)


def intersecting_mask(longitudes, latitudes, resolution: float, geometry: BaseGeometry):
    from shapely import box
    from shapely import intersects

    half = resolution / 2.0
    lon, lat = np.meshgrid(np.asarray(longitudes, dtype=float), np.asarray(latitudes, dtype=float))
    cells = box(lon - half, lat - half, lon + half, lat + half)
    return np.asarray(intersects(cells, geometry), dtype=bool)


def _hour(value: datetime) -> datetime:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    else:
        value = value.astimezone(timezone.utc)
    return value.replace(minute=0, second=0, microsecond=0)


def _parse_bound(text: str, *, is_end: bool) -> datetime:
    raw = str(text).strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    if len(raw) == 4 and raw.isdigit():
        year = int(raw)
        return datetime(year, 12, 31, 23, tzinfo=timezone.utc) if is_end else datetime(year, 1, 1, tzinfo=timezone.utc)
    if len(raw) == 7 and raw[4] == "-":
        year, month = int(raw[:4]), int(raw[5:7])
        last = calendar.monthrange(year, month)[1]
        return datetime(year, month, last, 23, tzinfo=timezone.utc) if is_end else datetime(year, month, 1, tzinfo=timezone.utc)
    if len(raw) == 10 and "T" not in raw:
        year, month, day = (int(part) for part in raw.split("-"))
        return datetime(year, month, day, 23 if is_end else 0, tzinfo=timezone.utc)
    return _hour(datetime.fromisoformat(raw))


def _as_tuple(value):
    if value is None:
        return None
    if isinstance(value, (str, int)):
        return (value,)
    return tuple(value)


def selection_hours(spec: DatasetSpec, *, time=None, start=None, end=None, year=None, month=None, day=None, hour=None, full_history=False, latest=None) -> list:
    """UTC hours for a time argument or for year, month, day, and hour."""

    if time is not None and any(value is not None for value in (start, end, year, month, day, hour)):
        raise ValueError("pass time, or start/end and year/month/day/hour, not both")
    if full_history and (time is not None or any(value is not None for value in (start, end, year, month, day, hour))):
        raise ValueError("full_history cannot be combined with another time argument")
    if full_history:
        if latest is None:
            raise ClimateAvailabilityError("full_history needs the latest hour published by CDS")
        start_at = datetime(spec.start_year, 1, 1, tzinfo=timezone.utc)
        return _hours_between(start_at, _hour(latest))
    if time is not None:
        if isinstance(time, (list, tuple)) and len(time) == 2:
            start, end = time
        else:
            start = end = time
    if start or end or year or month or day or hour:
        hours = _explicit_hours(start, end, year, month, day, hour)
    else:
        raise ValueError(
            "Pass time, start and end, or year, month, day, and hour. "
            "The historical archive is not downloaded unless full_history=True."
        )
    if not hours:
        raise ValueError("time selection does not include any hour")
    too_early = [item for item in hours if item.year < spec.start_year]
    if too_early:
        raise ValueError(f"{spec.name} starts in {spec.start_year}; {too_early[0].isoformat()} is before that")
    return hours


def _hours_between(start: datetime, end: datetime) -> list:
    start, end = _hour(start), _hour(end)
    if end < start:
        return []
    hours = []
    cursor = start
    while cursor <= end:
        hours.append(cursor)
        cursor += timedelta(hours=1)
    return hours


def _explicit_hours(start, end, year, month, day, hour) -> list:
    years = {int(item) for item in _as_tuple(year)} if year is not None else None
    months = {int(item) for item in _as_tuple(month)} if month is not None else None
    days = {int(item) for item in _as_tuple(day)} if day is not None else None
    clocks = {int(str(item).split(":")[0]) for item in _as_tuple(hour)} if hour is not None else None
    for clock in clocks or ():
        if not 0 <= clock <= 23:
            raise ValueError(f"hour must be 0..23, got {clock}")
    if start or end:
        if not start or not end:
            raise ValueError("pass both start and end")
        chosen = []
        for item in _hours_between(_parse_bound(start, is_end=False), _parse_bound(end, is_end=True)):
            if years is not None and item.year not in years:
                continue
            if months is not None and item.month not in months:
                continue
            if days is not None and item.day not in days:
                continue
            if clocks is not None and item.hour not in clocks:
                continue
            chosen.append(item)
        return chosen
    if not years:
        return []
    chosen = []
    for item_year in sorted(years):
        for item_month in sorted(months or range(1, 13)):
            if not 1 <= item_month <= 12:
                raise ValueError(f"month must be 1..12, got {item_month}")
            last = calendar.monthrange(item_year, item_month)[1]
            for item_day in sorted(days or range(1, last + 1)):
                if item_day < 1 or item_day > last:
                    continue
                for item_hour in sorted(clocks if clocks is not None else range(24)):
                    chosen.append(datetime(item_year, item_month, item_day, item_hour, tzinfo=timezone.utc))
    return chosen


def month_slices(hours: Sequence[datetime]) -> tuple:
    """Group hours into CDS day/time products, one product per month."""

    unique = sorted({_hour(item) for item in hours})
    by_month = {}
    for item in unique:
        by_month.setdefault((item.year, item.month), {}).setdefault(item.day, []).append(item.hour)
    slices = []
    for year, month in sorted(by_month):
        by_times = {}
        for day, clocks in by_month[(year, month)].items():
            key = tuple(f"{clock:02d}:00" for clock in clocks)
            by_times.setdefault(key, []).append(f"{day:02d}")
        for times, days in by_times.items():
            slices.append((str(year), f"{month:02d}", tuple(sorted(days)), times))
    return tuple(slices)


def redact(message: str, secret: str = "") -> str:
    text = message or ""
    if secret:
        text = text.replace(secret, "***")
    return _SECRET.sub(r"\1\2***", text)


def resolve_credentials(api_url=None, api_key=None):
    """Explicit values, then CDSAPI_*, CDS_*, then ~/.cdsapirc. The key may be empty."""

    from gigaspatial.config import config as project_config

    file_url, file_key = "", ""
    path = Path.home() / ".cdsapirc"
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if ":" not in line:
                continue
            name, value = line.split(":", 1)
            if name.strip() == "url":
                file_url = value.strip()
            elif name.strip() == "key":
                file_key = value.strip()
    env_url = os.environ.get("CDSAPI_URL") or os.environ.get("CDS_API_URL") or ""
    env_key = os.environ.get("CDSAPI_KEY") or os.environ.get("CDS_API_KEY") or ""
    config_url = getattr(project_config, "CDS_API_URL", "") or ""
    config_key = getattr(project_config, "CDS_API_KEY", "") or ""
    url = api_url or env_url or config_url or file_url or DEFAULT_API_URL
    key = api_key or env_key or config_key or file_key or ""
    return url, key


def latest_available(dataset_id: str, api_url: Optional[str] = None, getter=None) -> datetime:
    """Latest hour CDS publishes for this dataset. ERA5 and ERA5-Land are queried separately."""

    url, _key = resolve_credentials(api_url, None)
    fetch = getter or _get_json
    payload = fetch(f"{url.rstrip('/')}/catalogue/v1/collections/{dataset_id}")
    try:
        end = payload["extent"]["temporal"]["interval"][0][1]
    except (KeyError, IndexError, TypeError) as exc:
        raise ClimateAvailabilityError(f"{dataset_id} catalogue has no temporal extent") from exc
    if not end:
        raise ClimateAvailabilityError(f"{dataset_id} catalogue does not publish an end timestamp")
    return _parse_bound(end, is_end=True) if "T" not in str(end) else _hour(datetime.fromisoformat(str(end).replace("Z", "+00:00")))


def _get_json(url: str):
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return response.json()


def period_available(dataset_id: str, start, end, *, latest=None, api_url=None, getter=None) -> dict:
    """Say whether ``start``..``end`` is published, and which hours are not yet available."""

    published = latest or latest_available(dataset_id, api_url=api_url, getter=getter)
    hours = _hours_between(_parse_bound(start, is_end=False), _parse_bound(end, is_end=True))
    missing = [item for item in hours if item > published]
    return {
        "dataset_id": dataset_id,
        "latest_available": published.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "requested_start": hours[0].strftime("%Y-%m-%dT%H:%M:%SZ") if hours else None,
        "requested_end": hours[-1].strftime("%Y-%m-%dT%H:%M:%SZ") if hours else None,
        "available": not missing and bool(hours),
        "missing_hours": [item.strftime("%Y-%m-%dT%H:%M:%SZ") for item in missing],
    }


def build_request(variable_ids, year, month, days, times, area, data_format: str, product_type=None) -> dict:
    request = {
        "variable": list(variable_ids),
        "year": [str(year)],
        "month": [str(month)],
        "day": list(days),
        "time": list(times),
        "data_format": data_format,
        "download_format": "unarchived",
        "area": [float(value) for value in area],
    }
    if product_type:
        request["product_type"] = [product_type]
    return request


def plan_requests(spec: DatasetSpec, variables: Sequence[Variable], hours, areas, data_format: str, product_type=None) -> list:
    """CDS request bodies. Invariant variables use one hour. Others are grouped by month."""

    if product_type and not spec.product_types:
        raise ValueError(f"{spec.name} has no product_type field")
    if product_type and product_type not in spec.product_types:
        raise ValueError(f"{spec.name} product_type must be one of {spec.product_types}")
    chosen = product_type or spec.default_product
    requests = []
    varying = [item for item in variables if not item.time_invariant]
    invariant = [item for item in variables if item.time_invariant]
    for year, month, days, times in month_slices(hours):
        ids = [item.id for item in varying]
        for offset in range(0, len(ids), VARIABLES_PER_REQUEST):
            batch = ids[offset:offset + VARIABLES_PER_REQUEST]
            for area in areas:
                requests.append(build_request(batch, year, month, days, times, area, data_format, chosen))
    if invariant:
        first = hours[0]
        for item in invariant:
            for area in areas:
                requests.append(
                    build_request(
                        [item.id],
                        f"{first.year:04d}",
                        f"{first.month:02d}",
                        (f"{first.day:02d}",),
                        (f"{first.hour:02d}:00",),
                        area,
                        data_format,
                        chosen,
                    )
                )
    return requests


def _require_xarray():
    try:
        import xarray as xr
    except ImportError as exc:
        raise ImportError("ERA5 fetchers return an xarray.Dataset. Install them with: pip install 'giga-spatial[cds]'") from exc
    return xr


def open_climate(path, data_format: Optional[str] = None):
    """Open a CDS download as an in-memory dataset. Zip members are extracted beside the file."""

    xr = _require_xarray()
    path = Path(path)
    raw = path.read_bytes()[:8]
    if raw.startswith(b"PK"):
        with zipfile.ZipFile(path) as archive:
            members = [info for info in archive.infolist() if not info.is_dir()]
            if not members:
                raise ClimateError(f"{path} is an empty zip")
            member = max(members, key=lambda info: info.file_size)
            path = path.with_name(Path(member.filename).name)
            path.write_bytes(archive.read(member))
        raw = path.read_bytes()[:8]
    engine = None
    if data_format == "grib" or raw.startswith(b"GRIB"):
        engine = "cfgrib"
    try:
        if engine:
            dataset = xr.open_dataset(path, engine=engine)
            loaded = dataset.load()
            dataset.close()
        else:
            loaded = xr.load_dataset(path)
    except Exception as exc:
        if engine == "cfgrib":
            raise ImportError("Reading GRIB needs cfgrib and the ECMWF ecCodes library, in addition to giga-spatial[cds].") from exc
        raise
    return loaded


def _wrap_longitudes(dataset):
    if "longitude" not in dataset.coords and "lon" not in dataset.coords:
        return dataset
    name = "longitude" if "longitude" in dataset.coords else "lon"
    values = np.asarray(dataset[name].values, dtype=float)
    if values.size and np.nanmax(values) > 180:
        wrapped = np.where(values > 180, values - 360.0, values)
        order = np.argsort(wrapped)
        dataset = dataset.assign_coords({name: wrapped}).isel({name: order})
    return dataset


def standardize(dataset, spec: DatasetSpec, *, area_label: str, data_format: str, bounds):
    """One coordinate and attribute convention for NetCDF and GRIB."""

    renames = {}
    known = spec.by_id()
    for name in list(dataset.data_vars):
        if name in known:
            continue
        mapped = SHORT_NAMES.get(str(name))
        if mapped and mapped not in dataset.data_vars:
            renames[name] = mapped
    if renames:
        dataset = dataset.rename(renames)
    coord_renames = {}
    for old, new in (("lat", "latitude"), ("lon", "longitude"), ("valid_time", "time")):
        if old in dataset.coords or old in dataset.dims:
            coord_renames[old] = new
    if coord_renames:
        dataset = dataset.rename(coord_renames)
    dataset = _wrap_longitudes(dataset)
    for name, variable in known.items():
        if name not in dataset:
            continue
        dataset[name].attrs["long_name"] = variable.description
        dataset[name].attrs["accumulated"] = bool(variable.accumulated)
        if variable.units:
            dataset[name].attrs["units"] = variable.units
    west, south, east, north = bounds
    dataset.attrs["dataset_id"] = spec.dataset_id
    dataset.attrs["dataset_name"] = spec.name
    dataset.attrs["source_url"] = spec.documentation_url
    dataset.attrs["doi"] = spec.doi
    dataset.attrs["licence"] = "CC-BY-4.0"
    dataset.attrs["area"] = area_label
    dataset.attrs["spatial_coverage"] = f"west={west}, south={south}, east={east}, north={north}"
    dataset.attrs["temporal_resolution"] = "1 hour"
    dataset.attrs["spatial_resolution"] = f"{spec.resolution} degrees"
    dataset.attrs["data_format"] = data_format
    dataset.attrs["crs"] = "EPSG:4326"
    dataset.attrs["processing"] = (
        "Native CDS grid retained. Longitudes above 180 degrees are shifted into "
        "[-180, 180]. Cells outside the requested geometry are missing. "
        "The CDS area is only the download window."
    )
    if "time" in dataset.coords and dataset["time"].size:
        start = np.datetime_as_string(np.asarray(dataset["time"].values).reshape(-1)[0], unit="h")
        end = np.datetime_as_string(np.asarray(dataset["time"].values).reshape(-1)[-1], unit="h")
        dataset.attrs["start"] = start + "Z" if not str(start).endswith("Z") else start
        dataset.attrs["end"] = end + "Z" if not str(end).endswith("Z") else end
    return dataset


def clip(dataset, geometry: BaseGeometry, resolution: float):
    """Keep native cells that intersect ``geometry``. Interior misses stay missing."""

    if "longitude" not in dataset.coords or "latitude" not in dataset.coords:
        raise ClimateError("dataset has no latitude/longitude coordinates to clip")
    longitudes = np.asarray(dataset["longitude"].values, dtype=float)
    latitudes = np.asarray(dataset["latitude"].values, dtype=float)
    mask = intersecting_mask(longitudes, latitudes, resolution, geometry)
    if not mask.any():
        raise ClimateError("the geometry does not intersect the native grid")
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    subset = dataset.isel(
        latitude=slice(int(rows[0]), int(rows[-1]) + 1),
        longitude=slice(int(cols[0]), int(cols[-1]) + 1),
    )
    trimmed = mask[int(rows[0]): int(rows[-1]) + 1, int(cols[0]): int(cols[-1]) + 1]
    xr = _require_xarray()
    mask_da = xr.DataArray(trimmed, coords={"latitude": subset["latitude"], "longitude": subset["longitude"]}, dims=("latitude", "longitude"))
    return subset.where(mask_da)


def merge_climate(datasets):
    """Union datasets. The first finite value at a repeated timestamp is kept."""

    parts = [dataset for dataset in datasets if dataset is not None]
    if not parts:
        raise ClimateError("no data was retrieved")
    result = parts[0]
    for extra in parts[1:]:
        result = result.combine_first(extra)
    return result


def estimate_bytes(geometry, resolution: float, hours: int, variables: int) -> int:
    cells = 0
    for north, west, south, east in request_areas(geometry, resolution):
        cells += (int(round((east - west) / resolution)) + 1) * (int(round((north - south) / resolution)) + 1)
    return cells * max(hours, 1) * max(variables, 1) * 8


def _chunk(dataset):
    try:
        import dask  # noqa: F401
    except ImportError:
        return dataset
    if "time" not in dataset.dims:
        return dataset
    try:
        return dataset.chunk({"time": 24})
    except Exception:
        return dataset


class CDSRetriever:
    """Official cdsapi client. Credentials are never written into the request."""

    def __init__(self, url: str, key: str, timeout: float = 300):
        self.url = url or DEFAULT_API_URL
        self.key = key or ""
        self.timeout = timeout

    def latest_available(self, dataset_id: str) -> datetime:
        return latest_available(dataset_id, api_url=self.url)

    def retrieve(self, dataset_id: str, request: dict, target: Path) -> None:
        try:
            import cdsapi
        except ImportError as exc:
            raise ImportError("CDS downloads need the official client. Install it with: pip install 'giga-spatial[cds]'") from exc
        kwargs = {"quiet": True, "progress": False, "retry_max": 0, "timeout": self.timeout, "url": self.url}
        if self.key:
            kwargs["key"] = self.key
        client = cdsapi.Client(**kwargs)
        try:
            client.retrieve(dataset_id, request, str(target))
        except Exception as exc:
            raise ClimateRequestError(redact(str(exc), self.key)) from None


def retrieve_request(retriever, dataset_id: str, request: dict, target: Path, *, max_retries: int = 3) -> None:
    """Retrieve one request. A partial file is removed before the next attempt."""

    if max_retries < 1:
        raise ValueError("max_retries must be at least 1")
    for attempt in range(1, max_retries + 1):
        try:
            retriever.retrieve(dataset_id, request, target)
            if not target.exists() or target.stat().st_size <= 0:
                raise ClimateError("CDS returned an empty file")
            return
        except Exception as exc:
            if target.exists():
                try:
                    target.unlink()
                except OSError:
                    pass
            message = redact(str(exc))
            retryable = isinstance(exc, (TimeoutError, ConnectionError, requests.Timeout, requests.ConnectionError)) or any(
                token in message.lower() for token in ("timeout", "timed out", "429", "500", "502", "503", "504", "connection")
            )
            client_error = any(token in message.lower() for token in ("400", "401", "403", "404", "invalid", "required licence", "accept"))
            if client_error:
                retryable = False
            error = exc if isinstance(exc, ClimateError) else ClimateRequestError(message, variable=request["variable"][0] if len(request["variable"]) == 1 else None, retryable=retryable)
            if not retryable or attempt == max_retries:
                if len(request["variable"]) == 1:
                    raise ClimateRequestError(
                        f"{dataset_id} rejected {request['variable'][0]}: {message}",
                        variable=request["variable"][0],
                        retryable=retryable,
                    ) from exc
                raise error from exc
            time.sleep(min(2 ** (attempt - 1), 30))


def _split_failure(exc: BaseException) -> bool:
    text = str(exc).lower()
    return any(token in text for token in ("400", "invalid", "not compatible", "combination", "too large", "cost"))


def _retrieve_batch(retriever, spec, request, directory: Path, index: int, max_retries: int):
    if len(request["variable"]) > 1:
        try:
            return _retrieve_one(retriever, spec, request, directory, index, max_retries)
        except ClimateRequestError as exc:
            if not _split_failure(exc):
                raise
            left = dict(request)
            names = list(request["variable"])
            mid = max(1, len(names) // 2)
            left["variable"] = names[:mid]
            right = dict(request)
            right["variable"] = names[mid:]
            return _retrieve_batch(retriever, spec, left, directory, index, max_retries) + _retrieve_batch(
                retriever, spec, right, directory, index + mid, max_retries
            )
    return _retrieve_one(retriever, spec, request, directory, index, max_retries)


def _retrieve_one(retriever, spec, request, directory: Path, index: int, max_retries: int):
    target = directory / f"request-{index}.download"
    retrieve_request(retriever, spec.dataset_id, request, target, max_retries=max_retries)
    return [open_climate(target, request["data_format"])]


def fetch_dataset(spec: DatasetSpec, area, variables, time=None, *, start=None, end=None, year=None, month=None, day=None, hour=None, full_history=False, data_format="netcdf", product_type=None, retriever=None, max_bytes=DEFAULT_MAX_BYTES, api_url=None, api_key=None, check_availability=True, max_retries=3):
    """Retrieve ``spec`` for an area and return one in-memory dataset. Nothing is written to a data store."""

    if data_format not in ("netcdf", "grib"):
        raise ValueError("data_format must be 'netcdf' or 'grib'")
    _require_xarray()
    geometry, country_id, country_name = resolve_area(area)
    chosen = select_variables(spec, variables)
    if retriever is None:
        url, key = resolve_credentials(api_url, api_key)
        if not key:
            raise ClimateAuthError(
                "CDS API key is not configured. Set CDS_API_KEY or CDSAPI_KEY, or add key: to ~/.cdsapirc. "
                "Accept the dataset licence on the CDS website."
            )
        retriever = CDSRetriever(url, key)
    latest = None
    if check_availability or full_history:
        latest = retriever.latest_available(spec.dataset_id) if hasattr(retriever, "latest_available") else latest_available(spec.dataset_id, api_url=api_url)
    hours = selection_hours(
        spec, time=time, start=start, end=end, year=year, month=month, day=day, hour=hour, full_history=full_history, latest=latest
    )
    if latest is not None:
        too_late = [item for item in hours if item > latest]
        if too_late:
            raise ClimateAvailabilityError(
                f"{spec.dataset_id} publishes through {latest.strftime('%Y-%m-%dT%H:%M:%SZ')}; "
                f"{too_late[0].strftime('%Y-%m-%dT%H:%M:%SZ')} is not available yet"
            )
    needed = estimate_bytes(geometry, spec.resolution, len(hours), len(chosen))
    if needed > max_bytes:
        raise ClimateLimitError(
            f"this request is about {needed / 1024 ** 2:.0f} MiB and the limit is {max_bytes / 1024 ** 2:.0f} MiB. "
            "Request a smaller area, fewer variables, or a shorter time, or raise max_bytes."
        )
    areas = request_areas(geometry, spec.resolution)
    requests = plan_requests(spec, chosen, hours, areas, data_format, product_type)
    opened = []
    with tempfile.TemporaryDirectory(prefix="cds-") as temporary:
        directory = Path(temporary)
        for index, request in enumerate(requests):
            opened.extend(_retrieve_batch(retriever, spec, request, directory, index, max_retries))
    dataset = merge_climate(opened)
    bounds = geometry.bounds
    label = country_id or "geometry"
    if country_name:
        label = f"{country_name} ({country_id})"
    dataset = standardize(dataset, spec, area_label=label, data_format=data_format, bounds=bounds)
    dataset = clip(dataset, geometry, spec.resolution)
    longitudes = np.asarray(dataset["longitude"].values, dtype=float)
    latitudes = np.asarray(dataset["latitude"].values, dtype=float)
    dataset.attrs["spatial_coverage"] = (
        f"west={float(np.min(longitudes))}, south={float(np.min(latitudes))}, "
        f"east={float(np.max(longitudes))}, north={float(np.max(latitudes))}"
    )
    dataset.attrs["variables"] = ",".join(item.id for item in chosen)
    dataset.attrs["retrieved_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if latest is not None:
        dataset.attrs["latest_available"] = latest.strftime("%Y-%m-%dT%H:%M:%SZ")
    if dataset.nbytes > max_bytes:
        raise ClimateLimitError(
            f"the retrieved dataset is {dataset.nbytes / 1024 ** 2:.0f} MiB and the limit is {max_bytes / 1024 ** 2:.0f} MiB"
        )
    return _chunk(dataset)
