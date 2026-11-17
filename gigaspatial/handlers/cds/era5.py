"""
ERA5 hourly data on single levels, from the Copernicus Climate Data Store.

Dataset id: ``reanalysis-era5-single-levels``.
Documentation: https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=overview
Licence: CC BY 4.0. DOI: 10.24381/cds.adbb2d47.

The handler name is ``cds-era5``. A country is resolved with
``AdminBoundaries``. CDS is asked for the country's bounding boxes only, and
the stored grid is clipped to the country geometry at the native 0.25 degree
spacing. Variables are the dataset's own names. History starts in 1940.
``update`` reads this dataset's catalogue end, which is not assumed to match
ERA5-Land.
"""

from __future__ import annotations

import argparse
import calendar
import hashlib
import json
import math
import os
import re
import tempfile
import threading
import time
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, ClassVar, Iterable, Mapping, Optional, Sequence, Union

import numpy as np
import requests
from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass as pydantic_dataclass
from shapely import wkt
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
from gigaspatial.handlers.base.source import SourceKind, classify_source


ALL_TIMES = tuple(f"{hour:02d}:00" for hour in range(24))


@dataclass(frozen=True)
class TimeSlice:
    """One CDS time request: a single month, and the days and times to cross."""

    year: str
    month: str
    days: tuple[str, ...]
    times: tuple[str, ...]


def as_utc_hour(value: datetime) -> datetime:
    """Floor a timestamp to the UTC hour CDS uses."""

    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    else:
        value = value.astimezone(timezone.utc)
    return value.replace(minute=0, second=0, microsecond=0)


def parse_timestamp(value: str) -> datetime:
    """Parse an ISO timestamp from a catalogue or a status file."""

    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    return as_utc_hour(parsed)


def format_timestamp(value: datetime) -> str:
    return as_utc_hour(value).strftime("%Y-%m-%dT%H:%M:%SZ")


def iter_hours(start: datetime, end: datetime) -> list[datetime]:
    """Inclusive hourly range."""

    start = as_utc_hour(start)
    end = as_utc_hour(end)
    if end < start:
        return []
    hours = []
    cursor = start
    while cursor <= end:
        hours.append(cursor)
        cursor += timedelta(hours=1)
    return hours


def slice_hours(time_slice: TimeSlice) -> list[datetime]:
    """Expand one slice. Days and times are a cartesian product."""

    hours = []
    for day in time_slice.days:
        for clock in time_slice.times:
            hours.append(
                datetime(
                    int(time_slice.year),
                    int(time_slice.month),
                    int(day),
                    int(clock[:2]),
                    tzinfo=timezone.utc,
                )
            )
    return hours


def slices_from_hours(hours: Iterable[datetime]) -> tuple[TimeSlice, ...]:
    """Group hours into CDS requests whose day/time product matches them."""

    unique = sorted({as_utc_hour(hour) for hour in hours})
    if not unique:
        return ()
    by_month: dict[tuple[int, int], dict[int, list[datetime]]] = {}
    for hour in unique:
        by_month.setdefault((hour.year, hour.month), {}).setdefault(hour.day, []).append(hour)
    slices = []
    for year, month in sorted(by_month):
        by_times: dict[tuple[str, ...], list[str]] = {}
        for day, day_hours in by_month[(year, month)].items():
            times = tuple(f"{hour.hour:02d}:00" for hour in day_hours)
            by_times.setdefault(times, []).append(f"{day:02d}")
        for times, days in by_times.items():
            slices.append(
                TimeSlice(
                    year=str(year),
                    month=f"{month:02d}",
                    days=tuple(sorted(days)),
                    times=times,
                )
            )
    return tuple(slices)


def slice_token(time_slice: TimeSlice) -> str:
    """Stable file token. A complete month is ``YYYY-MM``."""

    year = int(time_slice.year)
    month = int(time_slice.month)
    last_day = calendar.monthrange(year, month)[1]
    all_days = tuple(f"{day:02d}" for day in range(1, last_day + 1))
    if time_slice.days == all_days and time_slice.times == ALL_TIMES:
        return f"{time_slice.year}-{time_slice.month}"
    raw = f"{time_slice.year}-{time_slice.month}_d{_compact(time_slice.days)}_h{_compact(tuple(clock[:2] for clock in time_slice.times))}"
    if len(raw) <= 120:
        return raw
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]
    return f"{time_slice.year}-{time_slice.month}_{digest}"


def _compact(parts: Sequence[str]) -> str:
    numbers = [int(part) for part in parts]
    ranges = []
    start = previous = numbers[0]
    for number in numbers[1:]:
        if number == previous + 1:
            previous = number
            continue
        ranges.append((start, previous))
        start = previous = number
    ranges.append((start, previous))
    bits = []
    for left, right in ranges:
        bits.append(f"{left:02d}" if left == right else f"{left:02d}-{right:02d}")
    return "_".join(bits)


def _parse_bound(text: str, *, is_end: bool) -> datetime:
    raw = text.strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    if len(raw) == 4 and raw.isdigit():
        year = int(raw)
        if is_end:
            return datetime(year, 12, 31, 23, tzinfo=timezone.utc)
        return datetime(year, 1, 1, tzinfo=timezone.utc)
    if len(raw) == 7 and raw[4] == "-":
        year, month = int(raw[:4]), int(raw[5:7])
        if is_end:
            last = calendar.monthrange(year, month)[1]
            return datetime(year, month, last, 23, tzinfo=timezone.utc)
        return datetime(year, month, 1, tzinfo=timezone.utc)
    if len(raw) == 10 and raw[4] == "-" and "T" not in raw:
        year, month, day = (int(part) for part in raw.split("-"))
        return datetime(year, month, day, 23 if is_end else 0, tzinfo=timezone.utc)
    return as_utc_hour(datetime.fromisoformat(raw))


def _number_set(values: Optional[Sequence[str]]) -> Optional[set[int]]:
    if not values:
        return None
    return {int(value) for value in values}


def _hour_set(values: Optional[Sequence[str]]) -> Optional[set[int]]:
    if not values:
        return None
    return {int(value[:2]) for value in values}


def selection_hours(options, *, start_year: int, latest: Optional[datetime]) -> list[datetime]:
    """
    Resolve a user's time selection to UTC hours.

    ``full_history`` runs from the dataset start through ``latest``. Any other
    selection uses ``start``/``end`` and/or year, month, day, and hour. Hours
    before ``start_year`` are rejected. An explicit range is not clipped to
    CDS availability; the update path does that separately.
    """

    partial = any(
        getattr(options, name)
        for name in ("start", "end", "year", "month", "day", "hour")
    )
    if options.full_history:
        if partial:
            raise ValueError(
                "full_history cannot be combined with start, end, year, month, day, or hour"
            )
        if latest is None:
            raise ValueError("full_history requires the latest timestamp published by CDS")
        start = datetime(start_year, 1, 1, tzinfo=timezone.utc)
        end = as_utc_hour(latest)
        hours = iter_hours(start, end)
    else:
        hours = _bounded_hours(options)
    if not hours and not partial:
        raise ValueError(
            "Pass start and end, or year, month, day, and hour, or full_history=True. "
            "The historical archive is not downloaded unless you ask for it."
        )
    if not hours:
        raise ValueError("time selection does not include any hour")
    too_early = [hour for hour in hours if hour.year < start_year]
    if too_early:
        raise ValueError(
            f"this dataset starts in {start_year}; "
            f"{format_timestamp(too_early[0])} is before that"
        )
    return hours


def _bounded_hours(options) -> list[datetime]:
    years = _number_set(options.year)
    months = _number_set(options.month)
    days = _number_set(options.day)
    clocks = _hour_set(options.hour)
    if options.start or options.end:
        if not options.start or not options.end:
            raise ValueError("pass both start and end")
        start = _parse_bound(options.start, is_end=False)
        end = _parse_bound(options.end, is_end=True)
        if end < start:
            raise ValueError("end is before start")
        chosen = []
        for hour in iter_hours(start, end):
            if years is not None and hour.year not in years:
                continue
            if months is not None and hour.month not in months:
                continue
            if days is not None and hour.day not in days:
                continue
            if clocks is not None and hour.hour not in clocks:
                continue
            chosen.append(hour)
        return chosen
    if not years:
        return []
    chosen = []
    for year in sorted(years):
        month_list = sorted(months) if months is not None else list(range(1, 13))
        for month in month_list:
            if not 1 <= month <= 12:
                raise ValueError(f"month must be 01..12, got {month}")
            last = calendar.monthrange(year, month)[1]
            day_list = sorted(days) if days is not None else list(range(1, last + 1))
            for day in day_list:
                if day > last or day < 1:
                    continue
                hour_list = sorted(clocks) if clocks is not None else list(range(24))
                for hour in hour_list:
                    chosen.append(datetime(year, month, day, hour, tzinfo=timezone.utc))
    return chosen


def forward_hours(last_downloaded: datetime, latest: datetime) -> list[datetime]:
    """Hours strictly after ``last_downloaded`` through ``latest``, inclusive."""

    start = as_utc_hour(last_downloaded) + timedelta(hours=1)
    end = as_utc_hour(latest)
    return iter_hours(start, end)


def hours_in_period(period: dict) -> list[datetime]:
    return iter_hours(parse_timestamp(period["start"]), parse_timestamp(period["end"]))

ALL_VARIABLES = "__all__"
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


def _variable(record: Mapping[str, Any]) -> Variable:
    units, accumulated = infer_units(record["id"])
    groups = tuple(record.get("groups") or ())
    return Variable(
        id=record["id"],
        description=record.get("description") or record["id"],
        units=units,
        accumulated=accumulated,
        time_invariant="Invariant" in groups,
        groups=groups,
    )


_CATALOGUES: dict[str, list] = {}


def variable_records(rows) -> list[dict]:
    """Variable id, CDS form label, and form groups."""

    return [
        {"id": name, "description": description, "groups": list(groups)}
        for name, description, groups in rows
    ]


def bundled_document(dataset_id: str) -> dict:
    """Return the bundled form snapshot for ``dataset_id``."""

    try:
        variables = _CATALOGUES[dataset_id]
    except KeyError as exc:
        raise ValueError(f"no bundled CDS catalogue for {dataset_id}") from exc
    return {"dataset_id": dataset_id, "variables": variables}


def bundled_records(dataset_id: str) -> list[dict]:
    return list(bundled_document(dataset_id)["variables"])


def variables_from_form(form: Sequence[Mapping[str, Any]]) -> list[dict]:
    """
    Read variable id, label, and group from a CDS download-form document.

    The form is the JSON document linked from the catalogue collection with
    ``rel="form"``.
    """

    widget = next(item for item in form if item.get("name") == "variable")
    grouped: dict[str, list[str]] = {}
    labels: dict[str, str] = {}
    order: list[str] = []
    for group in widget["details"]["groups"]:
        for value in group["values"]:
            grouped.setdefault(value, [])
            if group["label"] not in grouped[value]:
                grouped[value].append(group["label"])
            labels.setdefault(value, group["labels"][value])
            if value not in order:
                order.append(value)
    return [
        {"id": value, "description": labels[value], "groups": grouped[value]}
        for value in order
    ]


def merge_records(bundled: Sequence[Mapping[str, Any]], discovered: Sequence[Mapping[str, Any]]) -> list[dict]:
    """Keep bundled order, refresh labels from CDS, and append new ids."""

    by_id = {item["id"]: dict(item) for item in bundled}
    order = [item["id"] for item in bundled]
    for item in discovered:
        if item["id"] not in by_id:
            order.append(item["id"])
        by_id[item["id"]] = {
            "id": item["id"],
            "description": item.get("description") or item["id"],
            "groups": list(item.get("groups") or ()),
        }
    return [by_id[key] for key in order]


def discover_records(dataset_id: str, api_url: str, getter) -> list[dict]:
    """
    Download the live form for ``dataset_id``.

    ``getter`` accepts a URL and returns the parsed JSON document. The
    collection document's ``form`` link is followed, so a new variable on the
    form is picked up without a new catalogue file.
    """

    collection_url = f"{api_url.rstrip('/')}/catalogue/v1/collections/{dataset_id}"
    collection = getter(collection_url)
    form_url = next(link["href"] for link in collection.get("links", []) if link.get("rel") == "form")
    form = getter(form_url)
    return variables_from_form(form)


def load_variables(
    dataset_id: str,
    *,
    refresh: bool = False,
    api_url: str = "https://cds.climate.copernicus.eu/api",
    getter=None,
) -> tuple[Variable, ...]:
    """
    Return the variable catalogue.

    ``refresh=False`` reads only the bundled snapshot, so tests and offline
    use need no network. ``refresh=True`` merges the live form when ``getter``
    (or an HTTP GET) succeeds, and falls back to the snapshot otherwise.
    """

    records = bundled_records(dataset_id)
    if refresh:
        fetch = getter or _http_get_json
        try:
            records = merge_records(records, discover_records(dataset_id, api_url, fetch))
        except Exception:
            # The bundled form remains usable when CDS cannot be reached.
            pass
    return tuple(_variable(record) for record in records)


def _http_get_json(url: str) -> dict:
    import requests

    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return response.json()


def select_variables(catalogue: Iterable[Variable], requested: Optional[Sequence[str]]) -> tuple[Variable, ...]:
    """
    Select variables by CDS name.

    ``("__all__",)`` selects the whole catalogue. An empty request is refused
    so a call cannot download every variable by accident. Order follows the
    caller, with duplicates removed.
    """

    by_id = {variable.id: variable for variable in catalogue}
    if not requested:
        raise ValueError(
            "Pass variables='all' or one or more CDS variable names. "
            "The full variable list is not selected unless you ask for it."
        )
    if tuple(requested) == (ALL_VARIABLES,):
        return tuple(by_id.values())
    unknown = [name for name in requested if name not in by_id]
    if unknown:
        sample = ", ".join(repr(name) for name in unknown[:5])
        raise ValueError(
            f"unknown CDS variable(s): {sample}. "
            f"The catalogue has {len(by_id)} variables; "
            "use the dataset's own names, or refresh the catalogue."
        )
    selected = []
    seen = set()
    for name in requested:
        if name in seen:
            continue
        seen.add(name)
        selected.append(by_id[name])
    return tuple(selected)

def _round_grid(value: float, resolution: float) -> float:
    steps = round(value / resolution)
    return round(steps * resolution, 6)


def snap_bounds(bounds: Sequence[float], resolution: float) -> tuple[float, float, float, float]:
    """Expand a lon/lat box outward onto the native grid. Returns west, south, east, north."""

    minx, miny, maxx, maxy = bounds
    west = math.floor(minx / resolution) * resolution
    south = math.floor(miny / resolution) * resolution
    east = math.ceil(maxx / resolution) * resolution
    north = math.ceil(maxy / resolution) * resolution
    if east <= west:
        east = west + resolution
    if north <= south:
        north = south + resolution
    west = max(-180.0, west)
    east = min(180.0, east)
    south = max(-90.0, south)
    north = min(90.0, north)
    return (
        _round_grid(west, resolution),
        _round_grid(south, resolution),
        _round_grid(east, resolution),
        _round_grid(north, resolution),
    )


def geometry_parts(geometry: BaseGeometry) -> list[BaseGeometry]:
    """Polygons of a geometry, or the geometry itself when it is not a collection."""

    if geometry.is_empty:
        raise ValueError("source geometry is empty")
    kind = geometry.geom_type
    if kind == "Polygon":
        return [geometry]
    if kind == "MultiPolygon":
        return [part for part in geometry.geoms if not part.is_empty]
    if kind == "GeometryCollection":
        parts: list[BaseGeometry] = []
        for part in geometry.geoms:
            parts.extend(geometry_parts(part))
        return parts
    return [geometry]


def _merge_boxes(boxes: list[tuple[float, float, float, float]], resolution: float):
    """Merge boxes that touch. Distant parts, including across the antimeridian, stay separate."""

    pending = sorted(boxes, key=lambda item: (item[0], item[1]))
    merged: list[list[float]] = []
    for west, south, east, north in pending:
        placed = False
        for item in merged:
            lon_gap = west - item[2]
            lat_gap = max(item[1] - north, south - item[3])
            lon_close = lon_gap <= resolution
            lat_close = lat_gap <= resolution
            overlaps = west <= item[2] + resolution and east >= item[0] - resolution
            if overlaps and lat_close and lon_close:
                item[0] = min(item[0], west)
                item[1] = min(item[1], south)
                item[2] = max(item[2], east)
                item[3] = max(item[3], north)
                placed = True
                break
        if not placed:
            merged.append([west, south, east, north])
    changed = True
    while changed:
        changed = False
        again: list[list[float]] = []
        for item in merged:
            placed = False
            for other in again:
                overlaps = item[0] <= other[2] + resolution and item[2] >= other[0] - resolution
                lat_gap = max(other[1] - item[3], item[1] - other[3])
                if overlaps and lat_gap <= resolution:
                    other[0] = min(other[0], item[0])
                    other[1] = min(other[1], item[1])
                    other[2] = max(other[2], item[2])
                    other[3] = max(other[3], item[3])
                    placed = True
                    changed = True
                    break
            if not placed:
                again.append(item)
        merged = again
    return [tuple(round(value, 6) for value in item) for item in merged]


def request_areas(geometry: BaseGeometry, resolution: float) -> tuple[tuple[float, float, float, float], ...]:
    """
    CDS ``area`` values for a geometry.

    Each value is ``(north, west, south, east)`` in degrees, snapped out to
    ``resolution``. Separate parts are not joined into one box when a gap
    remains between them.
    """

    boxes = [snap_bounds(part.bounds, resolution) for part in geometry_parts(geometry)]
    merged = _merge_boxes(boxes, resolution)
    return tuple(
        (north, west, south, east) for west, south, east, north in merged
    )


def wrap_longitudes(longitudes: np.ndarray, arrays: list[np.ndarray]) -> tuple[np.ndarray, list[np.ndarray]]:
    """Convert longitudes above 180 to the -180..180 range and keep array columns aligned."""

    longitudes = np.asarray(longitudes, dtype=float)
    if longitudes.size == 0 or np.nanmax(longitudes) <= 180:
        return longitudes, arrays
    wrapped = np.where(longitudes > 180, longitudes - 360.0, longitudes)
    order = np.argsort(wrapped)
    return wrapped[order], [np.take(array, order, axis=-1) for array in arrays]


def intersecting_mask(
    longitudes: np.ndarray,
    latitudes: np.ndarray,
    resolution: float,
    geometry: BaseGeometry,
) -> np.ndarray:
    """True where the native cell intersects ``geometry``. Shape is (latitude, longitude)."""

    from shapely import box
    from shapely import intersects

    half = resolution / 2.0
    lon, lat = np.meshgrid(
        np.asarray(longitudes, dtype=float),
        np.asarray(latitudes, dtype=float),
    )
    cells = box(lon - half, lat - half, lon + half, lat + half)
    return np.asarray(intersects(cells, geometry), dtype=bool)

DEFAULT_API_URL = "https://cds.climate.copernicus.eu/api"
_SECRET_QUERY = re.compile(r"(?i)(key|token|private-token)(=|:)\s*([^\s&]+)")


class CDSRequestError(RuntimeError):
    """A CDS retrieve failed."""

    def __init__(
        self,
        message: str,
        *,
        retryable: bool = False,
        not_available: bool = False,
        licence: bool = False,
    ):
        super().__init__(message)
        self.retryable = retryable
        self.not_available = not_available
        self.licence = licence


class CDSAvailabilityError(RuntimeError):
    """The catalogue did not publish a usable end timestamp."""


class CDSIntegrityError(RuntimeError):
    """A download was incomplete, duplicated, or empty, and was not stored."""


def redact(message: str, secret: str = "") -> str:
    """Remove API keys from a message before it is logged or raised."""

    text = message or ""
    if secret:
        text = text.replace(secret, "***")
    return _SECRET_QUERY.sub(r"\1\2***", text)


def classify_message(message: str) -> CDSRequestError:
    """Turn a CDS client message into a retryable, licence, or availability error."""

    lower = message.lower()
    licence = "licence" in lower or "license" in lower or ("terms" in lower and "accept" in lower)
    not_available = (
        "not available" in lower
        or "no data is available" in lower
        or "none of the data" in lower
    )
    client_error = any(
        token in lower
        for token in ("400", "401", "403", "404", "invalid request", "bad request")
    )
    transient = any(
        token in lower
        for token in (
            "timeout",
            "timed out",
            "429",
            "500",
            "502",
            "503",
            "504",
            "temporarily",
            "connection reset",
            "connection aborted",
        )
    )
    retryable = transient and not licence and not not_available and not client_error
    return CDSRequestError(
        message,
        retryable=retryable,
        not_available=not_available,
        licence=licence,
    )


def is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, CDSRequestError):
        return exc.retryable
    if isinstance(exc, CDSIntegrityError):
        return False
    if isinstance(exc, (TimeoutError, ConnectionError, requests.Timeout, requests.ConnectionError)):
        return True
    return classify_message(str(exc)).retryable


def retrieve_with_retry(
    retriever,
    dataset_id: str,
    request: dict,
    target: Path,
    *,
    max_retries: int,
    sleep=None,
) -> None:
    """
    Call ``retriever.retrieve`` until it succeeds or the error is not retryable.

    The target file is removed when an attempt fails, so a partial file is not
    reused.
    """

    pause = time.sleep if sleep is None else sleep

    if max_retries < 1:
        raise ValueError("max_retries must be at least 1")
    last: Optional[BaseException] = None
    for attempt in range(1, max_retries + 1):
        try:
            retriever.retrieve(dataset_id, request, target)
            return
        except Exception as exc:
            if target.exists():
                try:
                    target.unlink()
                except OSError:
                    pass
            error = exc if isinstance(exc, CDSRequestError) else classify_message(redact(str(exc)))
            if isinstance(exc, (TimeoutError, ConnectionError, requests.Timeout, requests.ConnectionError)):
                error = CDSRequestError(redact(str(exc)), retryable=True)
            last = error
            if not is_retryable(error) or attempt == max_retries:
                raise error from exc
            pause(min(2 ** (attempt - 1), 30))
    if last is not None:
        raise last


def catalogue_collection_url(api_url: str, dataset_id: str) -> str:
    return f"{api_url.rstrip('/')}/catalogue/v1/collections/{dataset_id}"


def latest_from_collection(collection: dict, dataset_id: str) -> datetime:
    """
    Read the catalogue temporal extent.

    The end timestamp is whatever CDS publishes for this dataset. No fixed
    publication delay is applied. A missing end is an error.
    """

    try:
        end = collection["extent"]["temporal"]["interval"][0][1]
    except (KeyError, IndexError, TypeError) as exc:
        raise CDSAvailabilityError(f"{dataset_id} catalogue has no temporal extent") from exc
    if not end:
        raise CDSAvailabilityError(
            f"{dataset_id} catalogue does not publish an end timestamp"
        )
    return parse_timestamp(end)


def fetch_latest_available(dataset_id: str, api_url: str, getter=None) -> datetime:
    """GET the collection document and return its temporal end."""

    fetch = getter or _get_json
    payload = fetch(catalogue_collection_url(api_url, dataset_id))
    return latest_from_collection(payload, dataset_id)


def _get_json(url: str) -> dict:
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return response.json()


def read_cdsapirc(path: Optional[Path] = None) -> tuple[str, str]:
    """Read ``url`` and ``key`` from the CDS client file. Missing file yields empty strings."""

    target = path or (Path.home() / ".cdsapirc")
    if not target.is_file():
        return "", ""
    url = ""
    key = ""
    try:
        lines = target.read_text(encoding="utf-8").splitlines()
    except OSError:
        return "", ""
    for line in lines:
        if ":" not in line:
            continue
        name, value = line.split(":", 1)
        if name.strip() == "url":
            url = value.strip()
        elif name.strip() == "key":
            key = value.strip()
    return url, key


def resolve_credentials(
    api_url: Optional[str] = None,
    api_key: Optional[str] = None,
    *,
    cdsapirc: Optional[Path] = None,
) -> tuple[str, str]:
    """
    Resolve the CDS API URL and key.

    Explicit values win, then ``CDSAPI_URL`` / ``CDSAPI_KEY`` and
    ``CDS_API_URL`` / ``CDS_API_KEY``, then ``~/.cdsapirc``. The URL falls
    back to the official API. An empty key is returned when none is configured;
    it is never invented.
    """

    file_url, file_key = read_cdsapirc(cdsapirc)
    env_url = os.environ.get("CDSAPI_URL") or os.environ.get("CDS_API_URL") or ""
    env_key = os.environ.get("CDSAPI_KEY") or os.environ.get("CDS_API_KEY") or ""
    url = api_url or env_url or file_url or DEFAULT_API_URL
    key = api_key or env_key or file_key or ""
    return url, key


class CdsapiRetriever:
    """Official ``cdsapi`` client plus the public catalogue for availability."""

    def __init__(self, url: str, key: str, timeout: float):
        self.url = url or DEFAULT_API_URL
        self.key = key or ""
        self.timeout = timeout

    def retrieve(self, dataset_id: str, request: dict, target: Path) -> None:
        try:
            import cdsapi
        except ImportError as exc:
            raise ImportError(
                "CDS downloads need the official client. "
                "Install it with: pip install 'giga-spatial[cds]'"
            ) from exc
        kwargs = {
            "quiet": True,
            "progress": False,
            "retry_max": 0,
            "timeout": self.timeout,
            "url": self.url,
        }
        if self.key:
            kwargs["key"] = self.key
        client = cdsapi.Client(**kwargs)
        try:
            client.retrieve(dataset_id, request, str(target))
        except Exception as exc:
            raise classify_message(redact(str(exc), self.key)) from None

    def latest_available(self, dataset_id: str) -> datetime:
        return fetch_latest_available(dataset_id, self.url)

EPOCH = np.datetime64("1970-01-01T00:00:00")
PROCESSING_NOTE = (
    "Native CDS grid retained. Longitudes above 180 degrees are shifted into "
    "[-180, 180]. Cells that do not intersect the country geometry are missing. "
    "The request bounding box is the CDS download window. Accumulated variables "
    "keep the hourly values CDS publishes; a repeated timestamp keeps the stored value."
)


@dataclass
class GridCube:
    """Hourly values on the native latitude/longitude grid."""

    data: dict
    times: np.ndarray
    latitudes: np.ndarray
    longitudes: np.ndarray
    attrs: dict = field(default_factory=dict)
    variable_attrs: dict = field(default_factory=dict)


def dataset_chunks() -> Optional[dict]:
    """Dask chunks for the time dimension, when dask is installed."""

    if not _dask_available():
        return None
    return {"time": 24}


def _dask_available() -> bool:
    try:
        import dask  # noqa: F401
    except ImportError:
        return False
    return True


def _text(value) -> str:
    if isinstance(value, bytes):
        return value.decode()
    return str(value)


def times_to_hours(times: np.ndarray) -> np.ndarray:
    stamps = np.asarray(times, dtype="datetime64[h]")
    return (stamps - EPOCH).astype("timedelta64[h]").astype(np.float64)


def hours_to_times(hours: np.ndarray) -> np.ndarray:
    return EPOCH + np.atleast_1d(np.asarray(hours, dtype=np.float64)).astype("timedelta64[h]")


def stamp_key(value) -> np.datetime64:
    if isinstance(value, np.datetime64):
        return np.datetime64(value, "h")
    if isinstance(value, datetime):
        hour = as_utc_hour(value).replace(tzinfo=None)
        return np.datetime64(hour, "h")
    return np.datetime64(value, "h")


def duplicate_timestamps(times) -> list:
    seen = set()
    duplicates = []
    for value in np.atleast_1d(times):
        key = stamp_key(value)
        if key in seen:
            duplicates.append(key)
        seen.add(key)
    return duplicates


def missing_timestamps(expected, actual) -> list[datetime]:
    have = {stamp_key(value) for value in np.atleast_1d(actual)}
    missing = []
    for value in expected:
        if stamp_key(value) not in have:
            missing.append(as_utc_hour(value) if isinstance(value, datetime) else value)
    return missing


def write_cube(path: Union[str, Path], cube: GridCube) -> None:
    """Write a cube as NetCDF classic so it can be read without the CDS extra."""

    from scipy.io import netcdf_file

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = netcdf_file(path, "w", version=2)
    try:
        times = np.atleast_1d(np.asarray(cube.times, dtype="datetime64[h]"))
        latitudes = np.atleast_1d(np.asarray(cube.latitudes, dtype=float))
        longitudes = np.atleast_1d(np.asarray(cube.longitudes, dtype=float))
        handle.createDimension("time", int(times.size))
        handle.createDimension("latitude", int(latitudes.size))
        handle.createDimension("longitude", int(longitudes.size))
        time_var = handle.createVariable("time", "f8", ("time",))
        time_var[:] = times_to_hours(times)
        time_var.units = "hours since 1970-01-01 00:00:00"
        lat_var = handle.createVariable("latitude", "f8", ("latitude",))
        lat_var[:] = latitudes
        lat_var.units = "degrees_north"
        lon_var = handle.createVariable("longitude", "f8", ("longitude",))
        lon_var[:] = longitudes
        lon_var.units = "degrees_east"
        for name, values in cube.data.items():
            variable = handle.createVariable(name, "f8", ("time", "latitude", "longitude"))
            variable[:] = np.asarray(values, dtype=float)
            attrs = cube.variable_attrs.get(name, {})
            if attrs.get("units"):
                variable.units = str(attrs["units"])
            if attrs.get("description"):
                variable.long_name = str(attrs["description"])[:200]
            variable.cds_name = name
        for key in ("dataset_id", "handler_name", "processing", "crs"):
            if cube.attrs.get(key):
                setattr(handle, key, str(cube.attrs[key])[:500])
    finally:
        handle.close()


def read_cube(path: Union[str, Path]) -> GridCube:
    """Read a cube written by :func:`write_cube` or another NetCDF classic file."""

    from scipy.io import netcdf_file

    handle = netcdf_file(path, "r", mmap=False)
    try:
        times = hours_to_times(handle.variables["time"][:])
        latitudes = np.atleast_1d(np.array(handle.variables["latitude"][:], dtype=float))
        longitudes = np.atleast_1d(np.array(handle.variables["longitude"][:], dtype=float))
        data = {}
        variable_attrs = {}
        for name, variable in handle.variables.items():
            if name in {"time", "latitude", "longitude"}:
                continue
            data[name] = np.array(variable[:], dtype=float)
            units = getattr(variable, "units", None)
            description = getattr(variable, "long_name", None)
            variable_attrs[name] = {
                "units": _text(units) if units is not None else None,
                "description": _text(description) if description is not None else name,
            }
        attrs = {}
        for key in ("dataset_id", "handler_name", "processing", "crs"):
            if hasattr(handle, key):
                attrs[key] = _text(getattr(handle, key))
        return GridCube(
            data=data,
            times=np.atleast_1d(times).astype("datetime64[h]"),
            latitudes=latitudes,
            longitudes=longitudes,
            attrs=attrs,
            variable_attrs=variable_attrs,
        )
    finally:
        handle.close()


def clip_cube(cube: GridCube, geometry, resolution: float) -> GridCube:
    """
    Keep native cells that intersect ``geometry``.

    Edge rows and columns that miss the geometry are removed. A cell that misses
    the geometry between cells that hit it stays in the grid as a missing value,
    so the spacing of the original grid is unchanged. Values of kept cells are
    not resampled.
    """

    names = list(cube.data)
    longitudes, arrays = wrap_longitudes(
        np.asarray(cube.longitudes, dtype=float),
        [np.asarray(cube.data[name], dtype=float) for name in names],
    )
    latitudes = np.asarray(cube.latitudes, dtype=float)
    mask = intersecting_mask(longitudes, latitudes, resolution, geometry)
    if not mask.any():
        raise CDSIntegrityError("the geometry does not intersect the native grid")
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    row_slice = slice(int(rows[0]), int(rows[-1]) + 1)
    col_slice = slice(int(cols[0]), int(cols[-1]) + 1)
    trimmed = mask[row_slice, col_slice]
    clipped = {}
    for name, values in zip(names, arrays):
        piece = np.array(values[:, row_slice, col_slice], dtype=float, copy=True)
        piece[:, ~trimmed] = np.nan
        clipped[name] = piece
    return GridCube(
        data=clipped,
        times=np.atleast_1d(np.asarray(cube.times, dtype="datetime64[h]")),
        latitudes=latitudes[row_slice],
        longitudes=np.asarray(longitudes, dtype=float)[col_slice],
        attrs=dict(cube.attrs),
        variable_attrs={key: dict(value) for key, value in cube.variable_attrs.items()},
    )


def merge_cubes(cubes: list[GridCube]) -> GridCube:
    """
    Merge cubes on time and on the native grid.

    Coordinates are united. No new grid is interpolated. Where two cubes share
    a timestamp, the first finite value is kept, including for accumulated
    variables: later files do not add to a value that is already stored.
    """

    if not cubes:
        raise CDSIntegrityError("no cubes to merge")
    if len(cubes) == 1:
        return cubes[0]
    latitudes = _union_coords([cube.latitudes for cube in cubes])
    longitudes = _union_coords([cube.longitudes for cube in cubes])
    times = _union_times(cubes)
    names = []
    for cube in cubes:
        for name in cube.data:
            if name not in names:
                names.append(name)
    lat_index = {round(float(value), 6): index for index, value in enumerate(latitudes)}
    lon_index = {round(float(value), 6): index for index, value in enumerate(longitudes)}
    time_index = {stamp_key(value): index for index, value in enumerate(times)}
    arrays = {
        name: np.full((times.size, latitudes.size, longitudes.size), np.nan, dtype=float)
        for name in names
    }
    duplicates = 0
    for cube in cubes:
        time_pos = [time_index[stamp_key(value)] for value in np.atleast_1d(cube.times)]
        lat_pos = [lat_index[round(float(value), 6)] for value in np.atleast_1d(cube.latitudes)]
        lon_pos = [lon_index[round(float(value), 6)] for value in np.atleast_1d(cube.longitudes)]
        seen_in_cube = set()
        for local_time, global_time in enumerate(time_pos):
            if global_time in seen_in_cube:
                duplicates += 1
                continue
            seen_in_cube.add(global_time)
            for name, values in cube.data.items():
                incoming = np.asarray(values[local_time], dtype=float)
                destination = arrays[name][global_time]
                block = destination[np.ix_(lat_pos, lon_pos)]
                filled = np.where(np.isnan(block), incoming, block)
                replaced = np.isfinite(block) & np.isfinite(incoming)
                duplicates += int(np.count_nonzero(replaced))
                destination[np.ix_(lat_pos, lon_pos)] = filled
    variable_attrs = {}
    for cube in cubes:
        for name, attrs in cube.variable_attrs.items():
            variable_attrs.setdefault(name, dict(attrs))
    attrs = dict(cubes[0].attrs)
    attrs["duplicate_timestamps_kept_existing"] = duplicates
    return GridCube(
        data=arrays,
        times=times,
        latitudes=latitudes,
        longitudes=longitudes,
        attrs=attrs,
        variable_attrs=variable_attrs,
    )


def _union_coords(groups) -> np.ndarray:
    values = sorted({round(float(item), 6) for group in groups for item in np.atleast_1d(group)})
    return np.asarray(values, dtype=float)


def _union_times(cubes: list[GridCube]) -> np.ndarray:
    keys = sorted({stamp_key(value) for cube in cubes for value in np.atleast_1d(cube.times)})
    return np.asarray(keys, dtype="datetime64[h]")


def synthetic_cube(times, latitudes, longitudes, variable: str, fill: float = 1.0, **attrs) -> GridCube:
    """A rectangular cube of ``fill``. Used by tests and as a shape reference."""

    shape = (len(list(times)), len(list(latitudes)), len(list(longitudes)))
    stamps = np.asarray(
        [np.datetime64(as_utc_hour(value).replace(tzinfo=None), "h") for value in times],
        dtype="datetime64[h]",
    )
    return GridCube(
        data={variable: np.full(shape, fill, dtype=float)},
        times=stamps,
        latitudes=np.asarray(list(latitudes), dtype=float),
        longitudes=np.asarray(list(longitudes), dtype=float),
        attrs=dict(attrs),
        variable_attrs={variable: {"description": variable}},
    )


def _magic(path: Path) -> bytes:
    """Leading bytes, used to tell zip, GRIB, and NetCDF apart."""

    with open(path, "rb") as handle:
        return handle.read(8)


def open_downloaded(path: Union[str, Path]) -> GridCube:
    """Open a CDS download: zip, GRIB, NetCDF classic, or NetCDF4."""

    path = Path(path)
    if _magic(path).startswith(b"PK"):
        path = unzip_single(path)
    magic = _magic(path)
    if magic.startswith(b"GRIB"):
        return open_grib(path)
    if magic.startswith(b"CDF"):
        return read_cube(path)
    return open_netcdf4(path)


def unzip_single(path: Path) -> Path:
    """Extract the largest file from a zip CDS still returns for some requests."""

    with zipfile.ZipFile(path) as archive:
        members = [info for info in archive.infolist() if not info.is_dir()]
        if not members:
            raise CDSIntegrityError(f"{path} is an empty zip")
        member = max(members, key=lambda info: info.file_size)
        dest = path.with_name(Path(member.filename).name)
        dest.write_bytes(archive.read(member))
    return dest


def open_grib(path: Path) -> GridCube:
    try:
        import xarray as xr
    except ImportError as exc:
        raise ImportError(
            "Reading GRIB needs xarray and cfgrib. "
            "Install the CDS extra and cfgrib (which needs the ECMWF ecCodes library)."
        ) from exc
    try:
        dataset = xr.open_dataset(path, engine="cfgrib")
    except Exception as exc:
        raise ImportError(
            "Reading GRIB needs cfgrib and the ECMWF ecCodes library."
        ) from exc
    try:
        return dataset_to_cube(dataset)
    finally:
        dataset.close()


def open_netcdf4(path: Path) -> GridCube:
    try:
        import xarray as xr
    except ImportError as exc:
        raise ImportError(
            "Reading CDS NetCDF4 needs the CDS extra: pip install 'giga-spatial[cds]'"
        ) from exc
    dataset = xr.open_dataset(path)
    try:
        return dataset_to_cube(dataset)
    finally:
        dataset.close()


def dataset_to_cube(dataset) -> GridCube:
    """Convert an xarray dataset that uses CDS coordinate names into a cube."""

    import pandas as pd

    time_name = _pick(dataset, ("time", "valid_time"))
    lat_name = _pick(dataset, ("latitude", "lat"))
    lon_name = _pick(dataset, ("longitude", "lon"))
    times = pd.to_datetime(dataset[time_name].values, utc=True).tz_convert(None)
    stamps = times.to_numpy().astype("datetime64[h]")
    latitudes = np.atleast_1d(np.asarray(dataset[lat_name].values, dtype=float))
    longitudes = np.atleast_1d(np.asarray(dataset[lon_name].values, dtype=float))
    skip = {time_name, lat_name, lon_name}
    data = {}
    variable_attrs = {}
    for name in dataset.data_vars:
        if name in skip:
            continue
        values = np.asarray(dataset[name].values, dtype=float)
        values = _as_time_lat_lon(dataset[name], values, time_name, lat_name, lon_name)
        data[name] = values
        attrs = dataset[name].attrs
        variable_attrs[name] = {
            "units": attrs.get("units"),
            "description": attrs.get("long_name") or attrs.get("standard_name") or name,
        }
    if not data:
        raise CDSIntegrityError("the dataset has no data variables")
    return GridCube(
        data=data,
        times=np.atleast_1d(stamps),
        latitudes=latitudes,
        longitudes=longitudes,
        attrs={"processing": PROCESSING_NOTE, "crs": "EPSG:4326"},
        variable_attrs=variable_attrs,
    )


def _pick(dataset, names: tuple[str, ...]) -> str:
    present = set(dataset.coords) | set(dataset.dims)
    for name in names:
        if name in present:
            return name
    raise CDSIntegrityError(f"dataset coordinates {sorted(present)} have none of {names}")


def _as_time_lat_lon(variable, values, time_name, lat_name, lon_name):
    order = list(variable.dims)
    needed = [time_name, lat_name, lon_name]
    if order == needed:
        return values
    axes = [order.index(name) for name in needed]
    return np.moveaxis(values, axes, [0, 1, 2])


def maybe_dataset(cube: GridCube):
    """Return an xarray dataset when xarray is installed, otherwise the cube."""

    try:
        import xarray as xr
    except ImportError:
        return cube
    variables = {
        name: (("time", "latitude", "longitude"), np.asarray(values, dtype=float))
        for name, values in cube.data.items()
    }
    metadata = {key: _jsonable(value) for key, value in cube.attrs.items()}
    metadata.setdefault("processing", PROCESSING_NOTE)
    metadata.setdefault("crs", "EPSG:4326")
    dataset = xr.Dataset(
        variables,
        coords={
            "time": ("time", np.atleast_1d(cube.times).astype("datetime64[h]")),
            "latitude": ("latitude", np.atleast_1d(cube.latitudes).astype(float)),
            "longitude": ("longitude", np.atleast_1d(cube.longitudes).astype(float)),
        },
        attrs={"giga_spatial": json.dumps(metadata, default=str)},
    )
    dataset["latitude"].attrs["units"] = "degrees_north"
    dataset["longitude"].attrs["units"] = "degrees_east"
    dataset["time"].attrs["timezone"] = "UTC"
    for name, attrs in cube.variable_attrs.items():
        if name not in dataset:
            continue
        if attrs.get("units"):
            dataset[name].attrs["units"] = str(attrs["units"])
        if attrs.get("description"):
            dataset[name].attrs["long_name"] = str(attrs["description"])
        dataset[name].attrs["accumulated"] = bool(attrs.get("accumulated", False))
    chunks = dataset_chunks()
    if chunks:
        try:
            dataset = dataset.chunk(chunks)
        except Exception:
            return dataset
    return dataset


def _jsonable(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return str(value)


def metadata_of(data) -> dict:
    """Metadata dict from a cube or from an xarray dataset produced here."""

    if isinstance(data, GridCube):
        return data.attrs
    raw = data.attrs.get("giga_spatial")
    if isinstance(raw, str):
        return json.loads(raw)
    return dict(data.attrs)


def grid_for_area(area, resolution: float, times, variable: str, fill: float = 1.0) -> GridCube:
    """Cell centres on ``resolution`` covering a CDS area ``(north, west, south, east)``."""

    north, west, south, east = area
    longitudes = _centres(west, east, resolution)
    latitudes = _centres(south, north, resolution)
    stamps = [as_utc_hour(value) if isinstance(value, datetime) else value for value in times]
    return synthetic_cube(stamps, latitudes, longitudes, variable, fill=fill)


def _centres(start: float, stop: float, resolution: float) -> list[float]:
    values = []
    cursor = start
    while cursor <= stop + (resolution / 10.0):
        values.append(round(cursor, 6))
        cursor += resolution
    return values

def new_status(
    *,
    dataset_id: str,
    handler_name: str,
    country_id: str,
    country_name: Optional[str],
    boundary_id: Optional[str],
    update_frequency_hours: float,
) -> dict:
    return {
        "dataset_id": dataset_id,
        "handler_name": handler_name,
        "country_id": country_id,
        "country_name": country_name,
        "boundary_id": boundary_id,
        "update_frequency_hours": update_frequency_hours,
        "last_update_check": None,
        "latest_available_cds_timestamp": None,
        "variables": {},
    }


def ensure_variable(status: dict, variable: Variable) -> dict:
    record = status["variables"].get(variable.id)
    if record is None:
        record = {
            "description": variable.description,
            "units": variable.units,
            "accumulated": variable.accumulated,
            "time_invariant": variable.time_invariant,
            "last_downloaded_timestamp": None,
            "latest_available_timestamp": None,
            "last_update_check": None,
            "last_successful_update": None,
            "missing_periods": [],
            "pending_periods": [],
            "status": "missing",
            "error": None,
            "retries": 0,
            "periods": [],
        }
        status["variables"][variable.id] = record
    else:
        record["description"] = variable.description
        record["units"] = variable.units
        record["accumulated"] = variable.accumulated
        record["time_invariant"] = variable.time_invariant
    return record


def note_check(status: dict, latest: Optional[datetime], when: datetime) -> None:
    stamp = format_timestamp(when)
    status["last_update_check"] = stamp
    if latest is not None:
        status["latest_available_cds_timestamp"] = format_timestamp(latest)
    for record in status["variables"].values():
        record["last_update_check"] = stamp
        if latest is not None:
            record["latest_available_timestamp"] = format_timestamp(latest)


def check_is_due(status: dict, when: datetime, frequency_hours: float, force: bool) -> bool:
    if force or not status.get("last_update_check"):
        return True
    elapsed = when - parse_timestamp(status["last_update_check"])
    return elapsed.total_seconds() >= frequency_hours * 3600


def plan_hours(record: dict, latest: Optional[datetime]) -> list[datetime]:
    """
    Hours that still need a download.

    Complete history is left in place. Newly published hours are those after
    the last stored timestamp and at or before ``latest``. Missing and pending
    periods are requested again. An invariant variable that already succeeded
    is left alone.
    """

    if record.get("time_invariant") and record.get("last_downloaded_timestamp") and not record.get("missing_periods"):
        return []
    hours: list[datetime] = []
    last = record.get("last_downloaded_timestamp")
    if last and latest is not None:
        hours.extend(forward_hours(parse_timestamp(last), latest))
    for period in list(record.get("missing_periods") or []) + list(record.get("pending_periods") or []):
        hours.extend(hours_in_period(period))
    if latest is not None:
        limit = parse_timestamp(format_timestamp(latest))
        hours = [hour for hour in hours if hour <= limit]
    return sorted(set(hours))


def plan_slices(record: dict, latest: Optional[datetime]) -> tuple[TimeSlice, ...]:
    return slices_from_hours(plan_hours(record, latest))


def mark_complete(record: dict, time_slice: TimeSlice, path: str, when: datetime) -> None:
    hours = slice_hours(time_slice)
    end = format_timestamp(hours[-1])
    previous = record.get("last_downloaded_timestamp")
    if previous is None or parse_timestamp(previous) < hours[-1]:
        record["last_downloaded_timestamp"] = end
    record["last_successful_update"] = format_timestamp(when)
    record["error"] = None
    record["retries"] = 0
    record["status"] = "stored"
    token = path
    period = {
        "start": format_timestamp(hours[0]),
        "end": end,
        "path": path,
        "hours": len(hours),
        "complete": True,
    }
    replaced = False
    for existing in record["periods"]:
        if existing.get("path") == token:
            existing.update(period)
            replaced = True
            break
    if not replaced:
        record["periods"].append(period)
    record["missing_periods"] = [
        item for item in record.get("missing_periods", []) if item.get("path") != path
    ]
    record["pending_periods"] = [
        item for item in record.get("pending_periods", []) if item.get("path") != path
    ]


def mark_failed(
    record: dict,
    time_slice: TimeSlice,
    path: str,
    when: datetime,
    message: str,
    *,
    pending: bool = False,
) -> None:
    hours = slice_hours(time_slice)
    period = {
        "start": format_timestamp(hours[0]),
        "end": format_timestamp(hours[-1]),
        "path": path,
    }
    key = "pending_periods" if pending else "missing_periods"
    periods = [item for item in record.get(key, []) if item.get("path") != path]
    periods.append(period)
    record[key] = periods
    record["error"] = message
    record["retries"] = int(record.get("retries") or 0) + 1
    record["status"] = "pending" if pending else "error"
    record["last_update_check"] = format_timestamp(when)


def refresh_variable_status(record: dict, latest: Optional[datetime]) -> None:
    """Set current, partial, pending, or stored from the catalogue end and any gaps."""

    if record.get("pending_periods"):
        record["status"] = "pending"
        return
    if record.get("missing_periods"):
        record["status"] = "error" if record.get("error") else "partial"
        return
    last = record.get("last_downloaded_timestamp")
    if not last:
        record["status"] = "missing"
        return
    if latest is None:
        record["status"] = "stored"
        return
    if parse_timestamp(last) >= parse_timestamp(format_timestamp(latest)):
        record["status"] = "current"
    else:
        record["status"] = "partial"

def _string_tuple(value) -> Optional[tuple[str, ...]]:
    if value is None:
        return None
    if isinstance(value, (str, int)):
        return (str(value),)
    return tuple(str(item) for item in value)


def _variable_tuple(value) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        if value == "all":
            return (ALL_VARIABLES,)
        return (value,)
    items = []
    for item in value:
        if str(item) == "all":
            return (ALL_VARIABLES,)
        items.append(str(item))
    return tuple(items)


def _hour_tuple(value) -> Optional[tuple[str, ...]]:
    if value is None:
        return None
    raw = (value,) if isinstance(value, (str, int)) else tuple(value)
    hours = []
    for item in raw:
        text = str(item).strip()
        hour = int(text.split(":")[0])
        if not 0 <= hour <= 23:
            raise ValueError(f"hour must be 00..23, got {item!r}")
        hours.append(f"{hour:02d}:00")
    return tuple(hours)


@dataclass(frozen=True)
class CDSResolveOptions(ResolveOptions):
    """
    What to request from CDS.

    ``variables`` is one CDS name, a sequence of names, or ``"all"``.
    Time is a ``start``/``end`` pair, ``year``/``month``/``day``/``hour``,
    or ``full_history=True``. ``product_type`` applies to ERA5 only.
    """

    variables: tuple[str, ...] = ()
    product_type: Optional[str] = None
    start: Optional[str] = None
    end: Optional[str] = None
    year: Optional[tuple[str, ...]] = None
    month: Optional[tuple[str, ...]] = None
    day: Optional[tuple[str, ...]] = None
    hour: Optional[tuple[str, ...]] = None
    full_history: bool = False

    def __post_init__(self):
        object.__setattr__(self, "variables", _variable_tuple(self.variables))
        object.__setattr__(self, "year", _string_tuple(self.year))
        object.__setattr__(self, "month", _pad(self.month, 2))
        object.__setattr__(self, "day", _pad(self.day, 2))
        object.__setattr__(self, "hour", _hour_tuple(self.hour))
        object.__setattr__(self, "start", self.start or None)
        object.__setattr__(self, "end", self.end or None)
        object.__setattr__(self, "product_type", self.product_type or None)
        for month in self.month or ():
            if not 1 <= int(month) <= 12:
                raise ValueError(f"month must be 01..12, got {month}")
        for day in self.day or ():
            if not 1 <= int(day) <= 31:
                raise ValueError(f"day must be 01..31, got {day}")


def _pad(value, width: int) -> Optional[tuple[str, ...]]:
    items = _string_tuple(value)
    if items is None:
        return None
    padded = []
    for item in items:
        if item.isdigit():
            padded.append(item.zfill(width))
        else:
            padded.append(item)
    return tuple(padded)


@dataclass(frozen=True)
class CDSSource:
    """Country identity and the CDS areas for one source. Not the output geometry."""

    country_id: str
    country_name: Optional[str]
    boundary_id: Optional[str]
    geometry_path: str
    areas: tuple


def country_directory(base: Path, country_id: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", country_id or "unknown")
    return Path(base) / safe


def write_json(store, path, payload: dict) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True)
    partial = str(path) + ".partial"
    store.write_file(partial, text)
    if isinstance(store, LocalDataStore):
        os.replace(store._resolve_path(partial), store._resolve_path(str(path)))
        return
    store.write_file(str(path), text)
    if store.file_exists(partial):
        store.remove(partial)


def read_json(store, path) -> Optional[dict]:
    if not store.file_exists(str(path)):
        return None
    raw = store.read_file(str(path), encoding="utf-8")
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return json.loads(raw)


def read_geometry(store, path):
    raw = store.read_file(str(path), encoding="utf-8")
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return wkt.loads(raw)


def file_size(store, path) -> int:
    if isinstance(store, LocalDataStore):
        resolved = store._resolve_path(str(path))
        return resolved.stat().st_size if resolved.is_file() else 0
    if not store.file_exists(str(path)):
        return 0
    payload = store.read_file(str(path))
    return len(payload) if isinstance(payload, (bytes, str)) else 0


def local_file(store, path) -> Path:
    if isinstance(store, LocalDataStore):
        return Path(store._resolve_path(str(path)))
    payload = store.read_file(str(path))
    if isinstance(payload, str):
        payload = payload.encode("utf-8")
    handle = tempfile.NamedTemporaryFile(delete=False, suffix=Path(str(path)).suffix)
    handle.write(payload)
    handle.close()
    return Path(handle.name)


def time_slice_of(unit: dict) -> TimeSlice:
    return TimeSlice(
        year=str(unit["year"]),
        month=str(unit["month"]),
        days=tuple(unit["days"]),
        times=tuple(unit["times"]),
    )


def stored_path(config, unit: dict) -> Path:
    product = unit.get("product_type") or "hourly"
    return (
        country_directory(config.base_path, unit["country_id"])
        / product
        / unit["variable"]
        / unit["data_format"]
        / f"{slice_token(time_slice_of(unit))}.nc"
    )


def build_cds_request(unit: dict, area: Sequence[float], *, include_product_type: bool) -> dict:
    """
    Request body for the current CDS download form.

    Keys match the form: ``variable``, ``year``, ``month``, ``day``, ``time``,
    ``data_format``, ``download_format``, and ``area`` as north, west, south,
    east. ERA5 also sends ``product_type``. No ``grid`` is sent, so CDS returns
    the native grid. The old ``format`` key is not sent.
    """

    request = {
        "variable": [unit["variable"]],
        "year": [str(unit["year"])],
        "month": [str(unit["month"])],
        "day": list(unit["days"]),
        "time": list(unit["times"]),
        "data_format": unit["data_format"],
        "download_format": "unarchived",
        "area": [float(value) for value in area],
    }
    if include_product_type:
        request["product_type"] = [unit["product_type"]]
    return request


def publish_cube(store, path: Path, cube, meta: dict) -> Path:
    """
    Replace ``path`` only after ``cube`` has been written and closed.

    The previous file stays in place until ``os.replace``. The ``.ok`` marker
    is removed first and written last, so an interrupted publish is not treated
    as a successful download.
    """

    path = Path(path)
    marker = Path(str(path) + ".ok")
    partial = Path(str(path) + ".partial")
    if store.file_exists(str(marker)):
        store.remove(str(marker))
    if isinstance(store, LocalDataStore):
        destination = store._resolve_path(str(partial))
        write_cube(destination, cube)
        if destination.stat().st_size <= 0:
            destination.unlink(missing_ok=True)
            raise CDSIntegrityError("CDS file was empty and was not stored")
        os.replace(destination, store._resolve_path(str(path)))
    else:
        handle = tempfile.NamedTemporaryFile(delete=False, suffix=".nc")
        handle.close()
        try:
            write_cube(handle.name, cube)
            payload = Path(handle.name).read_bytes()
        finally:
            Path(handle.name).unlink(missing_ok=True)
        if not payload:
            raise CDSIntegrityError("CDS file was empty and was not stored")
        store.write_file(str(path), payload)
    write_json(store, Path(str(path) + ".meta.json"), meta)
    store.write_file(str(marker), "ok\n")
    return path


def _default_base_path() -> Path:
    return global_config.get_path("cds", "bronze")


@pydantic_dataclass(config=ConfigDict(arbitrary_types_allowed=True, extra="forbid"))
class CDSConfig(BaseHandlerConfig):
    """Configuration shared by the CDS handlers. Dataset modules set the class variables."""

    options_type = CDSResolveOptions
    HANDLER_NAME: ClassVar[str] = ""
    DATASET_ID: ClassVar[str] = ""
    DOCUMENTATION_URL: ClassVar[str] = ""
    DOI: ClassVar[str] = ""
    START_YEAR: ClassVar[int] = 1940
    RESOLUTION: ClassVar[float] = 0.25
    HAS_PRODUCT_TYPE: ClassVar[bool] = False
    PRODUCT_TYPES: ClassVar[tuple] = ()
    DEFAULT_PRODUCT_TYPE: ClassVar[Optional[str]] = None

    base_path: Path = Field(default_factory=_default_base_path)
    n_workers: int = 2
    api_url: str = Field(default="", repr=False)
    api_key: str = Field(default="", repr=False)
    data_format: str = "netcdf"
    update_frequency_hours: float = 24.0
    request_timeout: float = 300.0
    max_retries: int = 3
    refresh_variables: bool = False
    retriever: Optional[Any] = Field(default=None, repr=False)

    def credentials(self) -> tuple[str, str]:
        return resolve_credentials(self.api_url or None, self.api_key or None)

    def catalogue(self) -> tuple[Variable, ...]:
        url, _key = self.credentials()
        return load_variables(
            type(self).DATASET_ID,
            refresh=self.refresh_variables,
            api_url=url,
        )

    def latest_available(self) -> datetime:
        if self.retriever is not None and hasattr(self.retriever, "latest_available"):
            return self.retriever.latest_available(type(self).DATASET_ID)
        url, _key = self.credentials()
        return fetch_latest_available(type(self).DATASET_ID, url)

    def get_search_key(self, source, options: CDSResolveOptions) -> CDSSource:
        kind = classify_source(source)
        if kind is SourceKind.COUNTRY:
            from gigaspatial.handlers.boundaries import AdminBoundaries

            boundary_kwargs = {}
            if options.country_geom_path is not None:
                boundary_kwargs["path"] = options.country_geom_path
            if options.country_data_store is not None:
                boundary_kwargs["data_store"] = options.country_data_store
            table = AdminBoundaries.create(country_code=source, **boundary_kwargs)
            geometry = table.to_geoms()[0]
            entity = table.entities[0]
            country_id = entity.country_iso or entity.boundary_id
            country_name = getattr(entity, "name_en", None) or entity.name
            boundary_id = entity.boundary_id
        else:
            geometry = self.extract_search_geometry(source, options)
            import hashlib

            digest = hashlib.sha1(geometry.wkt.encode("utf-8")).hexdigest()[:12]
            country_id = f"geom_{digest}"
            country_name = None
            boundary_id = None
        areas = request_areas(geometry, type(self).RESOLUTION)
        directory = country_directory(self.base_path, country_id)
        geometry_path = directory / "geometry.wkt"
        self.data_store.write_file(str(geometry_path), geometry.wkt)
        write_json(
            self.data_store,
            directory / "source.json",
            {
                "country_id": country_id,
                "country_name": country_name,
                "boundary_id": boundary_id,
                "dataset_id": type(self).DATASET_ID,
                "handler_name": type(self).HANDLER_NAME,
                "source_dataset_url": type(self).DOCUMENTATION_URL,
            },
        )
        return CDSSource(
            country_id=country_id,
            country_name=country_name,
            boundary_id=boundary_id,
            geometry_path=str(geometry_path),
            areas=tuple(tuple(float(value) for value in area) for area in areas),
        )

    def get_relevant_data_units_by_key(self, search_key: CDSSource, options: CDSResolveOptions):
        if self.data_format not in {"netcdf", "grib"}:
            raise ValueError("data_format must be 'netcdf' or 'grib'")
        product = options.product_type or type(self).DEFAULT_PRODUCT_TYPE
        if type(self).HAS_PRODUCT_TYPE:
            if product not in type(self).PRODUCT_TYPES:
                raise ValueError(
                    f"product_type must be one of {type(self).PRODUCT_TYPES}, got {product!r}"
                )
        elif options.product_type:
            raise ValueError(f"{type(self).DATASET_ID} has no product_type parameter")
        selected = select_variables(self.catalogue(), options.variables)
        latest = self.latest_available() if options.full_history else None
        hours = selection_hours(options, start_year=type(self).START_YEAR, latest=latest)
        if len(hours) > 24 * 366:
            self.logger.warning(
                "CDS selection for %s covers %s hours",
                type(self).DATASET_ID,
                len(hours),
            )
        period_slices = slices_from_hours(hours)
        units = []
        for variable in selected:
            slices = period_slices
            if variable.time_invariant:
                first = slice_hours(period_slices[0])[0]
                slices = slices_from_hours([first])
            for time_slice in slices:
                units.append(
                    {
                        "dataset_id": type(self).DATASET_ID,
                        "handler_name": type(self).HANDLER_NAME,
                        "country_id": search_key.country_id,
                        "country_name": search_key.country_name,
                        "boundary_id": search_key.boundary_id,
                        "variable": variable.id,
                        "description": variable.description,
                        "units": variable.units,
                        "accumulated": variable.accumulated,
                        "time_invariant": variable.time_invariant,
                        "product_type": product,
                        "year": time_slice.year,
                        "month": time_slice.month,
                        "days": time_slice.days,
                        "times": time_slice.times,
                        "areas": search_key.areas,
                        "data_format": self.data_format,
                        "geometry_path": search_key.geometry_path,
                    }
                )
        return units

    def get_data_unit_path(self, unit: dict, options=None) -> Path:
        return stored_path(self, unit)

    def is_unit_available(self, unit, paths) -> bool:
        for path in paths:
            marker = Path(str(path) + ".ok")
            if not self.data_store.file_exists(str(path)):
                return False
            if not self.data_store.file_exists(str(marker)):
                return False
            if file_size(self.data_store, path) <= 0:
                return False
        return True


class CDSDownloader(BaseHandlerDownloader):
    """Retrieve one variable-month from CDS, clip it, and store the native grid."""

    options_type = DownloadOptions
    config_type: ClassVar[type] = CDSConfig

    def __init__(self, config=None, data_store=None, logger=None):
        super().__init__(config=config, data_store=data_store, logger=logger)
        self._guards: dict[str, threading.Lock] = {}
        self._guard_lock = threading.Lock()

    def download_data_unit(self, unit: dict, options: DownloadOptions) -> Path:
        path = self.config.get_data_unit_path(unit)
        with self._lock_for(str(path)):
            return self._download_once(unit, path)

    def _lock_for(self, path: str) -> threading.Lock:
        with self._guard_lock:
            return self._guards.setdefault(path, threading.Lock())

    def _retriever(self):
        if self.config.retriever is not None:
            return self.config.retriever
        url, key = self.config.credentials()
        if not key:
            raise CDSRequestError(
                "CDS API key is not configured. Set CDS_API_KEY or CDSAPI_KEY, "
                "or add key: to ~/.cdsapirc. Accept the dataset licence on the CDS website.",
                licence=True,
            )
        return CdsapiRetriever(url, key, self.config.request_timeout)

    def _download_once(self, unit: dict, path: Path) -> Path:
        geometry = read_geometry(self.data_store, unit["geometry_path"])
        cubes = []
        with tempfile.TemporaryDirectory(prefix="cds-") as temporary:
            for index, area in enumerate(unit["areas"]):
                target = Path(temporary) / f"area-{index}"
                request = build_cds_request(
                    unit,
                    area,
                    include_product_type=type(self.config).HAS_PRODUCT_TYPE,
                )
                retrieve_with_retry(
                    self._retriever(),
                    unit["dataset_id"],
                    request,
                    target,
                    max_retries=self.config.max_retries,
                )
                cubes.append(open_downloaded(target))
        combined = merge_cubes(cubes) if len(cubes) > 1 else cubes[0]
        variable = unit["variable"]
        if variable not in combined.data and len(combined.data) == 1:
            only = next(iter(combined.data))
            combined.data[variable] = combined.data.pop(only)
            combined.variable_attrs[variable] = combined.variable_attrs.pop(only, {})
        combined.variable_attrs[variable] = {
            "units": unit.get("units"),
            "description": unit.get("description") or variable,
            "accumulated": bool(unit.get("accumulated")),
        }
        clipped = clip_cube(combined, geometry, type(self.config).RESOLUTION)
        duplicates = duplicate_timestamps(clipped.times)
        if duplicates:
            raise CDSIntegrityError(
                f"download for {variable} has duplicate timestamps and was not stored"
            )
        missing = missing_timestamps(slice_hours(time_slice_of(unit)), clipped.times)
        if missing:
            raise CDSIntegrityError(
                f"download for {variable} is missing {len(missing)} hour(s); "
                f"first missing hour is {missing[0]}"
            )
        downloaded_at = format_timestamp(datetime.now(timezone.utc))
        meta = {
            "dataset_id": unit["dataset_id"],
            "handler_name": unit["handler_name"],
            "country_id": unit["country_id"],
            "country_name": unit["country_name"],
            "boundary_id": unit["boundary_id"],
            "variable": variable,
            "description": unit.get("description"),
            "units": unit.get("units"),
            "accumulated": bool(unit.get("accumulated")),
            "start": format_timestamp(slice_hours(time_slice_of(unit))[0]),
            "end": format_timestamp(slice_hours(time_slice_of(unit))[-1]),
            "temporal_resolution": "1 hour",
            "spatial_resolution": f"{type(self.config).RESOLUTION} degrees",
            "source_dataset_url": type(self.config).DOCUMENTATION_URL,
            "download_timestamp": downloaded_at,
            "processing": PROCESSING_NOTE,
            "crs": "EPSG:4326",
            "data_format": unit["data_format"],
        }
        clipped.attrs.update(
            {
                "dataset_id": unit["dataset_id"],
                "handler_name": unit["handler_name"],
                "processing": PROCESSING_NOTE,
                "crs": "EPSG:4326",
                "country_id": unit["country_id"],
            }
        )
        return publish_cube(self.data_store, path, clipped, meta)


class CDSReader(BaseHandlerReader):
    """Load stored country files and keep the native grid."""

    options_type = ReadOptions
    config_type: ClassVar[type] = CDSConfig

    def load_from_paths(self, source_data_path, options: ReadOptions):
        cubes = []
        order = []
        for path in source_data_path:
            if not str(path).endswith(".nc"):
                continue
            local = local_file(self.data_store, path)
            order.append((local.stat().st_mtime, local, path))
        for _mtime, local, path in sorted(order):
            cube = read_cube(local)
            meta = read_json(self.data_store, Path(str(path) + ".meta.json")) or {}
            cube.attrs.update({key: value for key, value in meta.items() if key not in cube.attrs})
            cubes.append(cube)
        if not cubes:
            return None
        merged = merge_cubes(cubes)
        merged.attrs.update(self._metadata(merged))
        return maybe_dataset(merged)

    def _metadata(self, cube) -> dict:
        country_id = cube.attrs.get("country_id")
        status = {}
        if country_id:
            status = (
                read_json(
                    self.data_store,
                    country_directory(self.config.base_path, country_id) / "_status.json",
                )
                or {}
            )
        catalogue = {variable.id: variable for variable in self.config.catalogue()}
        names = list(cube.data)
        descriptions = {}
        units = {}
        for name in names:
            if name in catalogue:
                descriptions[name] = catalogue[name].description
                units[name] = catalogue[name].units
            else:
                descriptions[name] = cube.variable_attrs.get(name, {}).get("description", name)
                units[name] = cube.variable_attrs.get(name, {}).get("units")
        stamps = list(cube.times)
        updates = [
            record.get("last_successful_update")
            for record in status.get("variables", {}).values()
            if record.get("last_successful_update")
        ]
        return {
            "dataset_id": type(self.config).DATASET_ID,
            "handler_name": type(self.config).HANDLER_NAME,
            "country_name": status.get("country_name") or cube.attrs.get("country_name"),
            "country_id": country_id or status.get("country_id"),
            "boundary_id": status.get("boundary_id") or cube.attrs.get("boundary_id"),
            "variable_names": names,
            "variable_descriptions": descriptions,
            "units": units,
            "start_date": _stamp_text(stamps[0]) if len(stamps) else None,
            "end_date": _stamp_text(stamps[-1]) if len(stamps) else None,
            "temporal_resolution": "1 hour",
            "spatial_resolution": f"{type(self.config).RESOLUTION} degrees",
            "source_dataset_url": type(self.config).DOCUMENTATION_URL,
            "download_timestamp": cube.attrs.get("download_timestamp"),
            "last_successful_update": max(updates) if updates else None,
            "latest_available_cds_timestamp": status.get("latest_available_cds_timestamp"),
            "processing": PROCESSING_NOTE,
            "crs": "EPSG:4326",
            "licence": "CC-BY-4.0",
            "doi": type(self.config).DOI,
        }

    def crop_to_geometry(self, data, geometry, predicate="intersects", **kwargs):
        meta = metadata_of(data)
        if isinstance(data, GridCube):
            clipped = clip_cube(data, geometry, type(self.config).RESOLUTION)
        else:
            clipped = clip_cube(dataset_to_cube(data), geometry, type(self.config).RESOLUTION)
        clipped.attrs.update(meta)
        return maybe_dataset(clipped)


def _stamp_text(value) -> str:
    import pandas as pd

    parsed = pd.Timestamp(value)
    if parsed.tzinfo is None:
        parsed = parsed.tz_localize("UTC")
    return parsed.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


class CDSHandler(BaseHandler):
    """Country-level CDS handler. Subclasses set ``name`` and the component classes."""

    name = ""
    config_class = CDSConfig
    downloader_class = CDSDownloader
    reader_class = CDSReader

    def create_config(self, data_store, logger, **kwargs):
        return self.config_class(data_store=data_store, logger=logger, **kwargs)

    def create_downloader(self, config, data_store, logger, **kwargs):
        return self.downloader_class(config=config, data_store=data_store, logger=logger)

    def create_reader(self, config, data_store, logger, **kwargs):
        return self.reader_class(config=config, data_store=data_store, logger=logger)

    def load_data(self, source, crop_to_source: bool = True, ensure_available: bool = True, **kwargs):
        """Load country data. The result is clipped to the source geometry."""

        from gigaspatial.handlers.base.options import sort_options

        resolve, download, _ = sort_options(
            kwargs, self.config, self.downloader, self.reader, entry="load_data"
        )
        if ensure_available:
            available = BaseHandler._ensure_available(self, source, False, resolve, download)
            try:
                self._record_download(source, resolve, available)
            except Exception as exc:
                self.logger.error("Could not write CDS update status: %s", exc)
            if not available:
                raise RuntimeError("Could not ensure data availability for loading")
        return self._read(source, crop_to_source, kwargs)

    def _record_download(self, source, resolve, ok: bool) -> None:
        units = self.config._resolve_units(source, resolve)
        if not units:
            return
        when = datetime.now(timezone.utc)
        by_country: dict[str, list] = {}
        for unit in units:
            by_country.setdefault(unit["country_id"], []).append(unit)
        catalogue = {variable.id: variable for variable in self.config.catalogue()}
        for country_id, country_units in by_country.items():
            path = country_directory(self.config.base_path, country_id) / "_status.json"
            status = read_json(self.data_store, path) or new_status(
                dataset_id=type(self.config).DATASET_ID,
                handler_name=self.name,
                country_id=country_id,
                country_name=country_units[0].get("country_name"),
                boundary_id=country_units[0].get("boundary_id"),
                update_frequency_hours=self.config.update_frequency_hours,
            )
            if country_units[0].get("product_type"):
                status["product_type"] = country_units[0]["product_type"]
            for unit in country_units:
                variable = catalogue[unit["variable"]]
                record = ensure_variable(status, variable)
                stored = self.config.get_data_unit_path(unit)
                if self.config.is_unit_available(unit, [stored]):
                    mark_complete(record, time_slice_of(unit), str(stored), when)
                else:
                    mark_failed(
                        record,
                        time_slice_of(unit),
                        str(stored),
                        when,
                        "download did not complete",
                    )
            if ok:
                for record in status["variables"].values():
                    if record.get("status") != "error":
                        refresh_variable_status(record, None)
            write_json(self.data_store, path, status)

    def configured_countries(self) -> list[str]:
        """Country ids that already have a status file under this handler's directory."""

        base = self.config.base_path
        if not self.data_store.is_dir(str(base)):
            return []
        found = []
        for name in self.data_store.list_directories(str(base)):
            if self.data_store.file_exists(str(Path(base) / name / "_status.json")):
                found.append(name)
        return sorted(found)

    def update(self, source=None, *, force: bool = False, variables=None) -> dict:
        """
        Download hours CDS has published since the last successful run.

        With no source, every country that already has a status file is
        checked. The check asks this dataset's catalogue for its own latest
        timestamp. Hours that are already stored are not requested again.
        ``update_frequency_hours`` skips a check that ran recently unless
        ``force`` is set. There is no in-process scheduler: run this method,
        or ``python -m gigaspatial.handlers.cds.era5 update``, from cron or another
        external scheduler.
        """

        when = datetime.now(timezone.utc)
        if source is None:
            countries = self.configured_countries()
        else:
            countries = [self._country_id_for_update(source)]
        reports = [
            self._update_country(country_id, when, force=force, variables=variables)
            for country_id in countries
        ]
        return {
            "handler_name": self.name,
            "dataset_id": type(self.config).DATASET_ID,
            "countries": reports,
        }

    def _country_id_for_update(self, source: str) -> str:
        direct = country_directory(self.config.base_path, source)
        if self.data_store.file_exists(str(direct / "_status.json")):
            return direct.name
        from gigaspatial.handlers.boundaries import AdminBoundaries

        table = AdminBoundaries.create(country_code=source)
        entity = table.entities[0]
        return entity.country_iso or entity.boundary_id

    def _update_country(self, country_id: str, when: datetime, *, force: bool, variables) -> dict:
        path = country_directory(self.config.base_path, country_id) / "_status.json"
        status = read_json(self.data_store, path)
        if status is None:
            return {"country_id": country_id, "status": "not_configured"}
        if not check_is_due(status, when, self.config.update_frequency_hours, force):
            return {
                "country_id": country_id,
                "status": "skipped",
                "latest_available_cds_timestamp": status.get("latest_available_cds_timestamp"),
            }
        latest = None
        availability_error = None
        try:
            latest = self._latest()
        except Exception as exc:
            availability_error = str(exc)
            self.logger.warning(
                "CDS availability for %s could not be read: %s",
                type(self.config).DATASET_ID,
                exc,
            )
        note_check(status, latest, when)
        wanted = None
        if variables not in (None, "all"):
            wanted = set(_variable_tuple(variables) or ())
        geometry = self._geometry_for_update(country_id, status)
        areas = request_areas(geometry, type(self.config).RESOLUTION)
        geometry_path = str(country_directory(self.config.base_path, country_id) / "geometry.wkt")
        downloaded = []
        errors = []
        for variable_id, record in list(status["variables"].items()):
            if wanted is not None and variable_id not in wanted:
                continue
            slices = plan_slices(record, latest)
            if not slices:
                refresh_variable_status(record, latest)
                continue
            for time_slice in slices:
                unit = self._unit_from_status(
                    status, record, variable_id, time_slice, areas, geometry_path
                )
                try:
                    stored = self.downloader.download_data_unit(
                        unit, DownloadOptions(show_progress=False)
                    )
                except CDSRequestError as exc:
                    mark_failed(
                        record,
                        time_slice,
                        str(self.config.get_data_unit_path(unit)),
                        when,
                        str(exc),
                        pending=exc.not_available,
                    )
                    errors.append({"variable": variable_id, "error": str(exc)})
                    continue
                except Exception as exc:
                    mark_failed(
                        record,
                        time_slice,
                        str(self.config.get_data_unit_path(unit)),
                        when,
                        str(exc),
                    )
                    errors.append({"variable": variable_id, "error": str(exc)})
                    continue
                mark_complete(record, time_slice, str(stored), when)
                downloaded.append(str(stored))
            refresh_variable_status(record, latest)
            write_json(self.data_store, path, status)
        write_json(self.data_store, path, status)
        return {
            "country_id": country_id,
            "status": "error" if errors else "updated",
            "downloaded": downloaded,
            "errors": errors,
            "availability_error": availability_error,
            "latest_available_cds_timestamp": status.get("latest_available_cds_timestamp"),
            "variables": {
                key: {
                    "status": value.get("status"),
                    "last_downloaded_timestamp": value.get("last_downloaded_timestamp"),
                    "latest_available_timestamp": value.get("latest_available_timestamp"),
                    "missing_periods": value.get("missing_periods"),
                    "pending_periods": value.get("pending_periods"),
                    "retries": value.get("retries"),
                    "error": value.get("error"),
                }
                for key, value in status["variables"].items()
            },
        }

    def _latest(self) -> datetime:
        if self.config.retriever is not None and hasattr(self.config.retriever, "latest_available"):
            return self.config.retriever.latest_available(type(self.config).DATASET_ID)
        url, _key = self.config.credentials()
        return fetch_latest_available(type(self.config).DATASET_ID, url)

    def _geometry_for_update(self, country_id: str, status: dict):
        path = country_directory(self.config.base_path, country_id) / "geometry.wkt"
        if not str(country_id).startswith("geom_"):
            try:
                from gigaspatial.handlers.boundaries import AdminBoundaries

                table = AdminBoundaries.create(country_code=country_id)
                geometry = table.to_geoms()[0]
                self.data_store.write_file(str(path), geometry.wkt)
                return geometry
            except Exception as exc:
                self.logger.warning(
                    "Stored geometry will be used for %s because AdminBoundaries failed: %s",
                    country_id,
                    exc,
                )
        if not self.data_store.file_exists(str(path)):
            raise CDSIntegrityError(f"no stored geometry for {country_id}")
        return read_geometry(self.data_store, path)

    def _unit_from_status(self, status, record, variable_id, time_slice, areas, geometry_path):
        return {
            "dataset_id": type(self.config).DATASET_ID,
            "handler_name": self.name,
            "country_id": status.get("country_id"),
            "country_name": status.get("country_name"),
            "boundary_id": status.get("boundary_id"),
            "variable": variable_id,
            "description": record.get("description"),
            "units": record.get("units"),
            "accumulated": record.get("accumulated"),
            "time_invariant": record.get("time_invariant"),
            "product_type": status.get("product_type") or type(self.config).DEFAULT_PRODUCT_TYPE,
            "year": time_slice.year,
            "month": time_slice.month,
            "days": time_slice.days,
            "times": time_slice.times,
            "areas": tuple(tuple(float(value) for value in area) for area in areas),
            "data_format": self.config.data_format,
            "geometry_path": geometry_path,
        }

    def get_available_data_info(self, source, **kwargs) -> dict:
        info = super().get_available_data_info(source, **kwargs)
        info["handler_name"] = self.name
        info["dataset_id"] = type(self.config).DATASET_ID
        info["source_dataset_url"] = type(self.config).DOCUMENTATION_URL
        return info

@pydantic_dataclass(config=ConfigDict(arbitrary_types_allowed=True, extra="forbid"))
class ERA5Config(CDSConfig):
    """ERA5 single-level configuration. Files live under ``bronze/cds/era5``."""

    HANDLER_NAME: ClassVar[str] = "cds-era5"
    DATASET_ID: ClassVar[str] = "reanalysis-era5-single-levels"
    DOCUMENTATION_URL: ClassVar[str] = (
        "https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=overview"
    )
    DOI: ClassVar[str] = "10.24381/cds.adbb2d47"
    START_YEAR: ClassVar[int] = 1940
    RESOLUTION: ClassVar[float] = 0.25
    HAS_PRODUCT_TYPE: ClassVar[bool] = True
    PRODUCT_TYPES: ClassVar[tuple] = (
        "reanalysis",
        "ensemble_members",
        "ensemble_mean",
        "ensemble_spread",
    )
    DEFAULT_PRODUCT_TYPE: ClassVar[Optional[str]] = "reanalysis"

    base_path: Path = Field(
        default_factory=lambda: global_config.get_path("cds", "bronze") / "era5"
    )


class ERA5Downloader(CDSDownloader):
    """Downloads ERA5 single-level fields."""

    config_type: ClassVar[type] = ERA5Config


class ERA5Reader(CDSReader):
    """Reads stored ERA5 country files."""

    config_type: ClassVar[type] = ERA5Config


class ERA5Handler(CDSHandler):
    """
    ERA5 hourly single levels for a country.

    The handler name is ``cds-era5``.

        ERA5Handler().load_data(
            "ITA",
            variables="2m_temperature",
            start="2024-11-01",
            end="2024-11-02",
        )
        ERA5Handler().load_data("ITA", variables=("2m_temperature", "total_precipitation"), year=2024, month=11)
        ERA5Handler().load_data("ITA", variables="all", year=2024, month=1, day=1, hour=0)
        ERA5Handler().update("ITA")
    """

    name = "cds-era5"
    config_class = ERA5Config
    downloader_class = ERA5Downloader
    reader_class = ERA5Reader


_CATALOGUES["reanalysis-era5-single-levels"] = variable_records((
    ('10m_u_component_of_wind', '10m u-component of wind', ('Popular', 'Wind')),
    ('10m_v_component_of_wind', '10m v-component of wind', ('Popular', 'Wind')),
    ('2m_dewpoint_temperature', '2m dewpoint temperature', ('Popular', 'Temperature and pressure')),
    ('2m_temperature', '2m temperature', ('Popular', 'Temperature and pressure')),
    ('mean_sea_level_pressure', 'Mean sea level pressure', ('Popular', 'Temperature and pressure')),
    ('mean_wave_direction', 'Mean wave direction', ('Popular', 'Ocean waves')),
    ('mean_wave_period', 'Mean wave period', ('Popular', 'Ocean waves')),
    ('sea_surface_temperature', 'Sea surface temperature', ('Popular', 'Temperature and pressure')),
    ('significant_height_of_combined_wind_waves_and_swell', 'Significant height of combined wind waves and swell', ('Popular', 'Ocean waves')),
    ('surface_pressure', 'Surface pressure', ('Popular', 'Temperature and pressure')),
    ('total_precipitation', 'Total precipitation', ('Popular', 'Precipitation and rain')),
    ('ice_temperature_layer_1', 'Ice temperature layer 1', ('Temperature and pressure',)),
    ('ice_temperature_layer_2', 'Ice temperature layer 2', ('Temperature and pressure',)),
    ('ice_temperature_layer_3', 'Ice temperature layer 3', ('Temperature and pressure',)),
    ('ice_temperature_layer_4', 'Ice temperature layer 4', ('Temperature and pressure',)),
    ('maximum_2m_temperature_since_previous_post_processing', 'Maximum 2m temperature since previous post-processing', ('Temperature and pressure',)),
    ('minimum_2m_temperature_since_previous_post_processing', 'Minimum 2m temperature since previous post-processing', ('Temperature and pressure',)),
    ('skin_temperature', 'Skin temperature', ('Temperature and pressure',)),
    ('100m_u_component_of_wind', '100m u-component of wind', ('Wind',)),
    ('100m_v_component_of_wind', '100m v-component of wind', ('Wind',)),
    ('10m_u_component_of_neutral_wind', '10m u-component of neutral wind', ('Wind',)),
    ('10m_v_component_of_neutral_wind', '10m v-component of neutral wind', ('Wind',)),
    ('10m_wind_gust_since_previous_post_processing', '10m wind gust since previous post-processing', ('Wind',)),
    ('instantaneous_10m_wind_gust', 'Instantaneous 10m wind gust', ('Wind',)),
    ('mean_boundary_layer_dissipation', 'Mean boundary layer dissipation', ('Mean rates',)),
    ('mean_convective_precipitation_rate', 'Mean convective precipitation rate', ('Mean rates',)),
    ('mean_convective_snowfall_rate', 'Mean convective snowfall rate', ('Mean rates',)),
    ('mean_eastward_gravity_wave_surface_stress', 'Mean eastward gravity wave surface stress', ('Mean rates',)),
    ('mean_eastward_turbulent_surface_stress', 'Mean eastward turbulent surface stress', ('Mean rates',)),
    ('mean_evaporation_rate', 'Mean evaporation rate', ('Mean rates',)),
    ('mean_gravity_wave_dissipation', 'Mean gravity wave dissipation', ('Mean rates',)),
    ('mean_large_scale_precipitation_fraction', 'Mean large-scale precipitation fraction', ('Mean rates',)),
    ('mean_large_scale_precipitation_rate', 'Mean large-scale precipitation rate', ('Mean rates',)),
    ('mean_large_scale_snowfall_rate', 'Mean large-scale snowfall rate', ('Mean rates',)),
    ('mean_northward_gravity_wave_surface_stress', 'Mean northward gravity wave surface stress', ('Mean rates',)),
    ('mean_northward_turbulent_surface_stress', 'Mean northward turbulent surface stress', ('Mean rates',)),
    ('mean_potential_evaporation_rate', 'Mean potential evaporation rate', ('Mean rates',)),
    ('mean_runoff_rate', 'Mean runoff rate', ('Mean rates',)),
    ('mean_snow_evaporation_rate', 'Mean snow evaporation rate', ('Mean rates',)),
    ('mean_snowfall_rate', 'Mean snowfall rate', ('Mean rates',)),
    ('mean_snowmelt_rate', 'Mean snowmelt rate', ('Mean rates',)),
    ('mean_sub_surface_runoff_rate', 'Mean sub-surface runoff rate', ('Mean rates',)),
    ('mean_surface_direct_short_wave_radiation_flux', 'Mean surface direct short-wave radiation flux', ('Mean rates',)),
    ('mean_surface_direct_short_wave_radiation_flux_clear_sky', 'Mean surface direct short-wave radiation flux, clear sky', ('Mean rates',)),
    ('mean_surface_downward_long_wave_radiation_flux', 'Mean surface downward long-wave radiation flux', ('Mean rates',)),
    ('mean_surface_downward_long_wave_radiation_flux_clear_sky', 'Mean surface downward long-wave radiation flux, clear sky', ('Mean rates',)),
    ('mean_surface_downward_short_wave_radiation_flux', 'Mean surface downward short-wave radiation flux', ('Mean rates',)),
    ('mean_surface_downward_short_wave_radiation_flux_clear_sky', 'Mean surface downward short-wave radiation flux, clear sky', ('Mean rates',)),
    ('mean_surface_downward_uv_radiation_flux', 'Mean surface downward UV radiation flux', ('Mean rates',)),
    ('mean_surface_latent_heat_flux', 'Mean surface latent heat flux', ('Mean rates',)),
    ('mean_surface_net_long_wave_radiation_flux', 'Mean surface net long-wave radiation flux', ('Mean rates',)),
    ('mean_surface_net_long_wave_radiation_flux_clear_sky', 'Mean surface net long-wave radiation flux, clear sky', ('Mean rates',)),
    ('mean_surface_net_short_wave_radiation_flux', 'Mean surface net short-wave radiation flux', ('Mean rates',)),
    ('mean_surface_net_short_wave_radiation_flux_clear_sky', 'Mean surface net short-wave radiation flux, clear sky', ('Mean rates',)),
    ('mean_surface_runoff_rate', 'Mean surface runoff rate', ('Mean rates',)),
    ('mean_surface_sensible_heat_flux', 'Mean surface sensible heat flux', ('Mean rates',)),
    ('mean_top_downward_short_wave_radiation_flux', 'Mean top downward short-wave radiation flux', ('Mean rates',)),
    ('mean_top_net_long_wave_radiation_flux', 'Mean top net long-wave radiation flux', ('Mean rates',)),
    ('mean_top_net_long_wave_radiation_flux_clear_sky', 'Mean top net long-wave radiation flux, clear sky', ('Mean rates',)),
    ('mean_top_net_short_wave_radiation_flux', 'Mean top net short-wave radiation flux', ('Mean rates',)),
    ('mean_top_net_short_wave_radiation_flux_clear_sky', 'Mean top net short-wave radiation flux, clear sky', ('Mean rates',)),
    ('mean_total_precipitation_rate', 'Mean total precipitation rate', ('Mean rates',)),
    ('mean_vertically_integrated_moisture_divergence', 'Mean vertically integrated moisture divergence', ('Mean rates',)),
    ('clear_sky_direct_solar_radiation_at_surface', 'Clear-sky direct solar radiation at surface', ('Radiation and heat',)),
    ('downward_uv_radiation_at_the_surface', 'Downward UV radiation at the surface', ('Radiation and heat',)),
    ('forecast_logarithm_of_surface_roughness_for_heat', 'Forecast logarithm of surface roughness for heat', ('Radiation and heat',)),
    ('instantaneous_surface_sensible_heat_flux', 'Instantaneous surface sensible heat flux', ('Radiation and heat',)),
    ('near_ir_albedo_for_diffuse_radiation', 'Near IR albedo for diffuse radiation', ('Radiation and heat',)),
    ('near_ir_albedo_for_direct_radiation', 'Near IR albedo for direct radiation', ('Radiation and heat',)),
    ('surface_latent_heat_flux', 'Surface latent heat flux', ('Radiation and heat',)),
    ('surface_net_solar_radiation', 'Surface net solar radiation', ('Radiation and heat',)),
    ('surface_net_solar_radiation_clear_sky', 'Surface net solar radiation, clear sky', ('Radiation and heat',)),
    ('surface_net_thermal_radiation', 'Surface net thermal radiation', ('Radiation and heat',)),
    ('surface_net_thermal_radiation_clear_sky', 'Surface net thermal radiation, clear sky', ('Radiation and heat',)),
    ('surface_sensible_heat_flux', 'Surface sensible heat flux', ('Radiation and heat',)),
    ('surface_solar_radiation_downward_clear_sky', 'Surface solar radiation downward, clear sky', ('Radiation and heat',)),
    ('surface_solar_radiation_downwards', 'Surface solar radiation downwards', ('Radiation and heat',)),
    ('surface_thermal_radiation_downward_clear_sky', 'Surface thermal radiation downward, clear sky', ('Radiation and heat',)),
    ('surface_thermal_radiation_downwards', 'Surface thermal radiation downwards', ('Radiation and heat',)),
    ('toa_incident_solar_radiation', 'TOA incident solar radiation', ('Radiation and heat',)),
    ('top_net_solar_radiation', 'Top net solar radiation', ('Radiation and heat',)),
    ('top_net_solar_radiation_clear_sky', 'Top net solar radiation, clear sky', ('Radiation and heat',)),
    ('top_net_thermal_radiation', 'Top net thermal radiation', ('Radiation and heat',)),
    ('top_net_thermal_radiation_clear_sky', 'Top net thermal radiation, clear sky', ('Radiation and heat',)),
    ('total_sky_direct_solar_radiation_at_surface', 'Total sky direct solar radiation at surface', ('Radiation and heat',)),
    ('uv_visible_albedo_for_diffuse_radiation', 'UV visible albedo for diffuse radiation', ('Radiation and heat',)),
    ('uv_visible_albedo_for_direct_radiation', 'UV visible albedo for direct radiation', ('Radiation and heat',)),
    ('cloud_base_height', 'Cloud base height', ('Clouds',)),
    ('high_cloud_cover', 'High cloud cover', ('Clouds',)),
    ('low_cloud_cover', 'Low cloud cover', ('Clouds',)),
    ('medium_cloud_cover', 'Medium cloud cover', ('Clouds',)),
    ('total_cloud_cover', 'Total cloud cover', ('Clouds',)),
    ('total_column_cloud_ice_water', 'Total column cloud ice water', ('Clouds',)),
    ('total_column_cloud_liquid_water', 'Total column cloud liquid water', ('Clouds',)),
    ('vertical_integral_of_divergence_of_cloud_frozen_water_flux', 'Vertical integral of divergence of cloud frozen water flux', ('Clouds', 'Vertical integrals')),
    ('vertical_integral_of_divergence_of_cloud_liquid_water_flux', 'Vertical integral of divergence of cloud liquid water flux', ('Clouds', 'Vertical integrals')),
    ('vertical_integral_of_eastward_cloud_frozen_water_flux', 'Vertical integral of eastward cloud frozen water flux', ('Clouds', 'Vertical integrals')),
    ('vertical_integral_of_eastward_cloud_liquid_water_flux', 'Vertical integral of eastward cloud liquid water flux', ('Clouds', 'Vertical integrals')),
    ('vertical_integral_of_northward_cloud_frozen_water_flux', 'Vertical integral of northward cloud frozen water flux', ('Clouds', 'Vertical integrals')),
    ('vertical_integral_of_northward_cloud_liquid_water_flux', 'Vertical integral of northward cloud liquid water flux', ('Clouds', 'Vertical integrals')),
    ('lake_bottom_temperature', 'Lake bottom temperature', ('Lakes',)),
    ('lake_cover', 'Lake cover', ('Lakes',)),
    ('lake_depth', 'Lake depth', ('Lakes',)),
    ('lake_ice_depth', 'Lake ice depth', ('Lakes',)),
    ('lake_ice_temperature', 'Lake ice temperature', ('Lakes',)),
    ('lake_mix_layer_depth', 'Lake mix-layer depth', ('Lakes',)),
    ('lake_mix_layer_temperature', 'Lake mix-layer temperature', ('Lakes',)),
    ('lake_shape_factor', 'Lake shape factor', ('Lakes',)),
    ('lake_total_layer_temperature', 'Lake total layer temperature', ('Lakes',)),
    ('evaporation', 'Evaporation', ('Evaporation and runoff',)),
    ('potential_evaporation', 'Potential evaporation', ('Evaporation and runoff',)),
    ('runoff', 'Runoff', ('Evaporation and runoff',)),
    ('sub_surface_runoff', 'Sub-surface runoff', ('Evaporation and runoff',)),
    ('surface_runoff', 'Surface runoff', ('Evaporation and runoff',)),
    ('convective_precipitation', 'Convective precipitation', ('Precipitation and rain',)),
    ('convective_rain_rate', 'Convective rain rate', ('Precipitation and rain',)),
    ('instantaneous_large_scale_surface_precipitation_fraction', 'Instantaneous large-scale surface precipitation fraction', ('Precipitation and rain',)),
    ('large_scale_rain_rate', 'Large scale rain rate', ('Precipitation and rain',)),
    ('large_scale_precipitation', 'Large-scale precipitation', ('Precipitation and rain',)),
    ('large_scale_precipitation_fraction', 'Large-scale precipitation fraction', ('Precipitation and rain',)),
    ('maximum_total_precipitation_rate_since_previous_post_processing', 'Maximum total precipitation rate since previous post-processing', ('Precipitation and rain',)),
    ('minimum_total_precipitation_rate_since_previous_post_processing', 'Minimum total precipitation rate since previous post-processing', ('Precipitation and rain',)),
    ('precipitation_type', 'Precipitation type', ('Precipitation and rain',)),
    ('total_column_rain_water', 'Total column rain water', ('Precipitation and rain',)),
    ('convective_snowfall', 'Convective snowfall', ('Snow',)),
    ('convective_snowfall_rate_water_equivalent', 'Convective snowfall rate water equivalent', ('Snow',)),
    ('large_scale_snowfall_rate_water_equivalent', 'Large scale snowfall rate water equivalent', ('Snow',)),
    ('large_scale_snowfall', 'Large-scale snowfall', ('Snow',)),
    ('snow_albedo', 'Snow albedo', ('Snow',)),
    ('snow_density', 'Snow density', ('Snow',)),
    ('snow_depth', 'Snow depth', ('Snow',)),
    ('snow_evaporation', 'Snow evaporation', ('Snow',)),
    ('snowfall', 'Snowfall', ('Snow',)),
    ('snowmelt', 'Snowmelt', ('Snow',)),
    ('temperature_of_snow_layer', 'Temperature of snow layer', ('Snow',)),
    ('total_column_snow_water', 'Total column snow water', ('Snow',)),
    ('soil_temperature_level_1', 'Soil temperature level 1', ('Soil',)),
    ('soil_temperature_level_2', 'Soil temperature level 2', ('Soil',)),
    ('soil_temperature_level_3', 'Soil temperature level 3', ('Soil',)),
    ('soil_temperature_level_4', 'Soil temperature level 4', ('Soil',)),
    ('soil_type', 'Soil type', ('Soil',)),
    ('volumetric_soil_water_layer_1', 'Volumetric soil water layer 1', ('Soil',)),
    ('volumetric_soil_water_layer_2', 'Volumetric soil water layer 2', ('Soil',)),
    ('volumetric_soil_water_layer_3', 'Volumetric soil water layer 3', ('Soil',)),
    ('volumetric_soil_water_layer_4', 'Volumetric soil water layer 4', ('Soil',)),
    ('vertical_integral_of_divergence_of_geopotential_flux', 'Vertical integral of divergence of geopotential flux', ('Vertical integrals',)),
    ('vertical_integral_of_divergence_of_kinetic_energy_flux', 'Vertical integral of divergence of kinetic energy flux', ('Vertical integrals',)),
    ('vertical_integral_of_divergence_of_mass_flux', 'Vertical integral of divergence of mass flux', ('Vertical integrals',)),
    ('vertical_integral_of_divergence_of_moisture_flux', 'Vertical integral of divergence of moisture flux', ('Vertical integrals',)),
    ('vertical_integral_of_divergence_of_ozone_flux', 'Vertical integral of divergence of ozone flux', ('Vertical integrals',)),
    ('vertical_integral_of_divergence_of_thermal_energy_flux', 'Vertical integral of divergence of thermal energy flux', ('Vertical integrals',)),
    ('vertical_integral_of_divergence_of_total_energy_flux', 'Vertical integral of divergence of total energy flux', ('Vertical integrals',)),
    ('vertical_integral_of_eastward_geopotential_flux', 'Vertical integral of eastward geopotential flux', ('Vertical integrals',)),
    ('vertical_integral_of_eastward_heat_flux', 'Vertical integral of eastward heat flux', ('Vertical integrals',)),
    ('vertical_integral_of_eastward_kinetic_energy_flux', 'Vertical integral of eastward kinetic energy flux', ('Vertical integrals',)),
    ('vertical_integral_of_eastward_mass_flux', 'Vertical integral of eastward mass flux', ('Vertical integrals',)),
    ('vertical_integral_of_eastward_ozone_flux', 'Vertical integral of eastward ozone flux', ('Vertical integrals',)),
    ('vertical_integral_of_eastward_total_energy_flux', 'Vertical integral of eastward total energy flux', ('Vertical integrals',)),
    ('vertical_integral_of_eastward_water_vapour_flux', 'Vertical integral of eastward water vapour flux', ('Vertical integrals',)),
    ('vertical_integral_of_energy_conversion', 'Vertical integral of energy conversion', ('Vertical integrals',)),
    ('vertical_integral_of_kinetic_energy', 'Vertical integral of kinetic energy', ('Vertical integrals',)),
    ('vertical_integral_of_mass_of_atmosphere', 'Vertical integral of mass of atmosphere', ('Vertical integrals',)),
    ('vertical_integral_of_mass_tendency', 'Vertical integral of mass tendency', ('Vertical integrals',)),
    ('vertical_integral_of_northward_geopotential_flux', 'Vertical integral of northward geopotential flux', ('Vertical integrals',)),
    ('vertical_integral_of_northward_heat_flux', 'Vertical integral of northward heat flux', ('Vertical integrals',)),
    ('vertical_integral_of_northward_kinetic_energy_flux', 'Vertical integral of northward kinetic energy flux', ('Vertical integrals',)),
    ('vertical_integral_of_northward_mass_flux', 'Vertical integral of northward mass flux', ('Vertical integrals',)),
    ('vertical_integral_of_northward_ozone_flux', 'Vertical integral of northward ozone flux', ('Vertical integrals',)),
    ('vertical_integral_of_northward_total_energy_flux', 'Vertical integral of northward total energy flux', ('Vertical integrals',)),
    ('vertical_integral_of_northward_water_vapour_flux', 'Vertical integral of northward water vapour flux', ('Vertical integrals',)),
    ('vertical_integral_of_potential_and_internal_energy', 'Vertical integral of potential and internal energy', ('Vertical integrals',)),
    ('vertical_integral_of_potential_internal_and_latent_energy', 'Vertical integral of potential, internal and latent energy', ('Vertical integrals',)),
    ('vertical_integral_of_temperature', 'Vertical integral of temperature', ('Vertical integrals',)),
    ('vertical_integral_of_thermal_energy', 'Vertical integral of thermal energy', ('Vertical integrals',)),
    ('vertical_integral_of_total_energy', 'Vertical integral of total energy', ('Vertical integrals',)),
    ('vertically_integrated_moisture_divergence', 'Vertically integrated moisture divergence', ('Vertical integrals',)),
    ('high_vegetation_cover', 'High vegetation cover', ('Vegetation',)),
    ('leaf_area_index_high_vegetation', 'Leaf area index, high vegetation', ('Vegetation',)),
    ('leaf_area_index_low_vegetation', 'Leaf area index, low vegetation', ('Vegetation',)),
    ('low_vegetation_cover', 'Low vegetation cover', ('Vegetation',)),
    ('type_of_high_vegetation', 'Type of high vegetation', ('Vegetation',)),
    ('type_of_low_vegetation', 'Type of low vegetation', ('Vegetation',)),
    ('air_density_over_the_oceans', 'Air density over the oceans', ('Ocean waves',)),
    ('coefficient_of_drag_with_waves', 'Coefficient of drag with waves', ('Ocean waves',)),
    ('free_convective_velocity_over_the_oceans', 'Free convective velocity over the oceans', ('Ocean waves',)),
    ('maximum_individual_wave_height', 'Maximum individual wave height', ('Ocean waves',)),
    ('mean_direction_of_total_swell', 'Mean direction of total swell', ('Ocean waves',)),
    ('mean_direction_of_wind_waves', 'Mean direction of wind waves', ('Ocean waves',)),
    ('mean_period_of_total_swell', 'Mean period of total swell', ('Ocean waves',)),
    ('mean_period_of_wind_waves', 'Mean period of wind waves', ('Ocean waves',)),
    ('mean_square_slope_of_waves', 'Mean square slope of waves', ('Ocean waves',)),
    ('mean_wave_direction_of_first_swell_partition', 'Mean wave direction of first swell partition', ('Ocean waves',)),
    ('mean_wave_direction_of_second_swell_partition', 'Mean wave direction of second swell partition', ('Ocean waves',)),
    ('mean_wave_direction_of_third_swell_partition', 'Mean wave direction of third swell partition', ('Ocean waves',)),
    ('mean_wave_period_based_on_first_moment', 'Mean wave period based on first moment', ('Ocean waves',)),
    ('mean_wave_period_based_on_first_moment_for_swell', 'Mean wave period based on first moment for swell', ('Ocean waves',)),
    ('mean_wave_period_based_on_first_moment_for_wind_waves', 'Mean wave period based on first moment for wind waves', ('Ocean waves',)),
    ('mean_wave_period_based_on_second_moment_for_swell', 'Mean wave period based on second moment for swell', ('Ocean waves',)),
    ('mean_wave_period_based_on_second_moment_for_wind_waves', 'Mean wave period based on second moment for wind waves', ('Ocean waves',)),
    ('mean_wave_period_of_first_swell_partition', 'Mean wave period of first swell partition', ('Ocean waves',)),
    ('mean_wave_period_of_second_swell_partition', 'Mean wave period of second swell partition', ('Ocean waves',)),
    ('mean_wave_period_of_third_swell_partition', 'Mean wave period of third swell partition', ('Ocean waves',)),
    ('mean_zero_crossing_wave_period', 'Mean zero-crossing wave period', ('Ocean waves',)),
    ('model_bathymetry', 'Model bathymetry', ('Ocean waves',)),
    ('normalized_energy_flux_into_ocean', 'Normalized energy flux into ocean', ('Ocean waves',)),
    ('normalized_energy_flux_into_waves', 'Normalized energy flux into waves', ('Ocean waves',)),
    ('normalized_stress_into_ocean', 'Normalized stress into ocean', ('Ocean waves',)),
    ('ocean_surface_stress_equivalent_10m_neutral_wind_direction', 'Ocean surface stress equivalent 10m neutral wind direction', ('Ocean waves',)),
    ('ocean_surface_stress_equivalent_10m_neutral_wind_speed', 'Ocean surface stress equivalent 10m neutral wind speed', ('Ocean waves',)),
    ('peak_wave_period', 'Peak wave period', ('Ocean waves',)),
    ('period_corresponding_to_maximum_individual_wave_height', 'Period corresponding to maximum individual wave height', ('Ocean waves',)),
    ('significant_height_of_total_swell', 'Significant height of total swell', ('Ocean waves',)),
    ('significant_height_of_wind_waves', 'Significant height of wind waves', ('Ocean waves',)),
    ('significant_wave_height_of_first_swell_partition', 'Significant wave height of first swell partition', ('Ocean waves',)),
    ('significant_wave_height_of_second_swell_partition', 'Significant wave height of second swell partition', ('Ocean waves',)),
    ('significant_wave_height_of_third_swell_partition', 'Significant wave height of third swell partition', ('Ocean waves',)),
    ('wave_spectral_directional_width', 'Wave spectral directional width', ('Ocean waves',)),
    ('wave_spectral_directional_width_for_swell', 'Wave spectral directional width for swell', ('Ocean waves',)),
    ('wave_spectral_directional_width_for_wind_waves', 'Wave spectral directional width for wind waves', ('Ocean waves',)),
    ('wave_spectral_kurtosis', 'Wave spectral kurtosis', ('Ocean waves',)),
    ('wave_spectral_peakedness', 'Wave spectral peakedness', ('Ocean waves',)),
    ('wave_spectral_skewness', 'Wave spectral skewness', ('Ocean waves',)),
    ('angle_of_sub_gridscale_orography', 'Angle of sub-gridscale orography', ('Other',)),
    ('anisotropy_of_sub_gridscale_orography', 'Anisotropy of sub-gridscale orography', ('Other',)),
    ('benjamin_feir_index', 'Benjamin-feir index', ('Other',)),
    ('boundary_layer_dissipation', 'Boundary layer dissipation', ('Other',)),
    ('boundary_layer_height', 'Boundary layer height', ('Other',)),
    ('charnock', 'Charnock', ('Other',)),
    ('convective_available_potential_energy', 'Convective available potential energy', ('Other',)),
    ('convective_inhibition', 'Convective inhibition', ('Other',)),
    ('duct_base_height', 'Duct base height', ('Other',)),
    ('eastward_gravity_wave_surface_stress', 'Eastward gravity wave surface stress', ('Other',)),
    ('eastward_turbulent_surface_stress', 'Eastward turbulent surface stress', ('Other',)),
    ('forecast_albedo', 'Forecast albedo', ('Other',)),
    ('forecast_surface_roughness', 'Forecast surface roughness', ('Other',)),
    ('friction_velocity', 'Friction velocity', ('Other',)),
    ('geopotential', 'Geopotential', ('Other',)),
    ('gravity_wave_dissipation', 'Gravity wave dissipation', ('Other',)),
    ('instantaneous_eastward_turbulent_surface_stress', 'Instantaneous eastward turbulent surface stress', ('Other',)),
    ('instantaneous_moisture_flux', 'Instantaneous moisture flux', ('Other',)),
    ('instantaneous_northward_turbulent_surface_stress', 'Instantaneous northward turbulent surface stress', ('Other',)),
    ('k_index', 'K index', ('Other',)),
    ('land_sea_mask', 'Land-sea mask', ('Other',)),
    ('mean_vertical_gradient_of_refractivity_inside_trapping_layer', 'Mean vertical gradient of refractivity inside trapping layer', ('Other',)),
    ('minimum_vertical_gradient_of_refractivity_inside_trapping_layer', 'Minimum vertical gradient of refractivity inside trapping layer', ('Other',)),
    ('northward_gravity_wave_surface_stress', 'Northward gravity wave surface stress', ('Other',)),
    ('northward_turbulent_surface_stress', 'Northward turbulent surface stress', ('Other',)),
    ('sea_ice_cover', 'Sea-ice cover', ('Other',)),
    ('skin_reservoir_content', 'Skin reservoir content', ('Other',)),
    ('slope_of_sub_gridscale_orography', 'Slope of sub-gridscale orography', ('Other',)),
    ('standard_deviation_of_filtered_subgrid_orography', 'Standard deviation of filtered subgrid orography', ('Other',)),
    ('standard_deviation_of_orography', 'Standard deviation of orography', ('Other',)),
    ('total_column_ozone', 'Total column ozone', ('Other',)),
    ('total_column_supercooled_liquid_water', 'Total column supercooled liquid water', ('Other',)),
    ('total_column_water', 'Total column water', ('Other',)),
    ('total_column_water_vapour', 'Total column water vapour', ('Other',)),
    ('total_totals_index', 'Total totals index', ('Other',)),
    ('trapping_layer_base_height', 'Trapping layer base height', ('Other',)),
    ('trapping_layer_top_height', 'Trapping layer top height', ('Other',)),
    ('u_component_stokes_drift', 'U-component stokes drift', ('Other',)),
    ('v_component_stokes_drift', 'V-component stokes drift', ('Other',)),
    ('zero_degree_level', 'Zero degree level', ('Other',)),
))


HANDLERS = {
    "cds-era5": ERA5Handler,
}


def build_parser() -> argparse.ArgumentParser:
    """Command line for one update check. An external scheduler runs it."""

    parser = argparse.ArgumentParser(
        prog="python -m gigaspatial.handlers.cds.era5",
        description="Update country-level ERA5 and ERA5-Land downloads from CDS.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    update = commands.add_parser(
        "update",
        help="Download hours published since the last successful country update.",
    )
    update.add_argument(
        "--handler",
        choices=["cds-era5", "cds-era5land", "all"],
        default="all",
        help="Which dataset to update. 'all' runs each dataset on its own.",
    )
    update.add_argument(
        "--country",
        action="append",
        default=None,
        help="Country name or ISO code. Repeat for several. Omit to update every configured country.",
    )
    update.add_argument(
        "--force",
        action="store_true",
        help="Check CDS even when the last check is inside update_frequency_hours.",
    )
    return parser


def _handler_map():
    mapping = dict(HANDLERS)
    if "cds-era5land" not in mapping:
        from gigaspatial.handlers.cds.era5land import ERA5LandHandler

        mapping["cds-era5land"] = ERA5LandHandler
    return mapping


def main(argv=None) -> int:
    """Run one update and print a JSON report. Exit 1 when a country fails."""

    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "update":
        mapping = _handler_map()
        names = list(mapping) if args.handler == "all" else [args.handler]
        reports = []
        for name in names:
            handler = mapping[name]()
            countries = args.country or [None]
            for country in countries:
                reports.append(handler.update(country, force=args.force))
        print(json.dumps(reports, indent=2, default=str))
        failed = any(
            country.get("status") == "error"
            for report in reports
            for country in report.get("countries", [])
        )
        return 1 if failed else 0
    parser.error(f"unknown command {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
