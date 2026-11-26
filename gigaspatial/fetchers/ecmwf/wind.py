"""
Wind speed from 10u and 10v, cropped to a tropical-cyclone track.

10u and 10v are the 10 m eastward and northward wind components, in m/s.
Wind speed is the horizontal magnitude, ``hypot(10u, 10v)``, on the native
grid. Values are not resampled.

Maximum 10 m gust is the IFS field ``10fg`` through step 90 h and ``10fg3``
after that. AIFS open data does not publish that gust field. When it is
present it is copied onto the same cells as the wind components, so each
storm window carries the track area, the wind speed, and the gust together.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry

from gigaspatial.fetchers.ecmwf.catalogue import GUST_SHORT_NAMES

DEFAULT_MARGIN_DEGREES = 5.0
U_NAMES = ("10u", "u10")
V_NAMES = ("10v", "v10")
LAT_NAMES = ("latitude", "lat")
LON_NAMES = ("longitude", "lon")


@dataclass
class StormWindow:
    """
    One storm's forecast fields on the box around its track.

    Attributes:
        storm_id: Storm name or invest identifier.
        member: Ensemble member, when the track table has one.
        system: Forecasting system that produced the track.
        area: Box around the track, in EPSG:4326. A track that crosses the
            antimeridian is a MultiPolygon of the two side boxes.
        fields: Native-grid subset. ``wind_speed`` is m/s. ``gust`` is the
            maximum 10 m gust, in m/s, when the forecast contained one.
    """

    storm_id: str
    member: Optional[int]
    system: Optional[str]
    area: BaseGeometry
    fields: xr.Dataset


def _pick(dataset: xr.Dataset, names: Sequence[str]) -> str:
    for name in names:
        if name in dataset:
            return name
    found = ", ".join(str(name) for name in dataset.data_vars)
    raise KeyError(f"None of {list(names)} are in the dataset (variables: {found}).")


def add_wind_speed(dataset: xr.Dataset) -> xr.Dataset:
    """
    Add ``wind_speed`` = hypot(10u, 10v), in m/s.

    Component names may be the open-data short names ``10u`` and ``10v`` or
    the cfgrib names ``u10`` and ``v10``.
    """
    u_name = _pick(dataset, U_NAMES)
    v_name = _pick(dataset, V_NAMES)
    speed = np.hypot(dataset[u_name], dataset[v_name])
    speed.name = "wind_speed"
    speed.attrs["units"] = "m s-1"
    speed.attrs["long_name"] = "10 metre wind speed"
    return dataset.assign(wind_speed=speed)


def add_gust(dataset: xr.Dataset) -> xr.Dataset:
    """
    Copy the maximum 10 m gust onto ``gust`` when the file has one.

    ``10fg`` is preferred. ``10fg3`` is used when that is the name present.
    A dataset with neither name is returned unchanged.
    """
    for name in GUST_SHORT_NAMES:
        if name in dataset:
            gust = dataset[name]
            gust.name = "gust"
            gust.attrs.setdefault("units", "m s-1")
            gust.attrs.setdefault("long_name", "Maximum 10 metre wind gust")
            return dataset.assign(gust=gust)
    return dataset


def signed_longitudes(values) -> np.ndarray:
    """Wrap longitudes into the range -180 to 180."""
    array = np.asarray(values, dtype=float)
    return np.mod(array + 180.0, 360.0) - 180.0


def track_boxes(
    longitudes,
    latitudes,
    margin_degrees: float = DEFAULT_MARGIN_DEGREES,
) -> List[BaseGeometry]:
    """
    Return the box, or two boxes, around a track.

    A track whose longitudes span more than 180 degrees is treated as
    crossing the antimeridian, and the result is the two boxes on either
    side. Latitude is clipped to -90 and 90.
    """
    lons = signed_longitudes(longitudes)
    lats = np.asarray(latitudes, dtype=float)
    if lons.size == 0 or np.all(np.isnan(lons)):
        raise ValueError("Track has no positions to build an area from.")
    finite = np.isfinite(lons) & np.isfinite(lats)
    lons = lons[finite]
    lats = lats[finite]
    if lons.size == 0:
        raise ValueError("Track has no positions to build an area from.")
    working = lons
    if float(np.nanmax(lons) - np.nanmin(lons)) > 180.0:
        working = np.where(lons < 0.0, lons + 360.0, lons)
    margin = float(margin_degrees)
    lon_min = float(np.min(working) - margin)
    lon_max = float(np.max(working) + margin)
    lat_min = max(-90.0, float(np.min(lats) - margin))
    lat_max = min(90.0, float(np.max(lats) + margin))
    if lon_max <= 180.0 and lon_min >= -180.0:
        return [box(lon_min, lat_min, lon_max, lat_max)]
    boxes = []
    if lon_min < -180.0:
        boxes.append(box(-180.0, lat_min, min(180.0, lon_max), lat_max))
        boxes.append(box(lon_min + 360.0, lat_min, 180.0, lat_max))
    elif lon_max > 180.0:
        boxes.append(box(max(-180.0, lon_min), lat_min, 180.0, lat_max))
        boxes.append(box(-180.0, lat_min, lon_max - 360.0, lat_max))
    else:
        boxes.append(box(lon_min, lat_min, lon_max, lat_max))
    return [item for item in boxes if not item.is_empty]


def _coord(dataset: xr.Dataset, names: Sequence[str]) -> str:
    for name in names:
        if name in dataset.coords or name in dataset.dims:
            return name
    raise KeyError(f"Dataset has none of the coordinates {list(names)}.")


def _descending(values: np.ndarray) -> bool:
    finite = values[np.isfinite(values)]
    return finite.size >= 2 and float(finite[0]) > float(finite[-1])


def _dataset_lon_window(lon_min: float, lon_max: float, coordinates: np.ndarray):
    """Express a -180..180 window in the dataset's longitude convention."""
    sample = coordinates[np.isfinite(coordinates)]
    if sample.size and float(np.nanmax(sample)) > 180.0:
        def shift(value: float) -> float:
            return value + 360.0 if value < 0.0 else value

        return shift(lon_min), shift(lon_max)
    return lon_min, lon_max


def crop_dataset(dataset: xr.Dataset, area: BaseGeometry) -> xr.Dataset:
    """
    Select native cells whose centres lie inside ``area``'s bounds.

    A MultiPolygon is cropped one part at a time and concatenated on
    longitude. Nothing is interpolated.
    """
    if area.geom_type == "MultiPolygon":
        parts = list(area.geoms)
    elif area.geom_type == "GeometryCollection":
        parts = [part for part in area.geoms if not part.is_empty]
    else:
        parts = [area]
    pieces = [_crop_bounds(dataset, part.bounds) for part in parts if not part.is_empty]
    if not pieces:
        return dataset.isel({_coord(dataset, LAT_NAMES): slice(0, 0)})
    if len(pieces) == 1:
        return pieces[0]
    lon_name = _coord(dataset, LON_NAMES)
    return xr.concat(pieces, dim=lon_name)


def _crop_bounds(dataset: xr.Dataset, bounds: Tuple[float, float, float, float]) -> xr.Dataset:
    lon_min, lat_min, lon_max, lat_max = bounds
    lat_name = _coord(dataset, LAT_NAMES)
    lon_name = _coord(dataset, LON_NAMES)
    latitudes = np.asarray(dataset[lat_name].values)
    lat_slice = (
        slice(lat_max, lat_min) if _descending(latitudes) else slice(lat_min, lat_max)
    )
    longitudes = np.asarray(dataset[lon_name].values, dtype=float)
    start, stop = _dataset_lon_window(lon_min, lon_max, longitudes)
    if start <= stop:
        return dataset.sel({lat_name: lat_slice, lon_name: slice(start, stop)})
    left = dataset.sel({lat_name: lat_slice, lon_name: slice(start, None)})
    right = dataset.sel({lat_name: lat_slice, lon_name: slice(None, stop)})
    return xr.concat([left, right], dim=lon_name)


def _column(frame: pd.DataFrame, names: Sequence[str]) -> Optional[str]:
    lowered = {str(column).lower(): column for column in frame.columns}
    for name in names:
        if name in lowered:
            return lowered[name]
    return None


def _member_value(raw) -> Optional[int]:
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return None
    try:
        if pd.isna(raw):
            return None
    except TypeError:
        pass
    return int(raw)


def storm_groups(tracks: pd.DataFrame):
    """Yield ``(storm_id, member, system, frame)`` for each track."""
    storm_col = _column(tracks, ("storm_id", "storm", "name"))
    if storm_col is None:
        raise KeyError("Track table needs a storm_id column.")
    member_col = _column(tracks, ("member", "number", "ensemble_member"))
    system_col = _column(tracks, ("system",))
    group_cols = [storm_col]
    if member_col is not None:
        group_cols.append(member_col)
    grouped = tracks.groupby(group_cols, dropna=False, sort=False)
    for key, frame in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        storm_id = str(key[0])
        member = _member_value(key[1]) if member_col is not None else None
        system = None
        if system_col is not None and len(frame):
            raw = frame[system_col].iloc[0]
            system = None if raw is None or (isinstance(raw, float) and math.isnan(raw)) else str(raw)
        yield storm_id, member, system, frame


def pair_storm_fields(
    tracks: pd.DataFrame,
    dataset: xr.Dataset,
    margin_degrees: float = DEFAULT_MARGIN_DEGREES,
) -> List[StormWindow]:
    """
    Crop wind and gust onto a box around each storm track.

    Wind speed is computed from 10u and 10v first, then each storm keeps the
    cells inside its own box. Gust, when the dataset has ``10fg`` or
    ``10fg3``, is the same cells.
    """
    prepared = add_gust(add_wind_speed(dataset))
    lat_col = _column(tracks, ("latitude", "lat"))
    lon_col = _column(tracks, ("longitude", "lon"))
    if lat_col is None or lon_col is None:
        if "geometry" not in getattr(tracks, "columns", []):
            raise KeyError("Track table needs latitude and longitude.")
        latitudes = tracks.geometry.y
        longitudes = tracks.geometry.x
        working = tracks.assign(_lat=latitudes, _lon=longitudes)
        lat_col, lon_col = "_lat", "_lon"
    else:
        working = tracks
    windows = []
    for storm_id, member, system, frame in storm_groups(working):
        boxes = track_boxes(frame[lon_col], frame[lat_col], margin_degrees)
        if len(boxes) == 1:
            area = boxes[0]
        else:
            area = boxes[0]
            for extra in boxes[1:]:
                area = area.union(extra)
        windows.append(
            StormWindow(
                storm_id=storm_id,
                member=member,
                system=system,
                area=area,
                fields=crop_dataset(prepared, area),
            )
        )
    return windows
