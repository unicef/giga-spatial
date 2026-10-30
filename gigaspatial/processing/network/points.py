"""Point inputs: GeoDataFrames, frames with latitude and longitude, or lists of (lat, lon) or Points."""

from typing import Any, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from pyproj import CRS

from gigaspatial.processing.geo import convert_to_geodataframe, detect_coordinate_columns
from gigaspatial.processing.network.geodesy import WGS84

RESERVED_NODE_COLUMNS = ("x", "y", "kind")


def is_wgs84(crs: Any) -> bool:
    return CRS.from_user_input(crs).to_epsg() == 4326


def point_coordinates(points: Any) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """The points' other columns, and their longitudes and latitudes in EPSG:4326.

    Args:
        points: A GeoDataFrame of points, a DataFrame with latitude and
            longitude columns (or a geometry column), or a list of
            ``(lat, lon)`` tuples or shapely Points.

    Returns:
        ``(attributes, x, y)``. ``attributes`` keeps the input's index and its
        columns other than the coordinates.
    """
    if isinstance(points, pd.DataFrame) and not isinstance(points, gpd.GeoDataFrame):
        if "geometry" in points.columns:
            points = convert_to_geodataframe(points)
        else:
            lat_column, lon_column = detect_coordinate_columns(points)
            x = points[lon_column].to_numpy(dtype=np.float64)
            y = points[lat_column].to_numpy(dtype=np.float64)
            attributes = points.drop(columns=[lat_column, lon_column])
            _check_missing(x, y)
            return attributes, x, y
    if isinstance(points, gpd.GeoDataFrame):
        gdf = points
        if gdf.crs is not None and not is_wgs84(gdf.crs):
            gdf = gdf.to_crs(WGS84)
        if not (gdf.geometry.geom_type == "Point").all():
            raise ValueError("Points must be Point geometries")
        x = gdf.geometry.x.to_numpy()
        y = gdf.geometry.y.to_numpy()
        attributes = pd.DataFrame(gdf.drop(columns=gdf.geometry.name))
    else:
        items = list(points)
        if items and all(isinstance(p, shapely.Point) for p in items):
            x = np.array([p.x for p in items], dtype=np.float64)
            y = np.array([p.y for p in items], dtype=np.float64)
        else:
            coords = np.asarray(items, dtype=np.float64).reshape(-1, 2)
            y, x = coords[:, 0], coords[:, 1]
        attributes = pd.DataFrame(index=pd.RangeIndex(len(x)))
    _check_missing(x, y)
    return attributes, x, y


def points_table(points: Any, id_column: Optional[str] = None) -> pd.DataFrame:
    """Points as a node table indexed by ``node_id``, with ``x``, ``y`` in EPSG:4326 and their attributes.

    Raises:
        ValueError: If IDs repeat, coordinates are missing, or a column is
            named ``x``, ``y``, or ``kind``.
    """
    attributes, x, y = point_coordinates(points)
    if id_column is not None:
        if id_column not in attributes.columns:
            raise ValueError(f"id_column {id_column!r} is not a column of the points")
        ids = pd.Index(attributes[id_column])
        attributes = attributes.drop(columns=id_column)
        attributes.index = ids
    attributes.index = attributes.index.rename("node_id")
    if not attributes.index.is_unique:
        duplicated = list(attributes.index[attributes.index.duplicated()].unique()[:5])
        raise ValueError(f"Point IDs must be unique; repeated: {duplicated}")
    clash = [c for c in RESERVED_NODE_COLUMNS if c in attributes.columns]
    if clash:
        raise ValueError(f"Point columns {clash} are reserved for node coordinates and kind")
    table = attributes.copy(deep=False)
    table.insert(0, "x", x)
    table.insert(1, "y", y)
    return table


def _check_missing(x: np.ndarray, y: np.ndarray) -> None:
    if np.isnan(x).any() or np.isnan(y).any():
        raise ValueError("Point coordinates must not be missing")
