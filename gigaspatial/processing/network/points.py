"""Point inputs: GeoDataFrames, frames with latitude and longitude, or lists of (lat, lon) or Points."""

from typing import Any, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from pyproj import CRS
from scipy.spatial import cKDTree

from gigaspatial.processing.geo import convert_to_geodataframe, detect_coordinate_columns
from gigaspatial.processing.network.geodesy import WGS84, haversine, meters_to_chord, unit_vectors

RESERVED_NODE_COLUMNS = ("x", "y", "kind")
RESERVED_POINT_SET_COLUMNS = (*RESERVED_NODE_COLUMNS, "bipartite")


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


def check_connect(connect: str, k: int, max_distance: Optional[float]) -> None:
    """Validate ``connect``, ``k``, and ``max_distance`` for point link searches."""
    if connect == "knn":
        if int(k) < 1:
            raise ValueError(f"k must be at least 1, got {k}")
    elif connect == "distance":
        if max_distance is None:
            raise ValueError('connect="distance" needs max_distance')
    else:
        raise ValueError(f'connect must be "knn" or "distance", got {connect!r}')


def within_set_links(
    lat: np.ndarray,
    lon: np.ndarray,
    connect: str,
    k: int,
    max_distance: Optional[float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unordered pairs ``(a, b)`` and great-circle lengths in meters within one point set."""
    check_connect(connect, k, max_distance)
    if connect == "knn":
        a, b = _knn_pairs(lat, lon, int(k))
    else:
        if len(lat) < 2:
            empty = np.empty(0, dtype=np.int64)
            a, b = empty, empty
        else:
            radius = float(meters_to_chord(max_distance)) * (1.0 + 1e-12) + 1e-15
            pairs = cKDTree(unit_vectors(lat, lon)).query_pairs(radius, output_type="ndarray")
            a, b = pairs[:, 0], pairs[:, 1]
    length = haversine(lat[a], lon[a], lat[b], lon[b])
    if max_distance is not None:
        keep = length <= max_distance
        a, b, length = a[keep], b[keep], length[keep]
    order = np.lexsort((b, a))
    return a[order], b[order], length[order]


def cross_set_links(
    lat: np.ndarray,
    lon: np.ndarray,
    target_lat: np.ndarray,
    target_lon: np.ndarray,
    connect: str,
    k: int,
    max_distance: Optional[float],
    tree: Optional[cKDTree] = None,
    candidates: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Origin rows, target positions, and great-circle lengths in meters across two point sets."""
    check_connect(connect, k, max_distance)
    n = len(lat)
    if n == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty, np.empty(0, dtype=np.float64)
    if candidates is None:
        candidates = np.arange(len(target_lat), dtype=np.int64)
    if tree is None:
        tree = cKDTree(unit_vectors(target_lat[candidates], target_lon[candidates]))
    xyz = unit_vectors(lat, lon)
    if connect == "knn":
        k_eff = min(int(k), len(candidates))
        _, j = tree.query(xyz, k=k_eff)
        j = np.asarray(j, dtype=np.int64).reshape(n, k_eff)
        point_row = np.repeat(np.arange(n, dtype=np.int64), k_eff)
        target_pos = candidates[j.ravel()]
    else:
        radius = float(meters_to_chord(max_distance)) * (1.0 + 1e-12) + 1e-15
        neighbours = tree.query_ball_point(xyz, radius)
        counts = np.array([len(nb) for nb in neighbours], dtype=np.int64)
        point_row = np.repeat(np.arange(n, dtype=np.int64), counts)
        picked = np.concatenate(neighbours).astype(np.int64) if counts.sum() else np.empty(0, dtype=np.int64)
        target_pos = candidates[picked]
    length = haversine(lat[point_row], lon[point_row], target_lat[target_pos], target_lon[target_pos])
    if max_distance is not None:
        keep = length <= max_distance
        point_row, target_pos, length = point_row[keep], target_pos[keep], length[keep]
    if connect == "distance" and len(point_row):
        order = np.lexsort((target_pos, length, point_row))
        point_row, target_pos, length = point_row[order], target_pos[order], length[order]
    return point_row, target_pos, length


def _knn_pairs(lat: np.ndarray, lon: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Unordered pairs joining each point to its k nearest neighbours."""
    n = len(lat)
    if n < 2:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty
    k = min(k, n - 1)
    xyz = unit_vectors(lat, lon)
    _, neighbours = cKDTree(xyz).query(xyz, k=k + 1)
    rows = np.broadcast_to(np.arange(n)[:, None], neighbours.shape)
    not_self = neighbours != rows
    keep = not_self & (np.cumsum(not_self, axis=1) <= k)
    a, b = rows[keep], neighbours[keep]
    pairs = np.unique(np.column_stack([np.minimum(a, b), np.maximum(a, b)]), axis=0)
    return pairs[:, 0], pairs[:, 1]
