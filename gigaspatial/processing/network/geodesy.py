"""Great-circle distances in meters, matching the formula and radius osmnx uses for road lengths."""

import numpy as np
import pandas as pd
import shapely

EARTH_RADIUS_M = 6_371_009.0
WGS84 = "EPSG:4326"


def haversine(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Great-circle distance in meters between coordinates in decimal degrees."""
    y1 = np.deg2rad(lat1)
    y2 = np.deg2rad(lat2)
    dy = y2 - y1
    dx = np.deg2rad(lon2) - np.deg2rad(lon1)
    h = np.sin(dy / 2) ** 2 + np.cos(y1) * np.cos(y2) * np.sin(dx / 2) ** 2
    return 2 * np.arcsin(np.sqrt(np.minimum(1.0, h))) * EARTH_RADIUS_M


def line_lengths(geometries) -> np.ndarray:
    """Great-circle length in meters along each line in EPSG:4326; NaN where the geometry is missing."""
    geoms = np.asarray(geometries, dtype=object)
    out = np.full(len(geoms), np.nan)
    present = ~pd.isna(geoms)
    if not present.any():
        return out
    coords, index = shapely.get_coordinates(geoms[present], return_index=True)
    same_line = index[1:] == index[:-1]
    segments = haversine(coords[:-1, 1], coords[:-1, 0], coords[1:, 1], coords[1:, 0])
    out[present] = np.bincount(
        index[1:][same_line], weights=segments[same_line], minlength=int(present.sum())
    )
    return out


def unit_vectors(lat, lon) -> np.ndarray:
    """Points on the unit sphere, so a KD-tree on them orders neighbours by great-circle distance."""
    y = np.deg2rad(np.asarray(lat, dtype=np.float64))
    x = np.deg2rad(np.asarray(lon, dtype=np.float64))
    cos_y = np.cos(y)
    return np.column_stack([cos_y * np.cos(x), cos_y * np.sin(x), np.sin(y)])


def chord_to_meters(chord) -> np.ndarray:
    """Great-circle meters for a chord length between unit vectors."""
    return 2 * EARTH_RADIUS_M * np.arcsin(np.clip(np.asarray(chord) / 2, 0.0, 1.0))


def meters_to_chord(meters) -> np.ndarray:
    """Chord length between unit vectors for a great-circle distance in meters."""
    return 2 * np.sin(np.asarray(meters, dtype=np.float64) / (2 * EARTH_RADIUS_M))
