"""Classification of the sources accepted by dataset handlers."""
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry


class SourceKind(Enum):
    """What a handler source refers to."""

    PATHS = "paths"
    COUNTRY = "country"
    GEOMETRY = "geometry"
    GEODATAFRAME = "geodataframe"
    DATAFRAME = "dataframe"
    POINTS = "points"
    OTHER = "other"


DATA_FILE_SUFFIXES = frozenset(
    {
        ".csv",
        ".fgb",
        ".feather",
        ".geojson",
        ".gpkg",
        ".gz",
        ".hgt",
        ".json",
        ".kml",
        ".nc",
        ".parquet",
        ".pbf",
        ".pmtiles",
        ".shp",
        ".tif",
        ".tiff",
        ".txt",
        ".xls",
        ".xlsx",
        ".zip",
    }
)


def _is_path_string(value: str) -> bool:
    return "/" in value or "\\" in value or Path(value).suffix.lower() in DATA_FILE_SUFFIXES


def classify_source(source: Any) -> SourceKind:
    """
    Classify a handler source.

    A string is a path when it contains a path separator or ends in a known
    data file suffix; any other string is a country name or code. A
    collection is a list of paths when every item is a string or ``Path``,
    and a list of points when every item is a ``Point`` or a coordinate pair.
    """
    if isinstance(source, Path):
        return SourceKind.PATHS
    if isinstance(source, str):
        return SourceKind.PATHS if _is_path_string(source) else SourceKind.COUNTRY
    if isinstance(source, BaseGeometry):
        return SourceKind.GEOMETRY
    if isinstance(source, gpd.GeoDataFrame):
        return SourceKind.GEODATAFRAME
    if isinstance(source, pd.DataFrame):
        return SourceKind.DATAFRAME
    if isinstance(source, (list, tuple, set)) and all(
        isinstance(p, (str, Path)) for p in source
    ):
        return SourceKind.PATHS
    if isinstance(source, Iterable) and not isinstance(source, dict):
        items = list(source)
        if all(isinstance(p, (Point, tuple, list)) for p in items):
            return SourceKind.POINTS
    return SourceKind.OTHER


def is_path_source(source: Any) -> bool:
    """Return True if the source names stored files rather than an area."""
    return classify_source(source) is SourceKind.PATHS
