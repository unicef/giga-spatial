import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Point, Polygon

from gigaspatial.processing.geo import aggregate_points_to_zones


def _make_zones() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {
            "zone_id": ["z1", "z2"],
            "zone_tag": ["alpha", "beta"],
            "geometry": [
                Polygon([(0, 0), (0, 1), (1, 1), (1, 0)]),
                Polygon([(2, 2), (2, 3), (3, 3), (3, 2)]),
            ],
        },
        crs="EPSG:4326",
    )


def test_aggregate_points_to_zones_count_without_value_columns():
    zones = _make_zones()
    pts = gpd.GeoDataFrame(
        {"tests": [10.0, np.nan], "geometry": [Point(0.5, 0.5), Point(0.6, 0.6)]},
        crs="EPSG:4326",
    )

    result = aggregate_points_to_zones(
        pts, zones, aggregation="count", zone_id_column="zone_id", drop_geometry=True
    )

    assert list(result.columns) == ["zone_id", "zone_tag", "point_count"]
    by_zone = result.set_index("zone_id")["point_count"]
    assert by_zone["z1"] == 2
    assert by_zone["z2"] == 0


def test_aggregate_points_to_zones_count_with_single_value_column():
    zones = _make_zones()
    pts = gpd.GeoDataFrame(
        {
            "tests": [10.0, np.nan, 20.0],
            "geometry": [Point(0.4, 0.4), Point(0.5, 0.5), Point(0.6, 0.6)],
        },
        crs="EPSG:4326",
    )

    result = aggregate_points_to_zones(
        pts,
        zones,
        value_columns="tests",
        aggregation="count",
        zone_id_column="zone_id",
        drop_geometry=True,
    )

    assert list(result.columns) == ["zone_id", "zone_tag", "tests"]
    by_zone = result.set_index("zone_id")["tests"]
    assert by_zone["z1"] == pytest.approx(2.0)
    assert by_zone["z2"] == pytest.approx(0.0)


def test_aggregate_points_to_zones_count_with_multiple_value_columns():
    zones = _make_zones()
    pts = gpd.GeoDataFrame(
        {
            "a": [1.0, np.nan],
            "b": ["x", "y"],
            "geometry": [Point(0.4, 0.4), Point(0.6, 0.6)],
        },
        crs="EPSG:4326",
    )

    result = aggregate_points_to_zones(
        pts,
        zones,
        value_columns=["a", "b"],
        aggregation="count",
        zone_id_column="zone_id",
        drop_geometry=True,
    )

    by_zone = result.set_index("zone_id")
    assert by_zone.loc["z1", "a"] == pytest.approx(1.0)
    assert by_zone.loc["z2", "a"] == pytest.approx(0.0)
    assert by_zone.loc["z1", "b"] == pytest.approx(2.0)
    assert by_zone.loc["z2", "b"] == pytest.approx(0.0)


def test_aggregate_points_to_zones_non_aggregated_zone_column_collision_safe():
    zones = _make_zones()
    pts = gpd.GeoDataFrame(
        {
            "tests": [10.0, 20.0],
            "zone_tag": ["pt1", "pt2"],
            "geometry": [Point(0.4, 0.4), Point(0.6, 0.6)],
        },
        crs="EPSG:4326",
    )

    result = aggregate_points_to_zones(
        pts,
        zones,
        value_columns="tests",
        aggregation="sum",
        zone_id_column="zone_id",
        drop_geometry=True,
    )

    by_zone = result.set_index("zone_id")
    assert by_zone.loc["z1", "zone_tag"] == "alpha"
    assert by_zone.loc["z1", "tests"] == pytest.approx(30.0)

