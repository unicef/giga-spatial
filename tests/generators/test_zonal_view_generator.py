import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Point, Polygon

from gigaspatial.core.io.local_data_store import LocalDataStore
from gigaspatial.generators.zonal.geometry import GeometryBasedZonalViewGenerator


def _make_generator(tmp_path) -> GeometryBasedZonalViewGenerator:
    zones = gpd.GeoDataFrame(
        {
            "zone_id": ["z1", "z2"],
            "geometry": [
                Polygon([(0, 0), (0, 1), (1, 1), (1, 0)]),
                Polygon([(2, 2), (2, 3), (3, 3), (3, 2)]),
            ],
        },
        crs="EPSG:4326",
    )
    return GeometryBasedZonalViewGenerator(
        zones, data_store=LocalDataStore(base_path=tmp_path)
    )


def test_map_points_value_columns_with_default_count_aggregation(tmp_path):
    gen = _make_generator(tmp_path)
    pts = gpd.GeoDataFrame(
        {
            "tests": [10.0, np.nan, 20.0],
            "geometry": [Point(0.4, 0.4), Point(0.5, 0.5), Point(0.6, 0.6)],
        },
        crs="EPSG:4326",
    )

    mapped = gen.map_points(pts, value_columns="tests")

    assert mapped["z1"] == pytest.approx(2.0)
    assert mapped["z2"] == pytest.approx(0.0)


def test_map_points_multiple_value_columns_with_count_aggregation(tmp_path):
    gen = _make_generator(tmp_path)
    pts = gpd.GeoDataFrame(
        {
            "a": [1.0, np.nan],
            "b": ["x", "y"],
            "geometry": [Point(0.4, 0.4), Point(0.6, 0.6)],
        },
        crs="EPSG:4326",
    )

    mapped = gen.map_points(pts, value_columns=["a", "b"], aggregation="count")

    assert mapped["a"]["z1"] == pytest.approx(1.0)
    assert mapped["a"]["z2"] == pytest.approx(0.0)
    assert mapped["b"]["z1"] == pytest.approx(2.0)
    assert mapped["b"]["z2"] == pytest.approx(0.0)

