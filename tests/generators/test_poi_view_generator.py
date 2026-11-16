import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point, Polygon

from gigaspatial import PoiViewGenerator
from gigaspatial.core.io.local_data_store import LocalDataStore


def _make_pois(tmp_path) -> PoiViewGenerator:
    pois = gpd.GeoDataFrame(
        {
            "poi_id": ["poi_1", "poi_2"],
            "geometry": [Point(0.0, 0.0), Point(1.0, 0.0)],
        },
        crs="EPSG:4326",
    )
    return PoiViewGenerator(pois, data_store=LocalDataStore(base_path=tmp_path))


def _make_point_data() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {
            "tests": [10.0, 20.0, 99.0],
            "geometry": [Point(0.001, 0.0), Point(0.002, 0.0), Point(0.1, 0.0)],
        },
        crs="EPSG:4326",
    )


def test_map_zonal_stats_point_data_with_value_column_and_custom_output(tmp_path):
    gen = _make_pois(tmp_path)
    result = gen.map_zonal_stats(
        _make_point_data(),
        stat="sum",
        map_radius_meters=500,
        output_column="tests_sum",
        value_column="tests",
    )

    assert list(result.columns) == ["poi_id", "tests_sum"]
    by_poi = result.set_index("poi_id")["tests_sum"]
    assert by_poi["poi_1"] == pytest.approx(30.0)
    assert pd.isna(by_poi["poi_2"])


def test_map_zonal_stats_point_data_with_value_column_default_output(tmp_path):
    gen = _make_pois(tmp_path)
    result = gen.map_zonal_stats(
        _make_point_data(),
        stat="mean",
        map_radius_meters=500,
        value_column="tests",
    )

    assert list(result.columns) == ["poi_id", "tests"]
    by_poi = result.set_index("poi_id")["tests"]
    assert by_poi["poi_1"] == pytest.approx(15.0)
    assert pd.isna(by_poi["poi_2"])


def test_map_zonal_stats_point_data_without_value_column_counts_points(tmp_path):
    gen = _make_pois(tmp_path)
    result = gen.map_zonal_stats(
        _make_point_data(),
        map_radius_meters=500,
        output_column="pts_count",
    )

    assert list(result.columns) == ["poi_id", "pts_count"]
    by_poi = result.set_index("poi_id")["pts_count"]
    assert by_poi["poi_1"] == 2
    assert by_poi["poi_2"] == 0


def test_map_zonal_stats_polygon_data_with_value_column(tmp_path):
    gen = _make_pois(tmp_path)
    polygons = gpd.GeoDataFrame(
        {
            "tests": [10.0, 20.0, 99.0],
            "geometry": [
                Polygon([(0.001, -0.001), (0.002, -0.001), (0.002, 0.001), (0.001, 0.001)]),
                Polygon([(0.002, -0.001), (0.003, -0.001), (0.003, 0.001), (0.002, 0.001)]),
                Polygon([(0.1, -0.001), (0.101, -0.001), (0.101, 0.001), (0.1, 0.001)]),
            ],
        },
        crs="EPSG:4326",
    )

    result = gen.map_zonal_stats(
        polygons,
        stat="sum",
        map_radius_meters=500,
        output_column="poly_tests",
        value_column="tests",
    )

    assert list(result.columns) == ["poi_id", "poly_tests"]
    by_poi = result.set_index("poi_id")["poly_tests"]
    assert by_poi["poi_1"] == pytest.approx(30.0)
    assert pd.isna(by_poi["poi_2"])


def test_map_zonal_stats_point_data_missing_value_column_raises(tmp_path):
    gen = _make_pois(tmp_path)
    with pytest.raises(ValueError, match="Value column 'missing' not found"):
        gen.map_zonal_stats(
            _make_point_data(),
            stat="sum",
            map_radius_meters=500,
            value_column="missing",
        )

