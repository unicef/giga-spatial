import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Point, Polygon

from gigaspatial.processing.geo import (
    aggregate_points_to_zones,
    aggregate_polygons_to_zones,
)


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


@pytest.mark.parametrize("predicate", ["intersects", "within", "fractional"])
def test_aggregate_polygons_to_zones_count_without_value_columns_and_non_mutation(predicate):
    zones = _make_zones()
    polys = gpd.GeoDataFrame(
        {
            "val": [5.0, 10.0],
            "geometry": [
                Polygon([(0.2, 0.2), (0.2, 0.4), (0.4, 0.4), (0.4, 0.2)]),
                Polygon([(0.5, 0.5), (0.5, 0.8), (0.8, 0.8), (0.8, 0.5)]),
            ],
        },
        crs="EPSG:4326",
    )
    original_cols = list(polys.columns)
    original_copy = polys.copy()

    result = aggregate_polygons_to_zones(
        polys,
        zones,
        value_columns=None,
        aggregation="count",
        predicate=predicate,
        zone_id_column="zone_id",
        drop_geometry=True,
    )

    assert list(polys.columns) == original_cols
    assert polys.equals(original_copy)
    assert list(result.columns) == ["zone_id", "zone_tag", "polygon_count"]
    by_zone = result.set_index("zone_id")["polygon_count"]
    assert by_zone["z1"] == 2
    assert by_zone["z2"] == 0


def test_aggregate_polygons_to_zones_count_output_column_and_suffix_and_defaults():
    zones = _make_zones()
    polys = gpd.GeoDataFrame(
        {
            "val": [5.0, 15.0],
            "geometry": [
                Polygon([(0.2, 0.2), (0.2, 0.4), (0.4, 0.4), (0.4, 0.2)]),
                Polygon([(0.5, 0.5), (0.5, 0.8), (0.8, 0.8), (0.8, 0.5)]),
            ],
        },
        crs="EPSG:4326",
    )

    # Default aggregation when omitted: "count" if value_columns is None, "sum" otherwise
    cnt_default = aggregate_polygons_to_zones(
        polys, zones, output_suffix="_n", drop_geometry=True
    )
    assert "polygon_count_n" in cnt_default.columns
    assert cnt_default.set_index("zone_id").loc["z1", "polygon_count_n"] == 2

    cnt_explicit = aggregate_polygons_to_zones(
        polys,
        zones,
        value_columns=None,
        aggregation="count",
        output_column="custom_cnt",
        drop_geometry=True,
    )
    assert "custom_cnt" in cnt_explicit.columns
    assert cnt_explicit.set_index("zone_id").loc["z1", "custom_cnt"] == 2

    sum_default = aggregate_polygons_to_zones(
        polys,
        zones,
        value_columns="val",
        output_column="total_val",
        drop_geometry=True,
    )
    by_zone = sum_default.set_index("zone_id")["total_val"]
    assert by_zone["z1"] == pytest.approx(20.0)
    assert np.isnan(by_zone["z2"])


def test_aggregate_polygons_to_zones_validation_errors():
    zones = _make_zones()
    polys = gpd.GeoDataFrame(
        {
            "a": [1.0],
            "b": [2.0],
            "geometry": [Polygon([(0.2, 0.2), (0.2, 0.4), (0.4, 0.4), (0.4, 0.2)])],
        },
        crs="EPSG:4326",
    )

    with pytest.raises(ValueError, match="value_columns"):
        aggregate_polygons_to_zones(polys, zones, value_columns=None, aggregation="sum")

    with pytest.raises(ValueError, match="output_column"):
        aggregate_polygons_to_zones(
            polys,
            zones,
            value_columns=["a", "b"],
            aggregation="sum",
            output_column="single",
        )


def test_aggregate_polygons_to_zones_empty_and_non_overlapping_preserve_columns():
    zones = _make_zones()
    empty_polys = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    far_polys = gpd.GeoDataFrame(
        {
            "val": [10.0],
            "geometry": [Polygon([(10, 10), (10, 11), (11, 11), (11, 10)])],
        },
        crs="EPSG:4326",
    )

    for polys in (empty_polys, far_polys):
        cnt = aggregate_polygons_to_zones(
            polys, zones, value_columns=None, aggregation="count", drop_geometry=True
        )
        assert list(cnt.columns) == ["zone_id", "zone_tag", "polygon_count"]
        assert "geometry" not in cnt.columns
        assert list(cnt["polygon_count"]) == [0, 0]

        val_cnt = aggregate_polygons_to_zones(
            polys,
            zones,
            value_columns="val",
            aggregation="count",
            output_suffix="_c",
            drop_geometry=True,
        )
        assert list(val_cnt.columns) == ["zone_id", "zone_tag", "val_c"]
        assert list(val_cnt["val_c"]) == [0, 0]

        for pred in ("intersects", "fractional"):
            val_sum = aggregate_polygons_to_zones(
                polys,
                zones,
                value_columns="val",
                aggregation="sum",
                predicate=pred,
                output_column="val_sum",
                drop_geometry=True,
            )
            assert list(val_sum.columns) == ["zone_id", "zone_tag", "val_sum"]
            assert val_sum["val_sum"].isna().all()


