import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Point, Polygon

from gigaspatial.processing import geo as geo_module
from gigaspatial.processing.geo import (
    add_area_in_meters,
    aggregate_points_to_zones,
    aggregate_polygons_to_zones,
    annotate_with_admin_regions,
    buffer_geodataframe,
    estimate_utm_crs_with_fallback,
    map_points_within_polygons,
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


def test_map_points_within_polygons_does_not_mutate_input():
    polygons = _make_zones()
    points = gpd.GeoDataFrame(
        {
            "pid": ["p_out", "p_in"],
            "geometry": [Point(5.0, 5.0), Point(0.5, 0.5)],
        },
        index=["row_b", "row_a"],
        crs="EPSG:4326",
    )
    original_columns = list(points.columns)
    original_index = points.index.copy()
    original_crs = points.crs
    original_snapshot = points.copy(deep=True)

    result = map_points_within_polygons(points, polygons)

    assert result is not points
    assert list(points.columns) == original_columns
    assert "is_within" not in points.columns
    assert points.index.equals(original_index)
    assert points.crs == original_crs
    assert points.equals(original_snapshot)

    assert list(result.index) == ["row_b", "row_a"]
    assert list(result["pid"]) == ["p_out", "p_in"]
    assert list(result["is_within"]) == [False, True]
    assert result.crs == original_crs


class _FakeAdminBoundaries:
    def __init__(self, gdf: gpd.GeoDataFrame):
        self._gdf = gdf

    def to_geodataframe(self) -> gpd.GeoDataFrame:
        return self._gdf.copy()


def _make_admin_fixtures(reverse_order: bool = False):
    admin1_rows = [
        {
            "boundary_id": "ADM1_B",
            "name": "Region B",
            "geometry": Polygon([(0, 0), (0, 2), (1, 2), (1, 0)]),
        },
        {
            "boundary_id": "ADM1_A",
            "name": "Region A",
            "geometry": Polygon([(1, 0), (1, 2), (2, 2), (2, 0)]),
        },
    ]
    admin2_rows = [
        {
            "boundary_id": "ADM2_B",
            "parent_id": "ADM1_B",
            "name": "District B",
            "geometry": Polygon([(0, 0), (0, 2), (1, 2), (1, 0)]),
        },
        {
            "boundary_id": "ADM2_A",
            "parent_id": "ADM1_A",
            "name": "District A",
            "geometry": Polygon([(1, 0), (1, 2), (2, 2), (2, 0)]),
        },
    ]
    if reverse_order:
        admin1_rows = list(reversed(admin1_rows))
        admin2_rows = list(reversed(admin2_rows))

    admin1_gdf = gpd.GeoDataFrame(admin1_rows, crs="EPSG:4326")
    admin2_gdf = gpd.GeoDataFrame(admin2_rows, crs="EPSG:4326")
    return admin1_gdf, admin2_gdf


@pytest.mark.parametrize("input_crs", [None, "EPSG:4326"])
def test_annotate_with_admin_regions_does_not_mutate_input_and_preserves_index(
    monkeypatch, input_crs
):
    from gigaspatial.handlers.boundaries import AdminBoundaries

    admin1_gdf, admin2_gdf = _make_admin_fixtures()

    def _fake_create(country_code, admin_level, data_store=None):
        return _FakeAdminBoundaries(admin1_gdf if admin_level == 1 else admin2_gdf)

    monkeypatch.setattr(AdminBoundaries, "create", staticmethod(_fake_create))

    points = gpd.GeoDataFrame(
        {
            "pid": ["p_second", "p_first", "p_outside"],
            "geometry": [Point(1.5, 1.0), Point(0.5, 1.0), Point(5.0, 5.0)],
        },
        index=["idx_b", "idx_a", "idx_c"],
        crs=input_crs,
    )
    original_columns = list(points.columns)
    original_index = points.index.copy()
    original_crs = points.crs
    original_snapshot = points.copy(deep=True)

    result = annotate_with_admin_regions(points, country_code="RWA")

    assert result is not points
    assert list(points.columns) == original_columns
    assert points.index.equals(original_index)
    assert points.crs == original_crs
    assert points.equals(original_snapshot)

    assert list(result.index) == ["idx_b", "idx_a", "idx_c"]
    assert list(result["pid"]) == ["p_second", "p_first", "p_outside"]
    assert list(result.columns) == [
        "pid",
        "geometry",
        "admin2_id",
        "admin2",
        "admin1_id",
        "admin1",
    ]
    assert result.loc["idx_b", "admin1_id"] == "ADM1_A"
    assert result.loc["idx_b", "admin2_id"] == "ADM2_A"
    assert result.loc["idx_a", "admin1_id"] == "ADM1_B"
    assert result.loc["idx_a", "admin2_id"] == "ADM2_B"
    assert result.loc["idx_c", "admin1_id"] is None or np.isnan(
        result.loc["idx_c", "admin1_id"]
    )
    assert result.loc["idx_c", "admin2_id"] is None or np.isnan(
        result.loc["idx_c", "admin2_id"]
    )


def test_annotate_with_admin_regions_shared_border_deterministic_and_preserves_order(
    monkeypatch,
):
    from gigaspatial.handlers.boundaries import AdminBoundaries

    points = gpd.GeoDataFrame(
        {
            "pid": ["in_b", "border", "outside", "in_a"],
            "geometry": [
                Point(0.5, 1.0),
                Point(1.0, 1.0),
                Point(10.0, 10.0),
                Point(1.5, 1.0),
            ],
        },
        index=[30, 10, 40, 20],
        crs="EPSG:4326",
    )
    original_snapshot = points.copy(deep=True)

    results = []
    for reverse_order in (False, True):
        admin1_gdf, admin2_gdf = _make_admin_fixtures(reverse_order=reverse_order)

        def _fake_create(country_code, admin_level, data_store=None, a1=admin1_gdf, a2=admin2_gdf):
            return _FakeAdminBoundaries(a1 if admin_level == 1 else a2)

        monkeypatch.setattr(AdminBoundaries, "create", staticmethod(_fake_create))
        results.append(annotate_with_admin_regions(points, country_code="RWA"))

    assert points.equals(original_snapshot)
    assert points.crs == original_snapshot.crs

    for result in results:
        assert len(result) == len(points)
        assert list(result.index) == [30, 10, 40, 20]
        assert list(result["pid"]) == ["in_b", "border", "outside", "in_a"]
        assert result.loc[30, "admin1_id"] == "ADM1_B"
        assert result.loc[30, "admin2_id"] == "ADM2_B"
        # Shared border tie-break deterministically picks smallest (admin1_id, admin2_id)
        assert result.loc[10, "admin1_id"] == "ADM1_A"
        assert result.loc[10, "admin2_id"] == "ADM2_A"
        assert result.loc[10, "admin1"] == "Region A"
        assert result.loc[10, "admin2"] == "District A"
        assert result.loc[20, "admin1_id"] == "ADM1_A"
        assert result.loc[20, "admin2_id"] == "ADM2_A"
        assert result.loc[40, "admin1_id"] is None or np.isnan(
            result.loc[40, "admin1_id"]
        )

    assert results[0].equals(results[1])


def test_annotate_with_admin_regions_preexisting_admin_id_with_tie(monkeypatch):
    from gigaspatial.handlers.boundaries import AdminBoundaries

    admin1_gdf, admin2_gdf = _make_admin_fixtures()

    def _fake_create(country_code, admin_level, data_store=None):
        return _FakeAdminBoundaries(admin1_gdf if admin_level == 1 else admin2_gdf)

    monkeypatch.setattr(AdminBoundaries, "create", staticmethod(_fake_create))

    points = gpd.GeoDataFrame(
        {
            "pid": ["border", "in_b"],
            "admin1_id": ["STALE_1", "STALE_2"],
            "geometry": [Point(1.0, 1.0), Point(0.5, 1.0)],
        },
        index=["row_1", "row_2"],
        crs="EPSG:4326",
    )
    original_snapshot = points.copy(deep=True)

    result = annotate_with_admin_regions(points, country_code="RWA")

    assert result is not points
    assert points.equals(original_snapshot)
    assert list(points["admin1_id"]) == ["STALE_1", "STALE_2"]

    assert len(result) == 2
    assert list(result.index) == ["row_1", "row_2"]
    assert list(result["pid"]) == ["border", "in_b"]
    assert result.loc["row_1", "admin1_id"] == "ADM1_A"
    assert result.loc["row_1", "admin2_id"] == "ADM2_A"
    assert result.loc["row_2", "admin1_id"] == "ADM1_B"
    assert result.loc["row_2", "admin2_id"] == "ADM2_B"


def test_buffer_geodataframe_high_latitude_matches_utm_not_web_mercator():
    oslo = gpd.GeoDataFrame(geometry=[Point(10.7522, 59.9139)], crs="EPSG:4326")
    radius_m = 1000.0

    buffered = buffer_geodataframe(oslo, radius_m)

    utm_crs = "EPSG:32632"
    actual_area = buffered.to_crs(utm_crs).geometry.area.iloc[0]
    utm_ref_area = (
        oslo.to_crs(utm_crs).geometry.buffer(radius_m).area.iloc[0]
    )
    mercator_ref_area = (
        oslo.to_crs("EPSG:3857")
        .geometry.buffer(radius_m)
        .to_crs(utm_crs)
        .area.iloc[0]
    )

    assert actual_area == pytest.approx(utm_ref_area, rel=1e-2)
    assert abs(actual_area - mercator_ref_area) / utm_ref_area > 0.5


def test_add_area_in_meters_high_latitude_matches_utm_not_web_mercator():
    oslo_poly = gpd.GeoDataFrame(
        geometry=[
            Polygon([(10.7, 59.9), (10.8, 59.9), (10.8, 59.95), (10.7, 59.95)])
        ],
        crs="EPSG:4326",
    )

    result = add_area_in_meters(oslo_poly)
    actual_area = result["area_in_meters"].iloc[0]
    utm_ref_area = oslo_poly.to_crs("EPSG:32632").geometry.area.iloc[0]
    mercator_ref_area = oslo_poly.to_crs("EPSG:3857").geometry.area.iloc[0]

    assert actual_area == pytest.approx(utm_ref_area, rel=1e-4)
    assert abs(actual_area - mercator_ref_area) / utm_ref_area > 2.0


def test_metric_operations_handle_none_utm_crs_via_helper(monkeypatch):
    oslo_poly = gpd.GeoDataFrame(
        {
            "val": [10.0],
            "geometry": [
                Polygon([(10.7, 59.9), (10.8, 59.9), (10.8, 59.95), (10.7, 59.95)])
            ],
        },
        crs="EPSG:4326",
    )
    oslo_zones = gpd.GeoDataFrame(
        {
            "zone_id": ["oslo_z1"],
            "geometry": [
                Polygon([(10.65, 59.85), (10.85, 59.85), (10.85, 60.0), (10.65, 60.0)])
            ],
        },
        crs="EPSG:4326",
    )

    monkeypatch.setattr(gpd.GeoDataFrame, "estimate_utm_crs", lambda self: None)

    buffered = buffer_geodataframe(oslo_poly, 100.0)
    assert len(buffered) == 1
    assert buffered.crs == "EPSG:4326"

    with_area = add_area_in_meters(oslo_poly)
    assert with_area["area_in_meters"].iloc[0] > 0

    counted = aggregate_polygons_to_zones(
        oslo_poly,
        oslo_zones,
        value_columns=None,
        aggregation="count",
        predicate="fractional",
        drop_geometry=True,
    )
    assert counted.set_index("zone_id").loc["oslo_z1", "polygon_count"] == 1

    aggregated = aggregate_polygons_to_zones(
        oslo_poly,
        oslo_zones,
        value_columns="val",
        aggregation="sum",
        predicate="fractional",
        drop_geometry=True,
    )
    assert aggregated.set_index("zone_id").loc["oslo_z1", "val"] == pytest.approx(10.0)


def test_metric_operations_delegate_to_estimate_utm_crs_with_fallback(monkeypatch):
    oslo_poly = gpd.GeoDataFrame(
        {
            "val": [10.0],
            "geometry": [
                Polygon([(10.7, 59.9), (10.8, 59.9), (10.8, 59.95), (10.7, 59.95)])
            ],
        },
        crs="EPSG:4326",
    )
    oslo_zones = gpd.GeoDataFrame(
        {
            "zone_id": ["oslo_z1"],
            "geometry": [
                Polygon([(10.65, 59.85), (10.85, 59.85), (10.85, 60.0), (10.65, 60.0)])
            ],
        },
        crs="EPSG:4326",
    )

    real_helper = estimate_utm_crs_with_fallback
    calls = []

    def _spy(gdf, logger=geo_module.LOGGER, fallback_crs="EPSG:3857"):
        calls.append(len(gdf))
        return real_helper(gdf, logger=logger, fallback_crs=fallback_crs)

    monkeypatch.setattr(geo_module, "estimate_utm_crs_with_fallback", _spy)

    buffer_geodataframe(oslo_poly, 100.0)
    add_area_in_meters(oslo_poly)
    aggregate_polygons_to_zones(
        oslo_poly,
        oslo_zones,
        value_columns=None,
        aggregation="count",
        predicate="fractional",
    )
    aggregate_polygons_to_zones(
        oslo_poly,
        oslo_zones,
        value_columns="val",
        aggregation="sum",
        predicate="fractional",
    )

    assert len(calls) == 4


def test_estimate_utm_crs_with_fallback_preserves_custom_fallback_crs(monkeypatch):
    oslo = gpd.GeoDataFrame(geometry=[Point(10.7522, 59.9139)], crs="EPSG:4326")
    monkeypatch.setattr(
        gpd.GeoDataFrame,
        "estimate_utm_crs",
        lambda self: (_ for _ in ()).throw(RuntimeError("utm failed")),
    )

    assert estimate_utm_crs_with_fallback(oslo, fallback_crs="EPSG:32632") == "EPSG:32632"





