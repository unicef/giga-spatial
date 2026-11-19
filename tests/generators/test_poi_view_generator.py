import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point, Polygon

from gigaspatial import PoiViewGenerator
from gigaspatial.core.io.local_data_store import LocalDataStore


def _make_pois(tmp_path, extra_cols=None) -> PoiViewGenerator:
    data = {
        "poi_id": ["poi_1", "poi_2"],
        "geometry": [Point(0.0, 0.0), Point(1.0, 0.0)],
    }
    if extra_cols:
        data.update(extra_cols)
    pois = gpd.GeoDataFrame(data, crs="EPSG:4326")
    return PoiViewGenerator(pois, data_store=LocalDataStore(base_path=tmp_path))


def _make_point_data() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {
            "tests": [10.0, 20.0, 99.0],
            "geometry": [Point(0.001, 0.0), Point(0.002, 0.0), Point(0.1, 0.0)],
        },
        crs="EPSG:4326",
    )


def _make_polygon_data() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
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
    result = gen.map_zonal_stats(
        _make_polygon_data(),
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


def test_map_zonal_stats_point_data_with_value_column_and_stat_count(tmp_path):
    gen = _make_pois(tmp_path)
    pts = gpd.GeoDataFrame(
        {
            "tests": [10.0, np.nan, 20.0, 99.0],
            "geometry": [
                Point(0.001, 0.0),
                Point(0.0015, 0.0),
                Point(0.002, 0.0),
                Point(0.1, 0.0),
            ],
        },
        crs="EPSG:4326",
    )

    result = gen.map_zonal_stats(
        pts,
        stat="count",
        map_radius_meters=500,
        output_column="tests_count",
        value_column="tests",
    )

    assert list(result.columns) == ["poi_id", "tests_count"]
    by_poi = result.set_index("poi_id")["tests_count"]
    assert by_poi["poi_1"] == pytest.approx(2.0)
    assert by_poi["poi_2"] == pytest.approx(0.0)


def test_map_zonal_stats_point_data_ignores_poi_attribute_collision(tmp_path):
    gen = _make_pois(tmp_path, extra_cols={"tests": [999.0, 888.0]})
    result = gen.map_zonal_stats(
        _make_point_data(),
        stat="sum",
        map_radius_meters=500,
        output_column="nearby_tests",
        value_column="tests",
    )

    assert list(result.columns) == ["poi_id", "nearby_tests"]
    by_poi = result.set_index("poi_id")["nearby_tests"]
    assert by_poi["poi_1"] == pytest.approx(30.0)
    assert pd.isna(by_poi["poi_2"])


def test_map_zonal_stats_polygon_data_ignores_poi_attribute_collision(tmp_path):
    gen = _make_pois(tmp_path, extra_cols={"tests": [999.0, 888.0]})
    result = gen.map_zonal_stats(
        _make_polygon_data(),
        stat="sum",
        map_radius_meters=500,
        output_column="nearby_poly_tests",
        value_column="tests",
    )

    assert list(result.columns) == ["poi_id", "nearby_poly_tests"]
    by_poi = result.set_index("poi_id")["nearby_poly_tests"]
    assert by_poi["poi_1"] == pytest.approx(30.0)
    assert pd.isna(by_poi["poi_2"])


# ─────────────────────────────────────────────────────────────────────────────
# Handler injection tests (TDD Slice 1)
# ─────────────────────────────────────────────────────────────────────────────


def _make_poi_gen_in_raster(tmp_path) -> PoiViewGenerator:
    pois = gpd.GeoDataFrame(
        {
            "poi_id": ["p1", "p2"],
            "geometry": [Point(0.015, 0.015), Point(0.035, 0.035)],
        },
        crs="EPSG:4326",
    )
    return PoiViewGenerator(pois, data_store=LocalDataStore(base_path=tmp_path))


def test_poi_map_wp_pop_uses_injected_handler(
    tmp_path, make_worldpop_handler, unexpected_ctor, monkeypatch
):
    import gigaspatial.generators.poi as poi_mod

    monkeypatch.setattr(poi_mod, "WPPopulationHandler", unexpected_ctor("WPPopulationHandler"))
    handler = make_worldpop_handler()
    gen = _make_poi_gen_in_raster(tmp_path)

    view = gen.map_wp_pop(country="RWA", map_radius_meters=300, handler=handler)
    assert "population" in view.columns


def test_poi_map_built_s_uses_injected_handler(
    tmp_path, make_ghsl_handler, unexpected_ctor, monkeypatch
):
    import gigaspatial.generators.poi as poi_mod

    monkeypatch.setattr(poi_mod, "GHSLDataHandler", unexpected_ctor("GHSLDataHandler"))
    handler = make_ghsl_handler(product="GHS_BUILT_S", coord_system=4326)
    gen = _make_poi_gen_in_raster(tmp_path)

    view = gen.map_built_s(map_radius_meters=300, handler=handler)
    assert "built_surface_m2" in view.columns


def test_poi_map_smod_uses_injected_handler(
    tmp_path, make_ghsl_handler, unexpected_ctor, monkeypatch
):
    import gigaspatial.generators.poi as poi_mod

    monkeypatch.setattr(poi_mod, "GHSLDataHandler", unexpected_ctor("GHSLDataHandler"))
    handler = make_ghsl_handler(product="GHS_SMOD", resolution=1000, coord_system=54009)
    gen = _make_poi_gen_in_raster(tmp_path)

    view = gen.map_smod(handler=handler)
    assert "smod_class" in view.columns


@pytest.mark.parametrize("find_globally", [False, True])
def test_poi_find_nearest_buildings_uses_injected_handler(
    tmp_path, unexpected_ctor, monkeypatch, find_globally
):
    from types import SimpleNamespace
    import gigaspatial.handlers.google_ms_combined_buildings as gms_mod

    monkeypatch.setattr(
        gms_mod,
        "GoogleMSBuildingsHandler",
        unexpected_ctor("GoogleMSBuildingsHandler"),
    )

    building_file = tmp_path / "rwa.parquet"
    fake_handler = SimpleNamespace(
        config=SimpleNamespace(),
        reader=SimpleNamespace(resolve_source_paths=lambda country, **kw: [building_file]),
        select_options=lambda kw, cfg=None: {},
        ensure_data_available=lambda country, **kw: True,
        load_centroids=lambda fp: {"coords": np.array([[0.0, 0.0], [1.0, 0.0]])},
    )

    gen = _make_pois(tmp_path)
    view = gen.find_nearest_buildings(
        country="RWA",
        search_radius=500,
        find_nearest_globally=find_globally,
        handler=fake_handler,
    )
    assert "nearest_building_distance_m" in view.columns
    assert view.set_index("poi_id").loc["poi_1", "nearest_building_distance_m"] == pytest.approx(0.0)


# ─────────────────────────────────────────────────────────────────────────────
# WorldPop (map_wp_pop) offline characterization tests (TDD Slice 2)
# ─────────────────────────────────────────────────────────────────────────────


def _make_raster_pois(tmp_path) -> PoiViewGenerator:
    """Create POIs at known positions relative to the (0.0, 0.0, 0.04, 0.04) 4x4 grid."""
    pois = gpd.GeoDataFrame(
        {
            "poi_id": ["center_pixel", "sw_corner", "outside"],
            "geometry": [
                Point(0.015, 0.015),  # center of row=2, col=1 (value 10)
                Point(0.01, 0.01),  # shared corner of [9, 10, 13, 14]
                Point(5.0, 5.0),  # outside raster extent
            ],
        },
        crs="EPSG:4326",
    )
    return PoiViewGenerator(pois, data_store=LocalDataStore(base_path=tmp_path))


def test_poi_map_wp_pop_intersects_sum(tmp_path, make_worldpop_handler):
    handler = make_worldpop_handler(project="pop", resolution=1000)
    gen = _make_raster_pois(tmp_path)

    view = gen.map_wp_pop(
        country="RWA",
        map_radius_meters=150,
        predicate="intersects",
        output_column="wp_pop_sum",
        handler=handler,
    )

    by_id = view.set_index("poi_id")["wp_pop_sum"]
    assert by_id["center_pixel"] == pytest.approx(10.0)
    assert by_id["sw_corner"] == pytest.approx(9.0 + 10.0 + 13.0 + 14.0)
    assert pd.isna(by_id["outside"])


def test_poi_map_wp_pop_degree_of_urbanization_forces_median(tmp_path, make_worldpop_handler):
    dug_data = np.array(
        [
            [11, 11, 12, 12],
            [11, 12, 21, 30],
            [10, 12, 21, 30],
            [13, 30, 21, 30],
        ],
        dtype=np.float32,
    )
    handler = make_worldpop_handler(
        release="GR2",
        project="degree_of_urbanization",
        year=2020,
        resolution=1000,
        constrained=True,
        dug_level="L1",
        country_rasters={
            "RWA": [
                (
                    "https://data.worldpop.org/GIS/dug/rwa_GRID_L1_2020.tif",
                    dug_data,
                    (0.0, 0.0, 0.04, 0.04),
                )
            ]
        },
    )
    gen = _make_raster_pois(tmp_path)

    view = gen.map_wp_pop(
        country="RWA",
        map_radius_meters=150,
        predicate="intersects",
        output_column="dug_class",
        handler=handler,
    )

    by_id = view.set_index("poi_id")["dug_class"]
    assert by_id["center_pixel"] == pytest.approx(12.0)
    # sw_corner intersects SW 4 cells: [10, 12, 13, 30] -> median is 12.5
    assert by_id["sw_corner"] == pytest.approx(12.5)
    assert pd.isna(by_id["outside"])


def test_poi_map_wp_pop_centroid_within_single_and_multi_country(
    tmp_path, make_worldpop_handler, default_raster_data
):
    handler = make_worldpop_handler(
        project="pop",
        resolution=1000,
        country_rasters={
            "RWA": [
                (
                    "https://data.worldpop.org/GIS/Population/rwa_2020.tif",
                    default_raster_data,
                    (0.0, 0.0, 0.04, 0.04),
                )
            ],
            "BDI": [
                (
                    "https://data.worldpop.org/GIS/Population/bdi_2020.tif",
                    np.full((4, 4), 100.0, dtype=np.float32),
                    (0.04, 0.0, 0.08, 0.04),
                )
            ],
        },
    )

    # Single country: 1200m buffer around (0.01, 0.01) contains centroids of [9, 10, 13, 14]
    gen_single = _make_raster_pois(tmp_path)
    single_view = gen_single.map_wp_pop(
        country="RWA",
        map_radius_meters=1200,
        predicate="centroid_within",
        output_column="pop_single",
        handler=handler,
    )
    by_single = single_view.set_index("poi_id")["pop_single"]
    # 1200m (~0.0108 deg) around (0.015, 0.015) includes centroids of 5 pixels: 6 + 9 + 10 + 11 + 14 = 50.0
    assert by_single["center_pixel"] == pytest.approx(50.0)
    assert by_single["sw_corner"] == pytest.approx(46.0)
    assert pd.isna(by_single["outside"])

    # Multi-country list: one POI in RWA and one POI in BDI
    pois_multi = gpd.GeoDataFrame(
        {
            "poi_id": ["in_rwa", "in_bdi"],
            "geometry": [Point(0.015, 0.015), Point(0.055, 0.015)],
        },
        crs="EPSG:4326",
    )
    gen_multi = PoiViewGenerator(pois_multi, data_store=LocalDataStore(base_path=tmp_path))
    multi_view = gen_multi.map_wp_pop(
        country=["RWA", "BDI"],
        map_radius_meters=150,
        predicate="centroid_within",
        output_column="pop_multi",
        handler=handler,
    )
    by_multi = multi_view.set_index("poi_id")["pop_multi"]
    assert by_multi["in_rwa"] == pytest.approx(10.0)
    assert by_multi["in_bdi"] == pytest.approx(100.0)


def test_poi_map_wp_pop_age_structures_sums_rasters_and_rejects_multi_country(
    tmp_path, make_worldpop_handler
):
    age_handler = make_worldpop_handler(
        release="GR1",
        project="age_structures",
        year=2020,
        resolution=100,
        constrained=False,
        un_adjusted=False,
        country_rasters={
            "RWA": [
                (
                    "https://data.worldpop.org/GIS/AgeSex/rwa_f_5_2020.tif",
                    np.full((4, 4), 2.0, dtype=np.float32),
                    (0.0, 0.0, 0.04, 0.04),
                ),
                (
                    "https://data.worldpop.org/GIS/AgeSex/rwa_m_5_2020.tif",
                    np.full((4, 4), 3.0, dtype=np.float32),
                    (0.0, 0.0, 0.04, 0.04),
                ),
            ]
        },
    )

    gen = _make_raster_pois(tmp_path)
    with pytest.raises(ValueError, match="age_structures"):
        gen.map_wp_pop(
            country=["RWA", "BDI"],
            map_radius_meters=150,
            predicate="centroid_within",
            handler=age_handler,
        )

    view = gen.map_wp_pop(
        country="RWA",
        map_radius_meters=1200,
        predicate="centroid_within",
        output_column="age_pop",
        handler=age_handler,
    )
    by_id = view.set_index("poi_id")["age_pop"]
    # 1200m around center_pixel covers 5 cell centroids (5*(2+3) = 25); sw_corner covers 4 cells (4*(2+3) = 20)
    assert by_id["center_pixel"] == pytest.approx(25.0)
    assert by_id["sw_corner"] == pytest.approx(20.0)
    assert pd.isna(by_id["outside"])


def test_poi_map_wp_pop_fractional_1000m_vs_100m_fallback_warning(
    tmp_path, make_worldpop_handler, caplog, monkeypatch
):
    import logging

    uniform_data = np.full((4, 4), 10.0, dtype=np.float32)

    handler_1000 = make_worldpop_handler(
        project="pop",
        resolution=1000,
        subdir="wp_1000",
        country_rasters={
            "RWA": [
                (
                    "https://data.worldpop.org/GIS/Population/rwa_1000.tif",
                    uniform_data,
                    (0.0, 0.0, 0.04, 0.04),
                )
            ]
        },
    )
    # POI at (0.015, 0.015) with 150m radius lies entirely within 1 pixel (~1.11km x 1.11km):
    # fractional area weight is pi * 150^2 / (pixel_area_m2) << 10.0, whereas intersects gives 10.0
    pois = gpd.GeoDataFrame(
        {"poi_id": ["p1"], "geometry": [Point(0.015, 0.015)]},
        crs="EPSG:4326",
    )
    gen_1000 = PoiViewGenerator(pois, data_store=LocalDataStore(base_path=tmp_path))
    view_1000 = gen_1000.map_wp_pop(
        country="RWA",
        map_radius_meters=150,
        resolution=1000,
        predicate="fractional",
        output_column="pop_frac",
        handler=handler_1000,
    )
    frac_val = view_1000.loc[0, "pop_frac"]
    assert 0.0 < frac_val < 2.0

    handler_100 = make_worldpop_handler(
        project="pop",
        resolution=100,
        subdir="wp_100",
        country_rasters={
            "RWA": [
                (
                    "https://data.worldpop.org/GIS/Population/rwa_100.tif",
                    uniform_data,
                    (0.0, 0.0, 0.04, 0.04),
                )
            ]
        },
    )
    gen_100 = PoiViewGenerator(pois, data_store=LocalDataStore(base_path=tmp_path))
    monkeypatch.setattr(gen_100.logger, "propagate", True)
    with caplog.at_level(logging.WARNING):
        view_100 = gen_100.map_wp_pop(
            country="RWA",
            map_radius_meters=150,
            resolution=100,
            predicate="fractional",
            output_column="pop_100",
            handler=handler_100,
        )

    assert any("intersects" in rec.message for rec in caplog.records)
    assert view_100.loc[0, "pop_100"] == pytest.approx(10.0)


# ─────────────────────────────────────────────────────────────────────────────
# GHSL (map_built_s, map_smod) offline characterization tests (TDD Slice 3)
# ─────────────────────────────────────────────────────────────────────────────


def test_poi_map_built_s_default_and_custom_stat_and_column(tmp_path, make_ghsl_handler):
    built_h = make_ghsl_handler(product="GHS_BUILT_S", year=2020, resolution=100, coord_system=4326)
    gen = _make_raster_pois(tmp_path)

    view = gen.map_built_s(map_radius_meters=1200, handler=built_h)
    by_id = view.set_index("poi_id")["built_surface_m2"]
    assert by_id["center_pixel"] == pytest.approx(50.0)
    assert by_id["sw_corner"] == pytest.approx(46.0)
    assert pd.isna(by_id["outside"])

    view = gen.map_built_s(
        map_radius_meters=1200,
        stat="mean",
        output_column="custom_built_mean",
        handler=built_h,
    )
    by_mean = view.set_index("poi_id")["custom_built_mean"]
    assert by_mean["center_pixel"] == pytest.approx(10.0)
    assert by_mean["sw_corner"] == pytest.approx(11.5)


def test_poi_map_smod_reprojects_to_mollweide_and_samples(tmp_path, make_ghsl_handler):
    smod_data = np.array(
        [
            [11, 11, 30, 30],
            [11, 11, 30, 30],
            [12, 12, 23, 23],
            [12, 12, 23, 23],
        ],
        dtype=np.float32,
    )
    smod_h = make_ghsl_handler(
        product="GHS_SMOD",
        year=2020,
        resolution=1000,
        coord_system=54009,
        data=smod_data,
    )
    assert smod_h.config.crs == "ESRI:54009"

    pois = gpd.GeoDataFrame(
        {
            "poi_id": ["sw_poi", "ne_poi"],
            "geometry": [Point(0.005, 0.005), Point(0.035, 0.035)],
        },
        crs="EPSG:4326",
    )
    gen = PoiViewGenerator(pois, data_store=LocalDataStore(base_path=tmp_path))

    # Default direct point sampling (reprojected from EPSG:4326 -> ESRI:54009)
    view = gen.map_smod(handler=smod_h)
    by_id = view.set_index("poi_id")["smod_class"]
    assert by_id["sw_poi"] == pytest.approx(12.0)
    assert by_id["ne_poi"] == pytest.approx(30.0)

    # Buffered sampling (median within buffer in ESRI:54009)
    view = gen.map_smod(map_radius_meters=300, output_column="smod_buf", handler=smod_h)
    by_buf = view.set_index("poi_id")["smod_buf"]
    assert by_buf["sw_poi"] == pytest.approx(12.0)
    assert by_buf["ne_poi"] == pytest.approx(30.0)


# ─────────────────────────────────────────────────────────────────────────────
# ID-column normalization, xfail bugs, and view lifecycle (TDD Slice 4)
# ─────────────────────────────────────────────────────────────────────────────


def test_poi_id_column_default_autogeneration_and_custom_rename_and_duplicate_raises(
    tmp_path,
):
    store = LocalDataStore(base_path=tmp_path)

    # Default poi_id_column="poi_id" absent -> auto-generated poi_0..N across input types
    df_gen = PoiViewGenerator(
        pd.DataFrame({"latitude": [0.01, 0.02], "longitude": [0.01, 0.02]}),
        data_store=store,
    )
    assert list(df_gen.view["poi_id"]) == ["poi_0", "poi_1"]

    gdf_gen = PoiViewGenerator(
        gpd.GeoDataFrame(geometry=[Point(0.01, 0.01), Point(0.02, 0.02)], crs="EPSG:4326"),
        data_store=store,
    )
    assert list(gdf_gen.view["poi_id"]) == ["poi_0", "poi_1"]

    tuple_gen = PoiViewGenerator([(0.01, 0.01), (0.02, 0.02)], data_store=store)
    assert list(tuple_gen.view["poi_id"]) == ["poi_0", "poi_1"]

    dict_gen = PoiViewGenerator(
        [{"lat": 0.01, "lon": 0.01}, {"lat": 0.02, "lon": 0.02}], data_store=store
    )
    assert list(dict_gen.view["poi_id"]) == ["poi_0", "poi_1"]

    # Custom poi_id_column present -> renamed to canonical 'poi_id'
    custom_gen = PoiViewGenerator(
        pd.DataFrame(
            {
                "school_id": ["sch_a", "sch_b"],
                "latitude": [0.01, 0.02],
                "longitude": [0.01, 0.02],
            }
        ),
        poi_id_column="school_id",
        data_store=store,
    )
    assert "poi_id" in custom_gen.view.columns
    assert "school_id" not in custom_gen.view.columns
    assert list(custom_gen.view["poi_id"]) == ["sch_a", "sch_b"]

    # Duplicate IDs -> ValueError
    dup_df = pd.DataFrame(
        {"school_id": ["dup", "dup"], "latitude": [0.01, 0.02], "longitude": [0.01, 0.02]}
    )
    with pytest.raises(ValueError, match="contains duplicate values"):
        PoiViewGenerator(dup_df, poi_id_column="school_id", data_store=store)


@pytest.mark.xfail(
    strict=True,
    reason="Ticket 03 fixes missing custom poi_id_column on DataFrame (poi.py:167-168) to raise ValueError",
)
def test_poi_missing_custom_id_column_on_dataframe_raises(tmp_path):
    df = pd.DataFrame({"latitude": [0.01], "longitude": [0.01]})
    with pytest.raises(ValueError):
        PoiViewGenerator(
            df,
            poi_id_column="school_id",
            data_store=LocalDataStore(base_path=tmp_path),
        )


@pytest.mark.xfail(
    strict=True,
    reason="Ticket 03 fixes projected GeoDataFrame coordinate extraction before to_crs (poi.py:135-149)",
)
def test_poi_projected_geodataframe_extracts_wgs84_lat_lon(tmp_path):
    projected = gpd.GeoDataFrame(
        {"poi_id": ["p1"]},
        geometry=[Point(30.0, -2.0)],
        crs="EPSG:4326",
    ).to_crs("EPSG:3857")

    gen = PoiViewGenerator(projected, data_store=LocalDataStore(base_path=tmp_path))
    assert gen.view.loc[0, "longitude"] == pytest.approx(30.0)
    assert gen.view.loc[0, "latitude"] == pytest.approx(-2.0)


def test_poi_view_chaining_exports_and_save_view(
    tmp_path, make_worldpop_handler, make_ghsl_handler
):
    from gigaspatial.generators.poi import PoiViewGeneratorConfig

    store = LocalDataStore(base_path=tmp_path)
    cfg = PoiViewGeneratorConfig(base_path=tmp_path / "views")
    wp_h = make_worldpop_handler(project="pop", resolution=1000, store=store, subdir="wp")
    built_h = make_ghsl_handler(
        product="GHS_BUILT_S",
        year=2020,
        resolution=100,
        coord_system=4326,
        store=store,
        subdir="ghsl_b",
    )
    smod_h = make_ghsl_handler(
        product="GHS_SMOD",
        year=2020,
        resolution=1000,
        coord_system=54009,
        store=store,
        subdir="ghsl_s",
    )

    pois = gpd.GeoDataFrame(
        {
            "poi_id": ["center_pixel", "sw_corner"],
            "geometry": [Point(0.015, 0.015), Point(0.01, 0.01)],
        },
        crs="EPSG:4326",
    )
    gen = PoiViewGenerator(pois, config=cfg, data_store=store)

    gen.map_wp_pop(
        country="RWA",
        map_radius_meters=150,
        output_column="wp_pop",
        handler=wp_h,
    )
    gen.map_built_s(
        map_radius_meters=150,
        output_column="built_m2",
        handler=built_h,
    )
    chained = gen.map_smod(output_column="smod_val", handler=smod_h)

    expected_cols = {"poi_id", "wp_pop", "built_m2", "smod_val"}
    assert expected_cols <= set(chained.columns)

    df = gen.to_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert not isinstance(df, gpd.GeoDataFrame)
    assert "geometry" not in df.columns
    assert expected_cols <= set(df.columns)

    gdf = gen.to_geodataframe()
    assert isinstance(gdf, gpd.GeoDataFrame)
    assert "geometry" in gdf.columns
    assert gdf.crs == "EPSG:4326"
    assert expected_cols <= set(gdf.columns)

    csv_path = gen.save_view("poi_out", output_format="csv")
    parquet_path = gen.save_view("poi_out", output_format="parquet")
    assert csv_path.exists()
    assert parquet_path.exists()

    df_csv = pd.read_csv(csv_path)
    df_pq = pd.read_parquet(parquet_path)
    assert expected_cols <= set(df_csv.columns)
    assert expected_cols <= set(df_pq.columns)
    assert df_pq.set_index("poi_id").loc["center_pixel", "wp_pop"] == pytest.approx(10.0)


@pytest.mark.xfail(
    strict=True,
    reason="Ticket 03 fixes save_view geojson export using to_geodataframe (poi.py:1042)",
)
def test_poi_save_view_geojson(tmp_path):
    from gigaspatial.generators.poi import PoiViewGeneratorConfig

    store = LocalDataStore(base_path=tmp_path)
    cfg = PoiViewGeneratorConfig(base_path=tmp_path / "views")
    pois = gpd.GeoDataFrame(
        {"poi_id": ["p1"], "geometry": [Point(0.015, 0.015)]},
        crs="EPSG:4326",
    )
    gen = PoiViewGenerator(pois, config=cfg, data_store=store)
    out_path = gen.save_view("poi_out", output_format="geojson")
    loaded = gpd.read_file(out_path)
    assert list(loaded["poi_id"]) == ["p1"]




