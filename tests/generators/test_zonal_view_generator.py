import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Point, Polygon

from gigaspatial import ZonalViewGenerator
from gigaspatial.core.io.local_data_store import LocalDataStore


def _make_generator(tmp_path) -> ZonalViewGenerator:
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
    return ZonalViewGenerator(
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

    view = gen.map_points(pts, value_columns="tests")
    assert view.equals(gen.view)
    by_zone = view.set_index("zone_id")["tests"]
    assert by_zone["z1"] == pytest.approx(2.0)
    assert by_zone["z2"] == pytest.approx(0.0)


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

    view = gen.map_points(pts, value_columns=["a", "b"], aggregation="count")
    by_zone = view.set_index("zone_id")
    assert by_zone.loc["z1", "a"] == pytest.approx(1.0)
    assert by_zone.loc["z2", "a"] == pytest.approx(0.0)
    assert by_zone.loc["z1", "b"] == pytest.approx(2.0)
    assert by_zone.loc["z2", "b"] == pytest.approx(0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Handler injection tests (TDD Slice 1)
# ─────────────────────────────────────────────────────────────────────────────


def _make_zone_gen_in_raster(tmp_path) -> ZonalViewGenerator:
    zones = gpd.GeoDataFrame(
        {
            "zone_id": ["z_sw", "z_ne", "z_out"],
            "geometry": [
                Polygon([(0.002, 0.002), (0.002, 0.018), (0.018, 0.018), (0.018, 0.002)]),
                Polygon([(0.022, 0.022), (0.022, 0.038), (0.038, 0.038), (0.038, 0.022)]),
                Polygon([(5.0, 5.0), (5.0, 5.1), (5.1, 5.1), (5.1, 5.0)]),
            ],
        },
        crs="EPSG:4326",
    )
    return ZonalViewGenerator(
        zones, data_store=LocalDataStore(base_path=tmp_path)
    )


def test_zonal_map_wp_pop_uses_injected_handler(
    tmp_path, make_worldpop_handler, unexpected_ctor, monkeypatch
):
    import gigaspatial.generators.base as gen_base_mod

    monkeypatch.setattr(
        gen_base_mod, "WPPopulationHandler", unexpected_ctor("WPPopulationHandler")
    )
    handler = make_worldpop_handler()
    gen = _make_zone_gen_in_raster(tmp_path)

    view = gen.map_wp_pop(country="RWA", handler=handler)
    assert "population" in view.columns


def test_zonal_map_built_s_and_smod_and_ghsl_use_injected_handler(
    tmp_path, make_ghsl_handler, unexpected_ctor, monkeypatch
):
    import gigaspatial.generators.base as gen_base_mod

    monkeypatch.setattr(
        gen_base_mod, "GHSLDataHandler", unexpected_ctor("GHSLDataHandler")
    )
    built_h = make_ghsl_handler(product="GHS_BUILT_S", coord_system=4326, subdir="ghsl_b")
    smod_h = make_ghsl_handler(
        product="GHS_SMOD", resolution=1000, coord_system=54009, subdir="ghsl_s"
    )
    pop_h = make_ghsl_handler(product="GHS_POP", resolution=100, coord_system=4326, subdir="ghsl_p")

    gen = _make_zone_gen_in_raster(tmp_path)
    gen.map_built_s(handler=built_h)
    gen.map_smod(handler=smod_h)
    gen.map_ghsl_pop(handler=pop_h)
    view = gen.map_ghsl(handler=built_h, stat="mean", output_column="custom_built_mean")

    assert {"built_surface_m2", "smod_class", "ghsl_pop", "custom_built_mean"} <= set(
        view.columns
    )


def test_zonal_map_ms_buildings_honors_injected_handler(
    tmp_path, unexpected_ctor, monkeypatch
):
    from types import SimpleNamespace
    import gigaspatial.generators.zonal.base as base_mod

    monkeypatch.setattr(base_mod, "MSBuildingsHandler", unexpected_ctor("MSBuildingsHandler"))
    buildings = gpd.GeoDataFrame(
        {
            "type": ["house", "house"],
            "geometry": [
                Polygon([(0.2, 0.2), (0.2, 0.3), (0.3, 0.3), (0.3, 0.2)]),
                Polygon([(0.4, 0.4), (0.4, 0.5), (0.5, 0.5), (0.5, 0.4)]),
            ],
        },
        crs="EPSG:4326",
    )
    fake_handler = SimpleNamespace(load_data=lambda gdf, ensure_available=True: buildings)
    gen = _make_generator(tmp_path)

    view = gen.map_ms_buildings(handler=fake_handler)
    by_zone = view.set_index("zone_id")
    assert by_zone.loc["z1", "ms_buildings_count"] == 2
    assert by_zone.loc["z2", "ms_buildings_count"] == 0


def test_zonal_map_buildings_uses_injected_handler(
    tmp_path, unexpected_ctor, monkeypatch
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
        load_centroids=lambda fp: {"coords": np.array([[0.5, 0.5], [0.6, 0.6], [2.5, 2.5]])},
    )
    gen = _make_generator(tmp_path)

    view = gen.map_buildings(country="RWA", handler=fake_handler)
    by_zone = view.set_index("zone_id")["building_count"]
    assert by_zone["z1"] == 2
    assert by_zone["z2"] == 1


def test_zonal_map_rwi_uses_injected_handler(tmp_path, unexpected_ctor, monkeypatch):
    from types import SimpleNamespace
    import gigaspatial.generators.base as gen_base_mod

    monkeypatch.setattr(gen_base_mod, "RWIHandler", unexpected_ctor("RWIHandler"))
    rwi_gdf = gpd.GeoDataFrame(
        {
            "rwi": [1.5],
            "geometry": [Polygon([(0.2, 0.2), (0.2, 0.8), (0.8, 0.8), (0.8, 0.2)])],
        },
        crs="EPSG:4326",
    )
    fake_handler = SimpleNamespace(
        config=SimpleNamespace(get_relevant_data_units=lambda c: ["rwa_rwi.csv"]),
        load_as_geodataframe=lambda c, ensure_available=True: rwi_gdf,
    )
    gen = _make_generator(tmp_path)

    view = gen.map_rwi(country="RWA", handler=fake_handler)
    assert view.set_index("zone_id").loc["z1", "rwi_mean"] == pytest.approx(1.5)


@pytest.mark.parametrize("grid_kind", ["h3", "s2", "mercator", "admin"])
def test_zonal_subclasses_forward_injected_handlers(
    tmp_path, make_worldpop_handler, unexpected_ctor, monkeypatch, grid_kind
):
    from types import SimpleNamespace
    import gigaspatial.generators.base as gen_base_mod
    import gigaspatial.handlers.google_ms_combined_buildings as gms_mod
    from gigaspatial.generators.zonal.admin import AdminBoundariesViewGenerator
    from gigaspatial.generators.zonal.h3 import H3ViewGenerator
    from gigaspatial.generators.zonal.mercator import MercatorViewGenerator
    from gigaspatial.generators.zonal.s2 import S2ViewGenerator

    monkeypatch.setattr(
        gen_base_mod, "WPPopulationHandler", unexpected_ctor("WPPopulationHandler")
    )
    monkeypatch.setattr(gen_base_mod, "RWIHandler", unexpected_ctor("RWIHandler"))
    monkeypatch.setattr(
        gms_mod,
        "GoogleMSBuildingsHandler",
        unexpected_ctor("GoogleMSBuildingsHandler"),
    )

    area = Polygon([(0.005, 0.005), (0.005, 0.035), (0.035, 0.035), (0.035, 0.005)])
    store = LocalDataStore(base_path=tmp_path)
    if grid_kind == "h3":
        gen = H3ViewGenerator(area, resolution=7, data_store=store)
    elif grid_kind == "s2":
        gen = S2ViewGenerator(area, level=12, data_store=store)
    elif grid_kind == "mercator":
        gen = MercatorViewGenerator(area, zoom_level=14, data_store=store)
    else:
        admin_path = tmp_path / "admin.geojson"
        gpd.GeoDataFrame(
            {"boundary_id": ["adm1"], "name": ["Area 1"], "geometry": [area]},
            crs="EPSG:4326",
        ).to_file(admin_path, driver="GeoJSON")
        gen = AdminBoundariesViewGenerator(
            country="RWA", admin_level=1, admin_path=admin_path, data_store=store
        )

    wp_handler = make_worldpop_handler()
    rwi_gdf = gpd.GeoDataFrame({"rwi": [2.0], "geometry": [area]}, crs="EPSG:4326")
    rwi_handler = SimpleNamespace(
        config=SimpleNamespace(get_relevant_data_units=lambda c: ["u"]),
        load_as_geodataframe=lambda c, ensure_available=True: rwi_gdf,
    )
    building_file = tmp_path / "bldg.parquet"
    bldg_handler = SimpleNamespace(
        config=SimpleNamespace(),
        reader=SimpleNamespace(resolve_source_paths=lambda c, **kw: [building_file]),
        select_options=lambda kw, cfg=None: {},
        ensure_data_available=lambda c, **kw: True,
        load_centroids=lambda fp: {"coords": np.array([[0.02, 0.02]])},
    )

    gen.map_wp_pop(country="RWA", output_column="wp_population", handler=wp_handler)
    gen.map_rwi(country="RWA", handler=rwi_handler)
    view = gen.map_buildings(country="RWA", handler=bldg_handler)
    assert {"wp_population", "rwi_mean", "building_count"} <= set(view.columns)


# ─────────────────────────────────────────────────────────────────────────────
# WorldPop (map_wp_pop) offline characterization tests (TDD Slice 2)
# ─────────────────────────────────────────────────────────────────────────────


def _make_raster_zones(tmp_path) -> ZonalViewGenerator:
    """Create polygon zones overlapping known cells of the (0.0, 0.0, 0.04, 0.04) 4x4 grid."""
    zones = gpd.GeoDataFrame(
        {
            "zone_id": ["sw_four", "half_top_c2", "outside"],
            "geometry": [
                # Covers SW 2x2 pixels [9, 10, 13, 14] and their centroids
                Polygon([(0.001, 0.001), (0.001, 0.019), (0.019, 0.019), (0.019, 0.001)]),
                # Exact left half of pixel (row=0, col=2), value=3
                Polygon([(0.020, 0.030), (0.020, 0.040), (0.025, 0.040), (0.025, 0.030)]),
                Polygon([(5.0, 5.0), (5.0, 5.1), (5.1, 5.1), (5.1, 5.0)]),
            ],
        },
        crs="EPSG:4326",
    )
    return ZonalViewGenerator(
        zones, data_store=LocalDataStore(base_path=tmp_path)
    )


def test_zonal_map_wp_pop_intersects_sum(tmp_path, make_worldpop_handler):
    handler = make_worldpop_handler(project="pop", resolution=1000)
    gen = _make_raster_zones(tmp_path)

    view = gen.map_wp_pop(
        country="RWA",
        predicate="intersects",
        output_column="wp_pop_sum",
        handler=handler,
    )

    by_id = view.set_index("zone_id")["wp_pop_sum"]
    assert by_id["sw_four"] == pytest.approx(46.0)
    # half_top_c2 shares x=0.020 and y=0.030 edges with pixels (0,1)=2, (1,1)=6, (1,2)=7 -> 2 + 3 + 6 + 7 = 18.0
    assert by_id["half_top_c2"] == pytest.approx(18.0)
    assert np.isnan(by_id["outside"])


def test_zonal_map_wp_pop_degree_of_urbanization_forces_median(
    tmp_path, make_worldpop_handler
):
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
    gen = _make_raster_zones(tmp_path)

    view = gen.map_wp_pop(
        country="RWA",
        predicate="intersects",
        output_column="dug_median",
        handler=handler,
    )

    by_id = view.set_index("zone_id")["dug_median"]
    assert by_id["sw_four"] == pytest.approx(12.5)
    assert by_id["half_top_c2"] == pytest.approx(12.0)
    assert np.isnan(by_id["outside"])


def test_zonal_map_wp_pop_centroid_within_single_and_multi_country(
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

    gen_single = _make_raster_zones(tmp_path)
    single_view = gen_single.map_wp_pop(
        country="RWA",
        predicate="centroid_within",
        output_column="pop_single",
        handler=handler,
    )
    by_single = single_view.set_index("zone_id")["pop_single"]
    # sw_four contains all 4 centroids of [9, 10, 13, 14] -> 46.0
    assert by_single["sw_four"] == pytest.approx(46.0)
    # half_top_c2 covers [0.020, 0.025], stopping just short of pixel centroid at 0.025
    assert np.isnan(by_single["outside"])

    zones_multi = gpd.GeoDataFrame(
        {
            "zone_id": ["in_rwa", "in_bdi"],
            "geometry": [
                Polygon([(0.001, 0.001), (0.001, 0.019), (0.019, 0.019), (0.019, 0.001)]),
                Polygon([(0.041, 0.001), (0.041, 0.019), (0.059, 0.019), (0.059, 0.001)]),
            ],
        },
        crs="EPSG:4326",
    )
    gen_multi = ZonalViewGenerator(
        zones_multi, data_store=LocalDataStore(base_path=tmp_path)
    )
    multi_view = gen_multi.map_wp_pop(
        country=["RWA", "BDI"],
        predicate="centroid_within",
        output_column="pop_multi",
        handler=handler,
    )
    by_multi = multi_view.set_index("zone_id")["pop_multi"]
    assert by_multi["in_rwa"] == pytest.approx(46.0)
    assert by_multi["in_bdi"] == pytest.approx(400.0)


def test_zonal_map_wp_pop_age_structures_sums_rasters_and_rejects_multi_country(
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

    gen = _make_raster_zones(tmp_path)
    with pytest.raises(ValueError, match="age_structures"):
        gen.map_wp_pop(
            country=["RWA", "BDI"],
            predicate="centroid_within",
            handler=age_handler,
        )

    view = gen.map_wp_pop(
        country="RWA",
        predicate="centroid_within",
        output_column="age_pop",
        handler=age_handler,
    )
    by_id = view.set_index("zone_id")["age_pop"]
    # sw_four has 4 cell centroids across 2 rasters: 4 * (2 + 3) = 20
    assert by_id["sw_four"] == pytest.approx(20.0)
    assert np.isnan(by_id["outside"])


def test_zonal_map_wp_pop_fractional_1000m_vs_100m_fallback_warning(
    tmp_path, make_worldpop_handler, caplog, monkeypatch
):
    import logging

    handler_1000 = make_worldpop_handler(
        project="pop", resolution=1000, subdir="wp_1000"
    )
    gen_1000 = _make_raster_zones(tmp_path)
    view_1000 = gen_1000.map_wp_pop(
        country="RWA",
        resolution=1000,
        predicate="fractional",
        output_column="pop_frac",
        handler=handler_1000,
    )
    # half_top_c2 is the exact left half of pixel (row=0, col=2) with value 3.0
    assert view_1000.set_index("zone_id").loc["half_top_c2", "pop_frac"] == pytest.approx(
        1.5, rel=1e-2
    )

    handler_100 = make_worldpop_handler(project="pop", resolution=100, subdir="wp_100")
    gen_100 = _make_raster_zones(tmp_path)
    monkeypatch.setattr(gen_100.logger, "propagate", True)
    with caplog.at_level(logging.WARNING):
        view_100 = gen_100.map_wp_pop(
            country="RWA",
            resolution=100,
            predicate="fractional",
            output_column="pop_100",
            handler=handler_100,
        )

    assert any("intersects" in rec.message for rec in caplog.records)
    # Falls back to edge-inclusive intersects (2 + 3 + 6 + 7 = 18.0)
    assert view_100.set_index("zone_id").loc["half_top_c2", "pop_100"] == pytest.approx(
        18.0
    )


# ─────────────────────────────────────────────────────────────────────────────
# GHSL (map_built_s, map_smod, map_ghsl_pop, map_ghsl) tests (TDD Slice 3)
# ─────────────────────────────────────────────────────────────────────────────


def test_zonal_map_built_s_and_map_ghsl(tmp_path, make_ghsl_handler):
    built_h = make_ghsl_handler(
        product="GHS_BUILT_S", year=2020, resolution=100, coord_system=4326
    )
    gen = _make_raster_zones(tmp_path)

    view = gen.map_built_s(handler=built_h)
    assert view.set_index("zone_id").loc["sw_four", "built_surface_m2"] == pytest.approx(
        46.0
    )
    assert np.isnan(view.set_index("zone_id").loc["outside", "built_surface_m2"])

    view = gen.map_ghsl(handler=built_h, stat="mean")
    assert "ghs_built_s_mean" in view.columns
    assert view.set_index("zone_id").loc["sw_four", "ghs_built_s_mean"] == pytest.approx(
        11.5
    )

    view = gen.map_ghsl(handler=built_h, stat="max", output_column="built_max")
    assert view.set_index("zone_id").loc["sw_four", "built_max"] == pytest.approx(14.0)


def test_zonal_map_smod_reprojects_zones_to_mollweide(tmp_path, make_ghsl_handler):
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

    zones = gpd.GeoDataFrame(
        {
            "zone_id": ["sw_zone", "ne_zone", "out_zone"],
            "geometry": [
                Polygon([(0.002, 0.002), (0.002, 0.018), (0.018, 0.018), (0.018, 0.002)]),
                Polygon([(0.022, 0.022), (0.022, 0.038), (0.038, 0.038), (0.038, 0.022)]),
                Polygon([(5.0, 5.0), (5.0, 5.1), (5.1, 5.1), (5.1, 5.0)]),
            ],
        },
        crs="EPSG:4326",
    )
    gen = ZonalViewGenerator(
        zones, data_store=LocalDataStore(base_path=tmp_path)
    )

    view = gen.map_smod(handler=smod_h)
    by_id = view.set_index("zone_id")["smod_class"]
    assert by_id["sw_zone"] == pytest.approx(12.0)
    assert by_id["ne_zone"] == pytest.approx(30.0)
    assert np.isnan(by_id["out_zone"])


def test_zonal_map_ghsl_pop_intersects_and_fractional_1000m_vs_100m_fallback(
    tmp_path, make_ghsl_handler, caplog, monkeypatch
):
    import logging

    pop_1000_h = make_ghsl_handler(
        product="GHS_POP",
        year=2020,
        resolution=1000,
        coord_system=4326,
        subdir="ghsl_pop_1000",
    )
    pop_100_h = make_ghsl_handler(
        product="GHS_POP",
        year=2020,
        resolution=100,
        coord_system=4326,
        subdir="ghsl_pop_100",
    )

    gen = _make_raster_zones(tmp_path)
    view = gen.map_ghsl_pop(
        resolution=1000,
        predicate="intersects",
        output_column="ghsl_pop_intersects",
        handler=pop_1000_h,
    )
    assert view.set_index("zone_id").loc["sw_four", "ghsl_pop_intersects"] == pytest.approx(
        46.0
    )

    view = gen.map_ghsl_pop(
        resolution=1000,
        predicate="fractional",
        output_column="ghsl_pop_frac",
        handler=pop_1000_h,
    )
    assert view.set_index("zone_id").loc["half_top_c2", "ghsl_pop_frac"] == pytest.approx(
        1.5, rel=1e-2
    )

    monkeypatch.setattr(gen.logger, "propagate", True)
    with caplog.at_level(logging.WARNING):
        view = gen.map_ghsl_pop(
            resolution=100,
            predicate="fractional",
            output_column="ghsl_pop_100_fallback",
            handler=pop_100_h,
        )

    assert any("intersects" in rec.message for rec in caplog.records)
    assert view.set_index("zone_id").loc[
        "sw_four", "ghsl_pop_100_fallback"
    ] == pytest.approx(46.0)


# ─────────────────────────────────────────────────────────────────────────────
# View chaining, exports, and save_view characterization tests (TDD Slice 4)
# ─────────────────────────────────────────────────────────────────────────────


def test_zonal_view_chaining_exports_and_save_view(
    tmp_path, make_worldpop_handler, make_ghsl_handler
):
    import pandas as pd
    from gigaspatial.generators.zonal.base import ZonalViewGeneratorConfig

    store = LocalDataStore(base_path=tmp_path)
    cfg = ZonalViewGeneratorConfig(base_path=tmp_path / "zonal_views")
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

    zones = gpd.GeoDataFrame(
        {
            "zone_id": ["sw_four", "outside"],
            "geometry": [
                Polygon([(0.001, 0.001), (0.001, 0.019), (0.019, 0.019), (0.019, 0.001)]),
                Polygon([(5.0, 5.0), (5.0, 5.1), (5.1, 5.1), (5.1, 5.0)]),
            ],
        },
        crs="EPSG:4326",
    )
    gen = ZonalViewGenerator(zones, config=cfg, data_store=store)

    gen.map_wp_pop(country="RWA", output_column="wp_pop", handler=wp_h)
    gen.map_built_s(output_column="built_m2", handler=built_h)
    chained = gen.map_smod(output_column="smod_val", handler=smod_h)

    expected_cols = {"zone_id", "wp_pop", "built_m2", "smod_val"}
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

    csv_path = gen.save_view("zonal_out", output_format="csv")
    parquet_path = gen.save_view("zonal_out", output_format="parquet")
    assert csv_path.exists()
    assert parquet_path.exists()

    df_csv = pd.read_csv(csv_path)
    df_pq = pd.read_parquet(parquet_path)
    assert expected_cols <= set(df_csv.columns)
    assert expected_cols <= set(df_pq.columns)
    assert df_pq.set_index("zone_id").loc["sw_four", "wp_pop"] == pytest.approx(46.0)


def test_zonal_save_view_geojson(tmp_path):
    from gigaspatial.generators.zonal.base import ZonalViewGeneratorConfig

    store = LocalDataStore(base_path=tmp_path)
    cfg = ZonalViewGeneratorConfig(base_path=tmp_path / "zonal_views")
    zones = gpd.GeoDataFrame(
        {
            "zone_id": ["z1"],
            "geometry": [Polygon([(0.001, 0.001), (0.001, 0.019), (0.019, 0.019), (0.019, 0.001)])],
        },
        crs="EPSG:4326",
    )
    gen = ZonalViewGenerator(zones, config=cfg, data_store=store)
    out_path = gen.save_view("zonal_out", output_format="geojson")
    loaded = gpd.read_file(out_path)
    assert list(loaded["zone_id"]) == ["z1"]


# ─────────────────────────────────────────────────────────────────────────────
# Hierarchy collapse, deprecation, GeoDataFrame sources & ID validation
# ─────────────────────────────────────────────────────────────────────────────


def test_zonal_view_generator_public_imports_and_geometry_based_deprecation(tmp_path):
    from gigaspatial import GeometryBasedZonalViewGenerator as TopDeprecated
    from gigaspatial import ZonalViewGenerator as TopZonal
    from gigaspatial.generators import (
        GeometryBasedZonalViewGenerator as GenDeprecated,
        ZonalViewGenerator as GenZonal,
    )
    from gigaspatial.generators.zonal import (
        GeometryBasedZonalViewGenerator as ZonalPkgDeprecated,
        ZonalViewGenerator as ZonalPkgZonal,
    )

    assert TopZonal is GenZonal is ZonalPkgZonal
    assert TopDeprecated is GenDeprecated is ZonalPkgDeprecated
    assert issubclass(TopDeprecated, TopZonal)

    zones = gpd.GeoDataFrame(
        {
            "zone_id": ["z1"],
            "geometry": [Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])],
        },
        crs="EPSG:4326",
    )
    with pytest.deprecated_call(match="GeometryBasedZonalViewGenerator"):
        legacy = TopDeprecated(zones, data_store=LocalDataStore(base_path=tmp_path))

    assert isinstance(legacy, TopZonal)
    assert list(legacy.view["zone_id"]) == ["z1"]


def test_zonal_subclasses_inherit_zonal_view_generator_without_override_duplicates():
    from gigaspatial.generators.zonal.admin import AdminBoundariesViewGenerator
    from gigaspatial.generators.zonal.h3 import H3ViewGenerator
    from gigaspatial.generators.zonal.mercator import MercatorViewGenerator
    from gigaspatial.generators.zonal.s2 import S2ViewGenerator

    for cls in (
        AdminBoundariesViewGenerator,
        H3ViewGenerator,
        S2ViewGenerator,
        MercatorViewGenerator,
    ):
        assert issubclass(cls, ZonalViewGenerator)
        for method_name in ("map_wp_pop", "map_rwi", "map_buildings"):
            assert method_name not in cls.__dict__


def test_zonal_country_default_resolution_and_missing_country_raises(
    tmp_path, make_worldpop_handler
):
    from gigaspatial.generators.zonal.admin import AdminBoundariesViewGenerator

    store = LocalDataStore(base_path=tmp_path)
    wp_handler = make_worldpop_handler(store=store)

    area = Polygon([(0.005, 0.005), (0.005, 0.035), (0.035, 0.035), (0.035, 0.005)])
    admin_path = tmp_path / "admin.geojson"
    gpd.GeoDataFrame(
        {"boundary_id": ["adm1"], "name": ["Area 1"], "geometry": [area]},
        crs="EPSG:4326",
    ).to_file(admin_path, driver="GeoJSON")
    admin_gen = AdminBoundariesViewGenerator(
        country="RWA", admin_level=1, admin_path=admin_path, data_store=store
    )
    view = admin_gen.map_wp_pop(output_column="wp_population", handler=wp_handler)
    assert view.set_index("zone_id").loc["adm1", "wp_population"] == pytest.approx(136.0)

    plain_gen = _make_raster_zones(tmp_path)
    with pytest.raises(ValueError, match="country"):
        plain_gen.map_wp_pop(handler=wp_handler)
    with pytest.raises(ValueError, match="country"):
        plain_gen.map_rwi()
    with pytest.raises(ValueError, match="country"):
        plain_gen.map_buildings()


def test_grid_generators_accept_geodataframe_source(tmp_path):
    from gigaspatial.generators.zonal.h3 import H3ViewGenerator
    from gigaspatial.generators.zonal.mercator import MercatorViewGenerator
    from gigaspatial.generators.zonal.s2 import S2ViewGenerator

    store = LocalDataStore(base_path=tmp_path)
    area_gdf = gpd.GeoDataFrame(
        {
            "name": ["area_a"],
            "geometry": [
                Polygon([(0.005, 0.005), (0.005, 0.035), (0.035, 0.035), (0.035, 0.005)])
            ],
        },
        crs="EPSG:4326",
    )

    h3_gen = H3ViewGenerator(area_gdf, resolution=7, data_store=store)
    s2_gen = S2ViewGenerator(area_gdf, level=12, data_store=store)
    merc_gen = MercatorViewGenerator(area_gdf, zoom_level=14, data_store=store)

    for gen in (h3_gen, s2_gen, merc_gen):
        assert len(gen.zone_gdf) > 0
        assert {"zone_id", "geometry"} <= set(gen.zone_gdf.columns)


def test_zonal_id_uniqueness_and_custom_column_collision_validation(tmp_path):
    store = LocalDataStore(base_path=tmp_path)
    geom1 = Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])
    geom2 = Polygon([(2, 2), (2, 3), (3, 3), (3, 2)])

    dup_default = gpd.GeoDataFrame(
        {"zone_id": ["dup", "dup"], "geometry": [geom1, geom2]},
        crs="EPSG:4326",
    )
    with pytest.raises(ValueError, match="duplicate"):
        ZonalViewGenerator(dup_default, data_store=store)

    dup_custom = gpd.GeoDataFrame(
        {"region_id": ["r1", "r1"], "geometry": [geom1, geom2]},
        crs="EPSG:4326",
    )
    with pytest.raises(ValueError, match="duplicate"):
        ZonalViewGenerator(dup_custom, zone_id_column="region_id", data_store=store)

    colliding = gpd.GeoDataFrame(
        {
            "zone_id": ["z1", "z2"],
            "region_id": ["r1", "r2"],
            "geometry": [geom1, geom2],
        },
        crs="EPSG:4326",
    )
    with pytest.raises(ValueError, match="zone_id"):
        ZonalViewGenerator(colliding, zone_id_column="region_id", data_store=store)


# ─────────────────────────────────────────────────────────────────────────────
# Ticket 03: BaseViewGenerator extraction and unified zonal contracts
# ─────────────────────────────────────────────────────────────────────────────


def test_zonal_inherits_base_and_preserves_projected_crs_on_to_geodataframe(tmp_path):
    import pandas as pd
    from gigaspatial.generators.base import BaseViewGenerator, BaseViewGeneratorConfig

    assert issubclass(ZonalViewGenerator, BaseViewGenerator)
    projected_zones = gpd.GeoDataFrame(
        {
            "zone_id": ["z1"],
            "geometry": [Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])],
        },
        crs="EPSG:4326",
    ).to_crs("EPSG:3857")

    gen = ZonalViewGenerator(
        projected_zones,
        zone_data_crs="EPSG:3857",
        data_store=LocalDataStore(base_path=tmp_path),
    )
    assert isinstance(gen, BaseViewGenerator)
    assert isinstance(gen.config, BaseViewGeneratorConfig)

    # Defensive copy checks
    v = gen.view
    v["hacked"] = 1
    assert "hacked" not in gen.view.columns
    df = gen.to_dataframe()
    assert isinstance(df, pd.DataFrame) and not isinstance(df, gpd.GeoDataFrame)
    df["hacked_2"] = 2
    assert "hacked_2" not in gen.to_dataframe().columns

    # Preserves projected CRS
    gdf = gen.to_geodataframe()
    assert gdf.crs == "EPSG:3857"
    assert list(gdf["zone_id"]) == ["z1"]


def test_zonal_spatial_primitives_update_view_and_enforce_overwrite(
    tmp_path, make_worldpop_handler, make_ghsl_handler, default_raster_data, write_raster
):
    from types import SimpleNamespace
    from gigaspatial.processing.tif_processor import TifProcessor

    store = LocalDataStore(base_path=tmp_path)
    gen = _make_zone_gen_in_raster(tmp_path)

    # map_points -> returns self.view
    pts = gpd.GeoDataFrame(
        {"val": [5.0], "geometry": [Point(0.01, 0.01)]}, crs="EPSG:4326"
    )
    v_pts = gen.map_points(pts, output_column="pt_cnt")
    assert v_pts.equals(gen.view)
    assert v_pts.set_index("zone_id").loc["z_sw", "pt_cnt"] == 1
    with pytest.raises(ValueError, match="already exist"):
        gen.map_points(pts, output_column="pt_cnt")
    gen.map_points(pts, output_column="pt_cnt", overwrite=True)

    # map_polygons with value_columns=None -> non-mutating count and returns self.view
    polys = gpd.GeoDataFrame(
        {
            "val": [10.0],
            "geometry": [
                Polygon([(0.005, 0.005), (0.005, 0.015), (0.015, 0.015), (0.015, 0.005)])
            ],
        },
        crs="EPSG:4326",
    )
    orig_cols = list(polys.columns)
    v_poly = gen.map_polygons(polys, output_column="poly_cnt")
    assert list(polys.columns) == orig_cols
    assert v_poly.equals(gen.view)
    assert v_poly.set_index("zone_id").loc["z_sw", "poly_cnt"] == 1
    assert v_poly.set_index("zone_id").loc["z_ne", "poly_cnt"] == 0
    with pytest.raises(ValueError, match="already exist"):
        gen.map_polygons(polys, output_column="poly_cnt")
    with pytest.raises(ValueError, match="value_columns"):
        gen.map_polygons(polys, value_columns=None, aggregation="sum")
    gen.map_polygons(polys, output_column="poly_cnt", overwrite=True)

    # map_rasters -> updates and returns self.view
    tif_path = write_raster(tmp_path / "r.tif", data=default_raster_data)
    tp = TifProcessor(dataset_path=tif_path, data_store=store)
    v_ras = gen.map_rasters(tp, stat="mean", output_column="r_mean")
    assert v_ras.equals(gen.view)
    assert "r_mean" in v_ras.columns
    with pytest.raises(ValueError, match="already exist"):
        gen.map_rasters(tp, stat="mean", output_column="r_mean")
    gen.map_rasters(tp, stat="max", output_column="r_mean", overwrite=True)
    tp.cleanup()

    # Dataset methods enforce overwrite=False and allow overwrite=True
    wp_h = make_worldpop_handler(store=store, subdir="wp_z_ov")
    gen.map_wp_pop(country="RWA", handler=wp_h)
    with pytest.raises(ValueError, match="already exist"):
        gen.map_wp_pop(country="RWA", handler=wp_h)
    gen.map_wp_pop(country="RWA", handler=wp_h, overwrite=True)

    built_h = make_ghsl_handler(
        product="GHS_BUILT_S", coord_system=4326, store=store, subdir="b_z_ov"
    )
    gen.map_built_s(handler=built_h)
    with pytest.raises(ValueError, match="already exist"):
        gen.map_built_s(handler=built_h)
    gen.map_built_s(handler=built_h, overwrite=True)

    # map_buildings overwrite
    b_file = tmp_path / "b.parquet"
    b_h = SimpleNamespace(
        config=SimpleNamespace(),
        reader=SimpleNamespace(resolve_source_paths=lambda c, **kw: [b_file]),
        select_options=lambda kw, cfg=None: {},
        ensure_data_available=lambda c, **kw: True,
        load_centroids=lambda fp: {"coords": np.array([[0.01, 0.01]])},
    )
    gen.map_buildings(country="RWA", handler=b_h)
    with pytest.raises(ValueError, match="already exist"):
        gen.map_buildings(country="RWA", handler=b_h)
    gen.map_buildings(country="RWA", handler=b_h, overwrite=True)


def test_zonal_shared_aqueduct_chain_coverage_and_rwi_output_column(
    tmp_path, default_raster_data, write_raster
):
    from types import SimpleNamespace
    from gigaspatial.processing.tif_processor import TifProcessor

    store = LocalDataStore(base_path=tmp_path)
    gen = _make_zone_gen_in_raster(tmp_path)

    tif_path = write_raster(tmp_path / "aq.tif", data=default_raster_data)
    aq_tp = TifProcessor(dataset_path=tif_path, data_store=store)
    aq_h = SimpleNamespace(
        config=SimpleNamespace(
            DATASET_URL="https://example.invalid/aq.tif",
            _get_filename=lambda: "inunriver_hist.tif",
        ),
        load_data=lambda gdf, ensure_available=True, **kw: aq_tp,
    )

    rwi_gdf = gpd.GeoDataFrame(
        {
            "rwi": [1.8],
            "geometry": [
                Polygon([(0.002, 0.002), (0.002, 0.018), (0.018, 0.018), (0.018, 0.002)])
            ],
        },
        crs="EPSG:4326",
    )
    rwi_h = SimpleNamespace(
        config=SimpleNamespace(get_relevant_data_units=lambda c: ["u"]),
        load_as_geodataframe=lambda c, ensure_available=True: rwi_gdf,
    )

    returned = gen.chain_operations(
        [
            {"method": "map_aqueduct_flood_hazard", "kwargs": {"handler": aq_h}},
            {
                "method": "map_rwi",
                "kwargs": {
                    "country": "RWA",
                    "aggregation": "median",
                    "output_column": "wealth_exact",
                    "handler": rwi_h,
                },
            },
            {
                "method": "map_rwi",
                "kwargs": {
                    "country": "RWA",
                    "aggregation": "max",
                    "handler": rwi_h,
                },
            },
        ]
    )
    assert returned is gen
    assert {"inunriver_hist_meters", "wealth_exact", "rwi_max"} <= set(gen.view.columns)
    assert gen.view.set_index("zone_id").loc["z_sw", "wealth_exact"] == pytest.approx(1.8)

    bounds = gpd.GeoDataFrame(
        geometry=[Polygon([(0.0, 0.0), (0.0, 0.04), (0.04, 0.04), (0.04, 0.0)])],
        crs="EPSG:4326",
    )
    cov = gen.validate_data_coverage(bounds)
    assert cov["total_zones"] == 3
    assert cov["covered_zones"] == 2
    assert cov["uncovered_zones"] == 1


def test_zonal_empty_data_contract_including_google_and_ms_buildings(tmp_path):
    import pandas as pd
    from types import SimpleNamespace

    gen = _make_generator(tmp_path)

    # Empty map_google_buildings -> self.view (DataFrame), NOT _zone_gdf.copy()
    empty_google = gpd.GeoDataFrame(
        columns=["full_plus_code", "area_in_meters", "geometry"],
        geometry="geometry",
        crs="EPSG:4326",
    )
    google_h = SimpleNamespace(
        load_points=lambda gdf, ensure_available=True, **kw: empty_google
    )
    v_g = gen.map_google_buildings(handler=google_h)
    assert isinstance(v_g, pd.DataFrame) and not isinstance(v_g, gpd.GeoDataFrame)
    assert v_g.equals(gen.view)
    assert list(v_g["google_buildings_count"]) == [0, 0]
    assert v_g["google_buildings_area_in_meters"].isna().all()

    # Empty map_ms_buildings -> self.view (DataFrame), NOT _zone_gdf.copy()
    empty_ms = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    ms_h = SimpleNamespace(load_data=lambda gdf, ensure_available=True: empty_ms)
    v_ms = gen.map_ms_buildings(handler=ms_h)
    assert isinstance(v_ms, pd.DataFrame) and not isinstance(v_ms, gpd.GeoDataFrame)
    assert v_ms.equals(gen.view)
    assert list(v_ms["ms_buildings_count"]) == [0, 0]
    assert v_ms["ms_buildings_area_in_meters"].isna().all()

    # Empty map_points -> 0 for count, NaN for sum
    v_pt_cnt = gen.map_points(
        gpd.GeoDataFrame(geometry=[], crs="EPSG:4326"), output_column="empty_pt_cnt"
    )
    assert list(v_pt_cnt["empty_pt_cnt"]) == [0, 0]
    v_pt_sum = gen.map_points(
        gpd.GeoDataFrame(geometry=[], crs="EPSG:4326"),
        value_columns="val",
        aggregation="sum",
        output_column="empty_pt_sum",
    )
    assert v_pt_sum["empty_pt_sum"].isna().all()

    # Empty map_polygons -> 0 for count, NaN for mean
    v_poly_cnt = gen.map_polygons(
        gpd.GeoDataFrame(geometry=[], crs="EPSG:4326"), output_column="empty_poly_cnt"
    )
    assert list(v_poly_cnt["empty_poly_cnt"]) == [0, 0]
    v_poly_mean = gen.map_polygons(
        gpd.GeoDataFrame(geometry=[], crs="EPSG:4326"),
        value_columns="val",
        aggregation="mean",
        output_column="empty_poly_mean",
    )
    assert v_poly_mean["empty_poly_mean"].isna().all()

    # Empty map_rasters([]) -> self.view with NaN
    v_r = gen.map_rasters([], stat="mean", output_column="empty_z_r")
    assert v_r["empty_z_r"].isna().all()

    # Empty building files in map_buildings -> 0 counts
    gms_empty = SimpleNamespace(
        config=SimpleNamespace(),
        reader=SimpleNamespace(resolve_source_paths=lambda c, **kw: []),
        select_options=lambda kw, cfg=None: {},
        ensure_data_available=lambda c, **kw: True,
    )
    v_b = gen.map_buildings(country="RWA", handler=gms_empty)
    assert list(v_b["building_count"]) == [0, 0]


def test_zonal_availability_and_resolution_failures_propagate(tmp_path):
    from types import SimpleNamespace

    gen = _make_generator(tmp_path)

    failing_gms = SimpleNamespace(
        config=SimpleNamespace(),
        reader=SimpleNamespace(resolve_source_paths=lambda c, **kw: []),
        select_options=lambda kw, cfg=None: {},
        ensure_data_available=lambda c, **kw: False,
    )
    with pytest.raises(RuntimeError, match="availability"):
        gen.map_buildings(country="RWA", handler=failing_gms)

    failing_rwi = SimpleNamespace(
        config=SimpleNamespace(get_relevant_data_units=lambda c: []),
    )
    with pytest.raises(ValueError, match="RWI"):
        gen.map_rwi(country="RWA", handler=failing_rwi)


def test_zonal_ensure_available_deprecations_and_direct_save_view(
    tmp_path, make_ghsl_handler
):
    import warnings
    import pandas as pd
    from types import SimpleNamespace
    from gigaspatial.generators.zonal.base import ZonalViewGeneratorConfig
    from gigaspatial.generators.zonal.h3 import H3ViewGenerator

    store = LocalDataStore(base_path=tmp_path)

    # ensure_available constructor & per-call override on subclass
    observed = []
    empty_ms = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    ms_h = SimpleNamespace(
        load_data=lambda gdf, ensure_available=True: (
            observed.append(ensure_available) or empty_ms
        )
    )
    area = Polygon([(0.005, 0.005), (0.005, 0.035), (0.035, 0.035), (0.035, 0.005)])
    h3_gen = H3ViewGenerator(area, resolution=7, data_store=store, ensure_available=False)
    h3_gen.map_ms_buildings(handler=ms_h)
    h3_gen.map_ms_buildings(handler=ms_h, ensure_available=True, overwrite=True)
    assert observed == [False, True]

    # Deprecated ZonalViewGeneratorConfig & save_view legacy signature
    zones = gpd.GeoDataFrame(
        {
            "zone_id": ["z1"],
            "geometry": [
                Polygon([(0.001, 0.001), (0.001, 0.019), (0.019, 0.019), (0.019, 0.001)])
            ],
        },
        crs="EPSG:4326",
    )
    cfg = ZonalViewGeneratorConfig(base_path=tmp_path / "cfg_zonal")
    with pytest.deprecated_call(match="ZonalViewGeneratorConfig"):
        gen_cfg = ZonalViewGenerator(zones, config=cfg, data_store=store)
    with pytest.deprecated_call(match="save_view"):
        legacy_p = gen_cfg.save_view("legacy_z")
    assert legacy_p.exists() and legacy_p.suffix == ".parquet"

    # Deprecated dataset_year / dataset_resolution on zonal
    gen = ZonalViewGenerator(zones, data_store=store)
    built_h = make_ghsl_handler(
        product="GHS_BUILT_S", year=2020, resolution=100, coord_system=4326
    )
    with pytest.deprecated_call(match="dataset_year"):
        gen.map_built_s(dataset_year=2020, output_column="bz", handler=built_h)
    smod_h = make_ghsl_handler(
        product="GHS_SMOD", year=2020, resolution=1000, coord_system=54009, subdir="sz_al"
    )
    with pytest.deprecated_call(match="dataset_resolution"):
        gen.map_smod(dataset_resolution=1000, output_column="sz", handler=smod_h)

    # Direct save_view(path) without DeprecationWarning
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        csv_p = gen.save_view(tmp_path / "direct" / "zones.csv")
        pq_p = gen.save_view(tmp_path / "direct" / "zones.parquet")
        pq_geo_p = gen.save_view(
            tmp_path / "direct" / "zones_geo.parquet", include_geometry=True
        )
        geojson_p = gen.save_view(tmp_path / "direct" / "zones.geojson")
        gpkg_p = gen.save_view(tmp_path / "direct" / "zones.gpkg")
    assert not [w for w in caught if issubclass(w.category, DeprecationWarning)]

    assert pd.read_csv(csv_p)["zone_id"].tolist() == ["z1"]
    assert "geometry" not in pd.read_parquet(pq_p).columns
    assert "geometry" in gpd.read_parquet(pq_geo_p).columns
    assert gpd.read_file(geojson_p)["zone_id"].tolist() == ["z1"]
    assert gpd.read_file(gpkg_p)["zone_id"].tolist() == ["z1"]


def test_zonal_handler_loaded_rasters_cleaned_up_deterministically(
    tmp_path, make_ghsl_handler, make_worldpop_handler, monkeypatch
):
    from pathlib import Path
    from gigaspatial.processing.tif_processor import TifProcessor

    store = LocalDataStore(base_path=tmp_path)
    created_temp_dirs = []
    orig_post_init = TifProcessor.__post_init__

    def _tracking_post_init(self):
        orig_post_init(self)
        created_temp_dirs.append(Path(self._temp_dir))

    monkeypatch.setattr(TifProcessor, "__post_init__", _tracking_post_init)

    gen = _make_raster_zones(tmp_path)
    built_h = make_ghsl_handler(
        product="GHS_BUILT_S", year=2020, resolution=100, coord_system=4326, store=store
    )
    gen.map_built_s(handler=built_h)

    wp_h = make_worldpop_handler(project="pop", resolution=1000, store=store)
    gen.map_wp_pop(country="RWA", predicate="centroid_within", handler=wp_h)

    assert len(created_temp_dirs) >= 2
    assert all(not d.exists() for d in created_temp_dirs)


def test_zonal_google_and_ms_buildings_non_empty_use_polygons_and_overwrite(tmp_path):
    from types import SimpleNamespace

    gen = _make_generator(tmp_path)

    google_pts = gpd.GeoDataFrame(
        {
            "full_plus_code": ["G_1", "G_2"],
            "area_in_meters": [50.0, 75.0],
            "geometry": [Point(0.3, 0.3), Point(0.5, 0.5)],
        },
        crs="EPSG:4326",
    )
    google_polys = gpd.GeoDataFrame(
        {
            "area_in_meters": [50.0, 75.0],
            "geometry": [
                Polygon([(0.2, 0.2), (0.2, 0.4), (0.4, 0.4), (0.4, 0.2)]),
                Polygon([(0.4, 0.4), (0.4, 0.6), (0.6, 0.6), (0.6, 0.4)]),
            ],
        },
        crs="EPSG:4326",
    )
    google_h = SimpleNamespace(
        load_points=lambda gdf, ensure_available=True, **kw: google_pts,
        load_polygons=lambda gdf, ensure_available=True, **kw: google_polys,
    )
    v_g = gen.map_google_buildings(handler=google_h, use_polygons=False)
    by_z = v_g.set_index("zone_id")
    assert by_z.loc["z1", "google_buildings_count"] == 2
    assert by_z.loc["z1", "google_buildings_area_in_meters"] == pytest.approx(125.0)
    assert by_z.loc["z2", "google_buildings_count"] == 0
    assert np.isnan(by_z.loc["z2", "google_buildings_area_in_meters"])

    with pytest.raises(ValueError, match="already exist"):
        gen.map_google_buildings(handler=google_h)
    v_gp = gen.map_google_buildings(
        handler=google_h, use_polygons=True, overwrite=True
    )
    assert v_gp.set_index("zone_id").loc["z1", "google_buildings_count"] == 2

    ms_polys = gpd.GeoDataFrame(
        {
            "geometry": [
                Polygon([(0.2, 0.2), (0.2, 0.3), (0.3, 0.3), (0.3, 0.2)]),
            ]
        },
        crs="EPSG:4326",
    )
    ms_h = SimpleNamespace(load_data=lambda gdf, ensure_available=True: ms_polys)
    v_ms = gen.map_ms_buildings(handler=ms_h, use_polygons=False)
    assert v_ms.set_index("zone_id").loc["z1", "ms_buildings_count"] == 1
    assert v_ms.set_index("zone_id").loc["z1", "ms_buildings_area_in_meters"] > 0.0
    with pytest.raises(ValueError, match="already exist"):
        gen.map_ms_buildings(handler=ms_h)
    v_msp = gen.map_ms_buildings(handler=ms_h, use_polygons=True, overwrite=True)
    assert v_msp.set_index("zone_id").loc["z1", "ms_buildings_count"] == 1








