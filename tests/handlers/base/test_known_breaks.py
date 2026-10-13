"""
Confirmed contract breaks in the current handlers.

Each test asserts the target behaviour and is marked ``xfail(strict=True)``
with the plan step that fixes it (.agents/plans/handler-base-refactor.md).
When a step lands, its tests start passing; strict mode then fails the run
until the marker is removed.

All tests run offline: network calls are replaced with fakes.
"""

import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point, box, mapping

from gigaspatial.core.io.local_data_store import LocalDataStore


def breaks(step: str, reason: str):
    return pytest.mark.xfail(strict=True, reason=f"[{step}] {reason}")


class _FakeResponse:
    def __init__(self, payload=None, content=b"data"):
        self._payload = payload
        self._content = content
        self.headers = {"content-length": str(len(content))}

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload

    def iter_content(self, chunk_size=8192):
        yield self._content


# ─────────────────────────────────────────────────────────────────────────────
# Base: CRS of the search geometry (L7)
# ─────────────────────────────────────────────────────────────────────────────


def test_crop_reprojects_geodataframe_source_without_crs_option(tmp_path):
    from _fakes import FakeHandler

    handler = FakeHandler(base_path=tmp_path / "fake")
    gdf = gpd.GeoDataFrame(geometry=[box(0, 0, 0.5, 0.5)], crs="EPSG:4326").to_crs(
        3857
    )

    data = handler.load_data(gdf, crop_to_source=True)

    assert len(data) == 1


# ─────────────────────────────────────────────────────────────────────────────
# Ookla (L3, L4)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def ookla(tmp_path):
    from gigaspatial.handlers.ookla_speedtest import (
        OoklaSpeedtestConfig,
        OoklaSpeedtestHandler,
    )

    store = LocalDataStore()
    config = OoklaSpeedtestConfig(
        type="fixed", year=2023, quarter=1, base_path=tmp_path, data_store=store
    )
    path = config.get_data_unit_path(config.DATASET_URL)
    pd.DataFrame(
        {
            "quadkey": ["1202102332221212", "0000000000000000"],
            "tile": ["POINT (0 0)", "POINT (0 0)"],
            "avg_d_kbps": [1, 2],
        }
    ).to_parquet(path)
    handler = OoklaSpeedtestHandler(
        type="fixed", year=2023, quarter=1, config=config, data_store=store
    )
    return SimpleNamespace(handler=handler, config=config, path=path)


def test_ookla_load_data_defaults_returns_full_dataset(ookla):
    assert len(ookla.handler.load_data()) == 2


def test_ookla_load_data_from_path(ookla):
    assert len(ookla.handler.load_data(ookla.path)) == 2


def test_ookla_ensure_available_false_skips_availability_check(ookla, monkeypatch):
    calls = []
    monkeypatch.setattr(
        ookla.handler, "ensure_data_available", lambda *a, **k: calls.append(1) or True
    )

    ookla.handler.load_data(ookla.path, ensure_available=False)

    assert calls == []


def test_ookla_filtering_keeps_unit_cache_readable(ookla):
    aoi = box(36.8, -1.3, 36.81, -1.29)
    ookla.handler.load_data(aoi, ensure_available=False)

    assert ookla.config.get_cached_search_geometry(aoi).equals(aoi)


# Inside the zoom 16 tile of quadkey 1202102332221212 (Berlin).
OOKLA_AOI = box(13.415, 52.517, 13.416, 52.518)


def test_ookla_crops_to_source_by_default(ookla):
    data = ookla.handler.load_data(OOKLA_AOI)

    assert list(data["quadkey"]) == ["1202102332221212"]


def test_ookla_no_crop_returns_whole_quarter(ookla):
    assert len(ookla.handler.load_data(OOKLA_AOI, crop_to_source=False)) == 2


def test_ookla_country_source_resolves_without_boundary_lookup(ookla, monkeypatch):
    from gigaspatial.handlers import boundaries

    def _no_lookup(*args, **kwargs):
        raise AssertionError("resolving a global dataset must not fetch boundaries")

    monkeypatch.setattr(boundaries.AdminBoundaries, "create", staticmethod(_no_lookup))

    assert ookla.config.get_relevant_data_units("KEN") == [ookla.config.DATASET_URL]
    assert len(ookla.handler.load_data("KEN", crop_to_source=False)) == 2


# ─────────────────────────────────────────────────────────────────────────────
# Download kwargs (L2)
# ─────────────────────────────────────────────────────────────────────────────


def test_google_ms_download_accepts_forwarded_kwargs(tmp_path, monkeypatch):
    from gigaspatial.handlers.google_ms_combined_buildings import (
        GoogleMSBuildingsConfig,
        GoogleMSBuildingsDownloader,
    )

    config = GoogleMSBuildingsConfig(
        base_path=tmp_path, n_workers=1, data_store=LocalDataStore()
    )
    downloader = GoogleMSBuildingsDownloader(config=config)
    monkeypatch.setattr(
        downloader, "_download_from_s3", lambda uri, save_path: save_path
    )
    unit = (
        f"{config.cloud_uri}/geoparquet/by_country/country_iso=KEN/KEN.parquet"
    )

    assert downloader.download_data_units([unit], merge_rasters=False)


@pytest.fixture
def google_open_buildings(tmp_path, monkeypatch):
    from gigaspatial.handlers import boundaries, google_open_buildings as gob

    tiles = {
        "features": [
            {
                "type": "Feature",
                "geometry": mapping(box(0, 0, 1, 1)),
                "properties": {
                    "tile_id": "abc",
                    "tile_url": "https://example.test/polygons_abc.csv.gz",
                    "size_mb": 1.0,
                },
            }
        ]
    }
    monkeypatch.setattr(gob.requests, "get", lambda url, **k: _FakeResponse(tiles))

    class _Boundary:
        def to_geoms(self):
            return [box(0.2, 0.2, 0.8, 0.8)]

    monkeypatch.setattr(
        boundaries.AdminBoundaries,
        "create",
        staticmethod(lambda country_code=None, **k: _Boundary()),
    )
    config = gob.GoogleOpenBuildingsConfig(
        base_path=tmp_path, n_workers=1, data_store=LocalDataStore()
    )
    return gob.GoogleOpenBuildingsDownloader(config=config)


def test_google_open_buildings_download_accepts_forwarded_kwargs(
    google_open_buildings,
):
    unit = google_open_buildings.config.tiles_gdf.iloc[0].to_dict()

    assert google_open_buildings.download_data_units([unit], merge_rasters=False)


def test_google_open_buildings_download_by_country(google_open_buildings):
    assert google_open_buildings.download_by_country("FAK")


def test_google_open_buildings_accepts_caller_crs(google_open_buildings):
    units = google_open_buildings.config.get_relevant_data_units(
        box(0.2, 0.2, 0.4, 0.4), crs="EPSG:4326"
    )

    assert [u["tile_id"] for u in units] == ["abc"]


def test_srtm_accepts_caller_crs(tmp_path):
    from gigaspatial.handlers.nasa.srtm import NasaSRTMConfig

    config = NasaSRTMConfig(
        base_path=tmp_path, earthdata_username="u", earthdata_password="p"
    )

    units = config.get_relevant_data_units(box(36.2, -1.8, 36.4, -1.6), crs="EPSG:4326")

    assert [u["tile_id"] for u in units] == ["S02E036"]


def test_rwi_caller_can_override_token_match(monkeypatch):
    pytest.importorskip("hdx")
    from gigaspatial.handlers.hdx import HDXConfig
    from gigaspatial.handlers.rwi import RWIConfig

    monkeypatch.setattr(HDXConfig, "configure_hdx", lambda self: None)
    monkeypatch.setattr(HDXConfig, "fetch_dataset", lambda self: None)
    seen = {}

    def _resources(self, filter=None, exact_match=False, token_match=False):
        seen["token_match"] = token_match
        return []

    monkeypatch.setattr(HDXConfig, "get_dataset_resources", _resources)
    config = RWIConfig()

    config.get_relevant_data_units("KEN")
    assert seen["token_match"] is True

    config.get_relevant_data_units("KEN", force_recompute=True, token_match=False)
    assert seen["token_match"] is False


def test_rwi_latest_only_is_applied_in_the_lookup(monkeypatch):
    pytest.importorskip("hdx")
    from gigaspatial.handlers.hdx import HDXConfig
    from gigaspatial.handlers.rwi import RWIConfig

    old = {"name": "ken_2021.csv", "created": "2021-01-01T00:00:00Z"}
    new = {"name": "ken_2023.csv", "created": "2023-01-01T00:00:00Z"}
    calls = []

    def _resources(self, filter=None, exact_match=False, token_match=False):
        calls.append(filter)
        return [old, new]

    monkeypatch.setattr(HDXConfig, "configure_hdx", lambda self: None)
    monkeypatch.setattr(HDXConfig, "fetch_dataset", lambda self: None)
    monkeypatch.setattr(HDXConfig, "get_dataset_resources", _resources)
    config = RWIConfig()

    assert config.get_relevant_data_units("KEN") == [new]
    assert config.get_relevant_data_units("KEN") == [new]
    assert len(calls) == 1


# ─────────────────────────────────────────────────────────────────────────────
# GHSL validation
# ─────────────────────────────────────────────────────────────────────────────


def test_ghsl_built_s_accepted_at_10m(tmp_path):
    from gigaspatial.handlers.ghsl import CoordSystem, GHSLDataConfig

    tiles = gpd.GeoDataFrame(
        {"tile_id": ["R1_C1"]}, geometry=[box(0, 0, 10, 10)], crs="ESRI:54009"
    )
    (tmp_path / "cache").mkdir()
    tiles.to_file(tmp_path / "cache" / "tiles_54009.geojson", driver="GeoJSON")

    config = GHSLDataConfig(
        product="GHS_BUILT_S",
        resolution=10,
        base_path=tmp_path,
        data_store=LocalDataStore(),
    )

    assert (config.year, config.coord_system) == (2018, CoordSystem.Mollweide)


def test_ghsl_rejects_10m_for_products_without_it(tmp_path):
    from gigaspatial.handlers.ghsl import GHSLDataConfig

    with pytest.raises(ValueError, match="not available at 10"):
        GHSLDataConfig(product="GHS_POP", resolution=10, base_path=tmp_path)


# ─────────────────────────────────────────────────────────────────────────────
# GHSL: stored paths are the extracted rasters (L2, L5)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def ghsl(tmp_path):
    from gigaspatial.handlers.ghsl import GHSLDataConfig, GHSLDataHandler

    store = LocalDataStore()
    tiles = gpd.GeoDataFrame(
        {"tile_id": ["R1_C1"]}, geometry=[box(0, 0, 10, 10)], crs="EPSG:4326"
    )
    (tmp_path / "cache").mkdir()
    tiles.to_file(tmp_path / "cache" / "tiles_4326.geojson", driver="GeoJSON")

    config = GHSLDataConfig(
        product="GHS_POP",
        year=2020,
        resolution=100,
        base_path=tmp_path,
        data_store=store,
        n_workers=1,
    )
    handler = GHSLDataHandler(
        product="GHS_POP", year=2020, resolution=100, config=config, data_store=store
    )
    return SimpleNamespace(handler=handler, config=config, store=store)


def test_ghsl_stored_tif_counts_as_available_without_file_ext(ghsl):
    ghsl.store.write_file(str(ghsl.config.get_data_unit_path("R1_C1")), b"tif")
    downloads = []
    ghsl.handler.downloader.download_data_units = lambda *a, **k: downloads.append(a)

    assert ghsl.handler.ensure_data_available(box(1, 1, 2, 2)) is True
    assert downloads == []


def test_ghsl_stored_tif_is_named_after_the_zip(ghsl):
    url = ghsl.config.compute_dataset_url(tile_id="R1_C1")

    assert ghsl.config.get_data_unit_path("R1_C1").name == (
        url.rsplit("/", 1)[1].replace(".zip", ".tif")
    )


def test_ghsl_download_extracts_to_the_stored_path(ghsl, monkeypatch):
    import io
    import zipfile

    from gigaspatial.handlers import ghsl as ghsl_module

    stored = ghsl.config.get_data_unit_path("R1_C1")
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr(stored.name, b"raster")
        zf.writestr("GHSL_Data_Package.pdf", b"doc")
    monkeypatch.setattr(
        ghsl_module.requests,
        "get",
        lambda *a, **k: _FakeResponse(content=archive.getvalue()),
    )

    assert ghsl.handler.ensure_data_available(box(1, 1, 2, 2)) is True
    assert ghsl.store.read_file(str(stored)) == b"raster"
    info = ghsl.handler.get_available_data_info(box(1, 1, 2, 2))
    assert info["available_paths"] == [stored] and info["all_available"]


# ─────────────────────────────────────────────────────────────────────────────
# WorldPop: zip resources are stored as their extracted rasters (L5)
# ─────────────────────────────────────────────────────────────────────────────

WP_ZIP = (
    "https://data.worldpop.org/GIS/AgeSex_structures/School_Age_Population/"
    "v1/2020/KEN/KEN_SAP_1km_2020.zip"
)


@pytest.fixture
def school_age(tmp_path):
    from gigaspatial.handlers.worldpop import WPPopulationConfig

    return WPPopulationConfig(
        release="GR1",
        project="age_structures",
        year=2020,
        resolution=1000,
        school_age=True,
        under_18=False,
        un_adjusted=False,
        constrained=False,
        base_path=tmp_path / "worldpop",
        data_store=LocalDataStore(base_path=tmp_path),
    )


def test_worldpop_zip_unit_has_no_stored_paths_before_extraction(school_age):
    assert school_age.get_stored_paths(WP_ZIP) == []
    assert school_age.is_unit_available(WP_ZIP, []) is False


def test_worldpop_zip_unit_is_stored_as_filtered_extracted_tifs(school_age):
    folder = school_age.get_data_unit_path(WP_ZIP).parent
    for name in ("KEN_F_M_PRIMARY_2020_1km.tif", "KEN_F_PRIMARY_2020_1km.tif"):
        school_age.data_store.write_file(str(folder / name), b"tif")

    default = school_age.get_stored_paths(WP_ZIP)
    female = school_age.get_stored_paths(WP_ZIP, sex="F")
    male = school_age.get_stored_paths(WP_ZIP, sex="M")

    assert [Path(p).name for p in default] == ["KEN_F_M_PRIMARY_2020_1km.tif"]
    assert [Path(p).name for p in female] == ["KEN_F_PRIMARY_2020_1km.tif"]
    assert male == [] and school_age.is_unit_available(WP_ZIP, male) is True


def test_worldpop_tif_unit_is_stored_as_itself(school_age):
    url = WP_ZIP.replace(".zip", ".tif")

    assert school_age.get_stored_paths(url) == [school_age.get_data_unit_path(url)]


# ─────────────────────────────────────────────────────────────────────────────
# MS Buildings: crop silently skipped for non-geometry sources (L1)
# ─────────────────────────────────────────────────────────────────────────────


def test_ms_buildings_crops_geodataframe_source(tmp_path, monkeypatch):
    from gigaspatial.grid.mercator_tiles import MercatorTiles
    from gigaspatial.handlers.microsoft_global_buildings import (
        MSBuildingsConfig,
        MSBuildingsHandler,
    )

    aoi = box(0.10, 0.10, 0.20, 0.20)
    quadkey = MercatorTiles.from_spatial(aoi, zoom_level=9).quadkeys[0]
    tile = MercatorTiles.from_quadkeys([quadkey]).to_geoms()[0]
    inside = Point(0.15, 0.15)
    outside = Point(tile.bounds[2] - 0.01, tile.bounds[3] - 0.01)

    def _load_tile_urls(self, url):
        self.df_tiles = pd.DataFrame(
            {
                "location": ["Fakeland"],
                "quadkey": [quadkey],
                "url": ["https://example.test/tile.csv.gz"],
                "size": ["1KB"],
                "upload_date": ["2024-01-01"],
            }
        )

    def _setup_location_mapping(self):
        self.df_tiles["country"] = "FAK"

    monkeypatch.setattr(MSBuildingsConfig, "_load_tile_urls", _load_tile_urls)
    monkeypatch.setattr(
        MSBuildingsConfig, "_setup_location_mapping", _setup_location_mapping
    )

    store = LocalDataStore()
    config = MSBuildingsConfig(
        MANIFEST_URL="https://bfppub.blob.core.windows.net/$web/2024-01-01/dataset-links.csv",
        base_path=tmp_path,
        data_store=store,
    )
    unit = config.df_tiles.iloc[0].to_dict()
    lines = "\n".join(
        json.dumps({"type": "Feature", "geometry": mapping(p), "properties": {}})
        for p in (inside, outside)
    )
    store.write_file(
        str(config.get_data_unit_path(unit)), gzip.compress(lines.encode())
    )
    handler = MSBuildingsHandler(config=config, data_store=store)

    gdf = gpd.GeoDataFrame(geometry=[aoi], crs="EPSG:4326")
    data = handler.load_data(gdf, crop_to_source=True)

    assert len(data) == 1
