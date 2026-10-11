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


@breaks("step 3", "GeoDataFrame source is only reprojected when crs= is passed")
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


@breaks("step 2", "load_data passes ensure_available positionally into crop_to_source")
def test_ookla_load_data_defaults_returns_full_dataset(ookla):
    assert len(ookla.handler.load_data()) == 2


@breaks("step 2", "load_data passes ensure_available positionally into crop_to_source")
def test_ookla_load_data_from_path(ookla):
    assert len(ookla.handler.load_data(ookla.path)) == 2


@breaks("step 2", "ensure_available=False is bound to crop_to_source and ignored")
def test_ookla_ensure_available_false_skips_availability_check(ookla, monkeypatch):
    calls = []
    monkeypatch.setattr(
        ookla.handler, "ensure_data_available", lambda *a, **k: calls.append(1) or True
    )

    ookla.handler.load_data(ookla.path, ensure_available=False)

    assert calls == []


@breaks("step 4", "load_data writes a bare quadkey list into the (units, geometry) cache")
def test_ookla_filtering_keeps_unit_cache_readable(ookla):
    aoi = box(36.8, -1.3, 36.81, -1.29)
    ookla.handler.load_data(aoi, ensure_available=False)

    ookla.config.get_cached_search_geometry(aoi)


# ─────────────────────────────────────────────────────────────────────────────
# Download kwargs (L2)
# ─────────────────────────────────────────────────────────────────────────────


@breaks("step 2", "download_data_unit(self, unit) rejects forwarded kwargs")
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


@breaks("step 2", "download_data_unit(tile_info, data_type) rejects forwarded kwargs")
def test_google_open_buildings_download_accepts_forwarded_kwargs(
    google_open_buildings,
):
    unit = google_open_buildings.config.tiles_gdf.iloc[0].to_dict()

    assert google_open_buildings.download_data_units([unit], merge_rasters=False)


@breaks("step 2", "download_by_country forwards data_store/path into download_data_unit")
def test_google_open_buildings_download_by_country(google_open_buildings):
    assert google_open_buildings.download_by_country("FAK")


# ─────────────────────────────────────────────────────────────────────────────
# GHSL: stored paths depend on the file_ext kwarg (L2, L5)
# ─────────────────────────────────────────────────────────────────────────────


@breaks("step 5", "without file_ext the base checks the never-stored .zip path")
def test_ghsl_stored_tif_counts_as_available_without_file_ext(tmp_path):
    from gigaspatial.handlers.ghsl import GHSLDataConfig, GHSLDataHandler

    store = LocalDataStore()
    tiles = gpd.GeoDataFrame(
        {"tile_id": ["R1_C1"]}, geometry=[box(0, 0, 10, 10)], crs="EPSG:4326"
    )
    (tmp_path / "cache").mkdir()
    tiles.to_file(tmp_path / "cache" / "tiles_4326.geojson", driver="GeoJSON")

    config = GHSLDataConfig(
        product="GHS_POP", year=2020, resolution=100, base_path=tmp_path, data_store=store
    )
    handler = GHSLDataHandler(
        product="GHS_POP", year=2020, resolution=100, config=config, data_store=store
    )
    tif = config.get_data_unit_path("R1_C1", file_ext=".tif")
    store.write_file(str(tif), b"tif")

    downloads = []
    handler.downloader.download_data_units = lambda *a, **k: downloads.append(a)

    assert handler.ensure_data_available(box(1, 1, 2, 2)) is True
    assert downloads == []


# ─────────────────────────────────────────────────────────────────────────────
# MS Buildings: crop silently skipped for non-geometry sources (L1)
# ─────────────────────────────────────────────────────────────────────────────


@breaks("step 3", "extract_search_geometry returns the raw source, so a GeoDataFrame is never cropped")
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
