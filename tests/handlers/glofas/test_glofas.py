"""
Tests for the GloFAS flood hazard handler.

Offline tests use an explicit geometry and a two-tile stand-in for the
published grid. The live test resolves that geometry against the real tile
index and checks the depth map exists.

Run with:
    pytest tests/handlers/glofas/test_glofas.py -v
    pytest tests/handlers/glofas/test_glofas.py -m live -v
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
import requests
from rasterio.transform import from_bounds
from shapely.geometry import box

from gigaspatial.handlers.glofas import GlofasHandler
import gigaspatial.handlers.glofas as module


# Inside published tile ID11 (N80_W140): lon -140..-130, lat 70..80.
SMALL_SOURCE = box(-135.2, 74.2, -134.8, 74.8)
TILE_BOUNDS = (-140.0, 70.0, -130.0, 80.0)
DEPTH_URL = (
    "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/CEMS-GLOFAS/flood_hazard/"
    "RP100/ID11_N80_W140_RP100_depth.tif"
)


def install_tiles(handler):
    """Stand in for tile_extents.geojson: ID11, plus a tile far from it."""
    handler.config.tiles_gdf = gpd.GeoDataFrame(
        {"id": [11, 99], "name": ["N80_W140", "N00_E000"]},
        geometry=[box(*TILE_BOUNDS), box(0, 0, 10, 10)],
        crs="EPSG:4326",
    )


@pytest.fixture
def handler(tmp_path):
    built = GlofasHandler(base_path=tmp_path / "glofas", n_workers=1)
    install_tiles(built)
    return built


@pytest.fixture
def stub_download(monkeypatch):
    """Write a coarse depth map instead of fetching the unit."""
    fetched = []

    def fake_stream_to_store(url, path, store, **kwargs):
        fetched.append(url)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        width, height = 20, 20
        transform = from_bounds(*TILE_BOUNDS, width, height)
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=1,
            dtype="float32",
            crs="EPSG:4326",
            transform=transform,
            nodata=-9999.0,
        ) as dataset:
            dataset.write(np.ones((1, height, width), dtype="float32"))
        return path

    monkeypatch.setattr(module, "stream_to_store", fake_stream_to_store)
    return fetched


def test_small_source_resolves_to_the_default_depth_tile(handler):
    units = handler.config.get_relevant_data_units(SMALL_SOURCE)

    assert len(units) == 1
    assert units[0]["tile_id"] == 11
    assert units[0]["layer"] == "depth"
    assert units[0]["return_period"] == 100
    assert units[0]["url"] == DEPTH_URL


def test_stored_path_distinguishes_layer_and_return_period(handler):
    (depth,) = handler.config.get_relevant_data_units(SMALL_SOURCE)
    (reclass,) = handler.config.get_relevant_data_units(
        SMALL_SOURCE, layer="depth_reclass", return_period=50
    )
    (water,) = handler.config.get_relevant_data_units(
        SMALL_SOURCE, layer="permanent_water"
    )

    depth_path = handler.config.get_data_unit_path(depth)
    reclass_path = handler.config.get_data_unit_path(reclass)
    water_path = handler.config.get_data_unit_path(water)

    assert depth_path.name == "ID11_N80_W140_RP100_depth.tif"
    assert {"depth", "RP100"} <= set(depth_path.parts)
    assert reclass_path.name == "ID11_N80_W140_RP50_depth_reclass.tif"
    assert {"depth_reclass", "RP50"} <= set(reclass_path.parts)
    assert water_path.name == "ID11_N80_W140_permanent_water.tif"
    assert "permanent_water" in water_path.parts
    assert "RP100" not in water_path.parts
    assert depth_path != reclass_path != water_path


def test_mask_urls_have_no_return_period(handler):
    (water,) = handler.config.get_relevant_data_units(
        SMALL_SOURCE, layer="permanent_water"
    )
    (spurious,) = handler.config.get_relevant_data_units(
        SMALL_SOURCE, layer="spurious_depth"
    )

    assert water["return_period"] is None
    assert water["url"].endswith(
        "/Permanent_WaterBodies/ID11_N80_W140_permanent_water.tif"
    )
    assert spurious["url"].endswith(
        "/Spurious_Depths/ID11_N80_W140_spurious_depth_areas.tif"
    )


def test_mask_layer_rejects_an_explicit_return_period(handler):
    with pytest.raises(ValueError, match="return_period"):
        handler.config.get_relevant_data_units(
            SMALL_SOURCE, layer="permanent_water", return_period=100
        )
    with pytest.raises(ValueError, match="return_period"):
        handler.config.get_relevant_data_units(
            SMALL_SOURCE, layer="spurious_depth", return_period=10
        )


def test_load_data_downloads_and_crops(handler, stub_download):
    data = handler.load_data(SMALL_SOURCE, crop_to_source=True, show_progress=False)

    assert stub_download == [DEPTH_URL]
    assert data.width < 20
    assert data.height < 20


def test_stored_units_are_not_downloaded_again(handler, stub_download):
    handler.load_data(SMALL_SOURCE, show_progress=False)
    handler.load_data(SMALL_SOURCE, show_progress=False)

    assert len(stub_download) == 1


def test_unknown_keyword_raises(handler):
    with pytest.raises(TypeError):
        handler.load_data(SMALL_SOURCE, not_an_option=True)


def test_downloader_and_reader_declare_config_type():
    assert module.GlofasDownloader.config_type is module.GlofasConfig
    assert module.GlofasReader.config_type is module.GlofasConfig


@pytest.mark.live
def test_resolved_units_exist_remotely(tmp_path):
    handler = GlofasHandler(base_path=tmp_path / "glofas", n_workers=1)
    units = handler.config.get_relevant_data_units(SMALL_SOURCE, return_period=10)

    assert units
    assert units[0]["tile_id"] == 11
    response = requests.head(units[0]["url"], allow_redirects=True, timeout=60)
    assert response.status_code == 200, units[0]["url"]
