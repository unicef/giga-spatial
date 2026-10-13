"""
Typed options of the migrated handlers: each keyword reaches the component
that declares it, and unknown keywords raise ``TypeError``.

All tests run offline: network calls are replaced with fakes.
"""

import io
import zipfile
from pathlib import Path
from types import SimpleNamespace

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

from gigaspatial.core.io.local_data_store import LocalDataStore


class _FakeResponse:
    def __init__(self, content=b"data"):
        self._content = content
        self.headers = {"content-length": str(len(content))}

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size=8192):
        yield self._content


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


@pytest.fixture
def worldpop(tmp_path, monkeypatch):
    from gigaspatial.handlers.worldpop import WPPopulationConfig, WPPopulationHandler

    files = [
        "https://data.worldpop.org/GIS/AgeSex_structures/ken_f_5_2020.tif",
        "https://data.worldpop.org/GIS/AgeSex_structures/ken_m_5_2020.tif",
        "https://data.worldpop.org/GIS/AgeSex_structures/ken_f_10_2020.tif",
    ]
    searches = []

    def _search(*args):
        searches.append(args)
        return [{"files": files}]

    config = WPPopulationConfig(
        release="GR1",
        project="age_structures",
        year=2020,
        resolution=1000,
        school_age=False,
        under_18=False,
        un_adjusted=False,
        constrained=False,
        base_path=tmp_path / "worldpop",
        data_store=LocalDataStore(base_path=tmp_path),
        n_workers=1,
    )
    monkeypatch.setattr(config.client, "search_datasets", _search)
    handler = WPPopulationHandler(config=config, data_store=config.data_store)
    return SimpleNamespace(handler=handler, config=config, searches=searches)


@pytest.fixture
def srtm(tmp_path):
    from gigaspatial.handlers.nasa.srtm import NasaSRTMConfig, NasaSRTMHandler

    config = NasaSRTMConfig(
        base_path=tmp_path, earthdata_username="u", earthdata_password="p"
    )
    return NasaSRTMHandler(config=config)


@pytest.fixture
def aqueduct(tmp_path):
    from gigaspatial.handlers.wri.aqueduct_flood import (
        AqueductFloodConfig,
        AqueductFloodHandler,
    )

    return AqueductFloodHandler(config=AqueductFloodConfig(base_path=tmp_path))


@pytest.mark.parametrize("name", ["ghsl", "worldpop", "srtm", "aqueduct"])
def test_unknown_option_raises_before_any_work(name, request):
    handler = request.getfixturevalue(name)
    handler = getattr(handler, "handler", handler)

    with pytest.raises(TypeError, match="'colour'"):
        handler.load_data(box(1, 1, 2, 2), colour="red")


# ─────────────────────────────────────────────────────────────────────────────
# GHSL
# ─────────────────────────────────────────────────────────────────────────────


def _zip_with(name: str) -> bytes:
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr(name, b"raster")
    return archive.getvalue()


def test_ghsl_merge_rasters_reaches_the_reader(ghsl, monkeypatch):
    ghsl.store.write_file(str(ghsl.config.get_data_unit_path("R1_C1")), b"tif")
    seen = []
    monkeypatch.setattr(
        ghsl.handler.reader,
        "_load_raster_data",
        lambda raster_paths, merge_rasters=False: seen.append(merge_rasters),
    )

    ghsl.handler.load_data(box(1, 1, 2, 2), merge_rasters=True)

    assert seen == [True]


def test_ghsl_extract_option_reaches_the_downloader(ghsl, monkeypatch):
    from gigaspatial.handlers import ghsl as ghsl_module

    stored = ghsl.config.get_data_unit_path("R1_C1")
    monkeypatch.setattr(
        ghsl_module.requests,
        "get",
        lambda *a, **k: _FakeResponse(_zip_with(stored.name)),
    )

    paths = ghsl.handler.downloader.download(box(1, 1, 2, 2), extract=False)

    assert paths == [stored.with_suffix(".zip")]
    assert ghsl.store.file_exists(str(stored.with_suffix(".zip")))
    assert not ghsl.store.file_exists(str(stored))


def test_ghsl_download_by_country_uses_the_given_boundary(ghsl, monkeypatch):
    from gigaspatial.handlers import boundaries
    from gigaspatial.handlers import ghsl as ghsl_module

    calls = []

    class _Boundary:
        def to_geoms(self):
            return [box(1, 1, 2, 2)]

    def _create(country_code=None, **kwargs):
        calls.append((country_code, kwargs))
        return _Boundary()

    monkeypatch.setattr(boundaries.AdminBoundaries, "create", staticmethod(_create))
    stored = ghsl.config.get_data_unit_path("R1_C1")
    monkeypatch.setattr(
        ghsl_module.requests,
        "get",
        lambda *a, **k: _FakeResponse(_zip_with(stored.name)),
    )

    ghsl.handler.downloader.download_by_country(
        "FAK", data_store=ghsl.store, country_geom_path="fak.geojson"
    )

    assert calls == [("FAK", {"path": "fak.geojson", "data_store": ghsl.store})]
    assert ghsl.store.file_exists(str(stored))


def test_ghsl_load_into_dataframe_splits_options_from_frame_kwargs(ghsl, monkeypatch):
    load_calls, frame_calls = [], []

    class _Raster:
        def to_dataframe(self, **kwargs):
            frame_calls.append(kwargs)
            return pd.DataFrame({"pixel_value": [1]})

    def _load_data(**kwargs):
        load_calls.append(kwargs)
        return [_Raster()]

    monkeypatch.setattr(ghsl.handler, "load_data", _load_data)

    data = ghsl.handler.load_into_dataframe(
        box(1, 1, 2, 2), merge_rasters=False, band_number=2
    )

    assert len(data) == 1
    assert load_calls[0]["merge_rasters"] is False
    assert "band_number" not in load_calls[0]
    assert frame_calls == [{"band_number": 2}]


# ─────────────────────────────────────────────────────────────────────────────
# WorldPop
# ─────────────────────────────────────────────────────────────────────────────


def test_worldpop_demographic_options_select_the_files_to_fetch(worldpop):
    names = lambda units: [u.rsplit("/", 1)[1] for u in units]

    everything = worldpop.config.get_relevant_data_units("KEN")
    female = worldpop.config.get_relevant_data_units("KEN", sex="F")
    young = worldpop.config.get_relevant_data_units("KEN", sex=["F"], max_age=5)

    assert names(everything) == [
        "ken_f_5_2020.tif",
        "ken_m_5_2020.tif",
        "ken_f_10_2020.tif",
    ]
    assert names(female) == ["ken_f_5_2020.tif", "ken_f_10_2020.tif"]
    assert names(young) == ["ken_f_5_2020.tif"]
    assert len(worldpop.searches) == 3


def test_worldpop_only_downloads_the_selected_cohort(worldpop, monkeypatch):
    fetched = []

    def _download(url, options):
        fetched.append(url.rsplit("/", 1)[1])
        path = worldpop.config.get_data_unit_path(url)
        worldpop.config.data_store.write_file(str(path), b"tif")
        return path

    monkeypatch.setattr(worldpop.handler.downloader, "download_data_unit", _download)

    assert worldpop.handler.ensure_data_available("KEN", sex="M") is True
    assert fetched == ["ken_m_5_2020.tif"]


def test_worldpop_config_has_no_demographic_side_channel(worldpop):
    assert not hasattr(worldpop.config, "demographic_filter")


def test_select_options_keeps_only_handler_options(worldpop):
    shared = {"sex": "F", "merge_rasters": True, "project": "pop", "stat": "sum"}

    assert worldpop.handler.select_options(shared) == {
        "sex": "F",
        "merge_rasters": True,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SRTM
# ─────────────────────────────────────────────────────────────────────────────


def test_srtm_read_options_reach_the_reader(srtm, tmp_path):
    missing = [str(tmp_path / "N00E000.SRTMGL1.hgt.zip")]

    assert srtm.reader.load(missing, as_dataframe=False) == []
    assert isinstance(srtm.reader.load(missing), pd.DataFrame)
