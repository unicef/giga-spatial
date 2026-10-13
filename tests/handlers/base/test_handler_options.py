"""
Typed options of the migrated handlers: each keyword reaches the component
that declares it, and unknown keywords raise ``TypeError``.

All tests run offline: network calls are replaced with fakes.
"""

import gzip
import io
import zipfile
from pathlib import Path
from types import SimpleNamespace

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point, box, mapping

from gigaspatial.core.io.local_data_store import LocalDataStore


class _FakeResponse:
    def __init__(self, content=b"data", payload=None):
        self._content = content
        self._payload = payload
        self.headers = {"content-length": str(len(content))}

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload

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


@pytest.fixture
def google_open_buildings(tmp_path, monkeypatch):
    from gigaspatial.handlers import google_open_buildings as gob

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
    urls = []

    def _get(url, **kwargs):
        urls.append(url)
        return _FakeResponse(payload=tiles)

    monkeypatch.setattr(gob.requests, "get", _get)
    store = LocalDataStore()
    config = gob.GoogleOpenBuildingsConfig(
        base_path=tmp_path, n_workers=1, data_store=store
    )
    handler = gob.GoogleOpenBuildingsHandler(config=config, data_store=store)
    return SimpleNamespace(handler=handler, config=config, store=store, urls=urls)


@pytest.fixture
def google_ms(tmp_path):
    from gigaspatial.handlers.google_ms_combined_buildings import (
        GoogleMSBuildingsConfig,
        GoogleMSBuildingsHandler,
    )

    store = LocalDataStore()
    config = GoogleMSBuildingsConfig(base_path=tmp_path, data_store=store)
    return GoogleMSBuildingsHandler(config=config, data_store=store)


@pytest.fixture
def ms_buildings(tmp_path, monkeypatch):
    from gigaspatial.handlers.microsoft_global_buildings import (
        MSBuildingsConfig,
        MSBuildingsHandler,
    )

    def _load_tile_urls(self, url):
        self.df_tiles = pd.DataFrame(
            {
                "location": ["Fakeland"],
                "quadkey": ["122"],
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
    return MSBuildingsHandler(config=config, data_store=store)


@pytest.fixture
def ookla(tmp_path):
    from gigaspatial.handlers.ookla_speedtest import (
        OoklaSpeedtestConfig,
        OoklaSpeedtestHandler,
    )

    store = LocalDataStore()
    config = OoklaSpeedtestConfig(
        type="fixed",
        year=2023,
        quarter=1,
        base_path=tmp_path,
        data_store=store,
        n_workers=1,
    )
    return OoklaSpeedtestHandler(
        type="fixed", year=2023, quarter=1, config=config, data_store=store
    )


@pytest.fixture
def opencellid(tmp_path):
    from gigaspatial.handlers.opencellid import OpenCellIDConfig, OpenCellIDHandler

    store = LocalDataStore()
    config = OpenCellIDConfig(base_path=tmp_path, api_token="token", data_store=store)
    return OpenCellIDHandler(config=config, data_store=store)


@pytest.fixture
def hdx(tmp_path, monkeypatch):
    pytest.importorskip("hdx")
    from gigaspatial.handlers.hdx import HDXConfig, HDXHandler

    seen = {}

    def _resources(self, filter=None, exact_match=False, token_match=False):
        seen.update(filter=filter, exact_match=exact_match, token_match=token_match)
        return [{"name": "ken.csv"}]

    monkeypatch.setattr(HDXConfig, "configure_hdx", lambda self: None)
    monkeypatch.setattr(HDXConfig, "fetch_dataset", lambda self: None)
    monkeypatch.setattr(HDXConfig, "get_dataset_resources", _resources)
    store = LocalDataStore()
    config = HDXConfig(dataset_name="fake", base_path=tmp_path, data_store=store)
    handler = HDXHandler(dataset_name="fake", config=config, data_store=store)
    handler.seen = seen
    return handler


@pytest.fixture
def rwi(tmp_path, monkeypatch):
    pytest.importorskip("hdx")
    from gigaspatial.handlers.hdx import HDXConfig
    from gigaspatial.handlers.rwi import RWIConfig, RWIHandler

    monkeypatch.setattr(HDXConfig, "configure_hdx", lambda self: None)
    monkeypatch.setattr(HDXConfig, "fetch_dataset", lambda self: None)
    store = LocalDataStore()
    config = RWIConfig(base_path=tmp_path, data_store=store)
    return RWIHandler(config=config, data_store=store)


@pytest.mark.parametrize(
    "name",
    [
        "hdx",
        "rwi",
        "ghsl",
        "worldpop",
        "srtm",
        "aqueduct",
        "google_open_buildings",
        "google_ms",
        "ms_buildings",
        "ookla",
        "opencellid",
    ],
)
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


def test_select_options_can_keep_one_components_options(worldpop):
    shared = {"sex": "F", "merge_rasters": True, "show_progress": False}

    assert worldpop.handler.select_options(shared, worldpop.config) == {"sex": "F"}


# ─────────────────────────────────────────────────────────────────────────────
# SRTM
# ─────────────────────────────────────────────────────────────────────────────


def test_srtm_read_options_reach_the_reader(srtm, tmp_path):
    missing = [str(tmp_path / "N00E000.SRTMGL1.hgt.zip")]

    assert srtm.reader.load(missing, as_dataframe=False) == []
    assert isinstance(srtm.reader.load(missing), pd.DataFrame)


# ─────────────────────────────────────────────────────────────────────────────
# Google Open Buildings
# ─────────────────────────────────────────────────────────────────────────────

AOI = box(0.2, 0.2, 0.4, 0.4)


def test_google_open_buildings_data_type_is_stored_on_the_unit(google_open_buildings):
    config = google_open_buildings.config

    points = config.get_relevant_data_units(AOI, data_type="points")
    polygons = config.get_relevant_data_units(AOI)

    assert [u["data_type"] for u in points] == ["points"]
    assert [u["data_type"] for u in polygons] == ["polygons"]
    assert config.get_data_unit_path(points[0]).name == (
        "points_s2_level_4_abc_buildings.csv.gz"
    )


def test_google_open_buildings_downloads_the_requested_data_type(
    google_open_buildings,
):
    gob = google_open_buildings

    assert gob.handler.ensure_data_available(AOI, data_type="points") is True

    assert [u for u in gob.urls if u.endswith(".csv.gz")] == [
        "https://example.test/points_abc.csv.gz"
    ]
    stored = gob.config.base_path / "points_s2_level_4_abc_buildings.csv.gz"
    assert gob.store.file_exists(str(stored))


def test_google_open_buildings_points_and_polygons_read_their_own_files(
    google_open_buildings, monkeypatch
):
    gob = google_open_buildings
    for data_type in ("polygons", "points"):
        name = f"{data_type}_s2_level_4_abc_buildings.csv.gz"
        gob.store.write_file(str(gob.config.base_path / name), b"csv")
    read = []

    def _load_tabular_data(file_paths):
        read.append([Path(p).name for p in file_paths])
        return pd.DataFrame()

    monkeypatch.setattr(gob.handler.reader, "_load_tabular_data", _load_tabular_data)

    gob.handler.load_polygons(AOI, ensure_available=False)
    gob.handler.load_points(AOI, ensure_available=False)

    assert read == [
        ["polygons_s2_level_4_abc_buildings.csv.gz"],
        ["points_s2_level_4_abc_buildings.csv.gz"],
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Google-Microsoft combined buildings
# ─────────────────────────────────────────────────────────────────────────────


def test_google_ms_source_and_confidence_filters_are_applied(google_ms, tmp_path):
    path = tmp_path / "KEN.parquet"
    gpd.GeoDataFrame(
        {"source": ["google", "microsoft", "google"], "confidence": [0.9, 0.9, 0.3]},
        geometry=[Point(0, 0)] * 3,
        crs="EPSG:4326",
    ).to_parquet(path)

    data = google_ms.load_data([str(path)], source_filter="google", min_confidence=0.5)

    assert list(data["source"]) == ["google"]
    assert list(data["confidence"]) == [0.9]


# ─────────────────────────────────────────────────────────────────────────────
# Microsoft Global Buildings
# ─────────────────────────────────────────────────────────────────────────────


def test_ms_buildings_unmapped_country_uses_the_given_boundary(
    ms_buildings, monkeypatch
):
    from gigaspatial.handlers import microsoft_global_buildings as msb

    calls = []

    def _create(country, zoom_level, **kwargs):
        calls.append((country, kwargs))
        return SimpleNamespace(
            filter_quadkeys=lambda quadkeys: SimpleNamespace(quadkeys=[])
        )

    monkeypatch.setattr(msb.CountryMercatorTiles, "create", staticmethod(_create))

    units = ms_buildings.config.get_relevant_data_units(
        "KEN", country_geom_path="ken.geojson"
    )

    assert units == []
    assert calls == [("KEN", {"data_store": None, "country_geom_path": "ken.geojson"})]


# ─────────────────────────────────────────────────────────────────────────────
# Ookla
# ─────────────────────────────────────────────────────────────────────────────


def test_ookla_show_progress_reaches_the_progress_bar(ookla, monkeypatch):
    from gigaspatial.handlers import ookla_speedtest

    disabled = []

    def _tqdm(iterable, **kwargs):
        disabled.append(kwargs["disable"])
        return iterable

    monkeypatch.setattr(ookla_speedtest.requests, "get", lambda *a, **k: _FakeResponse())
    monkeypatch.setattr(ookla_speedtest, "tqdm", _tqdm)

    ookla.downloader.download_data_units(
        [ookla.config.DATASET_URL], show_progress=False
    )

    assert disabled == [True]


# ─────────────────────────────────────────────────────────────────────────────
# OpenCellID
# ─────────────────────────────────────────────────────────────────────────────


def test_opencellid_read_options_reach_the_reader(opencellid, tmp_path):
    created_2010, created_2020 = 1262304000, 1577836800
    rows = [
        f"GSM,1,1,1,1,0,10.0,1.0,100,1,1,{created_2010},{created_2010},0",
        f"GSM,1,1,1,2,0,11.0,1.0,100,1,1,{created_2020},{created_2020},0",
    ]
    path = tmp_path / "ke" / "MCC-639.csv.gz"
    path.parent.mkdir()
    path.write_bytes(gzip.compress("\n".join(rows).encode()))

    everything = opencellid.load_data([str(path)])
    recent = opencellid.load_data(
        [str(path)], created_newer=2015, as_geodataframe=True
    )

    assert len(everything) == 2
    assert not isinstance(everything, gpd.GeoDataFrame)
    assert isinstance(recent, gpd.GeoDataFrame)
    assert list(recent["lon"]) == [11.0]


# ─────────────────────────────────────────────────────────────────────────────
# HDX
# ─────────────────────────────────────────────────────────────────────────────


def test_hdx_resolve_options_shape_the_resource_filter(hdx):
    hdx.config.get_relevant_data_units("KEN", key="name", exact_match=True)

    assert hdx.seen == {
        "filter": {"name": ["KEN", "KE", "Kenya"]},
        "exact_match": True,
        "token_match": False,
    }


def test_hdx_matching_options_are_part_of_the_cache_key(hdx):
    hdx.config.get_relevant_data_units("KEN")
    hdx.config.get_relevant_data_units("KEN", token_match=True)

    assert hdx.seen["token_match"] is True


def test_hdx_read_kwargs_reach_read_dataset(hdx, tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("a;b\n1;2\n")

    data = hdx.load_data([str(path)], read_kwargs={"sep": ";"})

    assert list(data.columns) == ["a", "b"]
