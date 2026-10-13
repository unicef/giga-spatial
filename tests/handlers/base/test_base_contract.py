"""
Contract tests for the handler base classes, run through a fake handler.

These pin the behaviour the refactor in .agents/plans/handler-base-refactor.md
must keep. They use only the public interface of BaseHandler and its
components, never real datasets or the network.

Run with:
    pytest tests/handlers/base -v
"""

from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point, box

from _fakes import (
    TILES,
    FakeConfig,
    FakeDownloader,
    FakeHandler,
    FakeReader,
    FakeReadOptions,
    FakeResolveOptions,
    tile_rows,
)
from gigaspatial.handlers.base import BaseHandlerConfig, DownloadOptions
from gigaspatial.handlers.base.source import SourceKind, classify_source

# Intersects only tile t00 and contains only its lower-left point.
SMALL_AOI = box(0.0, 0.0, 0.5, 0.5)
# Intersects t00 and t10.
WIDE_AOI = box(0.2, 0.2, 1.5, 0.8)


@pytest.fixture
def handler(tmp_path):
    return FakeHandler(base_path=tmp_path / "fake")


def stored(handler, unit: str) -> bool:
    return handler.data_store.file_exists(
        str(handler.config.get_data_unit_path(unit))
    )


@pytest.fixture
def fake_country(monkeypatch):
    """Make the country code 'FAK' resolve to tile t00 without the network."""
    from gigaspatial.handlers import boundaries

    class _Boundary:
        def to_geoms(self):
            return [box(0.05, 0.05, 0.95, 0.95)]

    def _create(country_code=None, **kwargs):
        assert country_code == "FAK"
        return _Boundary()

    monkeypatch.setattr(boundaries.AdminBoundaries, "create", staticmethod(_create))


@pytest.fixture
def small_country(monkeypatch):
    """Make 'FAK' resolve to a boundary holding only t00's lower-left point."""
    from gigaspatial.handlers import boundaries

    class _Boundary:
        def to_geoms(self):
            return [SMALL_AOI]

    monkeypatch.setattr(
        boundaries.AdminBoundaries,
        "create",
        staticmethod(lambda country_code=None, **kwargs: _Boundary()),
    )


class CountryFileConfig(FakeConfig):
    units_match_country = True


class FilterConfig(FakeConfig):
    """Resolves a ``{"tiles": [...]}`` filter dict, which has no geometry."""

    def get_search_key(self, source, options):
        return source["tiles"]

    def get_relevant_data_units_by_key(self, search_key, options):
        return list(search_key)


@pytest.fixture
def boundary_calls(monkeypatch):
    """Make 'FAK' resolve to SMALL_AOI and record every boundary lookup."""
    from gigaspatial.handlers import boundaries

    calls = []

    class _Boundary:
        def to_geoms(self):
            return [SMALL_AOI]

    def _create(country_code=None, **kwargs):
        calls.append(country_code)
        return _Boundary()

    monkeypatch.setattr(boundaries.AdminBoundaries, "create", staticmethod(_create))
    return calls


class CodeConfig(FakeConfig):
    """Looks units up by country code, like per-country datasets do."""

    def get_search_key(self, source, options):
        return source

    def get_relevant_data_units_by_key(self, search_key, options):
        self.lookups += 1
        return ["t00"]


class GlobalConfig(FakeConfig):
    """One global file: every source resolves to the same unit."""

    def get_search_key(self, source, options):
        return None

    def get_relevant_data_units_by_key(self, search_key, options):
        self.lookups += 1
        return ["t00"]


class PairConfig(FakeConfig):
    """Each unit is stored as a data file plus a sidecar file."""

    def get_stored_paths(self, unit, options=None):
        return [self.base_path / f"{unit}.csv", self.base_path / f"{unit}.meta"]


class PairDownloader(FakeDownloader):
    def download_data_unit(self, unit, options):
        path = super().download_data_unit(unit, options)
        self.data_store.write_file(str(self.config.base_path / f"{unit}.meta"), "m")
        return path


class ArchiveConfig(FakeConfig):
    """Stored files are known only once the unit is extracted, like WorldPop zips."""

    def get_stored_paths(self, unit, options=None):
        path = self.base_path / f"{unit}.csv"
        return [path] if self.data_store.file_exists(str(path)) else []

    def is_unit_available(self, unit, paths):
        return bool(paths)


class MercatorConfig(FakeConfig):
    @property
    def crs(self) -> str:
        return "EPSG:3857"


# ─────────────────────────────────────────────────────────────────────────────
# Availability and download
# ─────────────────────────────────────────────────────────────────────────────


def test_ensure_downloads_only_intersecting_units(handler):
    assert handler.ensure_data_available(WIDE_AOI) is True

    assert sorted(handler.downloader.downloaded) == ["t00", "t10"]
    assert stored(handler, "t00") and stored(handler, "t10")
    assert not stored(handler, "t01") and not stored(handler, "t11")


def test_ensure_skips_units_already_stored(handler):
    handler.ensure_data_available(WIDE_AOI)
    handler.downloader.downloaded.clear()

    assert handler.ensure_data_available(WIDE_AOI) is True
    assert handler.downloader.downloaded == []


def test_ensure_downloads_only_missing_units(handler):
    handler.ensure_data_available(SMALL_AOI)
    handler.downloader.downloaded.clear()

    handler.ensure_data_available(WIDE_AOI)
    assert handler.downloader.downloaded == ["t10"]


def test_force_download_refetches_stored_units(handler):
    handler.ensure_data_available(SMALL_AOI)
    handler.downloader.downloaded.clear()

    assert handler.ensure_data_available(SMALL_AOI, force_download=True) is True
    assert handler.downloader.downloaded == ["t00"]


def test_ensure_propagates_resolution_errors(handler):
    with pytest.raises(ValueError, match="Unsupported source type"):
        handler.ensure_data_available(object())


def test_ensure_returns_false_when_download_fails(handler):
    def _fail(unit, options):
        raise ConnectionError("network down")

    handler.downloader.download_data_unit = _fail

    assert handler.ensure_data_available(SMALL_AOI) is False


def test_ensure_propagates_programming_errors_from_download(handler):
    def _bad_signature(unit):
        return None

    handler.downloader.download_data_unit = _bad_signature

    with pytest.raises(TypeError):
        handler.ensure_data_available(SMALL_AOI)


@pytest.mark.parametrize("n_workers", [1, 2])
def test_one_failing_unit_does_not_stop_the_others(tmp_path, n_workers):
    handler = FakeHandler(base_path=tmp_path / "fake", n_workers=n_workers)
    download = handler.downloader.download_data_unit

    def _fail_first(unit, options):
        if unit == "t00":
            raise ConnectionError("network down")
        return download(unit, options)

    handler.downloader.download_data_unit = _fail_first

    assert handler.ensure_data_available(box(0, 0, 2, 2)) is False
    assert sorted(handler.downloader.downloaded) == ["t01", "t10", "t11"]
    assert handler.ensure_data_available(box(1.2, 1.2, 1.8, 1.8)) is True


def test_units_download_concurrently_in_threads(tmp_path):
    import threading

    handler = FakeHandler(base_path=tmp_path / "fake", n_workers=2)
    download = handler.downloader.download_data_unit
    both_running = threading.Barrier(2, timeout=5)

    def _wait_for_the_other(unit, options):
        both_running.wait()
        return download(unit, options)

    handler.downloader.download_data_unit = _wait_for_the_other

    assert handler.ensure_data_available(box(0, 0, 2, 1)) is True


def test_programming_errors_propagate_from_threads(tmp_path):
    handler = FakeHandler(base_path=tmp_path / "fake", n_workers=2)
    handler.downloader.download_data_unit = lambda unit: None

    with pytest.raises(TypeError):
        handler.ensure_data_available(box(0, 0, 2, 2))


def test_parallel_download_stores_every_unit(tmp_path):
    handler = FakeHandler(base_path=tmp_path / "fake", n_workers=2)

    assert handler.ensure_data_available(box(0, 0, 2, 2)) is True
    assert all(stored(handler, unit) for unit in TILES)


def test_download_data_units_flattens_lists_and_drops_none(handler):
    results = iter([["a", "b"], None, "c"])
    handler.downloader.download_data_unit = lambda unit, options: next(results)

    assert handler.downloader.download_data_units(["x", "y", "z"]) == ["a", "b", "c"]


# ─────────────────────────────────────────────────────────────────────────────
# Source kinds
# ─────────────────────────────────────────────────────────────────────────────


def test_load_geometry_source_returns_rows_of_intersecting_units(handler):
    data = handler.load_data(WIDE_AOI)

    assert sorted(data["tile_id"].unique()) == ["t00", "t10"]
    assert len(data) == 4


def test_load_point_source(handler):
    # (lat, lon) tuples
    data = handler.load_data([(0.5, 0.5), (1.5, 1.5)])

    assert sorted(data["tile_id"].unique()) == ["t00", "t11"]


def test_load_geodataframe_source(handler):
    gdf = gpd.GeoDataFrame(geometry=[WIDE_AOI], crs="EPSG:4326")

    data = handler.load_data(gdf)

    assert sorted(data["tile_id"].unique()) == ["t00", "t10"]


def test_load_country_source(handler, fake_country):
    data = handler.load_data("FAK")

    assert list(data["tile_id"].unique()) == ["t00"]


def test_load_path_source_reads_stored_file_without_resolution(handler):
    handler.ensure_data_available(SMALL_AOI)
    handler.downloader.downloaded.clear()
    path = handler.config.get_data_unit_path("t00")

    data = handler.load_data(path)

    pd.testing.assert_frame_equal(data, tile_rows("t00"))
    assert handler.downloader.downloaded == []


def test_load_list_of_paths_source(handler):
    handler.ensure_data_available(WIDE_AOI)
    paths = [handler.config.get_data_unit_path(u) for u in ("t00", "t10")]

    data = handler.load_data(paths)

    assert sorted(data["tile_id"].unique()) == ["t00", "t10"]


def test_missing_explicit_path_is_not_downloaded(handler):
    missing = handler.config.get_data_unit_path("t00")

    with pytest.raises(RuntimeError):
        handler.load_data(missing)
    assert handler.downloader.downloaded == []


def test_load_without_ensure_does_not_download(handler):
    handler.ensure_data_available(SMALL_AOI)
    handler.downloader.downloaded.clear()

    data = handler.load_data(SMALL_AOI, ensure_available=False)

    assert list(data["tile_id"].unique()) == ["t00"]
    assert handler.downloader.downloaded == []


# ─────────────────────────────────────────────────────────────────────────────
# Stored-file paths
# ─────────────────────────────────────────────────────────────────────────────


def test_force_download_on_stored_path_source_is_available(handler):
    handler.ensure_data_available(SMALL_AOI)
    handler.downloader.downloaded.clear()

    path = handler.config.get_data_unit_path("t00")

    assert handler.ensure_data_available(path, force_download=True) is True
    assert handler.downloader.downloaded == []


def _pair_handler(tmp_path):
    config = PairConfig(base_path=tmp_path / "fake")
    downloader = PairDownloader(config=config, data_store=config.data_store)
    return FakeHandler(base_path=config.base_path, config=config, downloader=downloader)


def test_unit_stored_as_several_files_is_missing_until_all_exist(tmp_path):
    handler = _pair_handler(tmp_path)
    handler.ensure_data_available(WIDE_AOI)
    handler.data_store.remove(str(handler.config.base_path / "t10.meta"))
    handler.downloader.downloaded.clear()

    assert handler.ensure_data_available(WIDE_AOI) is True
    assert handler.downloader.downloaded == ["t10"]


def test_data_unit_paths_are_flattened_per_unit(tmp_path):
    config = PairConfig(base_path=tmp_path)

    assert config.get_data_unit_paths(["t00", "t10"]) == [
        tmp_path / "t00.csv",
        tmp_path / "t00.meta",
        tmp_path / "t10.csv",
        tmp_path / "t10.meta",
    ]


def test_single_string_unit_is_not_split_into_characters(handler):
    assert handler.config.get_data_unit_paths("t00") == [
        handler.config.get_data_unit_path("t00")
    ]


def test_units_with_paths_known_only_after_download(tmp_path):
    config = ArchiveConfig(base_path=tmp_path / "fake")
    handler = FakeHandler(base_path=config.base_path, config=config)

    assert handler.ensure_data_available(SMALL_AOI) is True
    assert handler.downloader.downloaded == ["t00"]
    assert len(handler.load_data(SMALL_AOI)) == 2

    handler.downloader.downloaded.clear()
    assert handler.ensure_data_available(SMALL_AOI) is True
    assert handler.downloader.downloaded == []


def test_config_must_implement_a_path_hook(tmp_path):
    class NoPaths(BaseHandlerConfig):
        def get_relevant_data_units_by_key(self, search_key, options):
            return ["t00"]

    with pytest.raises(NotImplementedError, match="get_stored_paths"):
        NoPaths(base_path=tmp_path).get_data_unit_paths(["t00"])


# ─────────────────────────────────────────────────────────────────────────────
# Cropping
# ─────────────────────────────────────────────────────────────────────────────


def test_crop_to_source_keeps_only_rows_inside_geometry(handler):
    data = handler.load_data(SMALL_AOI, crop_to_source=True)

    assert len(data) == 1
    assert data.iloc[0]["latitude"] == pytest.approx(0.1)


def test_no_crop_returns_whole_units(handler):
    data = handler.load_data(SMALL_AOI)

    assert len(data) == 2


def test_crop_reprojects_geodataframe_source(handler):
    gdf = gpd.GeoDataFrame(geometry=[SMALL_AOI], crs="EPSG:4326").to_crs(3857)

    data = handler.load_data(gdf, crop_to_source=True, crs="EPSG:4326")

    assert len(data) == 1


def test_crop_reads_bare_geometry_in_crs_option(handler):
    aoi_3857 = gpd.GeoSeries([SMALL_AOI], crs=4326).to_crs(3857).iloc[0]

    data = handler.load_data(aoi_3857, crop_to_source=True, crs="EPSG:3857")

    assert list(data["tile_id"]) == ["t00"]
    assert data.iloc[0]["latitude"] == pytest.approx(0.1)


def test_crop_country_source(handler, small_country):
    data = handler.load_data("FAK", crop_to_source=True)

    assert len(data) == 1


def test_crop_skips_country_source_when_units_match_country(tmp_path, small_country):
    config = CountryFileConfig(base_path=tmp_path / "fake")
    handler = FakeHandler(base_path=tmp_path / "fake", config=config)

    data = handler.load_data("FAK", crop_to_source=True)

    assert len(data) == 2


def test_crop_skips_path_source(handler):
    handler.ensure_data_available(SMALL_AOI)
    path = handler.config.get_data_unit_path("t00")

    data = handler.load_data(path, crop_to_source=True)

    assert len(data) == 2


def test_crop_raises_for_source_without_geometry(tmp_path):
    handler = FakeHandler(
        base_path=tmp_path / "fake", config=FilterConfig(base_path=tmp_path / "fake")
    )

    assert len(handler.load_data({"tiles": ["t00"]})) == 2
    with pytest.raises(ValueError, match="needs a source with a geometry"):
        handler.load_data({"tiles": ["t00"]}, crop_to_source=True)


# ─────────────────────────────────────────────────────────────────────────────
# Search key and search geometry
# ─────────────────────────────────────────────────────────────────────────────


def test_search_geometry_is_in_config_crs(tmp_path):
    config = MercatorConfig(base_path=tmp_path)
    expected = gpd.GeoSeries([SMALL_AOI], crs=4326).to_crs(3857).iloc[0]

    geometry = config.extract_search_geometry(SMALL_AOI)

    assert geometry.equals_exact(expected, tolerance=1e-6)


def test_search_key_defaults_to_search_geometry(handler):
    options = handler.config.options_type()

    assert handler.config.get_search_key(SMALL_AOI, options).equals(SMALL_AOI)


def test_extract_search_geometry_rejects_sources_without_geometry(handler):
    with pytest.raises(ValueError, match="Unsupported source type"):
        handler.config.extract_search_geometry({"tiles": ["t00"]})


@pytest.mark.parametrize(
    "source, kind",
    [
        ("KEN", SourceKind.COUNTRY),
        ("St. Lucia", SourceKind.COUNTRY),
        ("t00.csv", SourceKind.PATHS),
        ("data/t00", SourceKind.PATHS),
        (Path("t00"), SourceKind.PATHS),
        (["a.tif", Path("b.tif")], SourceKind.PATHS),
        ([(0.1, 0.1), Point(0.2, 0.2)], SourceKind.POINTS),
        (SMALL_AOI, SourceKind.GEOMETRY),
        (gpd.GeoDataFrame(geometry=[SMALL_AOI]), SourceKind.GEODATAFRAME),
        (pd.DataFrame({"latitude": [0.1], "longitude": [0.1]}), SourceKind.DATAFRAME),
        ({"url": ["KEN"]}, SourceKind.OTHER),
    ],
)
def test_classify_source(source, kind):
    assert classify_source(source) is kind


# ─────────────────────────────────────────────────────────────────────────────
# Unit cache
# ─────────────────────────────────────────────────────────────────────────────


def test_unit_lookup_is_cached_per_source(handler):
    first = handler.config.get_relevant_data_units(WIDE_AOI)
    second = handler.config.get_relevant_data_units(WIDE_AOI)

    assert first == second
    assert handler.config.lookups == 1


def test_force_recompute_bypasses_cache(handler):
    handler.config.get_relevant_data_units(WIDE_AOI)
    handler.config.get_relevant_data_units(WIDE_AOI, force_recompute=True)

    assert handler.config.lookups == 2


def test_cached_search_geometry_matches_source(handler):
    handler.config.get_relevant_data_units(WIDE_AOI)

    assert handler.config.get_cached_search_geometry(WIDE_AOI).equals(WIDE_AOI)


def test_clear_unit_cache(handler):
    handler.config.get_relevant_data_units(WIDE_AOI)
    handler.config.clear_unit_cache()

    assert handler.config.get_cached_search_geometry(WIDE_AOI) is None


def test_crs_option_is_part_of_the_cache_key(handler):
    handler.config.get_relevant_data_units(WIDE_AOI)
    handler.config.get_relevant_data_units(WIDE_AOI, crs="EPSG:4326")

    assert handler.config.lookups == 2


def test_load_resolves_each_source_once(handler):
    handler.load_data(WIDE_AOI, crop_to_source=True)

    assert handler.config.lookups == 1


def test_search_geometry_is_computed_once_when_key_is_not_a_geometry(
    tmp_path, boundary_calls
):
    config = CodeConfig(base_path=tmp_path / "fake")
    handler = FakeHandler(base_path=tmp_path / "fake", config=config)

    handler.load_data("FAK", crop_to_source=True)
    handler.load_data("FAK", crop_to_source=True)

    assert boundary_calls == ["FAK"]
    assert config.lookups == 1


def test_global_dataset_resolves_without_search_geometry(tmp_path, boundary_calls):
    config = GlobalConfig(base_path=tmp_path / "fake")
    handler = FakeHandler(base_path=tmp_path / "fake", config=config)

    assert len(handler.load_data("FAK")) == 2
    assert boundary_calls == []

    assert len(handler.load_data("FAK", crop_to_source=True)) == 1
    assert boundary_calls == ["FAK"]


def test_resolve_options_are_part_of_the_cache_key(handler):
    assert handler.config.get_relevant_data_units(WIDE_AOI) == ["t00", "t10"]
    assert handler.config.get_relevant_data_units(WIDE_AOI, tile="t10") == ["t10"]

    assert handler.config.lookups == 2


# ─────────────────────────────────────────────────────────────────────────────
# Options
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "call",
    [
        lambda h: h.load_data(SMALL_AOI, colour="red"),
        lambda h: h.download_and_load(SMALL_AOI, colour="red"),
        lambda h: h.ensure_data_available(SMALL_AOI, colour="red"),
        lambda h: h.get_available_data_info(SMALL_AOI, colour="red"),
        lambda h: h.reader.load(SMALL_AOI, colour="red"),
        lambda h: h.reader.resolve_source_paths(SMALL_AOI, colour="red"),
        lambda h: h.downloader.download(SMALL_AOI, colour="red"),
        lambda h: h.downloader.download_data_units(["t00"], colour="red"),
        lambda h: h.config.get_relevant_data_units(SMALL_AOI, colour="red"),
        lambda h: h.config.get_data_unit_paths(["t00"], colour="red"),
        lambda h: h.config.get_search_geometry(SMALL_AOI, colour="red"),
    ],
)
def test_unknown_option_raises_type_error(handler, call):
    with pytest.raises(TypeError, match="'colour'"):
        call(handler)
    assert handler.downloader.downloaded == []


@pytest.mark.parametrize(
    "call",
    [
        lambda h: h.ensure_data_available(SMALL_AOI, columns=("tile_id",)),
        lambda h: h.downloader.download(SMALL_AOI, columns=("tile_id",)),
        lambda h: h.config.get_relevant_data_units(SMALL_AOI, show_progress=True),
    ],
)
def test_option_of_a_component_the_entry_does_not_drive_raises(handler, call):
    with pytest.raises(TypeError):
        call(handler)


def test_each_component_receives_only_its_options(handler):
    data = handler.load_data(
        WIDE_AOI, tile="t10", columns=("tile_id",), show_progress=True
    )

    assert list(data.columns) == ["tile_id"]
    assert set(data["tile_id"]) == {"t10"}
    assert handler.downloader.received == [DownloadOptions(show_progress=True)]
    assert handler.reader.received == [
        FakeReadOptions(columns=("tile_id",), show_progress=True)
    ]


def test_options_reach_parallel_downloads(tmp_path):
    handler = FakeHandler(base_path=tmp_path / "fake", n_workers=2)

    assert handler.ensure_data_available(box(0, 0, 2, 2), show_progress=False)
    assert all(stored(handler, unit) for unit in TILES)


# ─────────────────────────────────────────────────────────────────────────────
# Availability report
# ─────────────────────────────────────────────────────────────────────────────


def test_available_data_info_counts_stored_and_missing(handler):
    handler.ensure_data_available(SMALL_AOI)

    info = handler.get_available_data_info(WIDE_AOI)

    assert info["total_data_units"] == 2
    assert info["available_data_units"] == 1
    assert info["missing_paths"] == [handler.config.get_data_unit_path("t10")]
    assert info["all_available"] is False
