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
from shapely.geometry import box

from _fakes import TILES, FakeHandler, tile_rows

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
    def _fail(unit, **kwargs):
        raise ConnectionError("network down")

    handler.downloader.download_data_unit = _fail

    assert handler.ensure_data_available(SMALL_AOI) is False


def test_ensure_propagates_programming_errors_from_download(handler):
    def _bad_signature(unit):
        return None

    handler.downloader.download_data_unit = _bad_signature

    with pytest.raises(TypeError):
        handler.ensure_data_available(SMALL_AOI, merge_rasters=True)


def test_parallel_download_stores_every_unit(tmp_path):
    handler = FakeHandler(base_path=tmp_path / "fake", n_workers=2)

    assert handler.ensure_data_available(box(0, 0, 2, 2)) is True
    assert all(stored(handler, unit) for unit in TILES)


def test_download_data_units_flattens_lists_and_drops_none(handler):
    results = iter([["a", "b"], None, "c"])
    handler.downloader.download_data_unit = lambda unit, **kwargs: next(results)

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
