"""
Tests for the STORM tropical cyclone wind-speed fetcher.

Offline tests extract a stand-in GeoTIFF from a tiny zip. The live test checks
that the Zenodo archive is still published.

Run with:
    pytest tests/fetchers/test_storm.py -v
    pytest tests/fetchers/test_storm.py -m live -v
"""

from __future__ import annotations

import gc
import io
import zipfile

import numpy as np
import pytest
import requests
from rasterio.io import MemoryFile
from rasterio.transform import from_bounds
from shapely.geometry import box

from gigaspatial.fetchers.storm import (
    ARCHIVE_MD5,
    ARCHIVE_URL,
    STORMWindSpeedFetcher,
)
import gigaspatial.fetchers.storm as module


SMALL_SOURCE = box(0.2, 0.2, 0.8, 0.8)
RASTER_BOUNDS = (0.0, 0.0, 2.0, 2.0)


@pytest.fixture
def fetcher():
    built = STORMWindSpeedFetcher(show_progress=False)
    yield built
    built.close()


def _tif_bytes() -> bytes:
    width, height = 20, 20
    profile = dict(
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_bounds(*RASTER_BOUNDS, width, height),
        nodata=-9999.0,
    )
    with MemoryFile() as mem:
        with mem.open(**profile) as dataset:
            dataset.write(np.ones((1, height, width), dtype="float32"))
        return mem.read()


def _archive_bytes() -> bytes:
    """A zip with the members the offline tests extract, nested one directory down."""
    members = (
        module._member_name("constant", 100),
        module._member_name("constant", 50),
        module._member_name("HadGEM3-GC31-HM", 1000),
    )
    payload = _tif_bytes()
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zipped:
        for name in members:
            zipped.writestr(f"STORM_FIXED_RETURN_PERIODS/{name}", payload)
    return buffer.getvalue()


@pytest.fixture
def stub_download(monkeypatch):
    """Write a tiny zip instead of fetching the Zenodo archive."""
    fetched = []

    def fake_stream_to_store(url, path, store, **kwargs):
        fetched.append((url, kwargs.get("md5")))
        store.write_file(str(path), _archive_bytes())
        return path

    monkeypatch.setattr(module, "stream_to_store", fake_stream_to_store)
    return fetched


def test_fetch_returns_the_global_present_day_map(fetcher, stub_download):
    data = fetcher.fetch()

    assert stub_download == [(ARCHIVE_URL, ARCHIVE_MD5)]
    assert data.width == 20
    assert data.height == 20


def test_fetch_clips_to_the_source(fetcher, stub_download):
    data = fetcher.fetch(SMALL_SOURCE, model="HadGEM3-GC31-HM", return_period=1000)

    assert data.width < 20
    assert data.height < 20


def test_another_return_period_reuses_the_archive(fetcher, stub_download):
    fetcher.fetch()
    fetcher.fetch(return_period=50)

    assert stub_download == [(ARCHIVE_URL, ARCHIVE_MD5)]


def test_context_manager_removes_the_temp_dir(stub_download):
    with STORMWindSpeedFetcher(show_progress=False) as fetcher:
        root = fetcher._root
        data = fetcher.fetch()

    assert not root.exists()
    with data.open_dataset() as dataset:
        assert dataset.width == 20


def test_collected_fetcher_removes_the_temp_dir(stub_download):
    fetcher = STORMWindSpeedFetcher(show_progress=False)
    root = fetcher._root
    data = fetcher.fetch()
    del fetcher
    gc.collect()

    assert not root.exists()
    with data.open_dataset() as dataset:
        assert dataset.width == 20


def test_result_survives_close(fetcher, stub_download):
    data = fetcher.fetch()
    fetcher.close()

    with data.open_dataset() as dataset:
        assert dataset.width == 20


def test_closed_fetcher_raises(fetcher, stub_download):
    fetcher.close()

    with pytest.raises(RuntimeError, match="closed"):
        fetcher.fetch()


def test_path_source_raises(fetcher):
    with pytest.raises(ValueError, match="file path"):
        fetcher.fetch("storm.tif")


def test_unknown_model_and_return_period_raise(fetcher):
    with pytest.raises(ValueError, match="model"):
        fetcher.fetch(model="nope")
    with pytest.raises(ValueError, match="return_period"):
        fetcher.fetch(return_period=15)


def test_missing_member_raises(fetcher, stub_download):
    with pytest.raises(FileNotFoundError, match="10_YR_RP"):
        fetcher.fetch(return_period=10)


def _head(url: str) -> requests.Response:
    """Zenodo sometimes stalls a HEAD. Try again before treating that as absence."""
    last_error = None
    for _ in range(3):
        try:
            response = requests.head(url, allow_redirects=True, timeout=90)
        except requests.RequestException as error:
            last_error = error
            continue
        if response.status_code < 500:
            return response
        last_error = requests.RequestException(response.status_code)
    raise last_error


@pytest.mark.live
def test_archive_exists_remotely():
    response = _head(ARCHIVE_URL)
    assert response.status_code == 200, ARCHIVE_URL
