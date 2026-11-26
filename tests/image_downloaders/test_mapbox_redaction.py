"""Tests for token redaction in MapboxImageDownloader."""

import io
import logging
from types import SimpleNamespace

import requests

from gigaspatial.core.io.local_data_store import LocalDataStore
from gigaspatial.image_downloaders import mapbox_image as mapbox_mod
from gigaspatial.image_downloaders.mapbox_image import MapboxImageDownloader


def _capture_logger(name: str):
    stream = io.StringIO()
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(levelname)s:%(message)s"))
    logger.addHandler(handler)
    return logger, stream


def test_mapbox_failed_request_redacts_token_in_warning_log(tmp_path, monkeypatch):
    secret = "pk.mapbox-secret-token-987"
    downloader = MapboxImageDownloader(
        access_token=secret,
        data_store=LocalDataStore(base_path=tmp_path),
    )
    logger, stream = _capture_logger("test.mapbox.fail")
    downloader.logger = logger

    def fake_get(url, **kwargs):
        resp = requests.Response()
        resp.status_code = 401
        resp.url = url
        resp.reason = "Unauthorized"
        resp.raise_for_status()

    monkeypatch.setattr(mapbox_mod.requests, "get", fake_get)

    url = downloader._construct_url([0.0, 0.0, 1.0, 1.0], "256x256")
    assert secret in url
    assert downloader._download_single_image(url, tmp_path / "tile.png") is False

    logs = stream.getvalue()
    assert secret not in logs
    assert "access_token=***" in logs
    assert "attribution=false&logo=false" in logs


def test_mapbox_successful_request_uses_real_token_and_writes_file(tmp_path, monkeypatch):
    secret = "pk.mapbox-secret-token-987"
    seen_urls = []

    def fake_get(url, **kwargs):
        seen_urls.append(url)
        return SimpleNamespace(
            content=b"fake-png-bytes",
            raise_for_status=lambda: None,
        )

    monkeypatch.setattr(mapbox_mod.requests, "get", fake_get)

    downloader = MapboxImageDownloader(
        access_token=secret,
        data_store=LocalDataStore(base_path=tmp_path),
    )
    url = downloader._construct_url([0.0, 0.0, 1.0, 1.0], "256x256")
    out_path = tmp_path / "tile.png"

    assert downloader._download_single_image(url, out_path) is True
    assert len(seen_urls) == 1
    assert f"access_token={secret}" in seen_urls[0]
    assert out_path.read_bytes() == b"fake-png-bytes"


def test_mapbox_repr_masks_access_token(tmp_path):
    secret = "pk.mapbox-secret-token-987"
    downloader = MapboxImageDownloader(
        access_token=secret,
        style_id="mapbox/streets-v12",
        data_store=LocalDataStore(base_path=tmp_path),
    )
    rendered = repr(downloader)
    assert secret not in rendered
    assert "access_token='***'" in rendered
    assert "mapbox/streets-v12" in rendered
