"""Tests for token redaction in MaxarConfig and MaxarImageDownloader."""

import io
import logging
from types import SimpleNamespace

import pytest
import requests

from gigaspatial.core.io.local_data_store import LocalDataStore
from gigaspatial.image_downloaders import maxar_image as maxar_mod
from gigaspatial.image_downloaders.maxar_image import (
    MaxarConfig,
    MaxarImageDownloader,
)


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


def _make_downloader(tmp_path, monkeypatch, *, wms_factory=None, wfs_factory=None, **kwargs):
    seen_wms = []
    seen_wfs = []

    def fake_wms(url, version=None, headers=None):
        seen_wms.append({"url": url, "version": version, "headers": headers})
        if wms_factory is not None:
            return wms_factory(url, version, headers)
        return SimpleNamespace(getmap=lambda **params: io.BytesIO(b"png-bytes"))

    def fake_wfs(url, version=None, headers=None):
        seen_wfs.append({"url": url, "version": version, "headers": headers})
        if wfs_factory is not None:
            return wfs_factory(url, version, headers)
        return SimpleNamespace(getfeature=lambda **params: io.BytesIO(b"{}"))

    monkeypatch.setattr(maxar_mod, "_HAS_OWSLIB", True)
    monkeypatch.setattr(maxar_mod, "WebMapService", fake_wms)
    monkeypatch.setattr(maxar_mod, "WebFeatureService", fake_wfs)
    monkeypatch.setattr(maxar_mod, "sleep", lambda _: None)

    downloader = MaxarImageDownloader(
        data_store=LocalDataStore(base_path=tmp_path),
        **kwargs,
    )
    return downloader, seen_wms, seen_wfs


def test_maxar_config_repr_and_str_mask_secrets():
    api_secret = "maxar-secret-key-111"
    bearer_secret = "maxar-bearer-token-222"
    cfg = MaxarConfig(
        api_key=api_secret,
        bearer_token=bearer_secret,
        base_url="https://pro.gegd.com/streaming/v1/ogc/wms?connectid=conn-secret-333",
    )

    for rendered in (repr(cfg), str(cfg)):
        assert api_secret not in rendered
        assert bearer_secret not in rendered
        assert "conn-secret-333" not in rendered
        assert "connectid=***" in rendered

    assert cfg.api_key == api_secret
    assert cfg.bearer_token == bearer_secret


def test_maxar_downloader_repr_masks_secrets(tmp_path, monkeypatch):
    secret = "maxar-secret-key-111"
    downloader, _, _ = _make_downloader(tmp_path, monkeypatch, api_key=secret)
    rendered = repr(downloader)
    assert secret not in rendered


def test_maxar_failed_download_redacts_token_in_logs(tmp_path, monkeypatch):
    secret = "maxar-secret-key-111"
    failing_url = (
        f"https://pro.gegd.com/streaming/v1/ogc/wms?maxar_api_key={secret}&layers=Maxar:Imagery"
    )

    def failing_wms_factory(url, version, headers):
        def getmap(**params):
            raise requests.HTTPError(f"403 Forbidden for url: {failing_url}")

        return SimpleNamespace(getmap=getmap)

    downloader, seen_wms, _ = _make_downloader(
        tmp_path,
        monkeypatch,
        wms_factory=failing_wms_factory,
        api_key=secret,
        max_retries=2,
        retry_delay=0,
    )
    logger, stream = _capture_logger("test.maxar.download.fail")
    downloader.logger = logger

    assert f"maxar_api_key={secret}" in seen_wms[0]["url"]
    ok = downloader._download_single_image(
        (0.0, 0.0, 1.0, 1.0), tmp_path / "tile.png", (256, 256)
    )
    assert ok is False

    logs = stream.getvalue()
    assert secret not in logs
    assert "maxar_api_key=***&layers=Maxar:Imagery" in logs


def test_maxar_failed_metadata_redacts_token_in_log_and_exception(tmp_path, monkeypatch):
    secret = "maxar-secret-key-111"
    failing_url = (
        f"https://pro.gegd.com/streaming/v1/ogc/wfs?maxar_api_key={secret}&typename=Maxar:FinishedFeature"
    )

    def failing_wfs_factory(url, version, headers):
        def getfeature(**params):
            resp = requests.Response()
            resp.status_code = 401
            resp.url = failing_url
            resp.reason = "Unauthorized"
            resp.raise_for_status()

        return SimpleNamespace(getfeature=getfeature)

    downloader, _, seen_wfs = _make_downloader(
        tmp_path,
        monkeypatch,
        wfs_factory=failing_wfs_factory,
        api_key=secret,
    )
    logger, stream = _capture_logger("test.maxar.meta.fail")
    downloader.logger = logger

    with pytest.raises(requests.HTTPError) as exc_info:
        downloader.get_imagery_metadata(bbox=(0.0, 0.0, 1.0, 1.0))

    assert f"maxar_api_key={secret}" in seen_wfs[0]["url"]
    assert secret not in str(exc_info.value)
    assert secret not in exc_info.value.response.url
    assert "maxar_api_key=***" in str(exc_info.value)
    logs = stream.getvalue()
    assert secret not in logs
    assert "maxar_api_key=***" in logs


def test_maxar_failed_wms_init_redacts_token_in_exception(tmp_path, monkeypatch):
    secret = "maxar-secret-key-111"

    def failing_wms_factory(url, version, headers):
        raise requests.HTTPError(f"401 Unauthorized for url: {url}")

    with pytest.raises(requests.HTTPError) as exc_info:
        _make_downloader(
            tmp_path,
            monkeypatch,
            wms_factory=failing_wms_factory,
            api_key=secret,
        )

    assert secret not in str(exc_info.value)
    assert "maxar_api_key=***" in str(exc_info.value)


def test_maxar_successful_download_uses_real_token_and_writes_file(tmp_path, monkeypatch):
    secret = "maxar-secret-key-111"
    downloader, seen_wms, _ = _make_downloader(
        tmp_path, monkeypatch, api_key=secret
    )
    out_path = tmp_path / "tile.png"
    assert downloader._download_single_image((0.0, 0.0, 1.0, 1.0), out_path, (256, 256)) is True
    assert f"maxar_api_key={secret}" in seen_wms[0]["url"]
    assert out_path.read_bytes() == b"png-bytes"
