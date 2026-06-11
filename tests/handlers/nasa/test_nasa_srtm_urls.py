"""
Tests for NasaSRTMConfig HTTPS tile URL builder.

Run with:
    pytest tests/handlers/srtm/test_nasa_srtm_urls.py -v
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from gigaspatial.handlers.nasa.srtm import NasaSRTMConfig


@pytest.fixture
def mock_global_config():
    with patch("gigaspatial.handlers.srtm.nasa_srtm.global_config") as mock_cfg:
        mock_cfg.EARTHDATA_USERNAME = "user"
        mock_cfg.EARTHDATA_PASSWORD = "pass"
        mock_cfg.LPDAAC_S3_REGION = "us-west-2"
        mock_cfg.LPDAAC_S3_BUCKET = "lp-prod-protected"
        mock_cfg.get_path.return_value = Path("/tmp/nasa_srtm")
        yield mock_cfg


def test_get_tile_url_30m_protected(mock_global_config):
    cfg = NasaSRTMConfig(resolution="30m", s3_bucket="lp-prod-protected")
    url = cfg.get_tile_url("S16E035")
    assert (
        url
        == "https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/"
        "SRTMGL1.003/S16E035.SRTMGL1.hgt/S16E035.SRTMGL1.hgt.zip"
    )


def test_get_tile_url_90m_protected(mock_global_config):
    cfg = NasaSRTMConfig(resolution="90m", s3_bucket="lp-prod-protected")
    url = cfg.get_tile_url("N00E013")
    assert "/SRTMGL3.003/" in url
    assert url.endswith("N00E013.SRTMGL3.hgt.zip")


def test_get_tile_url_public_bucket(mock_global_config):
    cfg = NasaSRTMConfig(resolution="30m", s3_bucket="lp-prod-public")
    url = cfg.get_tile_url("S16E035")
    assert "/lp-prod-public/" in url
    assert "lp-prod-protected" not in url
