import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import requests

import gigaspatial.fetchers.healthsites as healthsites_mod
from gigaspatial.fetchers.healthsites import HealthSitesFetcher


def test_healthsites_fetcher_imports_without_tenacity():
    repo_root = str(Path(__file__).resolve().parents[2])
    code = (
        "import sys; "
        f"sys.path.insert(0, {repo_root!r}); "
        "sys.modules['tenacity'] = None; "
        "from gigaspatial import HealthSitesFetcher; "
        "assert HealthSitesFetcher is not None"
    )
    subprocess.run([sys.executable, "-W", "ignore", "-c", code], check=True)


def test_fetch_osm_country_name_retries_transient_errors(monkeypatch):
    sleeps = []
    monkeypatch.setattr(healthsites_mod.time, "sleep", sleeps.append)

    timeouts = []

    def fake_get_osm_countries(iso3_code=None, timeout=2000, **kwargs):
        timeouts.append(timeout)
        if len(timeouts) < 3:
            raise requests.exceptions.Timeout("timed out")
        return {"name:en": "Kenya"}

    with patch.object(
        healthsites_mod.OSMLocationFetcher,
        "get_osm_countries",
        side_effect=fake_get_osm_countries,
    ):
        fetcher = HealthSitesFetcher(country="KEN", api_key="test-key")

    assert fetcher.country == "Kenya"
    assert timeouts == [2000, 4000, 6000]
    assert sleeps == [2, 4]


def test_fetch_osm_country_name_exhausts_retries(monkeypatch):
    sleeps = []
    monkeypatch.setattr(healthsites_mod.time, "sleep", sleeps.append)

    with patch.object(
        healthsites_mod.OSMLocationFetcher,
        "get_osm_countries",
        side_effect=requests.exceptions.ConnectionError("network down"),
    ) as mock_get:
        with pytest.raises(requests.exceptions.ConnectionError, match="network down"):
            HealthSitesFetcher(country="KEN", api_key="test-key")

    assert mock_get.call_count == 3
    assert sleeps == [2, 4]


def test_fetch_osm_country_name_does_not_retry_deterministic_value_error(monkeypatch):
    sleeps = []
    monkeypatch.setattr(healthsites_mod.time, "sleep", sleeps.append)

    with patch.object(
        healthsites_mod.OSMLocationFetcher,
        "get_osm_countries",
        return_value={"name": "Kenya"},  # missing name:en
    ) as mock_get:
        with pytest.raises(ValueError, match="Could not find OSM English name"):
            HealthSitesFetcher(country="KEN", api_key="test-key")

    assert mock_get.call_count == 1
    assert sleeps == []
