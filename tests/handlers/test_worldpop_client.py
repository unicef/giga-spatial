"""Catalog lookups retry transient WorldPop failures and raise them otherwise."""

from __future__ import annotations

import requests
import pytest

from gigaspatial.handlers.worldpop import (
    WPPopulationConfig,
    WPResolveOptions,
    WorldPopRestClient,
)


def _response(status, body, headers=None):
    response = requests.Response()
    response.status_code = status
    response._content = body if isinstance(body, bytes) else body.encode()
    response.headers.update(headers or {})
    response.url = "https://www.worldpop.org/rest/data/pop/G2_CN_POP_R25A_1km?iso3=ZWE"
    response.reason = "Error" if status >= 400 else "OK"
    return response


def _json(payload, status=200, headers=None):
    import json

    return _response(status, json.dumps(payload), headers)


class _ScriptedSession:
    def __init__(self, results):
        self.results = list(results)
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append({"url": url, "params": params, "timeout": timeout})
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def _client(session, max_attempts=3, retry_backoff=0):
    client = WorldPopRestClient(max_attempts=max_attempts, retry_backoff=retry_backoff)
    client.session = session
    return client


ZWE_2025 = {
    "data": [
        {
            "popyear": "2025",
            "iso3": "ZWE",
            "files": [
                "https://data.worldpop.org/GIS/Population/Global_2015_2030/"
                "R2025A/2025/ZWE/v1/1km_ua/constrained/"
                "zwe_pop_2025_CN_1km_R2025A_UA_v1.tif"
            ],
        }
    ]
}


class TestGetDatasets:
    def test_returns_catalog_entries(self):
        session = _ScriptedSession([_json(ZWE_2025)])
        datasets = _client(session).get_datasets(
            "pop", "G2_CN_POP_R25A_1km", {"iso3": "ZWE"}
        )

        assert datasets == ZWE_2025["data"]
        assert session.calls[0]["params"] == {"iso3": "ZWE"}
        assert session.calls[0]["url"].endswith("/pop/G2_CN_POP_R25A_1km")

    def test_empty_catalog_is_an_empty_list(self):
        session = _ScriptedSession([_json({"data": []})])
        assert (
            _client(session).get_datasets(
                "pop", "G2_CN_POP_R25A_1km", {"iso3": "ZWE"}
            )
            == []
        )

    def test_rejects_a_response_that_is_not_an_object(self):
        session = _ScriptedSession([_json(["not", "an", "object"])])
        with pytest.raises(ValueError, match="not a JSON object"):
            _client(session).get_datasets(
                "pop", "G2_CN_POP_R25A_1km", {"iso3": "ZWE"}
            )

    def test_retries_a_server_error_then_returns_the_catalog(self, monkeypatch):
        slept = []
        monkeypatch.setattr(
            "gigaspatial.handlers.worldpop.time.sleep", lambda delay: slept.append(delay)
        )
        session = _ScriptedSession(
            [
                _response(500, "<!doctype html>"),
                _response(503, "<!doctype html>"),
                _json(ZWE_2025),
            ]
        )
        datasets = _client(session, retry_backoff=1).get_datasets(
            "pop", "G2_CN_POP_R25A_1km", {"iso3": "ZWE"}
        )

        assert datasets == ZWE_2025["data"]
        assert len(session.calls) == 3
        assert slept == [1.0, 2.0]

    @pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
    def test_raises_when_a_transient_status_persists(self, status):
        session = _ScriptedSession([_response(status, "<!doctype html>")] * 3)
        with pytest.raises(requests.HTTPError):
            _client(session).get_datasets("pop", "G2_CN_POP_R25A_1km", {"iso3": "ZWE"})
        assert len(session.calls) == 3

    def test_does_not_retry_a_rejected_request(self):
        session = _ScriptedSession([_response(404, "missing"), _json(ZWE_2025)])
        with pytest.raises(requests.HTTPError):
            _client(session).get_datasets("pop", "G2_CN_POP_R25A_1km", {"iso3": "ZWE"})
        assert len(session.calls) == 1

    def test_retries_a_connection_error_then_returns_the_catalog(self):
        session = _ScriptedSession(
            [requests.ConnectionError("connection reset"), _json(ZWE_2025)]
        )
        datasets = _client(session).get_datasets(
            "pop", "G2_CN_POP_R25A_1km", {"iso3": "ZWE"}
        )
        assert datasets == ZWE_2025["data"]
        assert len(session.calls) == 2

    def test_raises_the_connection_error_when_retries_are_exhausted(self):
        session = _ScriptedSession([requests.ConnectionError("timed out")] * 3)
        with pytest.raises(requests.ConnectionError, match="timed out"):
            _client(session).get_datasets("pop", "G2_CN_POP_R25A_1km", {"iso3": "ZWE"})
        assert len(session.calls) == 3

    def test_retries_an_html_body(self):
        session = _ScriptedSession([_response(200, "<!doctype html>"), _json(ZWE_2025)])
        datasets = _client(session).get_datasets(
            "pop", "G2_CN_POP_R25A_1km", {"iso3": "ZWE"}
        )
        assert datasets == ZWE_2025["data"]
        assert len(session.calls) == 2

    def test_honours_retry_after_on_rate_limit(self, monkeypatch):
        slept = []
        monkeypatch.setattr(
            "gigaspatial.handlers.worldpop.time.sleep", lambda delay: slept.append(delay)
        )
        session = _ScriptedSession(
            [_response(429, "slow down", headers={"Retry-After": "7"}), _json(ZWE_2025)]
        )
        _client(session, retry_backoff=1).get_datasets(
            "pop", "G2_CN_POP_R25A_1km", {"iso3": "ZWE"}
        )
        assert slept == [7.0]

    def test_rejects_fewer_than_one_attempt(self):
        with pytest.raises(ValueError, match="max_attempts"):
            WorldPopRestClient(max_attempts=0)


def _config(client):
    config = WPPopulationConfig(
        release="GR2",
        project="pop",
        year=2025,
        resolution=1000,
        un_adjusted=False,
        constrained=True,
        school_age=False,
        under_18=False,
    )
    config.client = client
    return config


class TestCatalogFailureIsNotAMissingDataset:
    def test_a_real_miss_still_says_no_datasets_were_found(self):
        client = _client(_ScriptedSession([_json({"data": []})]), max_attempts=1)
        with pytest.raises(
            RuntimeError, match="No WorldPop datasets found for country: ZWE"
        ):
            _config(client).get_relevant_data_units_by_key("ZWE", WPResolveOptions())

    def test_a_failed_catalog_request_is_raised(self):
        client = _client(
            _ScriptedSession([_response(500, "<!doctype html>")]),
            max_attempts=1,
        )
        with pytest.raises(requests.HTTPError):
            _config(client).get_relevant_data_units_by_key("ZWE", WPResolveOptions())
