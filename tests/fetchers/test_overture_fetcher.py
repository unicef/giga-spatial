"""Tests for OvertureAmenityFetcher release resolution and validation."""

from __future__ import annotations

import json
import logging
import socket
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests

import gigaspatial.fetchers.overture as overture_mod
from gigaspatial import OvertureAmenityFetcher

STAC_CATALOG_URL = "https://stac.overturemaps.org/catalog.json"
RELEASE_CALENDAR_URL = "https://docs.overturemaps.org/release-calendar"


@pytest.fixture(autouse=True)
def _block_network(monkeypatch):
    def fail_connect(*args, **kwargs):
        raise AssertionError(f"Unexpected network access: {args!r}")

    monkeypatch.setattr(socket, "create_connection", fail_connect)


def _stac_response(payload, status: int = 200) -> requests.Response:
    response = requests.Response()
    response.status_code = status
    response.url = STAC_CATALOG_URL
    response.reason = "Error" if status >= 400 else "OK"
    body = payload if isinstance(payload, (str, bytes)) else json.dumps(payload)
    response._content = body if isinstance(body, bytes) else body.encode()
    return response


def _mock_stac_get(monkeypatch, outcome):
    calls = []

    def fake_get(url, timeout=None, **kwargs):
        calls.append({"url": url, "timeout": timeout, **kwargs})
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    monkeypatch.setattr(requests, "get", fake_get)
    return calls


# ----- Seam 1: get_latest_release


def test_get_latest_release_fetches_stac_and_normalizes(monkeypatch):
    calls = _mock_stac_get(monkeypatch, _stac_response({"latest": " 2026-11-18.0/ "}))

    assert OvertureAmenityFetcher.get_latest_release() == "2026-11-18.0"
    assert calls == [{"url": STAC_CATALOG_URL, "timeout": 20}]


def test_get_latest_release_forwards_custom_timeout(monkeypatch):
    calls = _mock_stac_get(monkeypatch, _stac_response({"latest": "2026-11-18.0"}))

    assert OvertureAmenityFetcher.get_latest_release(timeout=5) == "2026-11-18.0"
    assert calls == [{"url": STAC_CATALOG_URL, "timeout": 5}]


@pytest.mark.parametrize(
    "outcome",
    [
        requests.ConnectionError("offline"),
        requests.Timeout("timed out"),
        _stac_response({"message": "unavailable"}, status=503),
        _stac_response("not json"),
        _stac_response(["2026-11-18.0"]),
        _stac_response({}),
        _stac_response({"latest": ""}),
        _stac_response({"latest": "   "}),
        _stac_response({"latest": 123}),
    ],
)
def test_get_latest_release_raises_runtime_error_on_bad_stac_response(monkeypatch, outcome):
    _mock_stac_get(monkeypatch, outcome)

    with pytest.raises(RuntimeError):
        OvertureAmenityFetcher.get_latest_release()


# ----- Seam 1: get_available_releases


def test_get_available_releases_parses_links_includes_latest_and_sorts_newest_first(monkeypatch):
    catalog = {
        "latest": "2026-11-18.0/",
        "links": [
            {"rel": "root", "href": "./catalog.json"},
            {"rel": "self", "href": "https://stac.overturemaps.org/catalog.json"},
            {"rel": "child", "href": "https://stac.overturemaps.org/2026-09-24.0/catalog.json"},
            {"rel": "child", "href": "https://mirror.example.org/overture/stac/2026-08-20.0/catalog.json"},
            {"rel": "child", "href": "./2026-10-21.0/"},
            {"rel": "child", "href": "./2026-10-21.0/catalog.json"},
            {"rel": "child", "href": "./catalog.json"},
            "not-a-dict",
            {"rel": "child", "href": 42},
            {"rel": "child", "href": ""},
        ],
    }
    calls = _mock_stac_get(monkeypatch, _stac_response(catalog))

    releases = OvertureAmenityFetcher.get_available_releases(timeout=7)

    assert releases == ["2026-11-18.0", "2026-10-21.0", "2026-09-24.0", "2026-08-20.0"]
    assert calls == [{"url": STAC_CATALOG_URL, "timeout": 7}]


def test_get_available_releases_works_when_latest_is_also_in_child_links(monkeypatch):
    catalog = {
        "latest": "2026-11-18.0",
        "links": [
            {"rel": "child", "href": "./2026-10-21.0/catalog.json"},
            {"rel": "child", "href": "./2026-11-18.0/catalog.json"},
        ],
    }
    _mock_stac_get(monkeypatch, _stac_response(catalog))

    assert OvertureAmenityFetcher.get_available_releases() == ["2026-11-18.0", "2026-10-21.0"]


@pytest.mark.parametrize(
    "outcome",
    [
        requests.ConnectionError("offline"),
        _stac_response("not json"),
        _stac_response({}),
        _stac_response({"links": [{"rel": "root", "href": "./catalog.json"}]}),
        _stac_response({"links": [{"rel": "child", "href": "./catalog.json"}], "latest": "  "}),
        _stac_response({"links": "bad"}),
    ],
)
def test_get_available_releases_raises_runtime_error_when_unavailable_or_empty(monkeypatch, outcome):
    _mock_stac_get(monkeypatch, outcome)

    with pytest.raises(RuntimeError):
        OvertureAmenityFetcher.get_available_releases()


# ----- Seam 2: OvertureAmenityFetcher instantiation


class _FakeDuckDBConnection:
    def install_extension(self, name: str) -> None:
        pass

    def load_extension(self, name: str) -> None:
        pass

    def execute(self, query: str) -> None:
        pass


@pytest.fixture
def fake_duckdb(monkeypatch):
    monkeypatch.setattr(overture_mod, "_HAS_DUCKDB", True)
    monkeypatch.setattr(
        overture_mod,
        "duckdb",
        SimpleNamespace(connect=lambda: _FakeDuckDBConnection()),
        raising=False,
    )


def test_no_stale_default_release_constant():
    assert not hasattr(OvertureAmenityFetcher, "_DEFAULT_RELEASE")


@pytest.mark.parametrize("release_arg", ["latest", None])
def test_auto_release_resolves_from_stac_and_formats_base_url(
    monkeypatch, fake_duckdb, release_arg
):
    _mock_stac_get(
        monkeypatch,
        _stac_response(
            {
                "latest": "2026-11-18.0",
                "links": [{"rel": "child", "href": "./2026-11-18.0/catalog.json"}],
            }
        ),
    )

    fetcher = OvertureAmenityFetcher(
        country="RWA",
        amenity_types=["school"],
        release=release_arg,
    )

    assert fetcher.release == "2026-11-18.0"
    assert (
        fetcher.base_url
        == "s3://overturemaps-us-west-2/release/2026-11-18.0/theme=places/*/*"
    )


@pytest.mark.parametrize("release_arg", ["latest", None])
def test_auto_release_raises_clear_runtime_error_when_stac_lookup_fails(
    monkeypatch, fake_duckdb, release_arg
):
    _mock_stac_get(monkeypatch, requests.ConnectionError("network down"))

    with pytest.raises(RuntimeError) as exc_info:
        OvertureAmenityFetcher(
            country="RWA",
            amenity_types=["school"],
            release=release_arg,
        )

    message = str(exc_info.value)
    assert exc_info.value.__cause__ is not None
    assert "release=" in message
    assert "2026-11-18.0" in message
    assert RELEASE_CALENDAR_URL in message


@contextmanager
def _capture_overture_logs():
    logger = logging.getLogger("OvertureAmenityFetcher")
    records = []

    class _ListHandler(logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = _ListHandler()
    logger.addHandler(handler)
    try:
        yield records
    finally:
        logger.removeHandler(handler)


@pytest.mark.parametrize("release_arg", ["2026-10-21.0", " 2026-10-21.0/ "])
def test_explicit_listed_release_is_validated_normalized_and_formatted(
    monkeypatch, fake_duckdb, release_arg
):
    _mock_stac_get(
        monkeypatch,
        _stac_response(
            {
                "latest": "2026-11-18.0",
                "links": [
                    {"rel": "child", "href": "./2026-11-18.0/catalog.json"},
                    {"rel": "child", "href": "./2026-10-21.0/catalog.json"},
                ],
            }
        ),
    )

    fetcher = OvertureAmenityFetcher(
        country="RWA",
        amenity_types=["school"],
        release=release_arg,
    )

    assert fetcher.release == "2026-10-21.0"
    assert (
        fetcher.base_url
        == "s3://overturemaps-us-west-2/release/2026-10-21.0/theme=places/*/*"
    )


def test_explicit_unlisted_release_raises_value_error_with_available_releases_and_docs(
    monkeypatch, fake_duckdb
):
    _mock_stac_get(
        monkeypatch,
        _stac_response(
            {
                "latest": "2026-11-18.0",
                "links": [
                    {"rel": "child", "href": "./2026-11-18.0/catalog.json"},
                    {"rel": "child", "href": "./2026-10-21.0/catalog.json"},
                ],
            }
        ),
    )

    with pytest.raises(ValueError) as exc_info:
        OvertureAmenityFetcher(
            country="RWA",
            amenity_types=["school"],
            release="2026-06-17.0",
        )

    message = str(exc_info.value)
    assert "2026-06-17.0" in message
    assert "2026-11-18.0" in message
    assert "2026-10-21.0" in message
    assert "60 days" in message
    assert RELEASE_CALENDAR_URL in message


@pytest.mark.parametrize(
    "outcome",
    [
        requests.ConnectionError("offline"),
        _stac_response({}),
    ],
)
def test_explicit_release_warns_and_proceeds_when_stac_catalog_is_unavailable(
    monkeypatch, fake_duckdb, outcome
):
    _mock_stac_get(monkeypatch, outcome)

    with _capture_overture_logs() as records:
        fetcher = OvertureAmenityFetcher(
            country="RWA",
            amenity_types=["school"],
            release="2026-10-21.0",
        )

    assert fetcher.release == "2026-10-21.0"
    assert (
        fetcher.base_url
        == "s3://overturemaps-us-west-2/release/2026-10-21.0/theme=places/*/*"
    )
    assert any(
        r.levelno == logging.WARNING and "2026-10-21.0" in r.getMessage()
        for r in records
    )


@pytest.mark.parametrize("bad_release", ["", "   ", "/"])
def test_empty_explicit_release_raises_value_error_without_stac_call(
    monkeypatch, fake_duckdb, bad_release
):
    calls = _mock_stac_get(monkeypatch, AssertionError("STAC should not be called"))

    with pytest.raises(ValueError):
        OvertureAmenityFetcher(
            country="RWA",
            amenity_types=["school"],
            release=bad_release,
        )

    assert calls == []


def test_custom_base_url_skips_stac_validation_and_normalizes_release(
    monkeypatch, fake_duckdb
):
    calls = _mock_stac_get(monkeypatch, AssertionError("STAC should not be called"))

    fetcher = OvertureAmenityFetcher(
        country="RWA",
        amenity_types=["school"],
        release="2025-01-22.0/",
        base_url="s3://my-archive/overture/{release}/theme=places/*/*",
    )

    assert calls == []
    assert fetcher.release == "2025-01-22.0"
    assert (
        fetcher.base_url == "s3://my-archive/overture/2025-01-22.0/theme=places/*/*"
    )


def test_invalid_country_raises_value_error_before_stac_call(monkeypatch, fake_duckdb):
    calls = _mock_stac_get(monkeypatch, AssertionError("STAC should not be called"))

    with pytest.raises(ValueError, match="Invalid country code"):
        OvertureAmenityFetcher(
            country="ZZZ",
            amenity_types=["school"],
            release="latest",
        )

    assert calls == []


# ----- Seam 3: lazy duckdb import isolation


def test_overture_fetcher_imports_and_queries_stac_without_duckdb_and_fails_cleanly_on_init():
    repo_root = str(Path(__file__).resolve().parents[2])
    code = "\n".join(
        [
            "import json, sys, requests",
            f"sys.path.insert(0, {repo_root!r})",
            "sys.modules['duckdb'] = None",
            "calls = []",
            "resp = requests.Response()",
            "resp.status_code = 200",
            (
                "resp._content = json.dumps({'latest': '2026-11-18.0', "
                "'links': [{'rel': 'child', 'href': './2026-10-21.0/catalog.json'}]}).encode()"
            ),
            "requests.get = lambda url, timeout=None, **kw: (calls.append(url), resp)[1]",
            "from gigaspatial import OvertureAmenityFetcher",
            "assert OvertureAmenityFetcher.get_latest_release() == '2026-11-18.0'",
            "assert OvertureAmenityFetcher.get_available_releases() == ['2026-11-18.0', '2026-10-21.0']",
            "calls.clear()",
            "try:",
            "    OvertureAmenityFetcher(country='RWA', amenity_types=['school'])",
            "except ImportError as exc:",
            "    assert 'giga-spatial[duckdb]' in str(exc)",
            "else:",
            "    raise AssertionError('Expected ImportError when duckdb is unavailable')",
            "assert calls == []",
        ]
    )
    subprocess.run([sys.executable, "-W", "ignore", "-c", code], check=True)
