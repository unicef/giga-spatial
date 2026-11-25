"""Tests for token redaction in GeoRepoClient."""

from types import SimpleNamespace

import pytest
import requests

from gigaspatial.handlers import georepo as georepo_mod
from gigaspatial.handlers.georepo import GeoRepoClient


def test_georepo_repr_masks_api_key():
    secret = "georepo-secret-token-555"
    client = GeoRepoClient(api_key=secret, email="user@example.org")
    rendered = repr(client)
    assert secret not in rendered
    assert "api_key='***'" in rendered


def test_georepo_failed_request_redacts_token_in_exception(monkeypatch):
    secret = "georepo-secret-token-555"
    client = GeoRepoClient(api_key=secret, email="user@example.org")

    def fake_request(method, endpoint, headers=None, params=None, json=None):
        resp = requests.Response()
        resp.status_code = 401
        resp.url = f"{endpoint}?token={secret}&page=1"
        resp.reason = "Unauthorized"
        resp.raise_for_status()

    monkeypatch.setattr(georepo_mod.requests, "request", fake_request)

    with pytest.raises(requests.exceptions.HTTPError) as exc_info:
        client.list_modules()

    msg = str(exc_info.value)
    assert secret not in msg
    assert "token=***&page=1" in msg
    if exc_info.value.__context__ is not None:
        assert secret not in str(exc_info.value.__context__)


def test_georepo_successful_request_uses_real_token_and_vector_tiles_url(monkeypatch):
    secret = "georepo-secret-token-555"
    seen = []

    def fake_request(method, endpoint, headers=None, params=None, json=None):
        seen.append({"method": method, "endpoint": endpoint, "headers": headers})
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"results": [{"uuid": "m1", "name": "Admin Boundaries"}]},
        )

    monkeypatch.setattr(georepo_mod.requests, "request", fake_request)

    client = GeoRepoClient(api_key=secret, email="user@example.org")
    result = client.list_modules()

    assert result["results"][0]["uuid"] == "m1"
    assert seen[0]["headers"]["Authorization"] == f"Token {secret}"

    tiles_url = client.get_vector_tiles_url(
        {"vector_tiles": "https://georepo.unicef.org/tiles/{z}/{x}/{y}.pbf?t=123"}
    )
    assert tiles_url == (
        f"https://georepo.unicef.org/tiles/{{z}}/{{x}}/{{y}}.pbf"
        f"?t=123&token={secret}&georepo_user_key=user@example.org"
    )
