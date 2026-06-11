"""
Tests for LPDAACS3CredentialProvider.

Run with:
    pytest tests/handlers/srtm/test_lpdaac_credentials.py -v
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests

from gigaspatial.handlers.nasa.utils import LPDAACS3CredentialProvider

SAMPLE_CREDS = {
    "accessKeyId": "AKIATEST",
    "secretAccessKey": "secret",
    "sessionToken": "token",
    "expiration": "2026-06-10 14:00:00+00:00",
}

OAUTH_URL = (
    "https://urs.earthdata.nasa.gov/oauth/authorize"
    "?client_id=test&redirect_uri=https%3A%2F%2Fdata.lpdaac.earthdatacloud.nasa.gov%2Fredirect"
)


def test_fetch_credentials_uses_basic_auth_get_flow():
    provider = LPDAACS3CredentialProvider(username="user", password="pass")

    login_resp = MagicMock()
    login_resp.headers = {"Location": OAUTH_URL}

    creds_resp = MagicMock()
    creds_resp.json.return_value = SAMPLE_CREDS

    with patch("gigaspatial.handlers.srtm.utils.EarthdataSession") as mock_session_cls:
        session = mock_session_cls.return_value
        session.get.side_effect = [login_resp, creds_resp]

        creds = provider._fetch_credentials()

    assert creds == SAMPLE_CREDS
    assert session.get.call_count == 2
    session.get.assert_any_call(
        provider.endpoint, allow_redirects=False, timeout=30
    )
    session.get.assert_any_call(OAUTH_URL, allow_redirects=True, timeout=30)
    mock_session_cls.assert_called_once_with("user", "pass")


def test_fetch_credentials_raises_on_missing_redirect_location():
    provider = LPDAACS3CredentialProvider(username="user", password="pass")

    login_resp = MagicMock()
    login_resp.headers = {}

    with patch("gigaspatial.handlers.srtm.utils.EarthdataSession") as mock_session_cls:
        session = mock_session_cls.return_value
        session.get.return_value = login_resp

        with pytest.raises(ValueError, match="No redirect Location"):
            provider._fetch_credentials()


def test_fetch_credentials_raises_on_missing_response_fields():
    provider = LPDAACS3CredentialProvider(username="user", password="pass")

    login_resp = MagicMock()
    login_resp.headers = {"Location": OAUTH_URL}

    creds_resp = MagicMock()
    creds_resp.json.return_value = {"accessKeyId": "only-key"}

    with patch("gigaspatial.handlers.srtm.utils.EarthdataSession") as mock_session_cls:
        session = mock_session_cls.return_value
        session.get.side_effect = [login_resp, creds_resp]

        with pytest.raises(ValueError, match="missing fields"):
            provider._fetch_credentials()


def test_fetch_credentials_raises_helpful_message_on_401():
    provider = LPDAACS3CredentialProvider(username="user", password="pass")

    response = MagicMock()
    response.status_code = 401
    response.url = OAUTH_URL
    http_error = requests.HTTPError(response=response)

    login_resp = MagicMock()
    login_resp.headers = {"Location": OAUTH_URL}
    login_resp.raise_for_status.side_effect = http_error

    with patch("gigaspatial.handlers.srtm.utils.EarthdataSession") as mock_session_cls:
        session = mock_session_cls.return_value
        session.get.return_value = login_resp

        with pytest.raises(PermissionError, match="Earthdata Login authentication failed"):
            provider._fetch_credentials()


def test_get_credentials_caches_until_expired():
    provider = LPDAACS3CredentialProvider(username="user", password="pass")

    with patch.object(provider, "_fetch_credentials", return_value=SAMPLE_CREDS) as mock_fetch:
        first = provider.get_credentials()
        second = provider.get_credentials()

    assert first == second == SAMPLE_CREDS
    mock_fetch.assert_called_once()
