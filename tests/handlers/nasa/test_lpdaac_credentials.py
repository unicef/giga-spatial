"""
Tests for LPDAACS3CredentialProvider and EarthdataSession.

Run with:
    pytest tests/handlers/nasa/test_lpdaac_credentials.py -v
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest
import requests

from gigaspatial.handlers.nasa.utils import (
    EarthdataSession,
    LPDAACS3CredentialProvider,
)

OAUTH_URL = (
    "https://urs.earthdata.nasa.gov/oauth/authorize"
    "?client_id=test&redirect_uri=https%3A%2F%2Fdata.lpdaac.earthdatacloud.nasa.gov%2Fredirect"
)


def make_creds(expires_in: timedelta = timedelta(hours=1)) -> dict:
    return {
        "accessKeyId": "AKIATEST",
        "secretAccessKey": "secret",
        "sessionToken": "token",
        "expiration": (datetime.now(timezone.utc) + expires_in).isoformat(),
    }


def test_fetch_credentials_follows_redirects_in_one_request():
    provider = LPDAACS3CredentialProvider(username="user", password="pass")
    creds = make_creds()
    creds_resp = MagicMock()
    creds_resp.json.return_value = creds

    with patch("gigaspatial.handlers.nasa.utils.EarthdataSession") as mock_session_cls:
        session = mock_session_cls.return_value
        session.get.return_value = creds_resp

        result = provider._fetch_credentials()

    assert result == creds
    session.get.assert_called_once_with(
        provider.endpoint, allow_redirects=True, timeout=30
    )
    mock_session_cls.assert_called_once_with("user", "pass")


def test_fetch_credentials_raises_on_missing_response_fields():
    provider = LPDAACS3CredentialProvider(username="user", password="pass")
    creds_resp = MagicMock()
    creds_resp.json.return_value = {"accessKeyId": "only-key"}

    with patch("gigaspatial.handlers.nasa.utils.EarthdataSession") as mock_session_cls:
        mock_session_cls.return_value.get.return_value = creds_resp

        with pytest.raises(ValueError, match="missing fields"):
            provider._fetch_credentials()


def test_fetch_credentials_raises_helpful_message_on_401():
    provider = LPDAACS3CredentialProvider(username="user", password="pass")

    response = MagicMock()
    response.status_code = 401
    response.url = OAUTH_URL
    http_error = requests.HTTPError(response=response)

    login_resp = MagicMock()
    login_resp.raise_for_status.side_effect = http_error

    with patch("gigaspatial.handlers.nasa.utils.EarthdataSession") as mock_session_cls:
        mock_session_cls.return_value.get.return_value = login_resp

        with pytest.raises(PermissionError, match="Earthdata Login authentication failed"):
            provider._fetch_credentials()


@pytest.mark.parametrize(
    "redirect_url, keeps_auth",
    [
        (OAUTH_URL, True),
        ("https://data.lpdaac.earthdatacloud.nasa.gov/redirect?code=x", True),
        ("https://example.com/elsewhere", False),
    ],
)
def test_session_keeps_auth_only_for_same_host_or_earthdata_login(
    redirect_url, keeps_auth
):
    session = EarthdataSession("user", "pass")
    response = MagicMock()
    response.request.url = LPDAACS3CredentialProvider.DEFAULT_ENDPOINT
    prepared = requests.Request("GET", redirect_url).prepare()
    prepared.headers["Authorization"] = "Basic abc"

    session.rebuild_auth(prepared, response)

    assert ("Authorization" in prepared.headers) is keeps_auth


def test_get_credentials_caches_until_expired():
    provider = LPDAACS3CredentialProvider(username="user", password="pass")
    creds = make_creds()

    with patch.object(provider, "_fetch_credentials", return_value=creds) as mock_fetch:
        first = provider.get_credentials()
        second = provider.get_credentials()

    assert first == second == creds
    mock_fetch.assert_called_once()


def test_get_credentials_refreshes_inside_buffer():
    provider = LPDAACS3CredentialProvider(username="user", password="pass")
    near_expiry = make_creds(expires_in=timedelta(seconds=60))

    with patch.object(
        provider, "_fetch_credentials", return_value=near_expiry
    ) as mock_fetch:
        provider.get_credentials()
        provider.get_credentials()

    assert mock_fetch.call_count == 2
