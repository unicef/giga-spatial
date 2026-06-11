import logging
import threading
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import requests


class EarthdataSession(requests.Session):
    """
    Custom requests.Session for NASA Earthdata authentication.

    Maintains Authorization headers through redirects to/from Earthdata hosts.
    """

    AUTH_HOST = "urs.earthdata.nasa.gov"

    def __init__(self, username: str, password: str):
        super().__init__()
        self.auth = (username, password)

    def rebuild_auth(self, prepared_request, response):
        """Keep auth header on redirects to Earthdata host, remove otherwise."""
        headers = prepared_request.headers
        url = prepared_request.url

        if "Authorization" in headers:
            original_parsed = requests.utils.urlparse(response.request.url)
            redirect_parsed = requests.utils.urlparse(url)

            # NASA Standard: Strip auth only if changing hosts AND the target is not URS
            if (original_parsed.hostname != redirect_parsed.hostname) and (
                redirect_parsed.hostname != self.AUTH_HOST
            ):
                del headers["Authorization"]


class LPDAACS3CredentialProvider:
    """
    Fetches and caches temporary AWS STS credentials from the LP DAAC
    /s3credentials endpoint using Earthdata Login.
    """

    DEFAULT_ENDPOINT = "https://data.lpdaac.earthdatacloud.nasa.gov/s3credentials"
    REFRESH_BUFFER_SECONDS = 300

    def __init__(
        self,
        username: str,
        password: str,
        endpoint: str = DEFAULT_ENDPOINT,
        region: str = "us-west-2",
        refresh_buffer_seconds: int = REFRESH_BUFFER_SECONDS,
    ):
        if not username or not password:
            raise ValueError(
                "Earthdata credentials are required for lp-prod-protected S3 access. "
                "Set EARTHDATA_USERNAME and EARTHDATA_PASSWORD."
            )

        self.username = username
        self.password = password
        self.endpoint = endpoint
        self.region = region
        self.refresh_buffer_seconds = refresh_buffer_seconds

        self._lock = threading.Lock()
        self._cached_creds: Optional[Dict[str, Any]] = None
        self._expiration: Optional[datetime] = None

    def get_credentials(self) -> Dict[str, Any]:
        """Return cached STS credentials, refreshing if expired or missing."""
        with self._lock:
            if self._cached_creds is None or self._is_expired():
                self._cached_creds = self._fetch_credentials()
                self._expiration = self._parse_expiration(
                    self._cached_creds.get("expiration")
                )
            return self._cached_creds

    def get_s3fs_kwargs(self) -> Dict[str, Any]:
        """Return keyword arguments for s3fs.S3FileSystem with temporary credentials."""
        creds = self.get_credentials()
        return {
            "key": creds["accessKeyId"],
            "secret": creds["secretAccessKey"],
            "token": creds["sessionToken"],
            "client_kwargs": {"region_name": self.region},
        }

    def _is_expired(self) -> bool:
        if self._expiration is None:
            return True
        now = datetime.now(timezone.utc)
        remaining = (self._expiration - now).total_seconds()
        return remaining <= self.refresh_buffer_seconds

    def _parse_expiration(self, expiration: Optional[str]) -> Optional[datetime]:
        if not expiration:
            return None
        try:
            return datetime.fromisoformat(expiration.replace("Z", "+00:00"))
        except ValueError:
            logging.warning("Could not parse S3 credential expiration: %s", expiration)
            return None

    def _validate_credentials(self, creds: Dict[str, Any]) -> None:
        required = ("accessKeyId", "secretAccessKey", "sessionToken")
        missing = [key for key in required if not creds.get(key)]
        if missing:
            raise ValueError(
                f"LP DAAC s3credentials response missing fields: {missing}"
            )

    def _fetch_credentials(self) -> Dict[str, Any]:
        """
        Authenticate with Earthdata Login and retrieve temporary S3 credentials.

        Leverages automated redirect handling built into the custom session.
        """
        session = EarthdataSession(self.username, self.password)

        try:
            # Let the custom session handle the entire redirection dance automatically
            resp = session.get(self.endpoint, allow_redirects=True, timeout=30)
            resp.raise_for_status()

            creds = resp.json()
            self._validate_credentials(creds)
            return creds

        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 401:
                raise PermissionError(
                    "Earthdata Login authentication failed (401). "
                    "Verify EARTHDATA_USERNAME and EARTHDATA_PASSWORD are correct. "
                    "If credentials are valid, ensure your account has approved LP DAAC "
                    "data access via https://search.earthdata.nasa.gov/. "
                    "Alternatively, set LPDAAC_S3_BUCKET=lp-prod-public for tiles "
                    "available without authentication."
                ) from e
            raise
