"""
Fetch PeeringDB facilities into memory.

``fetch_facilities`` returns a GeoDataFrame of facilities for a country, a
longitude/latitude box, or the whole default listing. The result stays in memory.
"""

from __future__ import annotations

import json
import time
from typing import Any, Optional

import geopandas as gpd
import pycountry
import requests
from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass
from shapely.geometry import Point

from gigaspatial.config import config as global_config

_URL = "https://www.peeringdb.com/api/fac"
_MAX_BYTES = 50_000_000


@dataclass(config=ConfigDict(arbitrary_types_allowed=True), kw_only=True)
class PeeringDBFetcher:
    """
    Fetch PeeringDB facilities into memory.

    One call is a single ``GET`` of ``https://www.peeringdb.com/api/fac``.
    The records stay in memory and are not written to a data store.

    Parameters
    ----------
    api_key : str, optional
        PeeringDB API key. Defaults to ``PEERINGDB_API_KEY``. An empty
        value sends no ``Authorization`` header.
    user_agent : str, optional
        ``User-Agent`` header. Defaults to ``giga-spatial/1.0``.
    max_retries : int, optional
        Retries after HTTP 429, HTTP 5xx, or a connection error. Defaults to 3.
    min_interval_seconds : float, optional
        Minimum gap between requests on this instance, including retries.
        Defaults to 2.
    timeout : float, optional
        Request timeout in seconds. Defaults to 60.

    Examples
    --------
    >>> fetcher = PeeringDBFetcher()
    >>> facilities = fetcher.fetch_facilities(country="RW")
    """

    api_key: str = Field(default_factory=lambda: global_config.PEERINGDB_API_KEY)
    user_agent: str = Field(default="giga-spatial/1.0", min_length=1)
    max_retries: int = Field(default=3, ge=0)
    min_interval_seconds: float = Field(default=2, ge=0)
    timeout: float = Field(default=60, gt=0)

    def __post_init__(self) -> None:
        self._next_request_at: Optional[float] = None

    def fetch_facilities(
        self,
        country: Optional[str] = None,
        extent: Optional[tuple[float, float, float, float]] = None,
    ) -> gpd.GeoDataFrame:
        """
        Fetch PeeringDB facilities.

        Parameters
        ----------
        country : str, optional
            Country name or ISO 3166 code. Sent as ISO alpha-2.
        extent : tuple of float, optional
            Bounding box as ``(min_lon, min_lat, max_lon, max_lat)``.

        Returns
        -------
        geopandas.GeoDataFrame
            Facilities in EPSG:4326. A facility with a null coordinate has a
            missing geometry. An empty response is an empty frame with that CRS.

        Raises
        ------
        ValueError
            If ``country`` cannot be resolved, or ``extent`` is not four
            numbers with each minimum below its maximum.
        requests.RequestException
            If the request fails, the body exceeds 50_000_000 bytes, or the
            JSON has no ``data`` array.
        """
        params = self._query_params(country, extent)
        body = self._get(params)
        return self._frame(self._records(body))

    def _query_params(
        self,
        country: Optional[str],
        extent: Optional[tuple[float, float, float, float]],
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"depth": 0}
        if country is not None:
            params["country"] = self._country_code(country)
        if extent is not None:
            params.update(self._extent_params(extent))
        return params

    def _country_code(self, country: str) -> str:
        try:
            return pycountry.countries.lookup(country).alpha_2
        except LookupError as exc:
            raise ValueError(f"Invalid country: {country}") from exc

    def _extent_params(
        self, extent: tuple[float, float, float, float]
    ) -> dict[str, float]:
        try:
            values = tuple(extent)
        except TypeError as exc:
            raise ValueError(
                "extent must be four numbers (min_lon, min_lat, max_lon, max_lat)"
            ) from exc
        if len(values) != 4 or any(not _is_number(value) for value in values):
            raise ValueError(
                "extent must be four numbers (min_lon, min_lat, max_lon, max_lat)"
            )
        min_lon, min_lat, max_lon, max_lat = (float(value) for value in values)
        if not (min_lon < max_lon and min_lat < max_lat):
            raise ValueError("extent minimum must be below its maximum")
        return {
            "longitude__gte": min_lon,
            "latitude__gte": min_lat,
            "longitude__lte": max_lon,
            "latitude__lte": max_lat,
        }

    def _headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/json",
            "User-Agent": self.user_agent,
        }
        if self.api_key:
            headers["Authorization"] = f"Api-Key {self.api_key}"
        return headers

    def _begin_request(self) -> None:
        if self._next_request_at is not None:
            remaining = self._next_request_at - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
        self._next_request_at = time.monotonic() + self.min_interval_seconds

    def _get(self, params: dict[str, Any]) -> bytes:
        headers = self._headers()
        attempts = self.max_retries + 1
        last_error: Optional[Exception] = None
        for attempt in range(attempts):
            self._begin_request()
            try:
                response = requests.get(
                    _URL,
                    params=params,
                    headers=headers,
                    timeout=self.timeout,
                    stream=True,
                )
            except (requests.ConnectionError, requests.Timeout) as exc:
                last_error = exc
                if attempt == attempts - 1:
                    raise
                continue
            try:
                if response.status_code == 429 or response.status_code >= 500:
                    last_error = requests.HTTPError(
                        f"PeeringDB request failed with HTTP {response.status_code}"
                    )
                    if attempt == attempts - 1:
                        raise requests.RequestException(str(last_error)) from last_error
                    continue
                if response.status_code >= 400:
                    raise requests.RequestException(
                        f"PeeringDB request failed with HTTP {response.status_code}"
                    )
                return self._read_body(response)
            finally:
                response.close()
        raise requests.RequestException("PeeringDB request failed") from last_error

    def _read_body(self, response: requests.Response) -> bytes:
        declared = response.headers.get("Content-Length")
        if declared is not None:
            try:
                size = int(declared)
            except ValueError:
                size = None
            if size is not None and size > _MAX_BYTES:
                raise requests.RequestException(
                    f"PeeringDB response is {size} bytes, over the {_MAX_BYTES} byte limit"
                )
        chunks: list[bytes] = []
        total = 0
        for chunk in response.iter_content(chunk_size=65_536):
            if not chunk:
                continue
            total += len(chunk)
            if total > _MAX_BYTES:
                raise requests.RequestException(
                    f"PeeringDB response exceeded {_MAX_BYTES} bytes"
                )
            chunks.append(chunk)
        return b"".join(chunks)

    def _records(self, body: bytes) -> list[dict[str, Any]]:
        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise requests.RequestException("PeeringDB response is not JSON") from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
            raise requests.RequestException("PeeringDB response has no data array")
        records = payload["data"]
        if any(not isinstance(record, dict) for record in records):
            raise requests.RequestException("PeeringDB data entry is not an object")
        return records

    def _frame(self, records: list[dict[str, Any]]) -> gpd.GeoDataFrame:
        geometry = [_point(record) for record in records]
        if not records:
            return gpd.GeoDataFrame(geometry=geometry, crs="EPSG:4326")
        return gpd.GeoDataFrame(records, geometry=geometry, crs="EPSG:4326")


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _point(record: dict[str, Any]) -> Optional[Point]:
    latitude = record.get("latitude")
    longitude = record.get("longitude")
    if latitude is None or longitude is None:
        return None
    if not _is_number(latitude) or not _is_number(longitude):
        return None
    return Point(float(longitude), float(latitude))
