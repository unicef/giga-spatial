# gigaspatial/handlers/healthsites/api_client.py

from typing import Any, Optional, Tuple, Union

import httpx
import logging

import geopandas as gpd
import pandas as pd
import pycountry
from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from gigaspatial.config import config as global_config
from gigaspatial.handlers import OSMLocationFetcher
from gigaspatial.core.http import BaseRestApiClient, RestApiClientConfig, BasePaginationStrategy, AuthConfig, AuthType


class _HealthSitesPaginationStrategy(BasePaginationStrategy):
    """
    Page-number pagination for the Healthsites API.

    Handles both GeoJSON (records under 'features') and JSON
    (records as a direct list) response formats.

    Parameters
    ----------
    page_size : int
        Expected number of records per full page.
    output_format : str
        Either 'geojson' or 'json'.
    """

    def __init__(self, page_size: int = 100, output_format: str = "geojson") -> None:
        self.page_size = page_size
        self.output_format = output_format

    def extract_records(self, response: httpx.Response) -> list[dict]:
        parsed = response.json()
        if self.output_format == "geojson":
            return parsed.get("features", [])
        return parsed if isinstance(parsed, list) else []

    def next_request(
        self,
        response: httpx.Response,
        current_params: dict[str, Any],
    ) -> Optional[dict[str, Any]]:
        records = self.extract_records(response)
        if len(records) < self.page_size:
            return None  # partial page — last page reached
        return {**current_params, "page": current_params.get("page", 1) + 1}


class _HealthSitesApiClient(BaseRestApiClient):
    """
    Internal HTTP client for the Healthsites.io API.

    Uses API key query-param auth and format-aware page-number pagination.
    """

    def __init__(
        self,
        config: RestApiClientConfig,
        page_size: int,
        output_format: str,
    ) -> None:
        super().__init__(config)
        self._page_size = page_size
        self._output_format = output_format

    @property
    def pagination_strategy(self) -> _HealthSitesPaginationStrategy:
        return _HealthSitesPaginationStrategy(
            page_size=self._page_size,
            output_format=self._output_format,
        )

_API_BASE_URL = "https://healthsites.io/api/v3/facilities"


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class HealthSitesFetcher:
    """
    Fetch and process health facility data from the Healthsites.io API.

    Supports both GeoJSON and JSON output formats, bounding box filtering,
    date range filtering, and per-facility lookups.

    Parameters
    ----------
    country : str, optional
        Country name or ISO 3166 code. Converted to OSM English name internally.
    api_key : str
        Healthsites API key. Defaults to HEALTHSITES_API_KEY from global config.
    extent : tuple of float, optional
        Bounding box as (minLng, minLat, maxLng, maxLat).
    page_size : int
        Number of records per API page. Defaults to 100.
    flat_properties : bool
        Return properties in flat format. Defaults to True.
    tag_format : str
        Tag format, either 'osm' or 'hxl'. Defaults to 'osm'.
    output_format : str
        Response format, either 'geojson' or 'json'. Defaults to 'geojson'.
    max_retries : int
        Maximum retry attempts on transient errors. Defaults to 3.
    logger : logging.Logger, optional
        Logger instance. Defaults to global config logger.

    Examples
    --------
    >>> fetcher = HealthSitesFetcher(country="Kenya")
    >>> gdf = fetcher.fetch_facilities()
    >>> gdf = fetcher.fetch_facilities(max_pages=3, output_format="geojson")
    >>> df = fetcher.fetch_facilities(output_format="json")
    """

    country: Optional[str] = Field(default=None, description="Country to filter")
    api_key: str = Field(default=global_config.HEALTHSITES_API_KEY)
    extent: Optional[Tuple[float, float, float, float]] = Field(
        default=None, description="Bounding box as (minLng, minLat, maxLng, maxLat)"
    )
    page_size: int = Field(default=100, description="Number of records per API page")
    flat_properties: bool = Field(default=True, description="Flat property format")
    tag_format: str = Field(default="osm", description="Tag format: 'osm' or 'hxl'")
    output_format: str = Field(default="geojson", description="Response format: 'geojson' or 'json'")
    max_retries: int = Field(default=3, description="Max retry attempts on errors")
    logger: Optional[logging.Logger] = Field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.logger is None:
            self.logger = global_config.get_logger(self.__class__.__name__)
        if self.country:
            self.country = self._convert_country(self.country)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_client(self, output_format: str) -> _HealthSitesApiClient:
        config = RestApiClientConfig(
            base_url=_API_BASE_URL,
            auth=AuthConfig(
                auth_type=AuthType.API_KEY_QUERY,
                api_key=self.api_key,
                api_key_param="api-key",
            ),
            max_retries=self.max_retries,
            default_headers={"Accept": "application/json"},
        )
        return _HealthSitesApiClient(config, page_size=self.page_size, output_format=output_format)

    def _build_base_params(
        self,
        country: Optional[str],
        extent: Optional[Tuple[float, float, float, float]],
        output_format: str,
        flat_properties: bool,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
    ) -> dict[str, Any]:
        """Assemble query parameters shared across paginated requests."""
        params: dict[str, Any] = {
            "tag-format": self.tag_format,
            "output": output_format,
            "page": 1,
        }
        if flat_properties:
            params["flat-properties"] = "true"
        if country:
            params["country"] = country
        if extent:
            if len(extent) != 4:
                raise ValueError("Extent must be (minLng, minLat, maxLng, maxLat)")
            params["extent"] = ",".join(map(str, extent))
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date
        return params
    
    def _convert_country(self, country: str) -> str:
        """Resolve any country identifier to its OSM English name."""
        try:
            iso3 = pycountry.countries.lookup(country).alpha_3
        except LookupError:
            raise ValueError(f"Invalid country code: {country}")

        return self._fetch_osm_country_name(iso3, country)

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def _fetch_osm_country_name(self, iso3: str, original_input: str) -> str:
        """
        Fetch the OSM English country name with retry and escalating timeout.

        Retried up to 3 times with exponential backoff (2s → 4s → 8s wait).
        Timeout escalates by 2000ms on each attempt.
        """
        attempt = self._fetch_osm_country_name.retry.statistics.get("attempt_number", 1)
        timeout = 2000 + (attempt - 1) * 2000  # 2000 → 4000 → 6000ms

        self.logger.debug(
            "Fetching OSM name for %s (attempt %d, timeout %dms)", iso3, attempt, timeout
        )

        osm_data = OSMLocationFetcher.get_osm_countries(iso3_code=iso3, timeout=timeout)
        osm_name = osm_data.get("name:en")

        if not osm_name:
            raise ValueError(f"Could not find OSM English name for: {original_input}")

        self.logger.info("Resolved country to OSM name: %s", osm_name)
        return osm_name

    @staticmethod
    def _format_timestamp(value: Any) -> str:
        """Normalise a date/datetime/string to ISO 8601 string."""
        if isinstance(value, str):
            return value
        try:
            return value.strftime("%Y-%m-%dT%H:%M:%S")
        except AttributeError:
            raise ValueError(f"Cannot format timestamp from: {type(value)}")

    def _to_geodataframe(self, features: list[dict]) -> gpd.GeoDataFrame:
        gdf = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")
        self.logger.info("Created GeoDataFrame with %d records", len(gdf))
        return gdf

    def _process_json_flat(self, records: list[dict]) -> pd.DataFrame:
        """Flatten nested 'attributes' and 'centroid' keys from JSON responses."""
        processed = []
        for record in records:
            row = {k: v for k, v in record.items() if k not in ("attributes", "centroid")}
            row.update(record.get("attributes", {}))
            coords = record.get("centroid", {}).get("coordinates", [])
            row["longitude"] = coords[0] if len(coords) == 2 else None
            row["latitude"] = coords[1] if len(coords) == 2 else None
            processed.append(row)
        return pd.DataFrame(processed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_facilities(
        self,
        max_pages: Optional[int] = None,
        country: Optional[str] = None,
        extent: Optional[Tuple[float, float, float, float]] = None,
        output_format: Optional[str] = None,
        flat_properties: Optional[bool] = None,
        from_date: Optional[Any] = None,
        to_date: Optional[Any] = None,
    ) -> Union[gpd.GeoDataFrame, pd.DataFrame]:
        """
        Fetch health facility locations.

        Parameters
        ----------
        max_pages : int, optional
            Limit the number of pages fetched. Useful for testing.
        country : str, optional
            Override the instance-level country filter.
        extent : tuple, optional
            Override the instance-level bounding box.
        output_format : str, optional
            Override the instance-level output format ('geojson' or 'json').
        flat_properties : bool, optional
            Override the instance-level flat_properties setting.
        from_date : str, date, or datetime, optional
            Return facilities modified after this timestamp.
        to_date : str, date, or datetime, optional
            Return facilities modified before this timestamp.

        Returns
        -------
        gpd.GeoDataFrame
            When output_format is 'geojson' (default).
        pd.DataFrame
            When output_format is 'json'.
        """
        _country = self._convert_country(country) if country else self.country
        _extent = extent or self.extent
        _format = output_format or self.output_format
        _flat = flat_properties if flat_properties is not None else self.flat_properties
        _from = self._format_timestamp(from_date) if from_date else None
        _to = self._format_timestamp(to_date) if to_date else None

        base_params = self._build_base_params(_country, _extent, _format, _flat, _from, _to)

        self.logger.info(
            "Fetching health facilities — country: %s, format: %s",
            _country or "all", _format,
        )
        all_records: list[dict] = []

        with self._build_client(_format) as client:
            for page_num, page in enumerate(
                client.paginate("/", params=base_params), start=1
            ):
                all_records.extend(page)
                self.logger.info("Fetched page %d — %d records", page_num, len(page))

                if max_pages and page_num >= max_pages:
                    self.logger.info("Reached max_pages limit (%d)", max_pages)
                    break

        self.logger.info("Total records fetched: %d", len(all_records))

        if not all_records:
            self.logger.warning("No data fetched")
            return gpd.GeoDataFrame() if _format == "geojson" else pd.DataFrame()

        if _format == "geojson":
            return self._to_geodataframe(all_records)

        return self._process_json_flat(all_records) if not _flat else pd.DataFrame(all_records)

    def fetch_statistics(
        self,
        country: Optional[str] = None,
        extent: Optional[Tuple[float, float, float, float]] = None,
        from_date: Optional[Any] = None,
        to_date: Optional[Any] = None,
    ) -> dict:
        """
        Fetch aggregate statistics for health facilities.

        Parameters
        ----------
        country : str, optional
            Override the instance-level country filter.
        extent : tuple, optional
            Override the instance-level bounding box.
        from_date : str, date, or datetime, optional
            Filter by modification timestamp start.
        to_date : str, date, or datetime, optional
            Filter by modification timestamp end.

        Returns
        -------
        dict
            Statistics returned by the API.
        """
        _country = self._convert_country(country) if country else self.country
        _extent = extent or self.extent

        params: dict[str, Any] = {}
        if _country:
            params["country"] = _country
        if _extent:
            params["extent"] = ",".join(map(str, _extent))
        if from_date:
            params["from"] = self._format_timestamp(from_date)
        if to_date:
            params["to"] = self._format_timestamp(to_date)

        # Statistics is a single non-paginated endpoint — use client directly
        with self._build_client(self.output_format) as client:
            response = client.get("/statistic/", params=params)

        return response.json()

    def fetch_facility_by_id(self, osm_type: str, osm_id: str) -> dict:
        """
        Fetch a single facility by its OSM type and ID.

        Parameters
        ----------
        osm_type : str
            OSM element type: 'node', 'way', or 'relation'.
        osm_id : str
            OSM element ID.

        Returns
        -------
        dict
            Facility detail record.
        """
        with self._build_client(self.output_format) as client:
            response = client.get(f"/{osm_type}/{osm_id}/")

        return response.json()
