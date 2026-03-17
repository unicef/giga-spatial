import logging
from typing import Optional, Union

import geopandas as gpd
import pandas as pd
import pycountry
from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass
from shapely.geometry import Point

from gigaspatial.config import config as global_config
from gigaspatial.core.http import (
    AuthConfig,
    AuthType,
    RestApiClientConfig,
)
from gigaspatial.handlers.giga.api_client import GigaApiClient

_API_URL_TEMPLATE = (
    "https://uni-ooi-giga-maps-service.azurewebsites.net/api/v1"
    "/schools_location/country/{isocode3}"
)


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class GigaSchoolLocationFetcher:
    """
    Fetch and process school location data from the Giga School Geolocation API.

    Wraps the Giga REST API with authentication, retry logic, and
    pagination handled transparently via gigaspatial's HTTP module.

    Parameters
    ----------
    country : str
        Country name or ISO 3166 alpha-2/alpha-3 code.
    api_key : str
        Bearer token for the Giga API. Defaults to GIGA_SCHOOL_LOCATION_API_KEY
        from global config.
    page_size : int
        Number of records per API page. Defaults to 1000.
    max_retries : int
        Maximum number of retry attempts on transient errors. Defaults to 3.
    logger : logging.Logger, optional
        Logger instance. Defaults to the global config logger.

    Examples
    --------
    >>> fetcher = GigaSchoolLocationFetcher(country="Kenya")
    >>> df = fetcher.fetch_locations()
    >>> gdf = fetcher.fetch_locations(process_geospatial=True)
    >>> gdf = fetcher.fetch_locations(process_geospatial=True, max_pages=5)
    """

    country: str = Field(...)
    api_key: str = Field(default=global_config.GIGA_SCHOOL_LOCATION_API_KEY)
    page_size: int = Field(default=1000, description="Number of records per API page")
    max_retries: int = Field(default=3, description="Max retry attempts on errors")
    logger: Optional[logging.Logger] = Field(default=None, repr=False)

    def __post_init__(self) -> None:
        try:
            self.country = pycountry.countries.lookup(self.country).alpha_3
        except LookupError:
            raise ValueError(f"Invalid country code provided: {self.country}")

        if self.logger is None:
            self.logger = global_config.get_logger(self.__class__.__name__)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_client(self) -> GigaApiClient:
        """Construct the API client for this country's endpoint."""
        base_url = _API_URL_TEMPLATE.format(isocode3=self.country)
        config = RestApiClientConfig(
            base_url=base_url,
            auth=AuthConfig(
                auth_type=AuthType.BEARER,
                api_key=self.api_key,
            ),
            max_retries=self.max_retries,
            default_headers={"Accept": "application/json"},
        )
        return GigaApiClient(config, page_size=self.page_size)

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------

    def fetch_locations(
        self,
        process_geospatial: bool = False,
        max_pages: Optional[int] = None,
    ) -> Union[pd.DataFrame, gpd.GeoDataFrame]:
        """
        Fetch school locations from the Giga API.

        Parameters
        ----------
        process_geospatial : bool
            If True, returns a GeoDataFrame with a Point geometry column.
            Defaults to False.
        max_pages : int, optional
            Limit the number of pages fetched. Useful for testing.

        Returns
        -------
        pd.DataFrame or gpd.GeoDataFrame
            School location records. Empty DataFrame if no data is returned.
        """
        self.logger.info("Fetching school locations for country: %s", self.country)
        all_records: list[dict] = []

        with self._build_client() as client:
            for page_num, page in enumerate(
                client.paginate("/", params={"size": self.page_size, "page": 1}),
                start=1,
            ):
                all_records.extend(page)
                self.logger.info("Fetched page %d — %d records", page_num, len(page))

                if max_pages and page_num >= max_pages:
                    self.logger.info("Reached max_pages limit (%d)", max_pages)
                    break

        self.logger.info("Total records fetched: %d", len(all_records))

        if not all_records:
            self.logger.warning("No data fetched — returning empty DataFrame")
            return pd.DataFrame()

        df = pd.DataFrame(all_records)
        return self._process_geospatial_data(df) if process_geospatial else df

    def _process_geospatial_data(self, df: pd.DataFrame) -> gpd.GeoDataFrame:
        """
        Attach Point geometry to each school record.

        Parameters
        ----------
        df : pd.DataFrame
            Raw records from the API.

        Returns
        -------
        gpd.GeoDataFrame
            GeoDataFrame with geometry column in EPSG:4326.
        """
        if df.empty:
            return df

        df["geometry"] = [
            Point(row.longitude, row.latitude) for row in df.itertuples()
        ]
        self.logger.info("Created geometry for %d records", len(df))
        return gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
