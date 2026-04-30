import logging
from typing import Optional

import pandas as pd
from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass

from gigaspatial.config import config as global_config
from gigaspatial.core.http import (
    AuthConfig,
    AuthType,
    RestApiClientConfig,
)
from gigaspatial.handlers.giga.api_client import GigaApiClient

_API_URL = "https://uni-ooi-giga-maps-service.azurewebsites.net/api/v1/countries"


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class GigaCountriesFetcher:
    """
    Fetch and process country data from the Giga Country API.

    Parameters
    ----------
    api_key : str
        Bearer token for the Giga API. Defaults to GIGA_COUNTRIES_API_KEY
        from global config.
    page_size : int
        Number of records per API page. Defaults to 100.
    max_retries : int
        Maximum number of retry attempts on transient errors. Defaults to 3.
    logger : logging.Logger, optional
        Logger instance. Defaults to the global config logger.

    Examples
    --------
    >>> fetcher = GigaCountryFetcher()
    >>> df = fetcher.fetch_countries()
    >>> df_mng = fetcher.fetch_countries(country_iso3_code="MNG")
    """

    api_key: str = Field(default=global_config.GIGA_COUNTRIES_API_KEY)
    max_retries: int = Field(default=3, description="Max retry attempts on errors")
    logger: Optional[logging.Logger] = Field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.logger is None:
            self.logger = global_config.get_logger(self.__class__.__name__)

    def _build_client(self) -> GigaApiClient:
        """Construct the API client for the countries endpoint."""
        config = RestApiClientConfig(
            base_url=_API_URL,
            auth=AuthConfig(
                auth_type=AuthType.BEARER,
                api_key=self.api_key,
            ),
            max_retries=self.max_retries,
            default_headers={"Accept": "application/json"},
        )
        return GigaApiClient(config, page_size=1000)

    def fetch_countries(
        self,
        id: Optional[int] = None,
        country_iso3_code: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch countries from the Giga API.

        Parameters
        ----------
        id : int, optional
            Filter by country ID.
        country_iso3_code : str, optional
            Filter by country ISO3 code (e.g., "BRA", "MNG").

        Returns
        -------
        pd.DataFrame
            Country records. Empty DataFrame if no data is returned.
        """
        # Fixed parameters to fetch all records in a single 'page'
        params = {"size": 1000, "page": 1}
        if id is not None:
            params["id"] = id
        if country_iso3_code is not None:
            params["country_iso3_code"] = country_iso3_code

        self.logger.info("Fetching countries from Giga API")
        
        with self._build_client() as client:
            # We use the standard paginate logic to handle the 'data' key wrapper,
            # but we only fetch the first page since countries list is small (~250).
            try:
                pages = client.paginate("/", params=params)
                all_records = next(pages, [])
            except StopIteration:
                all_records = []

        if not all_records:
            self.logger.warning("No records found in the Giga API response")
            return pd.DataFrame()

        self.logger.info("Total records fetched: %d", len(all_records))
        return pd.DataFrame(all_records)
