# gigaspatial/handlers/giga_school_profile.py

import logging
from collections import Counter
from typing import Optional

import pandas as pd
import pycountry
from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass

from gigaspatial.config import config as global_config
from gigaspatial.core.http import AuthConfig, AuthType, RestApiClientConfig, PageNumberPagination
from gigaspatial.handlers.giga.api_client import GigaApiClient

_API_URL = "https://uni-ooi-giga-maps-service.azurewebsites.net/api/v1/schools_profile"


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class GigaSchoolProfileFetcher:
    """
    Fetch and process school profile data from the Giga School Profile API.

    Includes connectivity information and other school details.

    Parameters
    ----------
    country : str
        Country name or ISO 3166 alpha-2/alpha-3 code.
    api_key : str
        Bearer token for the Giga API. Defaults to GIGA_SCHOOL_PROFILE_API_KEY.
    page_size : int
        Number of records per API page. Defaults to 1000.
    max_retries : int
        Maximum retry attempts on transient errors. Defaults to 3.
    giga_id_school : str, optional
        Filter results to a specific school by Giga ID.
    logger : logging.Logger, optional
        Logger instance. Defaults to the global config logger.

    Examples
    --------
    >>> fetcher = GigaSchoolProfileFetcher(country="Kenya")
    >>> df = fetcher.fetch_profiles()
    >>> df = fetcher.fetch_profiles(max_pages=5)
    >>> df = fetcher.fetch_profiles(giga_id_school="abc-123")
    """

    country: str = Field(...)
    api_key: str = Field(default=global_config.GIGA_SCHOOL_PROFILE_API_KEY)
    page_size: int = Field(default=1000, description="Number of records per API page")
    max_retries: int = Field(default=3, description="Max retry attempts on errors")
    giga_id_school: Optional[str] = Field(
        default=None, description="Optional specific Giga school ID to fetch"
    )
    logger: Optional[logging.Logger] = Field(default=None, repr=False)

    def __post_init__(self) -> None:
        try:
            self.country = pycountry.countries.lookup(self.country).alpha_3
        except LookupError:
            raise ValueError(f"Invalid country code provided: {self.country}")

        if self.logger is None:
            self.logger = global_config.get_logger(self.__class__.__name__)

    def _build_client(self) -> GigaApiClient:
        config = RestApiClientConfig(
            base_url=_API_URL,
            auth=AuthConfig(auth_type=AuthType.BEARER, api_key=self.api_key),
            max_retries=self.max_retries,
            default_headers={"Accept": "application/json"},
        )
        return GigaApiClient(config, page_size=self.page_size)

    def fetch_profiles(
        self,
        max_pages: Optional[int] = None,
        giga_id_school: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch school profiles including connectivity information.

        Parameters
        ----------
        max_pages : int, optional
            Limit the number of pages fetched. Useful for testing.
        giga_id_school : str, optional
            Override the instance-level school ID filter.

        Returns
        -------
        pd.DataFrame
            School profiles. Empty DataFrame if no data is returned.
        """
        school_id = giga_id_school or self.giga_id_school
        base_params: dict = {
            "country_iso3_code": self.country,
            "size": self.page_size,
            "page": 1,
        }
        if school_id:
            base_params["giga_id_school"] = school_id
            self.logger.info("Filtering for specific school ID: %s", school_id)

        self.logger.info("Fetching school profiles for country: %s", self.country)
        all_records: list[dict] = []

        with self._build_client() as client:
            for page_num, page in enumerate(
                client.paginate("/", params=base_params), start=1
            ):
                all_records.extend(page)
                self.logger.info("Fetched page %d — %d records", page_num, len(page))

                # Single school requests never need more than one page
                if school_id:
                    break

                if max_pages and page_num >= max_pages:
                    self.logger.info("Reached max_pages limit (%d)", max_pages)
                    break

        self.logger.info("Total records fetched: %d", len(all_records))

        if not all_records:
            self.logger.warning("No data fetched — returning empty DataFrame")
            return pd.DataFrame()

        return pd.DataFrame(all_records)

    def get_connectivity_summary(self, df: pd.DataFrame) -> dict:
        """
        Generate a summary of connectivity statistics from the fetched data.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame returned by fetch_profiles().

        Returns
        -------
        dict
            Summary statistics about school connectivity.
        """
        if df.empty:
            return {"error": "No data available"}

        summary = {
            "total_schools": len(df),
            "country": (
                df["country_iso3_code"].iloc[0]
                if "country_iso3_code" in df.columns
                else "Unknown"
            ),
        }

        if "admin1" in df.columns:
            summary["top_admin1_regions"] = df["admin1"].value_counts().head(10).to_dict()

        if "admin2" in df.columns:
            summary["top_admin2_regions"] = df["admin2"].value_counts().head(10).to_dict()

        if "connectivity" in df.columns:
            connected = df["connectivity"].sum()
            summary["schools_with_connectivity"] = int(connected)
            summary["connectivity_percentage"] = connected / len(df) * 100

        if "connectivity_RT" in df.columns:
            rt_connected = df["connectivity_RT"].sum()
            summary["schools_with_realtime_connectivity"] = int(rt_connected)
            summary["realtime_connectivity_percentage"] = rt_connected / len(df) * 100

        if "connectivity_type" in df.columns and not df["connectivity_type"].isna().all():
            summary["connectivity_types_breakdown"] = dict(
                Counter(df["connectivity_type"].dropna().tolist())
            )

        if "connectivity_RT_datasource" in df.columns:
            summary["realtime_connectivity_datasources"] = (
                df["connectivity_RT_datasource"].value_counts().to_dict()
            )

        if "school_data_source" in df.columns:
            summary["school_data_sources"] = df["school_data_source"].value_counts().to_dict()

        self.logger.info("Generated connectivity summary")
        return summary
