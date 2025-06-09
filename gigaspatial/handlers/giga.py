import requests
import pandas as pd
import time
from pydantic.dataclasses import dataclass, Field
from pydantic import ConfigDict
from shapely.geometry import Point
import pycountry
import logging

from gigaspatial.config import config as global_config


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class GigaSchoolLocationFetcher:
    """
    Fetch and process school location data from the Giga School Geolocation Data API.
    """

    country: str = Field(...)
    api_url: str = Field(
        default="https://uni-ooi-giga-maps-service.azurewebsites.net/api/v1/schools_location/country/{isocode3}",
        description="Base URL for the Giga School API",
    )
    api_key: str = global_config.GIGA_SCHOOL_LOCATION_API_KEY
    page_size: int = Field(default=1000, description="Number of records per API page")
    sleep_time: float = Field(
        default=0.2, description="Sleep time between API requests"
    )

    logger: logging.Logger = Field(default=None, repr=False)

    def __post_init__(self):
        try:
            self.country = pycountry.countries.lookup(self.country).alpha_3
        except LookupError:
            raise ValueError(f"Invalid country code provided: {self.country}")
        self.api_url = self.api_url.format(isocode3=self.country)
        if self.logger is None:
            self.logger = global_config.get_logger(self.__class__.__name__)

    def fetch_locations(self, **kwargs) -> pd.DataFrame:
        """
        Fetch and process school locations.

        Args:
            **kwargs: Additional parameters for customization
                - page_size: Override default page size
                - sleep_time: Override default sleep time between requests
                - max_pages: Limit the number of pages to fetch

        Returns:
            pd.DataFrame: School locations with geospatial info.
        """
        # Override defaults with kwargs if provided
        page_size = kwargs.get("page_size", self.page_size)
        sleep_time = kwargs.get("sleep_time", self.sleep_time)
        max_pages = kwargs.get("max_pages", None)

        # Prepare headers
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
        }

        all_data = []
        page = 1

        self.logger.info(
            f"Starting to fetch school locations for country: {self.country}"
        )

        while True:
            # Check if we've reached max_pages limit
            if max_pages and page > max_pages:
                self.logger.info(f"Reached maximum pages limit: {max_pages}")
                break

            params = {"page": page, "size": page_size}

            try:
                self.logger.debug(f"Fetching page {page} with params: {params}")
                response = requests.get(self.api_url, headers=headers, params=params)
                response.raise_for_status()

                parsed = response.json()
                data = parsed.get("data", [])

            except requests.exceptions.RequestException as e:
                self.logger.error(f"Request failed on page {page}: {e}")
                break
            except ValueError as e:
                self.logger.error(f"Failed to parse JSON response on page {page}: {e}")
                break

            # Check if we got any data
            if not data:
                self.logger.info(f"No data on page {page}. Stopping.")
                break

            all_data.extend(data)
            self.logger.info(f"Fetched page {page} with {len(data)} records")

            # If we got fewer records than page_size, we've reached the end
            if len(data) < page_size:
                self.logger.info("Reached end of data (partial page received)")
                break

            page += 1

            # Sleep to be respectful to the API
            if sleep_time > 0:
                time.sleep(sleep_time)

        self.logger.info(f"Finished fetching. Total records: {len(all_data)}")

        # Convert to DataFrame and process
        if not all_data:
            self.logger.warning("No data fetched, returning empty DataFrame")
            return pd.DataFrame()

        df = pd.DataFrame(all_data)

        df = self._process_geospatial_data(df)

        return df

    def _process_geospatial_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process and enhance the DataFrame with geospatial information.

        Args:
            df: Raw DataFrame from API

        Returns:
            pd.DataFrame: Enhanced DataFrame with geospatial data
        """
        if df.empty:
            return df

        df["geometry"] = df.apply(
            lambda row: Point(row["longitude"], row["latitude"]), axis=1
        )
        self.logger.info(f"Created geometry for all {len(df)} records")

        return df
