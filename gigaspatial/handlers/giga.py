import requests
import pandas as pd
import time
from datetime import datetime, date
from pydantic.dataclasses import dataclass, Field
from pydantic import ConfigDict
from shapely.geometry import Point
import pycountry
from typing import Optional, Union
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


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class GigaSchoolProfileFetcher:
    """
    Fetch and process school profile data from the Giga School Profile API.
    This includes connectivity information and other school details.
    """

    country: str = Field(...)
    api_url: str = Field(
        default="https://uni-ooi-giga-maps-service.azurewebsites.net/api/v1/schools_profile/",
        description="Base URL for the Giga School Profile API",
    )
    api_key: str = global_config.GIGA_SCHOOL_PROFILE_API_KEY
    page_size: int = Field(default=1000, description="Number of records per API page")
    sleep_time: float = Field(
        default=0.2, description="Sleep time between API requests"
    )
    giga_id_school: Optional[str] = Field(
        default=None, description="Optional specific giga school ID to fetch"
    )

    logger: logging.Logger = Field(default=None, repr=False)

    def __post_init__(self):
        try:
            self.country = pycountry.countries.lookup(self.country).alpha_3
        except LookupError:
            raise ValueError(f"Invalid country code provided: {self.country}")

        if self.logger is None:
            self.logger = global_config.get_logger(self.__class__.__name__)

    def fetch_profiles(self, **kwargs) -> pd.DataFrame:
        """
        Fetch and process school profiles including connectivity information.

        Args:
            **kwargs: Additional parameters for customization
                - page_size: Override default page size
                - sleep_time: Override default sleep time between requests
                - max_pages: Limit the number of pages to fetch
                - giga_id_school: Override default giga_id_school filter

        Returns:
            pd.DataFrame: School profiles with connectivity and geospatial info.
        """
        # Override defaults with kwargs if provided
        page_size = kwargs.get("page_size", self.page_size)
        sleep_time = kwargs.get("sleep_time", self.sleep_time)
        max_pages = kwargs.get("max_pages", None)
        giga_id_school = kwargs.get("giga_id_school", self.giga_id_school)

        # Prepare headers
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
        }

        all_data = []
        page = 1

        self.logger.info(
            f"Starting to fetch school profiles for country: {self.country}"
        )

        if giga_id_school:
            self.logger.info(f"Filtering for specific school ID: {giga_id_school}")

        while True:
            # Check if we've reached max_pages limit
            if max_pages and page > max_pages:
                self.logger.info(f"Reached maximum pages limit: {max_pages}")
                break

            # Build parameters
            params = {
                "country_iso3_code": self.country,
                "page": page,
                "size": page_size,
            }

            # Add giga_id_school filter if specified
            if giga_id_school:
                params["giga_id_school"] = giga_id_school

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

            # If filtering by specific school ID, we likely only need one page
            if giga_id_school:
                self.logger.info(
                    "Specific school ID requested, stopping after first page"
                )
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

        return df

    def get_connectivity_summary(self, df: pd.DataFrame) -> dict:
        """
        Generate a summary of connectivity statistics from the fetched data.

        Args:
            df: DataFrame with school profile data

        Returns:
            dict: Summary statistics about connectivity
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

        # Administrative region analysis
        if "admin1" in df.columns:
            admin1_counts = df["admin1"].value_counts().head(10).to_dict()
            summary["top_admin1_regions"] = admin1_counts

        if "admin2" in df.columns:
            admin2_counts = df["admin2"].value_counts().head(10).to_dict()
            summary["top_admin2_regions"] = admin2_counts

        # Connectivity analysis
        if "connectivity" in df.columns:
            connected_count = df["connectivity"].sum()
            summary["schools_with_connectivity"] = int(connected_count)
            summary["connectivity_percentage"] = connected_count / len(df) * 100

        if "connectivity_RT" in df.columns:
            rt_connected_count = df["connectivity_RT"].sum()
            summary["schools_with_realtime_connectivity"] = int(rt_connected_count)
            summary["realtime_connectivity_percentage"] = (
                rt_connected_count / len(df) * 100
            )

        # Connectivity type analysis
        if "connectivity_type" in df.columns:

            if not all(df.connectivity_type.isna()):
                from collections import Counter

                type_counts = dict(Counter(df.connectivity_type.dropna().to_list()))
                summary["connectivity_types_breakdown"] = type_counts

        # Data source analysis
        if "connectivity_RT_datasource" in df.columns:
            datasource_counts = (
                df["connectivity_RT_datasource"].value_counts().to_dict()
            )
            summary["realtime_connectivity_datasources"] = datasource_counts

        if "school_data_source" in df.columns:
            school_datasource_counts = df["school_data_source"].value_counts().to_dict()
            summary["school_data_sources"] = school_datasource_counts

        self.logger.info("Generated connectivity summary")
        return summary


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class GigaSchoolMeasurementsFetcher:
    """
    Fetch and process school daily realtime connectivity measurements from the Giga API.
    This includes download/upload speeds, latency, and connectivity performance data.
    """

    country: str = Field(...)
    start_date: Union[str, date, datetime] = Field(...)
    end_date: Union[str, date, datetime] = Field(...)
    api_url: str = Field(
        default="https://uni-ooi-giga-maps-service.azurewebsites.net/api/v1/all_measurements",
        description="Base URL for the Giga School Measurements API",
    )
    api_key: str = global_config.GIGA_SCHOOL_MEASUREMENTS_API_KEY
    page_size: int = Field(default=1000, description="Number of records per API page")
    sleep_time: float = Field(
        default=0.2, description="Sleep time between API requests"
    )
    giga_id_school: Optional[str] = Field(
        default=None, description="Optional specific giga school ID to fetch"
    )

    logger: logging.Logger = Field(default=None, repr=False)

    def __post_init__(self):
        try:
            self.country = pycountry.countries.lookup(self.country).alpha_3
        except LookupError:
            raise ValueError(f"Invalid country code provided: {self.country}")

        # Convert dates to string format if needed
        self.start_date = self._format_date(self.start_date)
        self.end_date = self._format_date(self.end_date)

        # Validate date range
        if self.start_date > self.end_date:
            raise ValueError("start_date must be before or equal to end_date")

        if self.logger is None:
            self.logger = global_config.get_logger(self.__class__.__name__)

    def _format_date(self, date_input: Union[str, date, datetime]) -> str:
        """
        Convert date input to string format expected by API (YYYY-MM-DD).

        Args:
            date_input: Date in various formats

        Returns:
            str: Date in YYYY-MM-DD format
        """
        if isinstance(date_input, str):
            # Assume it's already in correct format or parse it
            try:
                parsed_date = datetime.strptime(date_input, "%Y-%m-%d")
                return date_input
            except ValueError:
                try:
                    parsed_date = pd.to_datetime(date_input)
                    return parsed_date.strftime("%Y-%m-%d")
                except:
                    raise ValueError(
                        f"Invalid date format: {date_input}. Expected YYYY-MM-DD"
                    )
        elif isinstance(date_input, (date, datetime)):
            return date_input.strftime("%Y-%m-%d")
        else:
            raise ValueError(f"Invalid date type: {type(date_input)}")

    def fetch_measurements(self, **kwargs) -> pd.DataFrame:
        """
        Fetch and process school connectivity measurements.

        Args:
            **kwargs: Additional parameters for customization
                - page_size: Override default page size
                - sleep_time: Override default sleep time between requests
                - max_pages: Limit the number of pages to fetch
                - giga_id_school: Override default giga_id_school filter
                - start_date: Override default start_date
                - end_date: Override default end_date

        Returns:
            pd.DataFrame: School measurements with connectivity performance data.
        """
        # Override defaults with kwargs if provided
        page_size = kwargs.get("page_size", self.page_size)
        sleep_time = kwargs.get("sleep_time", self.sleep_time)
        max_pages = kwargs.get("max_pages", None)
        giga_id_school = kwargs.get("giga_id_school", self.giga_id_school)
        start_date = kwargs.get("start_date", self.start_date)
        end_date = kwargs.get("end_date", self.end_date)

        # Format dates if overridden
        if start_date != self.start_date:
            start_date = self._format_date(start_date)
        if end_date != self.end_date:
            end_date = self._format_date(end_date)

        # Prepare headers
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
        }

        all_data = []
        page = 1

        self.logger.info(
            f"Starting to fetch measurements for country: {self.country} "
            f"from {start_date} to {end_date}"
        )

        if giga_id_school:
            self.logger.info(f"Filtering for specific school ID: {giga_id_school}")

        while True:
            # Check if we've reached max_pages limit
            if max_pages and page > max_pages:
                self.logger.info(f"Reached maximum pages limit: {max_pages}")
                break

            # Build parameters
            params = {
                "country_iso3_code": self.country,
                "start_date": start_date,
                "end_date": end_date,
                "page": page,
                "size": page_size,
            }

            # Add giga_id_school filter if specified
            if giga_id_school:
                params["giga_id_school"] = giga_id_school

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

            # If filtering by specific school ID, we might only need one page
            if giga_id_school and len(all_data) > 0:
                self.logger.info(
                    "Specific school ID requested, checking if more data needed"
                )

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
        df = self._process_measurements_data(df)

        return df

    def _process_measurements_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process and enhance the DataFrame with measurement performance metrics.

        Args:
            df: Raw DataFrame from API

        Returns:
            pd.DataFrame: Enhanced DataFrame with processed measurement data
        """
        if df.empty:
            return df

        # Convert date column to datetime
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df["date_only"] = df["date"].dt.date
            df["year"] = df["date"].dt.year
            df["month"] = df["date"].dt.month
            df["day_of_week"] = df["date"].dt.day_name()
            self.logger.info("Processed date fields")

        # Process speed measurements
        numeric_columns = ["download_speed", "upload_speed", "latency"]
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Create performance categories
        if "download_speed" in df.columns:
            df["download_speed_category"] = pd.cut(
                df["download_speed"],
                bins=[0, 5, 25, 100, float("inf")],
                labels=[
                    "Very Slow (<5 Mbps)",
                    "Slow (5-25 Mbps)",
                    "Moderate (25-100 Mbps)",
                    "Fast (>100 Mbps)",
                ],
                include_lowest=True,
            )

        if "upload_speed" in df.columns:
            df["upload_speed_category"] = pd.cut(
                df["upload_speed"],
                bins=[0, 1, 10, 50, float("inf")],
                labels=[
                    "Very Slow (<1 Mbps)",
                    "Slow (1-10 Mbps)",
                    "Moderate (10-50 Mbps)",
                    "Fast (>50 Mbps)",
                ],
                include_lowest=True,
            )

        if "latency" in df.columns:
            df["latency_category"] = pd.cut(
                df["latency"],
                bins=[0, 50, 150, 300, float("inf")],
                labels=[
                    "Excellent (<50ms)",
                    "Good (50-150ms)",
                    "Fair (150-300ms)",
                    "Poor (>300ms)",
                ],
                include_lowest=True,
            )

        # Create quality flags
        if "download_speed" in df.columns and "upload_speed" in df.columns:
            df["has_broadband"] = (df["download_speed"] >= 25) & (
                df["upload_speed"] >= 3
            )
            df["has_basic_connectivity"] = (df["download_speed"] >= 1) & (
                df["upload_speed"] >= 0.5
            )

        # Flag measurements with missing data
        df["has_complete_measurement"] = (
            df["download_speed"].notna()
            & df["upload_speed"].notna()
            & df["latency"].notna()
        )

        self.logger.info(f"Processed measurement data for {len(df)} records")

        return df

    def get_performance_summary(self, df: pd.DataFrame) -> dict:
        """
        Generate a comprehensive summary of connectivity performance metrics.

        Args:
            df: DataFrame with measurement data

        Returns:
            dict: Summary statistics about connectivity performance
        """
        if df.empty:
            return {"error": "No data available"}

        summary = {
            "total_measurements": len(df),
            "country": (
                df["country_iso3_code"].iloc[0]
                if "country_iso3_code" in df.columns
                else "Unknown"
            ),
            "date_range": {
                "start": (
                    df["date"].min().strftime("%Y-%m-%d")
                    if "date" in df.columns
                    else None
                ),
                "end": (
                    df["date"].max().strftime("%Y-%m-%d")
                    if "date" in df.columns
                    else None
                ),
            },
        }

        # School coverage
        if "giga_id_school" in df.columns:
            unique_schools = df["giga_id_school"].nunique()
            summary["unique_schools_measured"] = unique_schools
            summary["avg_measurements_per_school"] = (
                len(df) / unique_schools if unique_schools > 0 else 0
            )

        # Speed statistics
        for speed_col in ["download_speed", "upload_speed"]:
            if speed_col in df.columns:
                speed_data = df[speed_col].dropna()
                if len(speed_data) > 0:
                    summary[f"{speed_col}_stats"] = {
                        "mean": float(speed_data.mean()),
                        "median": float(speed_data.median()),
                        "min": float(speed_data.min()),
                        "max": float(speed_data.max()),
                        "std": float(speed_data.std()),
                    }

        # Latency statistics
        if "latency" in df.columns:
            latency_data = df["latency"].dropna()
            if len(latency_data) > 0:
                summary["latency_stats"] = {
                    "mean": float(latency_data.mean()),
                    "median": float(latency_data.median()),
                    "min": float(latency_data.min()),
                    "max": float(latency_data.max()),
                    "std": float(latency_data.std()),
                }

        # Performance categories
        for cat_col in [
            "download_speed_category",
            "upload_speed_category",
            "latency_category",
        ]:
            if cat_col in df.columns:
                cat_counts = df[cat_col].value_counts().to_dict()
                summary[cat_col.replace("_category", "_breakdown")] = cat_counts

        # Quality metrics
        if "has_broadband" in df.columns:
            summary["broadband_capable_measurements"] = int(df["has_broadband"].sum())
            summary["broadband_percentage"] = float(df["has_broadband"].mean() * 100)

        if "has_basic_connectivity" in df.columns:
            summary["basic_connectivity_measurements"] = int(
                df["has_basic_connectivity"].sum()
            )
            summary["basic_connectivity_percentage"] = float(
                df["has_basic_connectivity"].mean() * 100
            )

        # Data completeness
        if "has_complete_measurement" in df.columns:
            summary["complete_measurements"] = int(df["has_complete_measurement"].sum())
            summary["data_completeness_percentage"] = float(
                df["has_complete_measurement"].mean() * 100
            )

        # Data sources
        if "data_source" in df.columns:
            source_counts = df["data_source"].value_counts().to_dict()
            summary["data_sources"] = source_counts

        # Temporal patterns
        if "day_of_week" in df.columns:
            day_counts = df["day_of_week"].value_counts().to_dict()
            summary["measurements_by_day_of_week"] = day_counts

        self.logger.info("Generated performance summary")
        return summary

    def get_school_performance_comparison(
        self, df: pd.DataFrame, top_n: int = 10
    ) -> dict:
        """
        Compare performance across schools.

        Args:
            df: DataFrame with measurement data
            top_n: Number of top/bottom schools to include

        Returns:
            dict: School performance comparison
        """
        if df.empty or "giga_id_school" not in df.columns:
            return {"error": "No school data available"}

        school_stats = (
            df.groupby("giga_id_school")
            .agg(
                {
                    "download_speed": ["mean", "median", "count"],
                    "upload_speed": ["mean", "median"],
                    "latency": ["mean", "median"],
                    "has_broadband": (
                        "mean" if "has_broadband" in df.columns else lambda x: None
                    ),
                }
            )
            .round(2)
        )

        # Flatten column names
        school_stats.columns = ["_".join(col).strip() for col in school_stats.columns]

        # Sort by download speed
        if "download_speed_mean" in school_stats.columns:
            top_schools = school_stats.nlargest(top_n, "download_speed_mean")
            bottom_schools = school_stats.nsmallest(top_n, "download_speed_mean")

            return {
                "top_performing_schools": top_schools.to_dict("index"),
                "bottom_performing_schools": bottom_schools.to_dict("index"),
                "total_schools_analyzed": len(school_stats),
            }

        return {"error": "Insufficient data for school comparison"}
