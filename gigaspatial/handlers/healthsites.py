import requests
import pandas as pd
import geopandas as gpd
import time
from typing import List, Optional, Union, Tuple
from pydantic.dataclasses import dataclass, Field
from pydantic import ConfigDict
import pycountry

from gigaspatial.config import config
from gigaspatial.handlers import OSMLocationFetcher


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class HealthSitesFetcher:
    """
    Fetch and process health facility location data from the Healthsites.io API.
    """

    country: Optional[str] = Field(default=None, description="Country to filter")
    api_url: str = Field(
        default="https://healthsites.io/api/v3/facilities/",
        description="Base URL for the Healthsites API",
    )
    api_key: str = config.HEALTHSITES_API_KEY
    extent: Optional[Tuple[float, float, float, float]] = Field(
        default=None, description="Bounding box as (minLng, minLat, maxLng, maxLat)"
    )
    page_size: int = Field(default=100, description="Number of records per API page")
    flat_properties: bool = Field(
        default=True, description="Show properties in flat format"
    )
    tag_format: str = Field(default="osm", description="Tag format (osm/hxl)")
    output_format: str = Field(
        default="geojson", description="Output format (json/geojson)"
    )
    sleep_time: float = Field(
        default=0.2, description="Sleep time between API requests"
    )

    def __post_init__(self):
        self.logger = config.get_logger(self.__class__.__name__)
        # Convert country code to OSM English name if provided
        if self.country:
            self.country = self._convert_country(self.country)

    def fetch_facilities(self, **kwargs) -> Union[pd.DataFrame, gpd.GeoDataFrame]:
        """
        Fetch and process health facility locations.

        Args:
            **kwargs: Additional parameters for customization
                - country: Override country filter
                - extent: Override extent filter
                - from_date: Get data modified from this timestamp (datetime or string)
                - to_date: Get data modified to this timestamp (datetime or string)
                - page_size: Override default page size
                - sleep_time: Override default sleep time between requests
                - max_pages: Limit the number of pages to fetch
                - output_format: Override output format ('json' or 'geojson')
                - flat_properties: Override flat properties setting

        Returns:
            Union[pd.DataFrame, gpd.GeoDataFrame]: Health facilities data.
                Returns GeoDataFrame for geojson format, DataFrame for json format.
        """
        # Override defaults with kwargs if provided
        country = kwargs.get("country", self.country)
        extent = kwargs.get("extent", self.extent)
        from_date = kwargs.get("from_date", None)
        to_date = kwargs.get("to_date", None)
        page_size = kwargs.get("page_size", self.page_size)
        sleep_time = kwargs.get("sleep_time", self.sleep_time)
        max_pages = kwargs.get("max_pages", None)
        output_format = kwargs.get("output_format", self.output_format)
        flat_properties = kwargs.get("flat_properties", self.flat_properties)

        # Convert country if provided in kwargs
        if country:
            country = self._convert_country(country)

        # Prepare base parameters
        base_params = {
            "api-key": self.api_key,
            "tag-format": self.tag_format,
            "output": output_format,
        }

        # Only add flat-properties if True (don't send it as false, as that makes it flat anyway)
        if flat_properties:
            base_params["flat-properties"] = "true"

        # Add optional filters
        if country:
            base_params["country"] = country

        if extent:
            if len(extent) != 4:
                raise ValueError(
                    "Extent must be a tuple of 4 values: (minLng, minLat, maxLng, maxLat)"
                )
            base_params["extent"] = ",".join(map(str, extent))

        if from_date:
            base_params["from"] = self._format_timestamp(from_date)

        if to_date:
            base_params["to"] = self._format_timestamp(to_date)

        all_data = []
        page = 1

        self.logger.info(
            f"Starting to fetch health facilities for country: {country or 'all countries'}"
        )
        self.logger.info(
            f"Output format: {output_format}, Flat properties: {flat_properties}"
        )

        while True:
            # Check if we've reached max_pages limit
            if max_pages and page > max_pages:
                self.logger.info(f"Reached maximum pages limit: {max_pages}")
                break

            # Add page parameter
            params = base_params.copy()
            params["page"] = page

            try:
                self.logger.debug(f"Fetching page {page} with params: {params}")
                response = requests.get(self.api_url, params=params)
                response.raise_for_status()

                parsed = response.json()

                # Handle different response structures based on output format
                if output_format == "geojson":
                    # GeoJSON returns FeatureCollection with features list
                    data = parsed.get("features", [])
                else:
                    # JSON returns direct list
                    data = parsed if isinstance(parsed, list) else []

            except requests.exceptions.RequestException as e:
                self.logger.error(f"Request failed on page {page}: {e}")
                break
            except ValueError as e:
                self.logger.error(f"Failed to parse JSON response on page {page}: {e}")
                break

            # Check if we got any data
            if not data or not isinstance(data, list):
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

        # Convert to DataFrame/GeoDataFrame based on format
        if not all_data:
            self.logger.warning("No data fetched, returning empty DataFrame")
            if output_format == "geojson":
                return gpd.GeoDataFrame()
            return pd.DataFrame()

        if output_format == "geojson":
            # Use GeoDataFrame.from_features for GeoJSON format
            gdf = gpd.GeoDataFrame.from_features(all_data)
            self.logger.info(f"Created GeoDataFrame with {len(gdf)} records")
            return gdf
        else:
            # For JSON format, handle nested structure if flat_properties is False
            if not flat_properties:
                df = self._process_json_with_centroid(all_data)
            else:
                df = pd.DataFrame(all_data)

            self.logger.info(f"Created DataFrame with {len(df)} records")
            return df

    def fetch_statistics(self, **kwargs) -> dict:
        """
        Fetch statistics for health facilities.

        Args:
            **kwargs: Same filtering parameters as fetch_facilities

        Returns:
            dict: Statistics data
        """
        country = kwargs.get("country", self.country)
        extent = kwargs.get("extent", self.extent)
        from_date = kwargs.get("from_date", None)
        to_date = kwargs.get("to_date", None)

        # Convert country if provided
        if country:
            country = self._convert_country(country)

        params = {
            "api-key": self.api_key,
        }

        # Add optional filters
        if country:
            params["country"] = country
        if extent:
            params["extent"] = ",".join(map(str, extent))
        if from_date:
            params["from"] = self._format_timestamp(from_date)
        if to_date:
            params["to"] = self._format_timestamp(to_date)

        try:
            response = requests.get(f"{self.api_url}/statistic/", params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Request failed for statistics: {e}")
            raise

    def fetch_facility_by_id(self, osm_type: str, osm_id: str) -> dict:
        """
        Fetch a specific facility by OSM type and ID.

        Args:
            osm_type: OSM type (node, way, relation)
            osm_id: OSM ID

        Returns:
            dict: Facility details
        """
        params = {"api-key": self.api_key}

        try:
            url = f"{self.api_url}/{osm_type}/{osm_id}"
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Request failed for facility {osm_type}/{osm_id}: {e}")
            raise

    def _create_dataframe(self, data: List[dict]) -> pd.DataFrame:
        """
        Create DataFrame from API response data.

        Args:
            data: List of facility records

        Returns:
            pd.DataFrame: Processed DataFrame
        """
        if self.output_format == "geojson":
            # Handle GeoJSON format
            records = []
            for feature in data:
                record = feature.get("properties", {}).copy()
                geometry = feature.get("geometry", {})
                coordinates = geometry.get("coordinates", [])

                if coordinates and len(coordinates) >= 2:
                    record["longitude"] = coordinates[0]
                    record["latitude"] = coordinates[1]

                records.append(record)
            return pd.DataFrame(records)
        else:
            # Handle regular JSON format
            return pd.DataFrame(data)

    def _process_json_with_centroid(self, data: List[dict]) -> pd.DataFrame:
        """
        Process JSON data to flatten 'attributes' and 'centroid' fields,
        and extract longitude/latitude from centroid.

        Args:
            data: List of facility records, where each record might contain
                  nested 'attributes' and 'centroid' dictionaries.

        Returns:
            pd.DataFrame: Processed DataFrame with flattened data.
        """
        processed_records = []
        for record in data:
            new_record = {}

            # Flatten top-level keys
            for key, value in record.items():
                if key not in ["attributes", "centroid"]:
                    new_record[key] = value

            # Flatten 'attributes'
            attributes = record.get("attributes", {})
            for attr_key, attr_value in attributes.items():
                new_record[f"{attr_key}"] = attr_value

            # Extract centroid coordinates
            centroid = record.get("centroid", {})
            coordinates = centroid.get("coordinates", [])
            if coordinates and len(coordinates) == 2:
                new_record["longitude"] = coordinates[0]
                new_record["latitude"] = coordinates[1]
            else:
                new_record["longitude"] = None
                new_record["latitude"] = None

            processed_records.append(new_record)

        return pd.DataFrame(processed_records)

    def _convert_country(self, country: str) -> str:
        try:
            # First convert to ISO3 format if needed
            country_obj = pycountry.countries.lookup(country)
            iso3_code = country_obj.alpha_3

            # Get OSM English name using OSMLocationFetcher
            osm_data = OSMLocationFetcher.get_osm_countries(iso3_code=iso3_code)
            osm_name_en = osm_data.get("name:en")

            if not osm_name_en:
                raise ValueError(
                    f"Could not find OSM English name for country: {country}"
                )

            self.logger.info(
                f"Converted country code to OSM English name: {osm_name_en}"
            )

            return osm_name_en

        except LookupError:
            raise ValueError(f"Invalid country code provided: {country}")
        except Exception as e:
            raise ValueError(f"Failed to get OSM English name: {e}")
