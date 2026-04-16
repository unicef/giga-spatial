"""
OpenStreetMap (OSM) data fetcher via Overpass API.

This module provides the `OSMLocationFetcher` class for retrieving points of interest
(POIs) and administrative boundaries from OSM. It supports:
- Keyword-based POI search (amenities, shops, etc.)
- ISO country or admin level area selection
- Historical change tracking (newer/changed filters)
- Metadata extraction (timestamp, version, user)
"""
import requests
import pandas as pd
from typing import List, Dict, Union, Optional, Literal
from pydantic.dataclasses import dataclass
from pydantic import Field
from time import sleep
from concurrent.futures import ThreadPoolExecutor
from requests.exceptions import RequestException
from shapely.geometry import Polygon, Point
import pycountry
from datetime import datetime

from gigaspatial.config import config


@dataclass
class OSMLocationFetcher:
    """
    Fetcher for OpenStreetMap POIs and boundaries using the Overpass API.

    Handles high-level queries for diverse OSM categories (amenities, buildings,
    shops, etc.) within specific administrative regions or countries.

    Attributes:
        country: ISO country code or name.
        admin_level: OSM administrative level for the search area.
        admin_value: Name of the administrative area.
        location_types: Mapping of category keys (e.g., 'amenity') to values.
        base_url: Overpass API endpoint URL.
        timeout: API request timeout in seconds.
        max_retries: Number of retry attempts for failed requests.
        retry_delay: Delay between retries in seconds.
    """

    country: Optional[str] = None
    admin_level: Optional[int] = None
    admin_value: Optional[str] = None
    location_types: Union[List[str], Dict[str, List[str]]] = Field(...)
    base_url: str = "http://overpass-api.de/api/interpreter"
    timeout: int = 600
    max_retries: int = 3
    retry_delay: int = 5

    def __post_init__(self):
        """Validates inputs, normalizes location_types, and initializes the logger.

        Raises:
            TypeError: If `location_types` is not a list or dictionary.
            ValueError: If neither `country` nor `admin_level/admin_value` are provided,
                or if the country code is invalid.
        """

        # Normalize location_types to always be a dictionary
        if isinstance(self.location_types, list):
            self.location_types = {"amenity": self.location_types}
        elif not isinstance(self.location_types, dict):
            raise TypeError(
                "location_types must be a list of strings or a dictionary mapping categories to type lists"
            )

        self.logger = config.get_logger(self.__class__.__name__)

        # Validate area selection
        if self.admin_level is not None and self.admin_value is not None:
            self.area_query = f'area["admin_level"={self.admin_level}]["name"="{self.admin_value}"]->.searchArea;'
            self.logger.info(
                f"Using admin_level={self.admin_level}, name={self.admin_value} for area selection."
            )
        elif self.country is not None:
            try:
                self.country = pycountry.countries.lookup(self.country).alpha_2
            except LookupError:
                raise ValueError(f"Invalid country code provided: {self.country}")
            self.area_query = f'area["ISO3166-1"={self.country}]->.searchArea;'
            self.logger.info(f"Using country={self.country} for area selection.")
        else:
            raise ValueError(
                "Either country or both admin_level and admin_value must be provided."
            )

    @staticmethod
    def get_admin_names(
        admin_level: int, country: Optional[str] = None, timeout: int = 120
    ) -> List[str]:
        """
        Fetch administrative area names for a specific OSM level.

        Args:
            admin_level: The OSM level to search (e.g., 2 for country, 4 for state).
            country: Optional country name/code to limit the search.
            timeout: API request timeout in seconds.

        Returns:
            Sorted list of unique administrative area names.
        """

        # Build area filter for country if provided
        if country:
            try:
                country_code = pycountry.countries.lookup(country).alpha_2
            except LookupError:
                raise ValueError(f"Invalid country code or name: {country}")
            area_filter = f'area["ISO3166-1"="{country_code}"]->.countryArea;'
            area_ref = "(area.countryArea)"
        else:
            area_filter = ""
            area_ref = ""

        # Overpass QL to get all admin areas at the specified level
        query = f"""
        [out:json][timeout:{timeout}];
        {area_filter}
        (
          relation["admin_level"="{admin_level}"]{area_ref};
        );
        out tags;
        """

        url = "http://overpass-api.de/api/interpreter"
        response = requests.get(url, params={"data": query}, timeout=timeout)
        response.raise_for_status()
        data = response.json()

        names = []
        for el in data.get("elements", []):
            tags = el.get("tags", {})
            name = tags.get("name")
            if name:
                names.append(name)
        return sorted(set(names))

    @staticmethod
    def get_osm_countries(
        iso3_code: Optional[str] = None, include_names: bool = True, timeout: int = 2000
    ) -> Union[str, Dict[str, str], List[str], List[Dict[str, str]]]:
        """
        Fetch country definitions and name variants directly from OSM.

        Args:
            iso3_code: Optional ISO 3166-1 alpha-3 code to fetch a specific country.
            include_names: If True, returns full name dictionaries with translations.
            timeout: API request timeout in seconds.

        Returns:
            A single name/dict if `iso3_code` is provided, otherwise a sorted list.
        """
        if iso3_code:
            # Filter for the specific ISO3 code provided
            iso3_upper = iso3_code.upper()
            country_filter = f'["ISO3166-1:alpha3"="{iso3_upper}"]'
        else:
            # Filter for the *existence* of an ISO3 code tag to limit results to actual countries
            country_filter = '["ISO3166-1:alpha3"]'

        # Query OSM for country-level boundaries
        query = f"""
        [out:json][timeout:{timeout}];
        (
          relation["boundary"="administrative"]["admin_level"="2"]{country_filter};
        );
        out tags;
        """

        url = "http://overpass-api.de/api/interpreter"
        response = requests.get(url, params={"data": query}, timeout=timeout)
        response.raise_for_status()
        data = response.json()

        countries = []
        for element in data.get("elements", []):
            tags = element.get("tags", {})

            if include_names:
                country_info = {
                    "name": tags.get("name", ""),
                    "name:en": tags.get("name:en", ""),
                    "official_name": tags.get("official_name", ""),
                    "official_name:en": tags.get("official_name:en", ""),
                    "ISO3166-1": tags.get("ISO3166-1", ""),
                    "ISO3166-1:alpha2": tags.get("ISO3166-1:alpha2", ""),
                    "ISO3166-1:alpha3": tags.get("ISO3166-1:alpha3", ""),
                }

                # Add any other name:* tags (translations)
                for key, value in tags.items():
                    if key.startswith("name:") and key not in country_info:
                        country_info[key] = value

                # Remove empty string values
                country_info = {k: v for k, v in country_info.items() if v}

                if country_info.get("name"):  # Only add if has a name
                    countries.append(country_info)
            else:
                name = tags.get("name")
                if name:
                    countries.append(name)

        # If looking for a specific country, return single result or raise error
        if iso3_code:
            if not countries:
                raise ValueError(
                    f"Country with ISO3 code '{iso3_code}' not found in OSM database"
                )
            return countries[0]  # Return single country, not a list

        # Return sorted list for all countries
        return sorted(
            countries, key=lambda x: x if isinstance(x, str) else x.get("name", "")
        )

    def _make_request(self, query: str) -> Dict:
        """
        Execute an Overpass API request with a retry mechanism.

        Args:
            query: The Overpass QL query string.

        Returns:
            The JSON response from the API.

        Raises:
            RuntimeError: If all retry attempts fail.
        """
        for attempt in range(self.max_retries):
            try:
                self.logger.debug(f"Executing query:\n{query}")
                response = requests.get(
                    self.base_url, params={"data": query}, timeout=self.timeout
                )
                response.raise_for_status()
                return response.json()
            except RequestException as e:
                self.logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < self.max_retries - 1:
                    sleep(self.retry_delay)
                else:
                    raise RuntimeError(
                        f"Failed to fetch data after {self.max_retries} attempts"
                    ) from e

    def _extract_matching_categories(self, tags: Dict[str, str]) -> Dict[str, str]:
        """
        Identify OSM tags that match the requested location types.

        Args:
            tags: Raw tags from an OSM element.

        Returns:
            A dictionary of matching category keys and values.
        """
        matches = {}
        for category, types in self.location_types.items():
            if category in tags and tags[category] in types:
                matches[category] = tags[category]
        return matches

    def _process_node_relation(self, element: Dict) -> List[Dict[str, any]]:
        """
        Convert an OSM node or relation element into a standardized POI record.

        Args:
            element: Raw JSON element from Overpass.

        Returns:
            A list of processed POI dictionaries.
        """
        try:
            tags = element.get("tags", {})
            matching_categories = self._extract_matching_categories(tags)

            if not matching_categories:
                self.logger.warning(
                    f"Element {element['id']} missing or not matching specified category tags"
                )
                return []

            _lat = element.get("lat") or element["center"]["lat"]
            _lon = element.get("lon") or element["center"]["lon"]
            point_geom = Point(_lon, _lat)

            # Extract metadata if available
            metadata = {}
            if "timestamp" in element:
                metadata["timestamp"] = element["timestamp"]
                metadata["version"] = element.get("version")
                metadata["changeset"] = element.get("changeset")
                metadata["user"] = element.get("user")
                metadata["uid"] = element.get("uid")

            # For each matching category, create a separate element
            results = []
            for category, value in matching_categories.items():
                result = {
                    "source_id": element["id"],
                    "category": category,
                    "category_value": value,
                    "name": tags.get("name", ""),
                    "name_en": tags.get("name:en", ""),
                    "type": element["type"],
                    "geometry": point_geom,
                    "latitude": _lat,
                    "longitude": _lon,
                    "matching_categories": list(matching_categories.keys()),
                }
                # Add metadata if available
                result.update(metadata)
                results.append(result)

            return results

        except KeyError as e:
            self.logger.error(f"Corrupt data received for node/relation: {str(e)}")
            return []

    def _process_way(self, element: Dict) -> List[Dict[str, any]]:
        """
        Convert an OSM way element into a standardized POI record with geometry.

        Args:
            element: Raw JSON element from Overpass.

        Returns:
            A list of processed POI dictionaries.
        """
        try:
            tags = element.get("tags", {})
            matching_categories = self._extract_matching_categories(tags)

            if not matching_categories:
                self.logger.warning(
                    f"Element {element['id']} missing or not matching specified category tags"
                )
                return []

            # Create polygon from geometry points
            polygon = Polygon([(p["lon"], p["lat"]) for p in element["geometry"]])
            centroid = polygon.centroid

            # Extract metadata if available
            metadata = {}
            if "timestamp" in element:
                metadata["timestamp"] = element["timestamp"]
                metadata["version"] = element.get("version")
                metadata["changeset"] = element.get("changeset")
                metadata["user"] = element.get("user")
                metadata["uid"] = element.get("uid")

            # For each matching category, create a separate element
            results = []
            for category, value in matching_categories.items():
                result = {
                    "source_id": element["id"],
                    "category": category,
                    "category_value": value,
                    "name": tags.get("name", ""),
                    "name_en": tags.get("name:en", ""),
                    "type": element["type"],
                    "geometry": polygon,
                    "latitude": centroid.y,
                    "longitude": centroid.x,
                    "matching_categories": list(matching_categories.keys()),
                }
                # Add metadata if available
                result.update(metadata)
                results.append(result)

            return results
        except (KeyError, ValueError) as e:
            self.logger.error(f"Error processing way geometry: {str(e)}")
            return []

    def _build_queries(
        self,
        date_filter_type: Optional[Literal["newer", "changed"]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        include_metadata: bool = False,
    ) -> List[str]:
        """
        Construct Overpass QL queries with optional date filtering and metadata.

        Args:
            date_filter_type: Type of date filter ('newer' or 'changed')
            start_date: Start date in ISO 8601 format
            end_date: End date in ISO 8601 format (required for 'changed')
            include_metadata: If True, include change metadata (timestamp, version, changeset, user)

        Returns:
            List[str]: List of [nodes_relations_query, ways_query]
        """
        # Build the date filter based on type
        if date_filter_type == "newer" and start_date:
            date_filter = f'(newer:"{start_date}")'
        elif date_filter_type == "changed" and start_date and end_date:
            date_filter = f'(changed:"{start_date}","{end_date}")'
        else:
            date_filter = ""

        # Determine output mode
        output_mode = "center meta" if include_metadata else "center"
        output_mode_geom = "geom meta" if include_metadata else "geom"

        # Query for nodes and relations
        nodes_relations_queries = []
        for category, types in self.location_types.items():
            nodes_relations_queries.extend(
                [
                    f"""node["{category}"~"^({"|".join(types)})"]{date_filter}(area.searchArea);""",
                    f"""relation["{category}"~"^({"|".join(types)})"]{date_filter}(area.searchArea);""",
                ]
            )

        nodes_relations_queries = "\n".join(nodes_relations_queries)

        nodes_relations_query = f"""
        [out:json][timeout:{self.timeout}];
        {self.area_query}
        (
            {nodes_relations_queries}
        );
        out {output_mode};
        """

        # Query for ways
        ways_queries = []
        for category, types in self.location_types.items():
            ways_queries.append(
                f"""way["{category}"~"^({"|".join(types)})"]{date_filter}(area.searchArea);"""
            )

        ways_queries = "\n".join(ways_queries)

        ways_query = f"""
        [out:json][timeout:{self.timeout}];
        {self.area_query}
        (
            {ways_queries}
        );
        out {output_mode_geom};
        """

        return [nodes_relations_query, ways_query]

    def fetch_locations(
        self,
        since_date: Optional[Union[str, datetime]] = None,
        handle_duplicates: Literal["separate", "combine", "primary"] = "separate",
        include_metadata: bool = False,
    ) -> pd.DataFrame:
        """Fetches OSM locations matching the configured categories.

        Args:
            since_date: Optional date filter (addition or modification date).
                Supports ISO 8601 strings or datetime objects.
            handle_duplicates: Strategy for elements matching multiple types.
                - 'separate': Duplicate records (default).
                - 'combine': Single record with list of categories.
                - 'primary': Keep only the first matching category.
            include_metadata: If True, includes timestamp and user metadata.

        Returns:
            A pandas DataFrame containing processed OSM locations.

        Raises:
            ValueError: If an invalid duplicate handling strategy is provided.
        """
        if handle_duplicates not in ("separate", "combine", "primary"):
            raise ValueError(
                "handle_duplicates must be one of: 'separate', 'combine', 'primary'"
            )

        self.logger.info(
            f"Fetching OSM locations from Overpass API for country: {self.country}"
        )
        self.logger.info(f"Location types: {self.location_types}")

        # Normalize date if provided
        since_str = self._normalize_date(since_date) if since_date else None

        if since_str:
            self.logger.info(f"Filtering for changes since: {since_str}")

        queries = self._build_queries(
            date_filter_type="newer" if since_str else None,
            start_date=since_str,
            include_metadata=include_metadata,
        )

        return self._execute_and_process_queries(queries, handle_duplicates)

    def fetch_locations_changed_between(
        self,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        handle_duplicates: Literal["separate", "combine", "primary"] = "separate",
        include_metadata: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch OSM locations that changed within a specific date range.

        Use this for historical analysis or tracking changes in a specific period.

        Args:
            start_date: Start date/time in ISO 8601 format (str: "YYYY-MM-DDThh:mm:ssZ")
                    or datetime object. Changes after this date will be included.
            end_date: End date/time in ISO 8601 format (str: "YYYY-MM-DDThh:mm:ssZ")
                    or datetime object. Changes before this date will be included.
            handle_duplicates: How to handle objects matching multiple categories:
                - 'separate': Create separate entries for each category (default)
                - 'combine': Use a single entry with a list of matching categories
                - 'primary': Keep only the first matching category
            include_metadata: If True, include change tracking metadata
                (timestamp, version, changeset, user, uid)
                Defaults to True since change tracking is the main use case.

        Returns:
            A pandas DataFrame of historical changes.
        """
        start_str = self._normalize_date(start_date)
        end_str = self._normalize_date(end_date)

        if start_str >= end_str:
            raise ValueError(
                f"start_date must be before end_date (got {start_str} >= {end_str})"
            )

        queries = self._build_queries(
            date_filter_type="changed",
            start_date=start_str,
            end_date=end_str,
            include_metadata=include_metadata,
        )

        return self._execute_and_process_queries(queries, handle_duplicates)
    
    def fetch_by_osmid(
        self,
        osmid: int,
        element_type: Literal["node", "way", "relation"] = "node",
        include_metadata: bool = False,
    ) -> Optional[Dict]:
        """
        Retrieve and process a single OSM element by its unique identifier.

        Args:
            osmid: The OpenStreetMap ID.
            element_type: The type of element ('node', 'way', or 'relation').
            include_metadata: If True, includes object-level metadata.

        Returns:
            A processed dictionary of the element, or None if not found.
        """
        valid_types = ("node", "way", "relation")
        if element_type not in valid_types:
            raise ValueError(
                f"element_type must be one of {valid_types}, got '{element_type}'"
            )

        output_mode = "center meta" if include_metadata else "center"
        output_mode_geom = "geom meta" if include_metadata else "geom"

        # Ways need geometry output; nodes and relations use center
        if element_type == "way":
            out_clause = output_mode_geom
        else:
            out_clause = output_mode

        query = f"""
        [out:json][timeout:{self.timeout}];
        {element_type}(id:{osmid});
        out {out_clause};
        """

        self.logger.info(f"Fetching OSM {element_type} with id={osmid}")
        response = self._make_request(query)
        elements = response.get("elements", [])

        if not elements:
            self.logger.warning(f"No OSM element found for {element_type} id={osmid}")
            return None

        element = elements[0]

        if element_type == "way":
            results = self._process_way(element)
        else:
            results = self._process_node_relation(element)

        if not results:
            self.logger.warning(
                f"OSM {element_type} id={osmid} found but could not be processed "
                "(no matching location_types tags). Returning raw tags only."
            )
            # Fall back to a minimal dict with raw tags and geometry when the element
            # doesn't match any configured location_types (common for ID lookups).
            tags = element.get("tags", {})
            if element_type == "way" and "geometry" in element:
                from shapely.geometry import Polygon
                polygon = Polygon([(p["lon"], p["lat"]) for p in element["geometry"]])
                centroid = polygon.centroid
                return {
                    "source_id": element["id"],
                    "type": element_type,
                    "name": tags.get("name", ""),
                    "name_en": tags.get("name:en", ""),
                    "geometry": polygon,
                    "latitude": centroid.y,
                    "longitude": centroid.x,
                    "tags": tags,
                }
            else:
                _lat = element.get("lat") or element.get("center", {}).get("lat")
                _lon = element.get("lon") or element.get("center", {}).get("lon")
                return {
                    "source_id": element["id"],
                    "type": element_type,
                    "name": tags.get("name", ""),
                    "name_en": tags.get("name:en", ""),
                    "geometry": Point(_lon, _lat) if _lat and _lon else None,
                    "latitude": _lat,
                    "longitude": _lon,
                    "tags": tags,
                }

        # Return the first matching processed result
        return results[0]

    def _normalize_date(self, date_input: Union[str, datetime]) -> str:
        """
        Standardize date input into an ISO 8601 string for Overpass.

        Args:
            date_input: String in ISO format or datetime object.

        Returns:
            An ISO 8601 formatted string.

        Raises:
            ValueError: If the string format is unrecognized.
            TypeError: If the input type is unsupported.
        """
        from datetime import datetime

        if isinstance(date_input, datetime):
            # Convert datetime to ISO 8601 string with Z (UTC) timezone
            return date_input.strftime("%Y-%m-%dT%H:%M:%SZ")

        elif isinstance(date_input, str):
            # Validate the string format
            try:
                # Try to parse it to ensure it's valid
                datetime.strptime(date_input, "%Y-%m-%dT%H:%M:%SZ")
                return date_input
            except ValueError:
                raise ValueError(
                    f"Invalid date format: '{date_input}'. "
                    "Expected format: 'YYYY-MM-DDThh:mm:ssZ' (e.g., '2024-03-15T14:30:00Z')"
                )
        else:
            raise TypeError(
                f"date_input must be str or datetime, got {type(date_input).__name__}"
            )

    def _execute_and_process_queries(
        self, queries: List[str], handle_duplicates: str
    ) -> pd.DataFrame:
        """
        Execute API queries and aggregate results into a DataFrame.

        Args:
            queries: List of [nodes/relations query, ways query].
            handle_duplicates: Duplicate handling strategy ('separate', 'combine', 'primary').

        Returns:
            A pandas DataFrame of processed locations.
        """
        nodes_relations_query, ways_query = queries

        # Fetch nodes and relations
        nodes_relations_response = self._make_request(nodes_relations_query)
        nodes_relations = nodes_relations_response.get("elements", [])

        # Fetch ways
        ways_response = self._make_request(ways_query)
        ways = ways_response.get("elements", [])

        if not nodes_relations and not ways:
            self.logger.warning("No locations found for the specified criteria")
            return pd.DataFrame()

        self.logger.info(
            f"Processing {len(nodes_relations)} nodes/relations and {len(ways)} ways..."
        )

        # Process nodes and relations
        with ThreadPoolExecutor() as executor:
            processed_nodes_relations = [
                item
                for sublist in executor.map(
                    self._process_node_relation, nodes_relations
                )
                for item in sublist
            ]

        # Process ways
        with ThreadPoolExecutor() as executor:
            processed_ways = [
                item
                for sublist in executor.map(self._process_way, ways)
                for item in sublist
            ]

        # Combine all processed elements
        all_elements = processed_nodes_relations + processed_ways

        if not all_elements:
            self.logger.warning("No matching elements found after processing")
            return pd.DataFrame()

        # Handle duplicates (reuse existing logic from fetch_locations)
        if handle_duplicates != "separate":
            grouped_elements = {}
            for elem in all_elements:
                source_id = elem["source_id"]
                if source_id not in grouped_elements:
                    grouped_elements[source_id] = elem
                elif handle_duplicates == "combine":
                    if grouped_elements[source_id]["category"] != elem["category"]:
                        if isinstance(grouped_elements[source_id]["category"], str):
                            grouped_elements[source_id]["category"] = [
                                grouped_elements[source_id]["category"]
                            ]
                            grouped_elements[source_id]["category_value"] = [
                                grouped_elements[source_id]["category_value"]
                            ]

                        if (
                            elem["category"]
                            not in grouped_elements[source_id]["category"]
                        ):
                            grouped_elements[source_id]["category"].append(
                                elem["category"]
                            )
                            grouped_elements[source_id]["category_value"].append(
                                elem["category_value"]
                            )

            all_elements = list(grouped_elements.values())

        locations = pd.DataFrame(all_elements)

        # Log statistics
        type_counts = locations["type"].value_counts()
        self.logger.info("\nElement type distribution:")
        for element_type, count in type_counts.items():
            self.logger.info(f"{element_type}: {count}")

        self.logger.info(f"Successfully processed {len(locations)} locations")
        return locations
