import requests
import pandas as pd
from typing import List, Dict, Union, Optional, Literal
from dataclasses import dataclass
from time import sleep
from concurrent.futures import ThreadPoolExecutor
from requests.exceptions import RequestException
from shapely.geometry import Polygon, Point
import pycountry

from gigaspatial.config import config


@dataclass
class OSMLocationFetcher:
    """
    A class to fetch and process location data from OpenStreetMap using the Overpass API.

    This class supports fetching various OSM location types including amenities, buildings,
    shops, and other POI categories.
    """

    country: str
    location_types: Union[List[str], Dict[str, List[str]]]
    base_url: str = "http://overpass-api.de/api/interpreter"
    timeout: int = 600
    max_retries: int = 3
    retry_delay: int = 5

    def __post_init__(self):
        """Validate inputs, normalize location_types, and set up logging."""
        try:
            self.country = pycountry.countries.lookup(self.country).alpha_2
        except LookupError:
            raise ValueError(f"Invalid country code provided: {self.country}")

        # Normalize location_types to always be a dictionary
        if isinstance(self.location_types, list):
            self.location_types = {"amenity": self.location_types}
        elif not isinstance(self.location_types, dict):
            raise TypeError(
                "location_types must be a list of strings or a dictionary mapping categories to type lists"
            )

        self.logger = config.get_logger(self.__class__.__name__)

    def _build_queries(self, since_year: Optional[int] = None) -> List[str]:
        """
        Construct separate Overpass QL queries for different element types and categories.
        Returns list of [nodes_relations_query, ways_query]
        """
        if since_year:
            date_filter = f'(newer:"{since_year}-01-01T00:00:00Z")'
        else:
            date_filter = ""

        # Query for nodes and relations (with center output)
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
        area["ISO3166-1"={self.country}]->.searchArea;
        (
            {nodes_relations_queries}
        );
        out center;
        """

        # Query for ways (with geometry output)
        ways_queries = []
        for category, types in self.location_types.items():
            ways_queries.append(
                f"""way["{category}"~"^({"|".join(types)})"]{date_filter}(area.searchArea);"""
            )

        ways_queries = "\n".join(ways_queries)

        ways_query = f"""
        [out:json][timeout:{self.timeout}];
        area["ISO3166-1"={self.country}]->.searchArea;
        (
            {ways_queries}
        );
        out geom;
        """

        return [nodes_relations_query, ways_query]

    def _make_request(self, query: str) -> Dict:
        """Make HTTP request to Overpass API with retry mechanism."""
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
        Extract all matching categories and their values from the tags.
        Returns:
            Dict mapping each matching category to its value
        """
        matches = {}
        for category, types in self.location_types.items():
            if category in tags and tags[category] in types:
                matches[category] = tags[category]
        return matches

    def _process_node_relation(self, element: Dict) -> List[Dict[str, any]]:
        """
        Process a node or relation element.
        May return multiple processed elements if the element matches multiple categories.
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

            # for each matching category, create a separate element
            results = []
            for category, value in matching_categories.items():
                results.append(
                    {
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
                )

            return results

        except KeyError as e:
            self.logger.error(f"Corrupt data received for node/relation: {str(e)}")
            return []

    def _process_way(self, element: Dict) -> List[Dict[str, any]]:
        """
        Process a way element with geometry.
        May return multiple processed elements if the element matches multiple categories.
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

            # For each matching category, create a separate element
            results = []
            for category, value in matching_categories.items():
                results.append(
                    {
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
                )

            return results
        except (KeyError, ValueError) as e:
            self.logger.error(f"Error processing way geometry: {str(e)}")
            return []

    def fetch_locations(
        self,
        since_year: Optional[int] = None,
        handle_duplicates: Literal["separate", "combine", "primary"] = "separate",
    ) -> pd.DataFrame:
        """
        Fetch and process OSM locations.

        Args:
            since_year (int, optional): Filter for locations added/modified since this year.
            handle_duplicates (str): How to handle objects matching multiple categories:
                - 'separate': Create separate entries for each category (default)
                - 'combine': Use a single entry with a list of matching categories
                - 'primary': Keep only the first matching category

        Returns:
            pd.DataFrame: Processed OSM locations
        """
        if handle_duplicates not in ("separate", "combine", "primary"):
            raise ValueError(
                "handle_duplicates must be one of: 'separate', 'combine', 'primary'"
            )

        self.logger.info(
            f"Fetching OSM locations from Overpass API for country: {self.country}"
        )
        self.logger.info(f"Location types: {self.location_types}")
        self.logger.info(f"Handling duplicate category matches as: {handle_duplicates}")

        # Get queries for different element types
        nodes_relations_query, ways_query = self._build_queries(since_year)

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

        # Handle duplicates based on the specified strategy
        if handle_duplicates != "separate":
            # Group by source_id
            grouped_elements = {}
            for elem in all_elements:
                source_id = elem["source_id"]
                if source_id not in grouped_elements:
                    grouped_elements[source_id] = elem
                elif handle_duplicates == "combine":
                    # Combine matching categories
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
                # For 'primary', just keep the first one we encountered

            all_elements = list(grouped_elements.values())

        locations = pd.DataFrame(all_elements)

        # Log element type distribution
        type_counts = locations["type"].value_counts()
        self.logger.info("\nElement type distribution:")
        for element_type, count in type_counts.items():
            self.logger.info(f"{element_type}: {count}")

        # Log category distribution
        if handle_duplicates == "combine":
            # Count each category separately when they're in lists
            category_counts = {}
            for cats in locations["category"]:
                if isinstance(cats, list):
                    for cat in cats:
                        category_counts[cat] = category_counts.get(cat, 0) + 1
                else:
                    category_counts[cats] = category_counts.get(cats, 0) + 1

            self.logger.info("\nCategory distribution:")
            for category, count in category_counts.items():
                self.logger.info(f"{category}: {count}")
        else:
            category_counts = locations["category"].value_counts()
            self.logger.info("\nCategory distribution:")
            for category, count in category_counts.items():
                self.logger.info(f"{category}: {count}")

        # Log elements with multiple matching categories
        multi_category = [e for e in all_elements if len(e["matching_categories"]) > 1]
        if multi_category:
            self.logger.info(
                f"\n{len(multi_category)} elements matched multiple categories"
            )

        self.logger.info(f"Successfully processed {len(locations)} locations")
        return locations
