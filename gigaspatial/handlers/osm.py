import requests
import pandas as pd
from typing import List, Dict
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from time import sleep
from requests.exceptions import RequestException
from shapely.geometry import Polygon, Point

from gigaspatial.utils.logging import get_logger


@dataclass
class OSMAmenityFetcher:
    """
    A class to fetch and process amenity locations from OpenStreetMap using the Overpass API.
    """

    country_iso2: str
    amenity_types: List[str]
    base_url: str = "http://overpass-api.de/api/interpreter"
    timeout: int = 600
    max_retries: int = 3
    retry_delay: int = 5

    def __post_init__(self):
        """Validate inputs and set up logging."""
        self.country_iso2 = self.country_iso2.upper()
        self.logger = get_logger(__name__)

    def _build_queries(self, since_year: int = None) -> List[str]:
        """
        Construct separate Overpass QL queries for different element types.
        Returns list of [nodes_relations_query, ways_query]
        """
        if since_year:
            date_filter = f'(newer:"{since_year}-01-01T00:00:00Z")'
        else:
            date_filter = ""

        # Query for nodes and relations (with center output)
        nodes_relations_queries = []
        for _type in self.amenity_types:
            nodes_relations_queries.extend(
                [
                    f"node[amenity={_type}]{date_filter}(area.searchArea);",
                    f"relation[amenity={_type}]{date_filter}(area.searchArea);",
                ]
            )

        nodes_relations_queries = "\n".join(nodes_relations_queries)

        nodes_relations_query = f"""
        [out:json][timeout:{self.timeout}];
        area["ISO3166-1"={self.country_iso2}]->.searchArea;
        (
            {nodes_relations_queries}
        );
        out center;
        """

        # Query for ways (with geometry output)
        ways_queries = []
        for _type in self.amenity_types:
            ways_queries.append(f"way[amenity={_type}]{date_filter}(area.searchArea);")

        ways_queries = "\n".join(ways_queries)

        ways_query = f"""
        [out:json][timeout:{self.timeout}];
        area["ISO3166-1"={self.country_iso2}]->.searchArea;
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

    def _process_node_relation(self, element: Dict) -> Dict[str, any]:
        """Process a node or relation element."""
        try:
            _lat = element.get("lat") or element["center"]["lat"]
            _lon = element.get("lon") or element["center"]["lon"]
            return {
                "source_id": element["id"],
                "amenity": element["tags"]["amenity"],
                "name": element["tags"].get("name", ""),
                "name_en": element["tags"].get("name:en", ""),
                "type": element["type"],
                "geometry": Point(_lon, _lat),
                "latitude": _lat,
                "longitude": _lon,
            }
        except KeyError as e:
            self.logger.error(f"Corrupt data received for node/relation: {str(e)}")
            return None

    def _process_way(self, element: Dict) -> Dict[str, any]:
        """Process a way element with geometry."""
        try:
            # Create polygon from geometry points
            polygon = Polygon([(p["lon"], p["lat"]) for p in element["geometry"]])
            centroid = polygon.centroid

            return {
                "source_id": element["id"],
                "amenity": element["tags"]["amenity"],
                "name": element["tags"].get("name", ""),
                "name_en": element["tags"].get("name:en", ""),
                "type": element["type"],
                "geometry": polygon,
                "latitude": centroid.y,
                "longitude": centroid.x,
            }
        except (KeyError, ValueError) as e:
            self.logger.error(f"Error processing way geometry: {str(e)}")
            return None

    def get_locations(self, since_year: int = None) -> pd.DataFrame:
        """
        Fetch and process amenity locations.

        Args:
            since_year (int, optional): Filter for amenities added/modified since this year.

        Returns:
            pd.DataFrame: Processed amenity locations
        """
        self.logger.info("Fetching amenity locations from Overpass API...")

        # Get queries for different element types
        nodes_relations_query, ways_query = self._build_queries(since_year)

        # Fetch nodes and relations
        nodes_relations_response = self._make_request(nodes_relations_query)
        nodes_relations = nodes_relations_response.get("elements", [])

        # Fetch ways
        ways_response = self._make_request(ways_query)
        ways = ways_response.get("elements", [])

        if not nodes_relations and not ways:
            self.logger.warning("No amenities found for the specified criteria")
            return pd.DataFrame()

        self.logger.info(
            f"Processing {len(nodes_relations)} nodes/relations and {len(ways)} ways..."
        )

        # Process nodes and relations
        with ThreadPoolExecutor() as executor:
            processed_nodes_relations = list(
                executor.map(self._process_node_relation, nodes_relations)
            )

        # Process ways
        with ThreadPoolExecutor() as executor:
            processed_ways = list(executor.map(self._process_way, ways))

        # Combine all processed elements
        all_elements = [
            p for p in processed_nodes_relations + processed_ways if p is not None
        ]

        if not all_elements:
            return pd.DataFrame()

        locations = pd.DataFrame(all_elements)

        # Log element type distribution
        type_counts = locations["type"].value_counts()
        self.logger.info("\nElement type distribution:")
        for element_type, count in type_counts.items():
            self.logger.info(f"{element_type}: {count}")

        self.logger.info(f"Successfully processed {len(locations)} amenity locations")
        return locations
