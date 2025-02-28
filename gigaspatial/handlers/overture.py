# import requests
import geopandas as gpd
from typing import List
from dataclasses import dataclass
from shapely.geometry import Polygon
import country_converter as coco
import duckdb

from gigaspatial.utils.logging import get_logger


@dataclass
class OvertureAmenityFetcher:
    """
    A class to fetch and process amenity locations from OpenStreetMap using the Overpass API.
    """

    country_iso3: str
    amenity_types: List[str]

    def __post_init__(self):
        """Validate inputs and set up logging."""
        self.country_iso2 = coco.convert(names=self.country_iso3, to="ISO2")
        self.logger = get_logger(__name__)

    def _set_connection(self):
        """Set the connection to the DB"""
        db = duckdb.connect()
        db.execute("INSTALL spatial")
        db.execute("INSTALL httpfs")
        db.execute(
            """
        LOAD spatial;
        LOAD httpfs;
        SET s3_region='us-west-2';
        """
        )
        return db

    def _build_query(self, release: str, geometry=None) -> str:
        """
        Constructs and returns the query
        """
        url = f"s3://overturemaps-us-west-2/release/{release}/theme=places/*/*"
        if geometry == None:
            query = f"""
                    SELECT
                        id as id,
                        names.primary AS name,
                        categories.primary as category,
                        ROUND(confidence,2) as confidence,
                        ST_AsText(ST_GeomFromWKB(geometry)) AS geometry
                    FROM 
                        read_parquet('{url}')
                    WHERE
                        json_extract_string(json_extract(addresses::json, '$[0]'), '$.country') = '{self.sountry_iso2}'
                        and (
                    """
        else:
            minx, miny, maxx, maxy = geometry.bounds
            query = f"""
                    SELECT
                        id as id,
                        names.primary AS name,
                        categories.primary as category,
                        ROUND(confidence,2) as confidence,
                        ST_AsText(ST_GeomFromWKB(geometry)) AS geometry
                    FROM 
                        read_parquet('{url}')
                    WHERE
                        bbox.xmin > {minx}
                        and bbox.ymin > {miny}
                        and bbox.xmax < {maxx}
                        and bbox.ymax < {maxy}
                        and (
                    """

        for amenity in self.amenity_types:
            query = query + f"category=='{amenity}' or "

        query = query[:-4] + ")"

        return query

    def get_locations(self, release: str, geometry: Polygon = None) -> gpd.GeoDataFrame:
        """
        Fetch and process amenity locations.

        Args:
            release (str): The overture release name.
            geometry (Polygon, optional): the polygon of the country of interest

        Returns:
            gpd.GeoDataFrame: Processed amenity locations
        """
        self.logger.info("Fetching amenity locations from Overture...")
        db = self._set_connection()
        query = self._build_query(release, geometry)

        df = db.execute(query).fetchdf()

        self.logger.info("Processing geometries")
        gdf = gpd.GeoDataFrame(df)
        gdf["geometry"] = gpd.GeoSeries.from_wkt(gdf["geometry"])
        gdf = gdf.set_geometry("geometry")

        if geometry != None:
            self.logger.info("Filtering within country border")
            locations = gdf[gdf["geometry"].within(geometry)]
        else:
            locations = gdf

        self.logger.info(f"Successfully processed {len(locations)} amenity locations")
        return locations
