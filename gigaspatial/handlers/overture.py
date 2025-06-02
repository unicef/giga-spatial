# import requests
import geopandas as gpd
from typing import List, Optional, Union
from pydantic.dataclasses import dataclass, Field
from pydantic import ConfigDict
from shapely.geometry import Polygon, MultiPolygon
from shapely.strtree import STRtree
from pathlib import Path
import pycountry
import duckdb

from gigaspatial.config import config
from gigaspatial.handlers.boundaries import AdminBoundaries
from gigaspatial.core.io.data_store import DataStore


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class OvertureAmenityFetcher:
    """
    A class to fetch and process amenity locations from Overture.
    """

    # constants
    release: Optional[str] = "2024-12-18.0"
    base_url: Optional[str] = (
        "s3://overturemaps-us-west-2/release/{release}/theme=places/*/*"
    )

    # user config
    country: str = Field(...)
    amenity_types: List[str] = Field(..., description="List of amenity types to fetch")
    geom: Union[Polygon, MultiPolygon] = None

    # config for country boundary access from data storage
    # if None GADM boundaries will be used
    data_store: DataStore = None
    country_geom_path: Optional[Union[str, Path]] = None

    def __post_init__(self):
        """Validate inputs and set up logging."""
        try:
            self.country = pycountry.countries.lookup(self.country).alpha_2
        except LookupError:
            raise ValueError(f"Invalid country code provided: {self.country}")

        self.base_url = self.base_url.format(release=self.release)
        self.logger = config.get_logger(self.__class__.__name__)

        self.connection = self._set_connection()

    def _set_connection(self):
        """Set the connection to the DB"""
        db = duckdb.connect()
        db.install_extension("spatial")
        db.load_extension("spatial")
        return db

    def _load_country_geometry(
        self,
    ) -> Union[Polygon, MultiPolygon]:
        """Load country boundary geometry from DataStore or GADM."""

        gdf_admin0 = AdminBoundaries.create(
            country_code=pycountry.countries.lookup(self.country).alpha_3,
            admin_level=0,
            data_store=self.data_store,
            path=self.country_geom_path,
        ).to_geodataframe()

        return gdf_admin0.geometry.iloc[0]

    def _build_query(self, match_pattern: bool = False, **kwargs) -> str:
        """Constructs and returns the query"""

        if match_pattern:
            amenity_query = " OR ".join(
                [f"category ilike '%{amenity}%'" for amenity in self.amenity_types]
            )
        else:
            amenity_query = " OR ".join(
                [f"category == '{amenity}'" for amenity in self.amenity_types]
            )

        query = """
        SELECT id,
            names.primary AS name,
            ROUND(confidence,2) as confidence,
            categories.primary AS category,
            ST_AsText(geometry) as geometry,
        FROM read_parquet('s3://overturemaps-us-west-2/release/2024-12-18.0/theme=places/type=place/*',
            hive_partitioning=1)
        WHERE bbox.xmin > {}
            AND bbox.ymin > {} 
            AND bbox.xmax <  {}
            AND bbox.ymax < {}
            AND ({})
        """

        if not self.geom:
            self.geom = self._load_country_geometry()

        return query.format(*self.geom.bounds, amenity_query)

    def fetch_locations(
        self, match_pattern: bool = False, **kwargs
    ) -> gpd.GeoDataFrame:
        """Fetch and process amenity locations."""
        self.logger.info("Fetching amenity locations from Overture DB...")

        query = self._build_query(match_pattern=match_pattern, **kwargs)

        df = self.connection.execute(query).df()

        self.logger.info("Processing geometries")
        gdf = gpd.GeoDataFrame(
            df, geometry=gpd.GeoSeries.from_wkt(df["geometry"]), crs="EPSG:4326"
        )

        # filter by geometry boundary
        s = STRtree(gdf.geometry)
        result = s.query(self.geom, predicate="intersects")

        locations = gdf.iloc[result].reset_index(drop=True)

        self.logger.info(f"Successfully processed {len(locations)} amenity locations")
        return locations
