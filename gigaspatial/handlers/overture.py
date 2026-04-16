"""
Overture Maps Places theme data handler.

This module provides specialized tools for querying Overture Maps datasets via
DuckDB's spatial and S3 extensions. It facilitates:
- Direct S3 querying of Overture Places GeoParquet files.
- Spatial filtering by administrative boundaries or custom geometries.
- Category-based filtering for points of interest (POIs).
- High-performance data acquisition without local mirroring.
"""
import geopandas as gpd
from typing import List, Optional, Union
from pydantic.dataclasses import dataclass, Field
from pydantic import ConfigDict
from shapely.geometry import Polygon, MultiPolygon
from shapely.strtree import STRtree
from pathlib import Path
import pycountry

try:
    import duckdb

    _HAS_DUCKDB = True
except ImportError:
    _HAS_DUCKDB = False

from gigaspatial.config import config
from gigaspatial.handlers.boundaries import AdminBoundaries
from gigaspatial.core.io.data_store import DataStore


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class OvertureAmenityFetcher:
    """
    Fetch and process amenity locations from the Overture Places theme.

    This handler queries the Overture Places GeoParquet on S3, filters by a
    country boundary, and returns a GeoDataFrame of point locations for the
    requested amenity categories.

    Attributes:
        release: Overture release version (e.g., "2024-11-13.1").
        base_url: S3 pattern for the Places theme.
        country: ISO 3166-1 alpha-2 or alpha-3 country code.
        amenity_types: List of primary categories to retrieve.
        geom: Optional spatial boundary for filtering.
        data_store: Optional storage interface for boundary retrieval.
        country_geom_path: Path to local country boundary file.
    """

    # constants
    release: Optional[str] = "2026-01-21.0"

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
        if not _HAS_DUCKDB:
            raise ImportError(
                "OvertureAmenityFetcher requires 'duckdb'. "
                "Install it with: pip install 'giga-spatial[duckdb]'"
            )
        try:
            self.country = pycountry.countries.lookup(self.country).alpha_2
        except LookupError:
            raise ValueError(f"Invalid country code provided: {self.country}")

        self.base_url = self.base_url.format(release=self.release)
        self.logger = config.get_logger(self.__class__.__name__)

        self.connection = self._set_connection()

    def _set_connection(self):
        """
        Establish a DuckDB connection with S3 and Spatial support.

        Returns:
            A configured duckdb.DuckDBPyConnection.
        """
        db = duckdb.connect()

        # CRITICAL: Install httpfs for S3 access
        db.install_extension("httpfs")
        db.load_extension("httpfs")

        db.install_extension("spatial")
        db.load_extension("spatial")

        # Configure S3 region
        db.execute("SET s3_region='us-west-2'")

        return db

    def _load_country_geometry(
        self,
    ) -> Union[Polygon, MultiPolygon]:
        """
        Locate and load the administrative boundary for the target country.

        Returns:
            A Shapely geometry representing the country boundary.
        """

        country_geom = AdminBoundaries.create(
            country_code=self.country,
            data_store=self.data_store,
            path=self.country_geom_path,
        ).to_geoms()[0]

        return country_geom

    def _build_query(self, match_pattern: bool = False, **kwargs) -> str:
        """
        Construct the SQL query for S3 Parquet access.

        Args:
            match_pattern: If True, uses ILIKE for category matching.
            **kwargs: Additional query parameters.

        Returns:
            A DuckDB SQL query string.
        """

        if match_pattern:
            amenity_query = " OR ".join(
                [f"category ilike '%{amenity}%'" for amenity in self.amenity_types]
            )
        else:
            amenity_query = " OR ".join(
                [f"category == '{amenity}'" for amenity in self.amenity_types]
            )

        if not self.geom:
            self.geom = self._load_country_geometry()

        bounds = self.geom.bounds

        query = f"""
        SELECT id,
            names.primary AS name,
            ROUND(confidence,2) as confidence,
            categories.primary AS category,
            ST_AsText(geometry) as geometry,
        FROM read_parquet('{self.base_url}')
        WHERE bbox.xmin > {bounds[0]}
            AND bbox.ymin > {bounds[1]}
            AND bbox.xmax < {bounds[2]}
            AND bbox.ymax < {bounds[3]}
            AND ({amenity_query})
        """

        return query

    def fetch_locations(
        self, match_pattern: bool = False, **kwargs
    ) -> gpd.GeoDataFrame:
        """
        Execute the query and return locations as a GeoDataFrame.

        Args:
            match_pattern: If True, uses ILIKE for category matching.
            **kwargs: Additional execution parameters.

        Returns:
            A GeoDataFrame of retrieved POIs.
        """
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
