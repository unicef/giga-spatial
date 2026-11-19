"""
Overture Maps Places theme fetcher.

This module provides specialized tools for querying Overture Maps datasets via
DuckDB's spatial and S3 extensions. It facilitates:
- Direct S3 querying of Overture Places GeoParquet files.
- Spatial filtering by administrative boundaries or custom geometries.
- Category-based filtering for points of interest (POIs).
- High-performance data acquisition without local mirroring.
"""

import geopandas as gpd
from typing import Any, ClassVar, Dict, List, Optional, Union
from pydantic.dataclasses import dataclass, Field
from pydantic import ConfigDict
from shapely.geometry import Polygon, MultiPolygon
from shapely.strtree import STRtree
from pathlib import Path
from urllib.parse import urlparse
import pycountry
import requests

try:
    import duckdb

    _HAS_DUCKDB = True
except ImportError:
    _HAS_DUCKDB = False

from gigaspatial.config import config
from gigaspatial.handlers.boundaries import AdminBoundaries
from gigaspatial.core.io.data_store import DataStore


@dataclass(config=ConfigDict(arbitrary_types_allowed=True), kw_only=True)
class OvertureAmenityFetcher:
    """
    Fetch and process amenity locations from the Overture Places theme.

    This fetcher queries the Overture Places GeoParquet on S3, filters by a
    country boundary, and returns a GeoDataFrame of point locations for the
    requested amenity categories.

    Attributes:
        release: Overture release version (e.g., "2026-11-18.0"), or "latest" / None
            for automatic STAC catalog resolution.
        base_url: S3 pattern for the Places theme.
        country: ISO 3166-1 alpha-2 or alpha-3 country code.
        amenity_types: List of primary categories to retrieve.
        geom: Optional spatial boundary for filtering.
        data_store: Optional storage interface for boundary retrieval.
        country_geom_path: Path to local country boundary file.
    """

    # constants
    _STAC_CATALOG_URL: ClassVar[str] = "https://stac.overturemaps.org/catalog.json"
    _RELEASE_CALENDAR_URL: ClassVar[str] = "https://docs.overturemaps.org/release-calendar"
    _DEFAULT_BASE_URL: ClassVar[str] = (
        "s3://overturemaps-us-west-2/release/{release}/theme=places/*/*"
    )

    release: Optional[str] = "latest"  # "latest" triggers auto-detection

    base_url: Optional[str] = _DEFAULT_BASE_URL

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

        self.logger = config.get_logger(self.__class__.__name__)

        try:
            self.country = pycountry.countries.lookup(self.country).alpha_2
        except LookupError:
            raise ValueError(f"Invalid country code provided: {self.country}")

        if self.release is None or self.release == "latest":
            try:
                self.release = self.get_latest_release()
                self.logger.info(f"Resolved latest Overture release: {self.release}")
            except RuntimeError as e:
                raise RuntimeError(
                    f"Could not resolve the latest Overture release from the STAC catalog ({e}). "
                    f"Pass an explicit release= (e.g., release='2026-11-18.0'); "
                    f"see {self._RELEASE_CALENDAR_URL}."
                ) from e
        else:
            normalized_release = self._normalize_release(self.release)
            if not normalized_release:
                raise ValueError(
                    "release must not be empty. Pass an explicit release "
                    "(e.g., release='2026-11-18.0') or 'latest'."
                )
            self.release = normalized_release
            if self.base_url == self._DEFAULT_BASE_URL:
                self._validate_release()

        self.base_url = self.base_url.format(release=self.release)

        self.connection = self._set_connection()

    @staticmethod
    def _normalize_release(release: Any) -> str:
        """Normalize a release tag by trimming whitespace and trailing slashes."""
        if not isinstance(release, str):
            return ""
        return release.strip().rstrip("/")

    def _validate_release(self) -> None:
        """Validate an explicit release against the Overture STAC catalog."""
        try:
            available = self.get_available_releases()
        except RuntimeError as e:
            self.logger.warning(
                f"Could not validate Overture release '{self.release}' against STAC catalog ({e}); "
                f"proceeding with '{self.release}'"
            )
            return

        if self.release not in available:
            raise ValueError(
                f"Overture release '{self.release}' is not listed in the STAC catalog. "
                f"Available releases: {', '.join(available)}. "
                f"Overture keeps releases for only ~60 days; see {self._RELEASE_CALENDAR_URL}."
            )

    @classmethod
    def _fetch_stac_catalog(cls, timeout: int = 20) -> Dict[str, Any]:
        """Fetch and parse the root Overture STAC catalog JSON."""
        try:
            response = requests.get(cls._STAC_CATALOG_URL, timeout=timeout)
            response.raise_for_status()
        except requests.RequestException as e:
            raise RuntimeError(f"Failed to fetch Overture STAC catalog: {e}") from e

        try:
            catalog = response.json()
        except ValueError as e:
            raise RuntimeError(f"Failed to parse Overture STAC catalog: {e}") from e

        if not isinstance(catalog, dict):
            raise RuntimeError("Overture STAC catalog did not return a JSON object")

        return catalog

    @classmethod
    def get_latest_release(cls, timeout: int = 20) -> str:
        """
        Query the Overture STAC catalog for the most recent release.

        Args:
            timeout: Request timeout in seconds.

        Returns:
            The latest release string, e.g. "2026-11-18.0".

        Raises:
            RuntimeError: If the catalog can't be fetched, parsed, or does
                not contain a "latest" release entry.
        """
        catalog = cls._fetch_stac_catalog(timeout=timeout)
        normalized = cls._normalize_release(catalog.get("latest"))
        if not normalized:
            raise RuntimeError(
                "Overture STAC catalog did not return a 'latest' release"
            )

        return normalized

    @classmethod
    def get_available_releases(cls, timeout: int = 20) -> List[str]:
        """
        Query the Overture STAC catalog for all currently available releases.

        Args:
            timeout: Request timeout in seconds.

        Returns:
            Release strings sorted newest-first, e.g. ``["2026-11-18.0", "2026-10-21.0"]``.

        Raises:
            RuntimeError: If the catalog can't be fetched, parsed, or contains
                no available releases.
        """
        catalog = cls._fetch_stac_catalog(timeout=timeout)
        releases = set()

        links = catalog.get("links")
        if isinstance(links, list):
            for link in links:
                if not isinstance(link, dict) or link.get("rel") != "child":
                    continue
                href = link.get("href")
                if not isinstance(href, str):
                    continue
                segments = [s for s in urlparse(href).path.split("/") if s and s != "."]
                if not segments:
                    continue
                if segments[-1].endswith(".json"):
                    raw_candidate = segments[-2] if len(segments) >= 2 else ""
                else:
                    raw_candidate = segments[-1]
                candidate = cls._normalize_release(raw_candidate)
                if candidate and candidate != ".." and not candidate.endswith(".json"):
                    releases.add(candidate)

        normalized_latest = cls._normalize_release(catalog.get("latest"))
        if normalized_latest:
            releases.add(normalized_latest)

        if not releases:
            raise RuntimeError(
                "Overture STAC catalog did not list any available releases"
            )

        return sorted(releases, reverse=True)

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
