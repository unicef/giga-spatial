import pandas as pd
import geopandas as gpd
import mercantile
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry
from shapely.strtree import STRtree
from shapely import Point
import json
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Union, Iterable, Optional, Tuple, ClassVar
import pycountry

from gigaspatial.core.io.data_store import DataStore
from gigaspatial.core.io.local_data_store import LocalDataStore
from gigaspatial.config import config


class MercatorTiles(BaseModel):
    """
    Representation of a collection of Web Mercator tiles as quadkeys.

    Provides utility methods to create, filter, and manipulate Web Mercator
    grids for spatial analysis. Handles conversion between coordinate pairs,
    geometries, and quadkey strings.

    Attributes:
        zoom_level: Web Mercator zoom level (0-20).
        quadkeys: List of quadkey strings.
        data_store: Storage interface for I/O operations.
        logger: Class-level logger.
    """

    zoom_level: int = Field(..., ge=0, le=20)
    quadkeys: List[str] = Field(default_factory=list)
    data_store: DataStore = Field(default_factory=LocalDataStore, exclude=True)
    logger: ClassVar = config.get_logger("MercatorTiles")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def from_quadkeys(cls, quadkeys: Iterable[Union[str, int]]) -> "MercatorTiles":
        """
        Create MercatorTiles from a list of quadkeys.

        Args:
            quadkeys: Iterable of quadkey strings or integers.

        Returns:
            A new MercatorTiles instance.
        """
        quadkeys = list(set(str(q) for q in quadkeys))
        if not quadkeys:
            cls.logger.warning("No quadkeys provided to from_quadkeys.")
            return cls(zoom_level=0, quadkeys=[])
        cls.logger.info(
            f"Initializing MercatorTiles from {len(quadkeys)} provided quadkeys."
        )
        return cls(zoom_level=len(quadkeys[0]), quadkeys=quadkeys)

    @classmethod
    def from_bounds(
        cls, xmin: float, ymin: float, xmax: float, ymax: float, zoom_level: int
    ) -> "MercatorTiles":
        """
        Create MercatorTiles covering the specified geographic bounding box.

        Args:
            xmin: Minimum longitude.
            ymin: Minimum latitude.
            xmax: Maximum longitude.
            ymax: Maximum latitude.
            zoom_level: Web Mercator zoom level.

        Returns:
            A new MercatorTiles instance covering the bounds.
        """
        cls.logger.info(
            f"Creating MercatorTiles from bounds: ({xmin}, {ymin}, {xmax}, {ymax}) at zoom level: {zoom_level}"
        )
        return cls(
            zoom_level=zoom_level,
            quadkeys=[
                mercantile.quadkey(tile)
                for tile in mercantile.tiles(xmin, ymin, xmax, ymax, zoom_level)
            ],
        )

    @classmethod
    def from_spatial(
        cls,
        source: Union[
            BaseGeometry,
            gpd.GeoDataFrame,
            List[Union[Point, Tuple[float, float]]],
        ],
        zoom_level: int,
        predicate: str = "intersects",
        **kwargs,
    ) -> "MercatorTiles":
        """
        Factory method to create MercatorTiles from various spatial sources.

        Args:
            source: Spatial data source (Geometry, GDF, or points).
            zoom_level: Web Mercator zoom level.
            predicate: Spatial predicate for containment ('intersects', 'within').
            **kwargs: Forwarded to specific factory methods.

        Returns:
            A new MercatorTiles instance.
        """
        cls.logger.info(
            f"Creating MercatorTiles from spatial source (type: {type(source)}) at zoom level: {zoom_level} with predicate: {predicate}"
        )
        if isinstance(source, gpd.GeoDataFrame):
            if source.crs != "EPSG:4326":
                source = source.to_crs("EPSG:4326")
            source = source.geometry.unary_union

        if isinstance(source, BaseGeometry):
            return cls.from_geometry(
                geometry=source, zoom_level=zoom_level, predicate=predicate, **kwargs
            )
        elif isinstance(source, Iterable) and all(
            isinstance(pt, Point) or len(pt) == 2 for pt in source
        ):
            return cls.from_points(geometry=source, zoom_level=zoom_level, **kwargs)
        else:
            raise

    @classmethod
    def from_geometry(
        cls,
        geometry: BaseGeometry,
        zoom_level: int,
        predicate: str = "intersects",
        **kwargs,
    ) -> "MercatorTiles":
        """
        Create MercatorTiles from a Shapely geometry.

        Args:
            geometry: Input geometry.
            zoom_level: Web Mercator zoom level.
            predicate: Spatial join predicate.
            **kwargs: Additional metadata parameters.

        Returns:
            MercatorTiles instance covering the geometry.
        """
        cls.logger.info(
            f"Creating MercatorTiles from geometry (bounds: {geometry.bounds}) at zoom level: {zoom_level} with predicate: {predicate}"
        )
        tiles = list(mercantile.tiles(*geometry.bounds, zoom_level))
        quadkeys_boxes = [
            (mercantile.quadkey(t), box(*mercantile.bounds(t))) for t in tiles
        ]
        quadkeys, boxes = zip(*quadkeys_boxes) if quadkeys_boxes else ([], [])

        if not boxes:
            cls.logger.warning(
                "No boxes generated from geometry bounds. Returning empty MercatorTiles."
            )
            return MercatorTiles(zoom_level=zoom_level, quadkeys=[])

        s = STRtree(boxes)
        result_indices = s.query(geometry, predicate=predicate)
        filtered_quadkeys = [quadkeys[i] for i in result_indices]
        cls.logger.info(
            f"Filtered down to {len(filtered_quadkeys)} quadkeys using spatial predicate."
        )
        return cls(zoom_level=zoom_level, quadkeys=filtered_quadkeys, **kwargs)

    @classmethod
    def from_points(
        cls, points: List[Union[Point, Tuple[float, float]]], zoom_level: int, **kwargs
    ) -> "MercatorTiles":
        """Creates a MercatorTiles collection from a list of points.

        Args:
            points: List of points as Shapely Points or (lon, lat) tuples.
            zoom_level: Web Mercator zoom level (0-20).
            **kwargs: Additional metadata parameters.

        Returns:
            A new MercatorTiles instance containing tiles for all points.
        """
        cls.logger.info(
            f"Creating MercatorTiles from {len(points)} points at zoom level: {zoom_level}"
        )
        quadkeys = set(cls.get_quadkeys_from_points(points, zoom_level))
        cls.logger.info(f"Generated {len(quadkeys)} unique quadkeys from points.")
        return cls(zoom_level=zoom_level, quadkeys=list(quadkeys), **kwargs)

    @classmethod
    def from_json(
        cls, data_store: DataStore, file: Union[str, Path], **kwargs
    ) -> "MercatorTiles":
        """
        Load MercatorTiles from a JSON file in storage.

        Args:
            data_store: DataStore instance for the file.
            file: Path to the JSON file.
            **kwargs: Metadata overrides.

        Returns:
            MercatorTiles instance loaded from file.
        """
        cls.logger.info(
            f"Loading MercatorTiles from JSON file: {file} using data store: {type(data_store).__name__}"
        )
        with data_store.open(str(file), "r") as f:
            data = json.load(f)
            if isinstance(data, list):  # If file contains only quadkeys
                data = {
                    "zoom_level": len(data[0]) if data else 0,
                    "quadkeys": data,
                    **kwargs,
                }
            else:
                data.update(kwargs)
            instance = cls(**data)
            instance.data_store = data_store
            cls.logger.info(
                f"Successfully loaded {len(instance.quadkeys)} quadkeys from JSON file."
            )
            return instance

    def filter_quadkeys(self, quadkeys: Iterable[str]) -> "MercatorTiles":
        """Filters the current collection against a provided set of quadkeys.

        Args:
            quadkeys: An iterable of quadkey strings to keep.

        Returns:
            A new MercatorTiles instance containing only the intersecting tiles.
        """
        original_count = len(self.quadkeys)
        incoming_count = len(
            list(quadkeys)
        )  # Convert to list to get length if it's an iterator

        self.logger.info(
            f"Filtering {original_count} quadkeys with an incoming set of {incoming_count} quadkeys."
        )
        filtered_quadkeys = list(set(self.quadkeys) & set(quadkeys))
        self.logger.info(f"Resulting in {len(filtered_quadkeys)} filtered quadkeys.")
        return MercatorTiles(
            zoom_level=self.zoom_level,
            quadkeys=filtered_quadkeys,
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Converts the tile collection to a pandas DataFrame.

        Returns:
            A DataFrame with 'quadkey', 'latitude', and 'longitude' columns.
        """
        self.logger.info(
            f"Converting {len(self.quadkeys)} quadkeys to pandas DataFrame."
        )
        if not self.quadkeys:
            self.logger.warning(
                "No quadkeys to convert to DataFrame. Returning empty DataFrame."
            )
            return pd.DataFrame(columns=["quadkey", "latitude", "longitude"])
        tiles_data = [mercantile.quadkey_to_tile(q) for q in self.quadkeys]
        bounds_data = [mercantile.bounds(tile) for tile in tiles_data]

        centroids = [
            (
                (bounds.south + bounds.north) / 2,  # latitude
                (bounds.west + bounds.east) / 2,  # longitude
            )
            for bounds in bounds_data
        ]

        self.logger.info(f"Successfully converted to DataFrame.")

        return pd.DataFrame(
            {
                "quadkey": self.quadkeys,
                "latitude": [c[0] for c in centroids],
                "longitude": [c[1] for c in centroids],
            }
        )

    def to_geoms(self) -> List[box]:
        """Converts quadkeys into a list of Shapely box (polygon) geometries.

        Returns:
            A list of boxes representing the tile boundaries.
        """
        self.logger.info(
            f"Converting {len(self.quadkeys)} quadkeys to shapely box geometries."
        )
        return [
            box(*mercantile.bounds(mercantile.quadkey_to_tile(q)))
            for q in self.quadkeys
        ]

    def to_geodataframe(self) -> gpd.GeoDataFrame:
        """Converts the tile collection to a GeoPandas GeoDataFrame.

        Returns:
            A GeoDataFrame with 'quadkey' and 'geometry' columns.
        """
        return gpd.GeoDataFrame(
            {"quadkey": self.quadkeys, "geometry": self.to_geoms()}, crs="EPSG:4326"
        )

    @staticmethod
    def get_quadkeys_from_points(
        points: List[Union[Point, Tuple[float, float]]], zoom_level: int
    ) -> List[str]:
        """Get list of quadkeys for the provided points at specified zoom level.

        Args:
            points: List of points as either shapely Points or (lon, lat) tuples
            zoom_level: Zoom level for the quadkeys

        Returns:
            List of quadkey strings
        """
        quadkeys = [
            (
                mercantile.quadkey(mercantile.tile(p.x, p.y, zoom_level))
                if isinstance(p, Point)
                else mercantile.quadkey(mercantile.tile(p[1], p[0], zoom_level))
            )
            for p in points
        ]
        return quadkeys

    def save(self, file: Union[str, Path], format: str = "json") -> None:
        """Saves the MercatorTiles collection to persistent storage.

        Args:
            file: The destination file path.
            format: The output format. Supported: 'json', 'parquet', 'geojson'.
                Defaults to 'json'.

        Raises:
            ValueError: If an unsupported format is provided.
        """
        with self.data_store.open(str(file), "wb" if format == "parquet" else "w") as f:
            if format == "parquet":
                self.to_geodataframe().to_parquet(f, index=False)
            elif format == "geojson":
                f.write(self.to_geodataframe().to_json(drop_id=True))
            elif format == "json":
                json.dump(self.quadkeys, f)
            else:
                raise ValueError(f"Unsupported format: {format}")

    def __len__(self) -> int:
        return len(self.quadkeys)


class CountryMercatorTiles(MercatorTiles):
    """
    MercatorTiles specialized for country-level operations.

    Extends MercatorTiles to work specifically with country boundaries retrieved
    from the Giga administrative boundary dataset.

    Note:
        Instances should be created using the `create()` factory method.

    Attributes:
        country: ISO 3166-1 alpha-3 country code.
    """

    country: str = Field(..., exclude=True)

    def __init__(self, *args, **kwargs):
        raise TypeError(
            "CountryMercatorTiles cannot be instantiated directly. "
            "Use CountryMercatorTiles.create() instead."
        )

    @classmethod
    def create(
        self,
        country: str,
        zoom_level: int,
        predicate: str = "intersects",
        data_store: Optional[DataStore] = None,
        country_geom_path: Optional[Union[str, Path]] = None,
    ) -> "CountryMercatorTiles":
        """Factory method to create MercatorTiles for a specific country's boundary.

        Args:
            country: ISO country code (3-letter alpha-3) or name.
            zoom_level: Target Web Mercator zoom level (0-20).
            predicate: Spatial join predicate for the boundary.
                Defaults to 'intersects'.
            data_store: Optional storage interface for boundary lookup.
            country_geom_path: Optional path override for the boundary file.

        Returns:
            A new CountryMercatorTiles instance fully populated for the country.
        """
        from gigaspatial.handlers.boundaries import AdminBoundaries

        instance = super().__new__(cls)
        super(CountryMercatorTiles, instance).__init__(
            zoom_level=zoom_level,
            quadkeys=[],
            data_store=data_store or LocalDataStore(),
            country=pycountry.countries.lookup(country).alpha_3,
        )

        cls.logger.info(
            f"Initializing Mercator zones for country: {country} at zoom level {zoom_level}"
        )

        country_geom = AdminBoundaries.create(
            country_code=country,
            data_store=data_store,
            path=country_geom_path,
        ).to_geoms()[0]

        tiles = MercatorTiles.from_geometry(country_geom, zoom_level, predicate)

        instance.quadkeys = tiles.quadkeys
        return instance
