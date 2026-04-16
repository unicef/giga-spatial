import pandas as pd
import geopandas as gpd
import h3
from shapely.geometry import Polygon, Point, shape
from shapely.geometry.base import BaseGeometry
from shapely.strtree import STRtree
import json
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Union, Iterable, Optional, Tuple, ClassVar, Literal
import pycountry

from gigaspatial.core.io.data_store import DataStore
from gigaspatial.core.io.local_data_store import LocalDataStore
from gigaspatial.config import config


class H3Hexagons(BaseModel):
    """
    Representation of a collection of H3 hexagons at a specific resolution.

    Provides utility methods to create, filter, manipulate, and export H3
    hexagonal grids for spatial analysis. Handles conversion between
    coordinate pairs, geometries, and H3 cell IDs.

    Attributes:
        resolution: H3 resolution level (0-15).
        hexagons: List of H3 cell ID strings.
        data_store: Storage interface for I/O operations.
        logger: Class-level logger.
    """

    resolution: int = Field(..., ge=0, le=15)
    hexagons: List[str] = Field(default_factory=list)
    data_store: DataStore = Field(default_factory=LocalDataStore, exclude=True)
    logger: ClassVar = config.get_logger("H3Hexagons")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def from_hexagons(cls, hexagons: List[str]) -> "H3Hexagons":
        """
        Create H3Hexagons from a list of H3 cell IDs.

        Args:
            hexagons: List of H3 cell hashes.

        Returns:
            A new H3Hexagons instance.
        """
        if not hexagons:
            cls.logger.warning("No hexagons provided to from_hexagons.")
            return cls(resolution=0, hexagons=[])

        cls.logger.info(
            f"Initializing H3Hexagons from {len(hexagons)} provided hexagons."
        )
        # Get resolution from first hexagon
        resolution = h3.get_resolution(hexagons[0])
        return cls(resolution=resolution, hexagons=list(set(hexagons)))

    @classmethod
    def from_bounds(
        cls, xmin: float, ymin: float, xmax: float, ymax: float, resolution: int
    ) -> "H3Hexagons":
        """
        Create H3Hexagons covering the specified geographic bounding box.

        Args:
            xmin: Minimum longitude.
            ymin: Minimum latitude.
            xmax: Maximum longitude.
            ymax: Maximum latitude.
            resolution: H3 resolution level.

        Returns:
            A new H3Hexagons instance covering the bounds.
        """
        cls.logger.info(
            f"Creating H3Hexagons from bounds: ({xmin}, {ymin}, {xmax}, {ymax}) at resolution: {resolution}"
        )

        # Create a LatLong bounding box polygon
        latlong_bbox_coords = [
            [ymin, xmin],
            [ymax, xmin],
            [ymax, xmax],
            [ymin, xmax],
            [ymin, xmin],
        ]

        # Get H3 cells that intersect with the bounding box
        poly = h3.LatLngPoly(latlong_bbox_coords)
        hexagons = h3.h3shape_to_cells(poly, res=resolution)

        return cls(resolution=resolution, hexagons=list(hexagons))

    @classmethod
    def from_spatial(
        cls,
        source: Union[
            BaseGeometry,
            gpd.GeoDataFrame,
            List[Union[Point, Tuple[float, float]]],
        ],
        resolution: int,
        contain: Literal["center", "full", "overlap", "bbox_overlap"] = "overlap",
        **kwargs,
    ) -> "H3Hexagons":
        """
        Factory method to create H3Hexagons from various spatial sources.

        Handles GeoDataFrames, Shapely geometries, and lists of coordinate points.

        Args:
            source: Spatial data source.
            resolution: Target H3 resolution.
            contain: H3 spatial containment logic.
            **kwargs: Forwarded to specific factory methods (e.g., from_geometry).

        Returns:
            A new H3Hexagons instance.
        """
        cls.logger.info(
            f"Creating H3Hexagons from spatial source (type: {type(source)}) at resolution: {resolution} with predicate: {contain}"
        )
        if isinstance(source, gpd.GeoDataFrame):
            if source.crs != "EPSG:4326":
                source = source.to_crs("EPSG:4326")

            is_point_series = source.geometry.geom_type == "Point"
            all_are_points = is_point_series.all()

            if all_are_points:
                source = source.geometry.to_list()
            else:
                source = source.geometry.unary_union

        if isinstance(source, BaseGeometry):
            return cls.from_geometry(
                geometry=source, resolution=resolution, contain=contain, **kwargs
            )
        elif isinstance(source, Iterable) and all(
            isinstance(pt, Point) or len(pt) == 2 for pt in source
        ):
            return cls.from_points(points=source, resolution=resolution, **kwargs)
        else:
            raise ValueError("Unsupported source type for H3Hexagons.from_spatial")

    @classmethod
    def from_geometry(
        cls,
        geometry: BaseGeometry,
        resolution: int,
        contain: Literal["center", "full", "overlap", "bbox_overlap"] = "overlap",
        **kwargs,
    ) -> "H3Hexagons":
        """
        Create H3Hexagons from a Shapely geometry.

        Args:
            geometry: Input geometry (Polygon, MultiPolygon, etc.).
            resolution: H3 resolution level.
            contain: Containment logic for H3 cell generation.
            **kwargs: Additional metadata parameters.

        Returns:
            H3Hexagons instance covering the geometry.
        """
        cls.logger.info(
            f"Creating H3Hexagons from geometry (bounds: {geometry.bounds}) at resolution: {resolution} with predicate: {contain}"
        )

        if isinstance(geometry, Point):
            return cls.from_points([geometry])

        # Convert shapely geometry to GeoJSON-like format
        if hasattr(geometry, "__geo_interface__"):
            geojson_geom = geometry.__geo_interface__
        else:
            # Fallback for complex geometries
            import json
            from shapely.geometry import mapping

            geojson_geom = mapping(geometry)

        h3_geom = h3.geo_to_h3shape(geojson_geom)

        hexagons = h3.h3shape_to_cells_experimental(
            h3_geom, resolution, contain=contain
        )

        cls.logger.info(
            f"Generated {len(hexagons)} hexagons using `{contain}` spatial predicate."
        )
        return cls(resolution=resolution, hexagons=list(hexagons), **kwargs)

    @classmethod
    def from_points(
        cls, points: List[Union[Point, Tuple[float, float]]], resolution: int, **kwargs
    ) -> "H3Hexagons":
        """
        Create H3Hexagons from a list of point coordinates.

        Args:
            points: List of points as Shapely Points or (lon, lat) tuples.
            resolution: H3 resolution level.
            **kwargs: Additional metadata parameters.

        Returns:
            H3Hexagons instance containing cells for all points.
        """
        cls.logger.info(
            f"Creating H3Hexagons from {len(points)} points at resolution: {resolution}"
        )
        hexagons = set(cls.get_hexagons_from_points(points, resolution))
        cls.logger.info(f"Generated {len(hexagons)} unique hexagons from points.")
        return cls(resolution=resolution, hexagons=list(hexagons), **kwargs)

    @classmethod
    def from_json(
        cls, data_store: DataStore, file: Union[str, Path], **kwargs
    ) -> "H3Hexagons":
        """
        Load H3Hexagons from a JSON file in storage.

        Args:
            data_store: DataStore instance for the file.
            file: Path to the JSON file.
            **kwargs: Metadata overrides.

        Returns:
            H3Hexagons instance loaded from file.
        """
        cls.logger.info(
            f"Loading H3Hexagons from JSON file: {file} using data store: {type(data_store).__name__}"
        )
        with data_store.open(str(file), "r") as f:
            data = json.load(f)
            if isinstance(data, list):  # If file contains only hexagon IDs
                # Get resolution from first hexagon if available
                resolution = h3.get_resolution(data[0]) if data else 0
                data = {
                    "resolution": resolution,
                    "hexagons": data,
                    **kwargs,
                }
            else:
                data.update(kwargs)
            instance = cls(**data)
            instance.data_store = data_store
            cls.logger.info(
                f"Successfully loaded {len(instance.hexagons)} hexagons from JSON file."
            )
            return instance

    @property
    def average_hexagon_area(self) -> float:
        """Calculates the average area of hexagons at the current resolution.

        Returns:
            Average area in square kilometers.
        """
        return h3.average_hexagon_area(self.resolution)

    @property
    def average_hexagon_edge_length(self) -> float:
        """Calculates the average edge length of hexagons at the current resolution.

        Returns:
            Average edge length in kilometers.
        """
        return h3.average_hexagon_edge_length(self.resolution)

    def filter_hexagons(self, hexagons: Iterable[str]) -> "H3Hexagons":
        """Filters the current collection against a provided set of hexagon IDs.

        Args:
            hexagons: An iterable of H3 cell ID strings to keep.

        Returns:
            A new H3Hexagons instance containing only the intersecting cells.
        """
        original_count = len(self.hexagons)
        incoming_count = len(
            list(hexagons)
        )  # Convert to list to get length if it's an iterator

        self.logger.info(
            f"Filtering {original_count} hexagons with an incoming set of {incoming_count} hexagons."
        )
        filtered_hexagons = list(set(self.hexagons) & set(hexagons))
        self.logger.info(f"Resulting in {len(filtered_hexagons)} filtered hexagons.")
        return H3Hexagons(
            resolution=self.resolution,
            hexagons=filtered_hexagons,
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Converts the hexagon collection to a pandas DataFrame.

        Returns:
            A DataFrame with 'hexagon', 'latitude', and 'longitude' columns.
        """
        self.logger.info(
            f"Converting {len(self.hexagons)} hexagons to pandas DataFrame."
        )
        if not self.hexagons:
            self.logger.warning(
                "No hexagons to convert to DataFrame. Returning empty DataFrame."
            )
            return pd.DataFrame(columns=["hexagon", "latitude", "longitude"])

        centroids = [h3.cell_to_latlng(hex_id) for hex_id in self.hexagons]

        self.logger.info(f"Successfully converted to DataFrame.")

        return pd.DataFrame(
            {
                "hexagon": self.hexagons,
                "latitude": [c[0] for c in centroids],
                "longitude": [c[1] for c in centroids],
            }
        )

    def to_geoms(self) -> List[Polygon]:
        """Converts hexagons to a list of Shapely Polygon geometries.

        Returns:
            A list of Polygons representing the cell boundaries.
        """
        self.logger.info(
            f"Converting {len(self.hexagons)} hexagons to shapely Polygon geometries."
        )
        return [shape(h3.cells_to_geo([hex_id])) for hex_id in self.hexagons]

    def to_geodataframe(self) -> gpd.GeoDataFrame:
        """Converts the hexagon collection to a GeoPandas GeoDataFrame.

        Returns:
            A GeoDataFrame with 'h3' and 'geometry' columns.
        """
        return gpd.GeoDataFrame(
            {"h3": self.hexagons, "geometry": self.to_geoms()}, crs="EPSG:4326"
        )

    @staticmethod
    def get_hexagons_from_points(
        points: List[Union[Point, Tuple[float, float]]], resolution: int
    ) -> List[str]:
        """Get list of H3 hexagon IDs for the provided points at specified resolution.

        Args:
            points: List of points as either shapely Points or (lon, lat) tuples
            resolution: H3 resolution level

        Returns:
            List of H3 hexagon ID strings
        """
        hexagons = []
        for p in points:
            if isinstance(p, Point):
                # Shapely Point has x=lon, y=lat
                hex_id = h3.latlng_to_cell(p.y, p.x, resolution)
            else:
                # Assume tuple is (lon, lat) - convert to (lat, lon) for h3
                hex_id = h3.latlng_to_cell(p[1], p[0], resolution)
            hexagons.append(hex_id)
        return hexagons

    def get_neighbors(self, k: int = 1) -> "H3Hexagons":
        """Calculates k-ring neighbors for all hexagons in the collection.

        Args:
            k: The distance of neighbors to retrieve (1 for immediate neighbors).
                Defaults to 1.

        Returns:
            A new H3Hexagons instance containing the neighbors.
        """
        self.logger.info(
            f"Getting k-ring neighbors (k={k}) for {len(self.hexagons)} hexagons."
        )

        all_neighbors = set()
        for hex_id in self.hexagons:
            neighbors = h3.grid_ring(hex_id, k)
            all_neighbors.update(neighbors)

        self.logger.info(
            f"Found {len(all_neighbors)} total hexagons including neighbors."
        )
        return H3Hexagons(resolution=self.resolution, hexagons=list(all_neighbors))

    def get_compact_representation(self) -> "H3Hexagons":
        """Merges adjacent hexagons into parent cells where possible (compacting).

        Returns:
            A new H3Hexagons instance in compacted form. Note that this may
            result in a set containing cells of multiple resolutions.
        """
        self.logger.info(f"Compacting {len(self.hexagons)} hexagons.")

        # Convert to set for h3.compact
        hex_set = set(self.hexagons)
        compacted = h3.compact_cells(hex_set)

        self.logger.info(f"Compacted to {len(compacted)} hexagons.")

        # Note: compacted representation may have mixed resolutions
        # We'll keep the original resolution as the "target" resolution
        return H3Hexagons(resolution=self.resolution, hexagons=list(compacted))

    def get_children(self, target_resolution: int) -> "H3Hexagons":
        """Generates all child hexagons at a higher resolution.

        Args:
            target_resolution: The target H3 resolution (must be > current).

        Returns:
            A new H3Hexagons instance with children at the target resolution.

        Raises:
            ValueError: If `target_resolution` is not higher than current resolution.
        """
        if target_resolution <= self.resolution:
            raise ValueError("Target resolution must be higher than current resolution")

        self.logger.info(
            f"Getting children at resolution {target_resolution} for {len(self.hexagons)} hexagons."
        )

        all_children = []
        for hex_id in self.hexagons:
            children = h3.cell_to_children(hex_id, target_resolution)
            all_children.extend(children)

        self.logger.info(f"Generated {len(all_children)} children hexagons.")
        return H3Hexagons(resolution=target_resolution, hexagons=all_children)

    def get_parents(self, target_resolution: int) -> "H3Hexagons":
        """Retrieves parent hexagons at a lower resolution.

        Args:
            target_resolution: The target H3 resolution (must be < current).

        Returns:
            A new H3Hexagons instance with parents at the target resolution.

        Raises:
            ValueError: If `target_resolution` is not lower than current resolution.
        """
        if target_resolution >= self.resolution:
            raise ValueError("Target resolution must be lower than current resolution")

        self.logger.info(
            f"Getting parents at resolution {target_resolution} for {len(self.hexagons)} hexagons."
        )

        parents = set()
        for hex_id in self.hexagons:
            parent = h3.cell_to_parent(hex_id, target_resolution)
            parents.add(parent)

        self.logger.info(f"Generated {len(parents)} parent hexagons.")
        return H3Hexagons(resolution=target_resolution, hexagons=list(parents))

    def save(self, file: Union[str, Path], format: str = "json") -> None:
        """Saves the H3Hexagons collection to persistent storage.

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
                json.dump(self.hexagons, f)
            else:
                raise ValueError(f"Unsupported format: {format}")

    def __len__(self) -> int:
        return len(self.hexagons)


class CountryH3Hexagons(H3Hexagons):
    """
    H3Hexagons specialized for country-level operations.

    Extends H3Hexagons to work specifically with country boundaries retrieved
    from the Giga administrative boundary dataset.

    Note:
        Instances should be created using the `create()` factory method.

    Attributes:
        country: ISO 3166-1 alpha-3 country code.
    """

    country: str = Field(..., exclude=True)

    def __init__(self, *args, **kwargs):
        raise TypeError(
            "CountryH3Hexagons cannot be instantiated directly. "
            "Use CountryH3Hexagons.create() instead."
        )

    @classmethod
    def create(
        self,
        country: str,
        resolution: int,
        contain: Literal["center", "full", "overlap", "bbox_overlap"] = "overlap",
        data_store: Optional[DataStore] = None,
        country_geom_path: Optional[Union[str, Path]] = None,
    ) -> "CountryH3Hexagons":
        """Factory method to create H3Hexagons for a specific country's boundary.

        Args:
            country: ISO country code (3-letter alpha-3) or name.
            resolution: Target H3 resolution (0-15).
            contain: Containment logic for mapping the geometry to cells.
                Defaults to 'overlap'.
            data_store: Optional storage interface for boundary lookup.
            country_geom_path: Optional path override for the boundary file.

        Returns:
            A new CountryH3Hexagons instance fully populated for the country.
        """
        from gigaspatial.handlers.boundaries import AdminBoundaries

        instance = super().__new__(cls)
        super(CountryH3Hexagons, instance).__init__(
            resolution=resolution,
            hexagons=[],
            data_store=data_store or LocalDataStore(),
            country=pycountry.countries.lookup(country).alpha_3,
        )

        cls.logger.info(
            f"Initializing H3 hexagons for country: {country} at resolution {resolution}"
        )

        country_geom = AdminBoundaries.create(
            country_code=country,
            data_store=data_store,
            path=country_geom_path,
        ).to_geoms()[0]

        hexagons = H3Hexagons.from_geometry(country_geom, resolution, contain=contain)

        instance.hexagons = hexagons.hexagons
        return instance
