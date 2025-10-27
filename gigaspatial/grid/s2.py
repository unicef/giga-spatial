import pandas as pd
import geopandas as gpd
from s2sphere import CellId, Cell, LatLng, RegionCoverer, LatLngRect
from shapely.geometry import Polygon, Point, shape
from shapely.geometry.base import BaseGeometry
import json
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Union, Iterable, Optional, Tuple, ClassVar, Literal
import pycountry
from gigaspatial.core.io.data_store import DataStore
from gigaspatial.core.io.local_data_store import LocalDataStore
from gigaspatial.config import config


class S2Cells(BaseModel):
    """S2Cells class for generating and managing Google S2 cell grids.

    S2 uses levels 0-30, where higher levels represent finer resolution.
    Level 0 covers the largest area and level 30 the smallest.
    """

    level: int = Field(..., ge=0, le=30)
    cells: List[int] = Field(default_factory=list)  # S2 cell IDs as integers
    data_store: DataStore = Field(default_factory=LocalDataStore, exclude=True)
    logger: ClassVar = config.get_logger("S2Cells")

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def from_cells(cls, cells: List[Union[int, str]]):
        """Create S2Cells from list of S2 cell IDs (integers or tokens)."""
        if not cells:
            cls.logger.warning("No cells provided to from_cells.")
            return cls(level=0, cells=[])

        cls.logger.info(f"Initializing S2Cells from {len(cells)} provided cells.")

        # Convert tokens to integers if needed
        cell_ids = []
        for cell in cells:
            if isinstance(cell, str):
                cell_ids.append(CellId.from_token(cell).id())
            else:
                cell_ids.append(cell)

        # Get level from first cell
        level = CellId(cell_ids[0]).level()
        return cls(level=level, cells=list(set(cell_ids)))

    @classmethod
    def from_bounds(
        cls,
        xmin: float,
        ymin: float,
        xmax: float,
        ymax: float,
        level: int,
        max_cells: int = 100,
    ):
        """Create S2Cells from boundary coordinates.

        Args:
            xmin, ymin, xmax, ymax: Bounding box coordinates in degrees
            level: S2 level (0-30)
            max_cells: Maximum number of cells to generate
        """
        cls.logger.info(
            f"Creating S2Cells from bounds: ({xmin}, {ymin}, {xmax}, {ymax}) at level: {level}"
        )

        # Create a LatLngRect for the bounding box
        rect = LatLngRect(
            LatLng.from_degrees(ymin, xmin), LatLng.from_degrees(ymax, xmax)
        )

        # Use RegionCoverer to get cells
        coverer = RegionCoverer()
        coverer.min_level = level
        coverer.max_level = level
        coverer.max_cells = max_cells

        covering = coverer.get_covering(rect)
        cells = [cell.id() for cell in covering]

        cls.logger.info(f"Generated {len(cells)} cells from bounds.")
        return cls(level=level, cells=cells)

    @classmethod
    def from_spatial(
        cls,
        source: Union[
            BaseGeometry,
            gpd.GeoDataFrame,
            List[Union[Point, Tuple[float, float]]],
        ],
        level: int,
        max_cells: int = 1000,
        **kwargs,
    ):
        """Create S2Cells from various spatial sources."""
        cls.logger.info(
            f"Creating S2Cells from spatial source (type: {type(source)}) at level: {level}"
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
                geometry=source, level=level, max_cells=max_cells, **kwargs
            )
        elif isinstance(source, Iterable) and all(
            isinstance(pt, Point) or len(pt) == 2 for pt in source
        ):
            return cls.from_points(points=source, level=level, **kwargs)
        else:
            raise ValueError("Unsupported source type for S2Cells.from_spatial")

    @classmethod
    def from_geometry(
        cls,
        geometry: BaseGeometry,
        level: int,
        max_cells: int = 1000,
        **kwargs,
    ):
        """Create S2Cells from a geometry.

        Args:
            geometry: Shapely geometry
            level: S2 level (0-30)
            max_cells: Maximum number of cells to generate
        """
        cls.logger.info(
            f"Creating S2Cells from geometry (bounds: {geometry.bounds}) at level: {level}"
        )

        if isinstance(geometry, Point):
            return cls.from_points([geometry], level)

        # For polygons and other shapes, use bounding box with RegionCoverer
        # Then filter to actual intersection
        minx, miny, maxx, maxy = geometry.bounds

        rect = LatLngRect(
            LatLng.from_degrees(miny, minx), LatLng.from_degrees(maxy, maxx)
        )

        coverer = RegionCoverer()
        coverer.min_level = level
        coverer.max_level = level
        coverer.max_cells = max_cells

        covering = coverer.get_covering(rect)

        # Filter cells that actually intersect the geometry
        cells = []
        for cell_id in covering:
            cell = Cell(cell_id)
            # Create polygon from cell vertices
            vertices = []
            for i in range(4):
                vertex = cell.get_vertex(i)
                lat_lng = LatLng.from_point(vertex)
                vertices.append((lat_lng.lng().degrees, lat_lng.lat().degrees))
            vertices.append(vertices[0])  # Close the polygon

            cell_polygon = Polygon(vertices)
            if cell_polygon.intersects(geometry):
                cells.append(cell_id.id())

        cls.logger.info(f"Generated {len(cells)} cells from geometry.")
        return cls(level=level, cells=cells, **kwargs)

    @classmethod
    def from_points(
        cls, points: List[Union[Point, Tuple[float, float]]], level: int, **kwargs
    ) -> "S2Cells":
        """Create S2Cells from a list of points or lat-lon pairs."""
        cls.logger.info(f"Creating S2Cells from {len(points)} points at level: {level}")

        cells = set(cls.get_cells_from_points(points, level))
        cls.logger.info(f"Generated {len(cells)} unique cells from points.")
        return cls(level=level, cells=list(cells), **kwargs)

    @classmethod
    def from_json(
        cls, data_store: DataStore, file: Union[str, Path], **kwargs
    ) -> "S2Cells":
        """Load S2Cells from a JSON file."""
        cls.logger.info(
            f"Loading S2Cells from JSON file: {file} using data store: {type(data_store).__name__}"
        )

        with data_store.open(str(file), "r") as f:
            data = json.load(f)

        if isinstance(data, list):  # If file contains only cell IDs
            # Get level from first cell if available
            level = CellId(data[0]).level() if data else 0
            data = {
                "level": level,
                "cells": data,
                **kwargs,
            }
        else:
            data.update(kwargs)

        instance = cls(**data)
        instance.data_store = data_store
        cls.logger.info(
            f"Successfully loaded {len(instance.cells)} cells from JSON file."
        )
        return instance

    @property
    def average_cell_area(self):
        """Average area of cells at this level in square meters."""
        # Approximate area calculation based on S2 geometry
        # Earth surface area is ~510 trillion square meters
        # Each level quadruples the number of cells
        earth_area = 510_000_000_000_000  # m^2
        num_cells_at_level = 6 * (4**self.level)  # 6 faces, each subdivided
        return earth_area / num_cells_at_level

    def filter_cells(self, cells: Iterable[int]) -> "S2Cells":
        """Filter cells by a given set of cell IDs."""
        original_count = len(self.cells)
        incoming_count = len(list(cells))

        self.logger.info(
            f"Filtering {original_count} cells with an incoming set of {incoming_count} cells."
        )

        filtered_cells = list(set(self.cells) & set(cells))
        self.logger.info(f"Resulting in {len(filtered_cells)} filtered cells.")

        return S2Cells(
            level=self.level,
            cells=filtered_cells,
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame with cell ID and centroid coordinates."""
        self.logger.info(f"Converting {len(self.cells)} cells to pandas DataFrame.")

        if not self.cells:
            self.logger.warning(
                "No cells to convert to DataFrame. Returning empty DataFrame."
            )
            return pd.DataFrame(
                columns=["cell_id", "cell_token", "latitude", "longitude"]
            )

        data = []
        for cell_id in self.cells:
            cell = Cell(CellId(cell_id))
            center = LatLng.from_point(cell.get_center())
            data.append(
                {
                    "cell_id": cell_id,
                    "cell_token": CellId(cell_id).to_token(),
                    "latitude": center.lat().degrees,
                    "longitude": center.lng().degrees,
                }
            )

        self.logger.info(f"Successfully converted to DataFrame.")
        return pd.DataFrame(data)

    def to_geoms(self) -> List[Polygon]:
        """Convert cells to shapely Polygon geometries."""
        self.logger.info(
            f"Converting {len(self.cells)} cells to shapely Polygon geometries."
        )

        polygons = []
        for cell_id in self.cells:
            cell = Cell(CellId(cell_id))
            vertices = []
            for i in range(4):
                vertex = cell.get_vertex(i)
                lat_lng = LatLng.from_point(vertex)
                vertices.append((lat_lng.lng().degrees, lat_lng.lat().degrees))
            vertices.append(vertices[0])  # Close the polygon
            polygons.append(Polygon(vertices))

        return polygons

    def to_geodataframe(self) -> gpd.GeoDataFrame:
        """Convert to GeoPandas GeoDataFrame."""
        return gpd.GeoDataFrame(
            {
                "cell_id": self.cells,
                "cell_token": [CellId(c).to_token() for c in self.cells],
                "geometry": self.to_geoms(),
            },
            crs="EPSG:4326",
        )

    @staticmethod
    def get_cells_from_points(
        points: List[Union[Point, Tuple[float, float]]], level: int
    ) -> List[int]:
        """Get list of S2 cell IDs for the provided points at specified level.

        Args:
            points: List of points as either shapely Points or (lon, lat) tuples
            level: S2 level

        Returns:
            List of S2 cell IDs as integers
        """
        cells = []
        for p in points:
            if isinstance(p, Point):
                # Shapely Point has x=lon, y=lat
                lat_lng = LatLng.from_degrees(p.y, p.x)
            else:
                # Assume tuple is (lon, lat)
                lat_lng = LatLng.from_degrees(p[1], p[0])

            cell_id = CellId.from_lat_lng(lat_lng).parent(level)
            cells.append(cell_id.id())

        return cells

    def get_neighbors(self, direct_only: bool = True) -> "S2Cells":
        """Get neighbors of all cells.

        Args:
            direct_only: If True, get only direct edge neighbors (4 per cell).
                        If False, get all 8 neighbors including corners.

        Returns:
            New S2Cells instance with neighbors included
        """
        self.logger.info(
            f"Getting neighbors for {len(self.cells)} cells (direct_only={direct_only})."
        )

        all_neighbors = set()
        for cell_id in self.cells:
            cell = CellId(cell_id)
            # Get edge neighbors
            for i in range(4):
                neighbors = cell.get_edge_neighbors()
                all_neighbors.update([n.id() for n in neighbors])

            if not direct_only:
                # Get corner neighbors
                for i in range(4):
                    vertex_neighbors = cell.get_vertex_neighbors(i)
                    all_neighbors.update([n.id() for n in vertex_neighbors])

        self.logger.info(f"Found {len(all_neighbors)} total cells including neighbors.")

        return S2Cells(level=self.level, cells=list(all_neighbors))

    def get_children(self, target_level: int) -> "S2Cells":
        """Get children cells at higher level.

        Args:
            target_level: Target level (must be higher than current)

        Returns:
            New S2Cells instance with children at target level
        """
        if target_level <= self.level:
            raise ValueError("Target level must be higher than current level")

        self.logger.info(
            f"Getting children at level {target_level} for {len(self.cells)} cells."
        )

        all_children = []
        for cell_id in self.cells:
            cell = CellId(cell_id)
            # Get all children at target level
            child = cell.child_begin(target_level)
            end = cell.child_end(target_level)

            while child != end:
                all_children.append(child.id())
                child = child.next()

        self.logger.info(f"Generated {len(all_children)} children cells.")
        return S2Cells(level=target_level, cells=all_children)

    def get_parents(self, target_level: int) -> "S2Cells":
        """Get parent cells at lower level.

        Args:
            target_level: Target level (must be lower than current)

        Returns:
            New S2Cells instance with parents at target level
        """
        if target_level >= self.level:
            raise ValueError("Target level must be lower than current level")

        self.logger.info(
            f"Getting parents at level {target_level} for {len(self.cells)} cells."
        )

        parents = set()
        for cell_id in self.cells:
            parent = CellId(cell_id).parent(target_level)
            parents.add(parent.id())

        self.logger.info(f"Generated {len(parents)} parent cells.")
        return S2Cells(level=target_level, cells=list(parents))

    def save(self, file: Union[str, Path], format: str = "json") -> None:
        """Save S2Cells to file in specified format."""
        with self.data_store.open(str(file), "wb" if format == "parquet" else "w") as f:
            if format == "parquet":
                self.to_geodataframe().to_parquet(f, index=False)
            elif format == "geojson":
                f.write(self.to_geodataframe().to_json(drop_id=True))
            elif format == "json":
                json.dump(self.cells, f)
            else:
                raise ValueError(f"Unsupported format: {format}")

    def __len__(self) -> int:
        return len(self.cells)


class CountryS2Cells(S2Cells):
    """S2Cells specialized for country-level operations.

    This class extends S2Cells to work specifically with country boundaries.
    It can only be instantiated through the create() classmethod.
    """

    country: str = Field(..., exclude=True)

    def __init__(self, *args, **kwargs):
        raise TypeError(
            "CountryS2Cells cannot be instantiated directly. "
            "Use CountryS2Cells.create() instead."
        )

    @classmethod
    def create(
        cls,
        country: str,
        level: int,
        max_cells: int = 1000,
        data_store: Optional[DataStore] = None,
        country_geom_path: Optional[Union[str, Path]] = None,
    ):
        """Create CountryS2Cells for a specific country."""
        from gigaspatial.handlers.boundaries import AdminBoundaries

        instance = super().__new__(cls)
        super(CountryS2Cells, instance).__init__(
            level=level,
            cells=[],
            data_store=data_store or LocalDataStore(),
            country=pycountry.countries.lookup(country).alpha_3,
        )

        cls.logger.info(
            f"Initializing S2 cells for country: {country} at level {level}"
        )

        country_geom = (
            AdminBoundaries.create(
                country_code=country,
                data_store=data_store,
                path=country_geom_path,
            )
            .boundaries[0]
            .geometry
        )

        cells = S2Cells.from_geometry(country_geom, level, max_cells=max_cells)
        instance.cells = cells.cells

        return instance
