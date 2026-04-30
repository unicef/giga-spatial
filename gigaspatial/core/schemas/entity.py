"""
Base entity schemas and table containers.
Provides the foundation for all geospatial and network entities in Giga Spatial,
including point-based (GigaEntity) and geometry-based (GigaGeoEntity) records,
along with generic table containers (EntityTable).
"""

from __future__ import annotations

from functools import wraps
from typing import (
    Tuple,
    List,
    Set,
    Dict,
    Any,
    Type,
    TypeVar,
    Generic,
    Union,
    Optional,
    ClassVar,
)
from pathlib import Path
from pydantic import (
    BaseModel,
    Field,
    PrivateAttr,
    ConfigDict,
    field_validator,
    model_validator,
)
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely import wkt, wkb
from scipy.spatial import cKDTree
import networkx as nx
import unicodedata
import uuid
from tqdm import tqdm

from .shared import NULL_LIKE_VALUES, ENTITY_UUID_NAMESPACE
from gigaspatial.processing.entity_processor import EntityProcessor
from gigaspatial.processing.geo import (
    detect_coordinate_columns,
    convert_to_geodataframe,
    annotate_with_admin_regions,
)
from gigaspatial.processing.algorithms import build_distance_graph
from gigaspatial.core.io.data_store import DataStore
from gigaspatial.core.io.local_data_store import LocalDataStore
from gigaspatial.core.io.readers import read_dataset
from gigaspatial.core.io.writers import write_dataset
from gigaspatial.config import config

logger = config.get_logger("EntityManager")

# -------------------------------------------------------------------
# Base config — applied to all entities universally
# Used on: BaseGigaEntity (inherited by all)
# -------------------------------------------------------------------
BASE_ENTITY_CONFIG = ConfigDict(
    populate_by_name=True,
    str_strip_whitespace=True,
)

# -------------------------------------------------------------------
# Geo entity config — for polygon/multipolygon geometry entities
# Used on: GigaGeoEntity (inherited by AdminBoundary, MobileCoverage,
#          BuildingFootprint)
# Adds arbitrary_types_allowed for Shapely geometry field
# -------------------------------------------------------------------
GEO_ENTITY_CONFIG = ConfigDict(
    populate_by_name=True,
    str_strip_whitespace=True,
    arbitrary_types_allowed=True,
)

# -------------------------------------------------------------------
# Enum entity config — for point entities with enum fields
# Used on: CellTower, Cell, TransmissionNode, School (future)
# -------------------------------------------------------------------
ENUM_ENTITY_CONFIG = ConfigDict(
    populate_by_name=True,
    str_strip_whitespace=True,
    use_enum_values=True,
)

# -------------------------------------------------------------------
# Geo enum entity config — for polygon entities with enum fields
# Used on: MobileCoverage, AdminBoundary, BuildingFootprint
# -------------------------------------------------------------------
GEO_ENUM_ENTITY_CONFIG = ConfigDict(
    populate_by_name=True,
    str_strip_whitespace=True,
    arbitrary_types_allowed=True,
    use_enum_values=True,
)


# Base class for all Giga entities
class BaseGigaEntity(BaseModel, ABC):
    """Base class for all Giga entities with common fields."""

    model_config = BASE_ENTITY_CONFIG

    source: Optional[str] = Field(None, max_length=100, description="Source reference")
    source_detail: Optional[str] = Field(
        None, description="Detailed source information"
    )

    @property
    @abstractmethod
    def id(self) -> str:
        """Abstract property that must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement id property")


class GigaEntity(BaseGigaEntity):
    """Entity with location data."""

    latitude: float = Field(
        ..., ge=-90, le=90, description="Latitude coordinate of the entity"
    )
    longitude: float = Field(
        ..., ge=-180, le=180, description="Longitude coordinate of the entity"
    )
    admin1: Optional[str] = Field(
        None, max_length=100, description="Primary administrative division name"
    )
    admin1_id: Optional[str] = Field(
        None,
        max_length=50,
        description="Unique identifier for the primary administrative division",
    )
    admin2: Optional[str] = Field(
        None, max_length=100, description="Secondary administrative division name"
    )
    admin2_id: Optional[str] = Field(
        None,
        max_length=50,
        description="Unique identifier for the secondary administrative division",
    )

    @field_validator("admin1", "admin2", mode="before")
    @classmethod
    def normalize_admin(cls, v: Optional[str]) -> Optional[str]:
        """
        Normalize administrative division names.

        Converts "unknown", "n/a" variations to None, and title-cases valid names.
        """
        if v is None or str(v).strip().lower() in ("unknown", "n/a", "none", ""):
            return None
        return str(v).strip().title()

    @model_validator(mode="after")
    def check_null_island(self) -> "GigaEntity":
        """Validate that coordinates are not (0.0, 0.0)."""
        if self.latitude == 0.0 and self.longitude == 0.0:
            raise ValueError("Null island coordinates (0.0, 0.0) are not valid.")
        return self


GeometryType = Union[Polygon, MultiPolygon]


class GigaGeoEntity(BaseGigaEntity):
    """
    Entity with polygon or multipolygon geometry.

    Used for spatially bounded records such as coverage areas,
    administrative boundaries, and building footprints where a
    point representation is insufficient.
    """

    model_config = GEO_ENTITY_CONFIG

    geometry: GeometryType = Field(
        ..., description="Polygon or MultiPolygon geometry of the entity"
    )

    @field_validator("geometry", mode="before")
    @classmethod
    def parse_geometry(cls, v: Any) -> GeometryType:
        """
        Accept Shapely geometries, WKT strings, or WKB bytes.

        Args:
            v: Raw geometry data.

        Returns:
            Parsed Polygon or MultiPolygon.

        Raises:
            ValueError: If parsing fails or geometry type is invalid.
        """
        if isinstance(v, (Polygon, MultiPolygon)):
            return v

        try:
            if isinstance(v, str):
                parsed = wkt.loads(v)
            elif isinstance(v, (bytes, bytearray)):
                parsed = wkb.loads(v)
            else:
                raise ValueError(
                    f"Cannot parse geometry from type {type(v).__name__}. "
                    "Expected Shapely geometry, WKT string, or WKB bytes."
                )
        except Exception as e:
            raise ValueError(f"Failed to parse geometry: {e}") from e

        if not isinstance(parsed, (Polygon, MultiPolygon)):
            raise ValueError(
                f"Expected Polygon or MultiPolygon geometry, got {type(parsed).__name__}. "
                "If you have Point or LineString geometries, use GigaEntity instead."
            )
        return parsed

    @property
    def centroid(self) -> tuple[float, float]:
        """
        Return (latitude, longitude) of the geometry centroid.

        Returns:
            Tuple of (lat, lon).
        """
        c = self.geometry.centroid
        return c.y, c.x

    @property
    def area_sq_km(self) -> float:
        """
        Approximate area in square kilometres.

        Calculated using the centroid UTM projection.

        Returns:
            Area in square km.
        """
        from gigaspatial.processing.geo import estimate_utm_crs_with_fallback
        import geopandas as gpd

        gdf = gpd.GeoDataFrame(geometry=[self.geometry], crs="EPSG:4326")
        utm_crs = estimate_utm_crs_with_fallback(gdf)
        return gdf.to_crs(utm_crs).geometry.area.iloc[0] / 1e6


class GigaEntityNoLocation(BaseGigaEntity):
    """Entity without location data."""

    pass


# Define a generic type bound to GigaEntity
E = TypeVar("E", bound=BaseGigaEntity)


class EntityTable(BaseModel, Generic[E]):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    entities: List[E] = Field(default_factory=list)
    entity_class: Optional[Type[E]] = Field(default=None, exclude=True)
    _cached_kdtree: Optional[cKDTree] = PrivateAttr(
        default=None
    )  # Internal cache for the KDTree

    @model_validator(mode="after")
    def _infer_entity_class(self) -> "EntityTable":
        """
        Infer entity_class from entities list if not explicitly provided.

        Returns:
            Table with potentially updated entity_class.
        """
        if self.entity_class is None and self.entities:
            self.entity_class = type(self.entities[0])
        return self

    @property
    def is_point_entity(self) -> bool:
        """
        Whether entities carry lat/lon point location.

        Returns:
            True if entities are GigaEntity but not GigaGeoEntity.
        """
        if self.entity_class is None:
            return False
        return issubclass(self.entity_class, GigaEntity) and not issubclass(
            self.entity_class, GigaGeoEntity
        )

    @property
    def is_geo_entity(self) -> bool:
        """
        Whether entities carry complex polygon/multipolygon geometry.

        Returns:
            True if entities are GigaGeoEntity.
        """
        if self.entity_class is None:
            return False
        return issubclass(self.entity_class, GigaGeoEntity)

    @classmethod
    def from_file(
        cls,
        file_path: Union[str, Path],
        entity_class: Type[E],
        data_store: Optional[DataStore] = None,
        processor: Optional[EntityProcessor] = None,
        **kwargs,
    ) -> "EntityTable":
        """
        Create an EntityTable from a file, with optional pre-validation cleaning.

        Args:
            file_path: Path to the dataset file.
            entity_class: The Pydantic entity class to validate each row against.
            data_store: DataStore instance for file access. Defaults to LocalDataStore.
            processor: Optional EntityCleaner subclass to apply before validation.
            **kwargs: Additional arguments forwarded to read_dataset.

        Returns:
            EntityTable instance containing successfully validated entities.

        Raises:
            FileNotFoundError: If the file does not exist in the data store.
            ValueError: If the file cannot be read or parsed.
        """

        df = read_dataset(file_path, data_store=data_store, **kwargs)
        logger.debug("Loaded %d rows from %s", len(df), file_path)

        if processor is not None:
            df = processor.process(df, **kwargs)
            logger.debug(
                "Cleaned data has %d rows after %s.", len(df), processor.__name__
            )

        try:
            return cls.from_dataframe(df, entity_class)
        except Exception as e:
            raise ValueError(f"Error reading or processing the file: {e}")

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        entity_class: Type[E],
        clean: bool = False,
        processor: Optional[EntityProcessor] = None,
        **kwargs,
    ) -> "EntityTable[E]":
        """
        Create an EntityTable from an existing DataFrame.

        Args:
            df: DataFrame containing entity data.
            entity_class: The Pydantic entity class to validate each row against.
            clean: Whether to apply an EntityProcessor before validation.
            processor: Optional processor instance to use if clean=True.
                Defaults to the base EntityProcessor if None.
            **kwargs: Additional arguments passed to the processor.

        Returns:
            EntityTable instance. Rows that fail validation are skipped with a warning.
        """
        if clean:
            if processor is None:
                processor = EntityProcessor()
            df = processor.process(df, **kwargs)

        entities: List[E] = []
        failed_rows: List[dict] = []
        errors: List[str] = []

        tqdm_stream = config.get_tqdm_logger_stream(logger)
        # Convert NaN to None so Pydantic handles missing Optional fields correctly.
        # Otherwise, np.nan triggers numeric validations (like ge=0) and fails.
        records = df.replace({np.nan: None}).to_dict(orient="records")

        for row in tqdm(
            records,
            file=tqdm_stream,
            desc=f"Validating {entity_class.__name__} entities",
            total=len(df),
        ):
            try:
                entities.append(entity_class(**row))
            except Exception as e:
                failed_rows.append(row)
                errors.append(str(e))

        if errors:
            logger.warning(
                "%d of %d rows failed validation and were skipped. First 5 errors: %s",
                len(errors),
                len(df),
                errors[:5],
            )

        logger.debug("EntityTable created with %d valid entities.", len(entities))
        return cls(entities=entities, entity_class=entity_class)

    @classmethod
    def from_files(
        cls,
        file_paths: List[Union[str, Path]],
        entity_class: Type[E],
        data_store: Optional[DataStore] = None,
        processor: Optional[Type[EntityProcessor]] = None,
        **kwargs,
    ) -> "EntityTable[E]":
        """
        Load and merge multiple source files into a single EntityTable.

        Each file is independently cleaned and validated before merging.
        Failed rows in individual files are skipped and logged but do not
        abort the remaining files.

        Args:
            file_paths: List of paths to source files.
            entity_class: The Pydantic entity class to validate each row against.
            data_store: DataStore instance for file access. Defaults to LocalDataStore.
            processor: Optional EntityProcessor subclass applied to each file.
            **kwargs: Additional arguments forwarded to read_dataset.

        Returns:
            Merged EntityTable instance.

        Raises:
            ValueError: If file_paths is empty.
        """
        if not file_paths:
            raise ValueError("file_paths must not be empty.")

        tables = []
        for path in file_paths:
            try:
                table = cls.from_file(
                    file_path=path,
                    entity_class=entity_class,
                    data_store=data_store,
                    processor=processor,
                    **kwargs,
                )
                tables.append(table)
                logger.debug("Loaded %d entities from %s.", len(table), path)
            except Exception as e:
                logger.warning("Failed to load file '%s': %s. Skipping.", path, e)

        if not tables:
            logger.warning("No files were successfully loaded.")
            return cls(entities=[], entity_class=entity_class)

        return cls.merge(tables)

    @classmethod
    def merge(
        cls,
        tables: List["EntityTable[E]"],
        deduplicate_by_id: bool = True,
    ) -> "EntityTable[E]":
        """
        Merge multiple EntityTable instances into one.

        Args:
            tables: List of EntityTable instances to merge.
            deduplicate_by_id: If True, deduplicate merged entities by their
                `id` property, keeping the first occurrence. Defaults to True.

        Returns:
            Merged EntityTable instance.

        Raises:
            ValueError: If tables list is empty.
        """
        if not tables:
            raise ValueError("Cannot merge an empty list of tables.")

        entity_class = next(
            (t.entity_class for t in tables if t.entity_class is not None), None
        )

        all_entities: List[E] = []
        for table in tables:
            all_entities.extend(table.entities)

        if deduplicate_by_id:
            seen_ids: set = set()
            unique_entities: List[E] = []
            for entity in all_entities:
                if entity.id not in seen_ids:
                    seen_ids.add(entity.id)
                    unique_entities.append(entity)

            duplicates_removed = len(all_entities) - len(unique_entities)
            if duplicates_removed:
                logger.info(
                    "Merge removed %d duplicate entities by id.", duplicates_removed
                )
            all_entities = unique_entities

        logger.info(
            "Merged %d tables into %d entities.", len(tables), len(all_entities)
        )
        return cls(entities=all_entities, entity_class=entity_class)

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert the entity table to a pandas DataFrame.

        Returns:
            A DataFrame where each row corresponds to an entity model dump.
        """
        return pd.DataFrame([e.model_dump() for e in self.entities])

    def to_geodataframe(self) -> gpd.GeoDataFrame:
        """
        Convert the entity table to a GeoDataFrame.

        Returns:
            GeoDataFrame with point or polygon geometry.

        Raises:
            ValueError: If entities have no associated geometry fields.
        """
        if not self.is_point_entity and not self.is_geo_entity:
            raise ValueError("Cannot create GeoDataFrame: entities have no geometry.")

        if self.is_point_entity:
            df = self.to_dataframe()
            return gpd.GeoDataFrame(
                df,
                geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
                crs="EPSG:4326",
            )
        else:
            # GigaGeoEntity — geometry column already present in model_dump()
            df = self.to_dataframe()
            return gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")

    def to_geoms(self) -> List:
        """
        Return a list of Shapely geometry objects for all entities.

        - ``is_point_entity``: constructs ``Point(lon, lat)`` per entity,
        skipping rows where either coordinate is ``None``.
        - ``is_geo_entity`` (``GigaGeoEntity``): returns each entity's
        ``geometry`` field directly, skipping ``None`` values.
        - Neither: returns an empty list.

        Returns:
            List of Shapely ``BaseGeometry`` objects, never ``None``.
        """
        if self.is_point_entity:
            return [
                Point(e.longitude, e.latitude)
                for e in self.entities
                if e.longitude is not None and e.latitude is not None
            ]

        if self.is_geo_entity:
            return [e.geometry for e in self.entities if e.geometry is not None]

        logger.debug(
            "%s is neither a point nor a geo entity; to_geoms returns [].",
            self.__class__.__name__,
        )
        return []

    def to_coordinate_vector(self) -> np.ndarray:
        """
        Transform the entity table into a numpy array of (lat, lon) coordinates.

        Returns:
            Numpy array of shape (N, 2). Empty array if not a point entity.
        """
        if not self.is_point_entity or not self.entities:
            return np.zeros((0, 2))
        return np.array([[e.latitude, e.longitude] for e in self.entities])

    def get_lat_array(self) -> np.ndarray:
        """
        Get an array of latitude values.

        Returns:
            Numpy array of floats.
        """
        if not self.is_point_entity:
            return np.array([])
        return np.array([e.latitude for e in self.entities])

    def get_lon_array(self) -> np.ndarray:
        """
        Get an array of longitude values.

        Returns:
            Numpy array of floats.
        """
        if not self.is_point_entity:
            return np.array([])
        return np.array([e.longitude for e in self.entities])

    def filter_by_admin1(self, admin1_id: str) -> "EntityTable[E]":
        """
        Filter entities by primary administrative division ID.

        Args:
            admin1_id: Identifier for the division.

        Returns:
            New EntityTable with filtered entities.
        """
        return self.__class__(
            entities=[e for e in self.entities if e.admin1_id == admin1_id]
        )

    def filter_by_admin2(self, admin2_id: str) -> "EntityTable[E]":
        """
        Filter entities by secondary administrative division ID.

        Args:
            admin2_id: Identifier for the division.

        Returns:
            New EntityTable with filtered entities.
        """
        return self.__class__(
            entities=[e for e in self.entities if e.admin2_id == admin2_id]
        )

    def filter_by_polygon(self, polygon: Polygon) -> "EntityTable[E]":
        """
        Filter entities whose location falls within a polygon.

        Args:
            polygon: Shapely Polygon to filter by.

        Returns:
            New EntityTable with entities inside the polygon.
        """
        if not self.is_point_entity:
            return self.__class__(entities=[])
        return self.__class__(
            entities=[
                e
                for e in self.entities
                if polygon.contains(Point(e.longitude, e.latitude))
            ]
        )

    def filter_by_bounds(
        self, min_lat: float, max_lat: float, min_lon: float, max_lon: float
    ) -> "EntityTable[E]":
        """
        Filter entities whose coordinates fall within a bounding box.

        Args:
            min_lat: Minimum latitude.
            max_lat: Maximum latitude.
            min_lon: Minimum longitude.
            max_lon: Maximum longitude.

        Returns:
            New EntityTable with entities within bounds.
        """
        if not self.is_point_entity:
            return self.__class__(entities=[])
        return self.__class__(
            entities=[
                e
                for e in self.entities
                if min_lat <= e.latitude <= max_lat
                and min_lon <= e.longitude <= max_lon
            ]
        )

    def get_nearest_neighbors(
        self, lat: float, lon: float, k: int = 5
    ) -> "EntityTable[E]":
        """
        Find k nearest neighbors to a given set of coordinates.

        Uses an internal KDTree cache to accelerate repeated lookups.

        Args:
            lat: Query latitude.
            lon: Query longitude.
            k: Number of neighbors to return. Defaults to 5.

        Returns:
            New EntityTable containing the k nearest entities.
        """
        if not self.is_point_entity:
            return self.__class__(entities=[])

        if not self._cached_kdtree:
            self._build_kdtree()

        if not self._cached_kdtree:
            return self.__class__(entities=[])

        _, indices = self._cached_kdtree.query([[lat, lon]], k=k)
        return self.__class__(entities=[self.entities[i] for i in indices[0]])

    def to_distance_graph(
        self,
        distance_threshold: float,
        max_k: int = 100,
        return_dataframe: bool = False,
        verbose: bool = True,
    ) -> Union[nx.Graph, Tuple[nx.Graph, pd.DataFrame]]:
        """
        Build a spatial distance graph of entities within this table.

        Computes pairwise distances between all entities and returns a NetworkX
        graph where edges represent pairs within the given distance threshold.
        Self-matches are automatically excluded.

        Args:
            distance_threshold: Maximum distance in meters for two entities to be connected.
            max_k: Maximum number of neighbors to consider per entity. Default is 100.
            return_dataframe: If True, also return the matches DataFrame alongside the graph.
            verbose: If True, log graph construction statistics.

        Returns:
            NetworkX Graph, or tuple of (Graph, DataFrame) if return_dataframe=True.

        Raises:
            ValueError: If entities do not have location data.
        """
        if not self.is_point_entity:
            raise ValueError(
                "Cannot build distance graph: entities do not have location data."
            )
        return build_distance_graph(
            left_df=self.to_geodataframe(),
            right_df=self.to_geodataframe(),
            distance_threshold=distance_threshold,
            max_k=max_k,
            return_dataframe=return_dataframe,
            verbose=verbose,
            exclude_same_index=True,
        )

    def to_distance_graph_with(
        self,
        other: "EntityTable",
        distance_threshold: float,
        max_k: int = 100,
        return_dataframe: bool = False,
        verbose: bool = True,
    ) -> Union[nx.Graph, Tuple[nx.Graph, pd.DataFrame]]:
        """
        Build a spatial distance graph between this table and another EntityTable.

        Matches entities in this table (left) against entities in another table (right)
        and returns a bipartite-style graph where edges connect entities within the
        given distance threshold.

        Args:
            other: The right-hand EntityTable to match against.
            distance_threshold: Maximum distance in meters for two entities to be connected.
            max_k: Maximum number of neighbors to consider per entity. Default is 100.
            return_dataframe: If True, also return the matches DataFrame alongside the graph.
            verbose: If True, log graph construction statistics.

        Returns:
            NetworkX Graph, or tuple of (Graph, DataFrame) if return_dataframe=True.

        Raises:
            ValueError: If either table's entities do not have location data.
        """
        if not self.is_point_entity:
            raise ValueError(
                "Cannot build distance graph: this table's entities do not have location data."
            )
        if not other.is_point_entity:
            raise ValueError(
                "Cannot build distance graph: other table's entities do not have location data."
            )
        return build_distance_graph(
            left_df=self.to_geodataframe(),
            right_df=other.to_geodataframe(),
            distance_threshold=distance_threshold,
            max_k=max_k,
            return_dataframe=return_dataframe,
            verbose=verbose,
            exclude_same_index=False,
        )

    def _build_kdtree(self):
        """Build and cache the KDTree from entity point coordinates."""
        if not self.is_point_entity:
            self._cached_kdtree = None
            return
        coords = self.to_coordinate_vector()
        if coords.size > 0:
            logger.debug("Building KDTree for %d entities.", len(self.entities))
            self._cached_kdtree = cKDTree(coords)
        else:
            logger.warning("EntityTable is empty, skipping KDTree build.")

    def clear_cache(self):
        """Clear the internal KDTree spatial index cache."""
        self._cached_kdtree = None

    def to_file(
        self,
        file_path: Union[str, Path],
        data_store: Optional[DataStore] = None,
        as_geodataframe: bool = False,
        **kwargs,
    ) -> None:
        """
        Save the entity table to a file.

        Args:
            file_path: Destination file path.
            data_store: DataStore to use for writing. Defaults to LocalDataStore.
            as_geodataframe: If True, uses GeoDataFrame for spatial formats (e.g. GeoJSON).
            **kwargs: Additional arguments for writers.write_dataset.

        Raises:
            ValueError: If the table is empty.
        """
        if not self.entities:
            raise ValueError("Cannot write to file: no entities available.")
        data_store = data_store or LocalDataStore()
        df = self.to_geodataframe() if as_geodataframe else self.to_dataframe()
        write_dataset(df, data_store, file_path, **kwargs)

    def __len__(self) -> int:
        return len(self.entities)

    def __iter__(self):
        return iter(self.entities)

    def __repr__(self) -> str:
        entity_type = (
            self.entities[0].__class__.__name__ if self.entities else "unknown"
        )
        return f"{self.__class__.__name__}(n={len(self.entities)}, type={entity_type})"

    def __getitem__(self, index: int) -> E:
        return self.entities[index]
