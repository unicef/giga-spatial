from typing import List, Type, TypeVar, Generic, Union, Optional
from pathlib import Path
from pydantic import BaseModel, Field, ValidationError, PrivateAttr
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, Polygon
from scipy.spatial import cKDTree

from gigaspatial.core.io.data_store import DataStore
from gigaspatial.core.io.local_data_store import LocalDataStore
from gigaspatial.core.io.readers import read_dataset
from gigaspatial.core.io.writers import write_dataset


# Base class for all Giga entities
class BaseGigaEntity(BaseModel):
    """Base class for all Giga entities with common fields."""

    source: Optional[str] = Field(None, max_length=100, description="Source reference")
    source_detail: Optional[str] = None

    @property
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
        "Unknown", max_length=100, description="Primary administrative division"
    )
    admin1_id_giga: Optional[str] = Field(
        None,
        max_length=50,
        description="Unique identifier for the primary administrative division",
    )
    admin2: Optional[str] = Field(
        "Unknown", max_length=100, description="Secondary administrative division"
    )
    admin2_id_giga: Optional[str] = Field(
        None,
        max_length=50,
        description="Unique identifier for the secondary administrative division",
    )


class GigaEntityNoLocation(BaseGigaEntity):
    """Entity without location data."""

    pass


# Define a generic type bound to GigaEntity
E = TypeVar("E", bound=BaseGigaEntity)


class EntityTable(BaseModel, Generic[E]):
    entities: List[E] = Field(default_factory=list)
    _cached_kdtree: Optional[cKDTree] = PrivateAttr(
        default=None
    )  # Internal cache for the KDTree

    @classmethod
    def from_file(
        cls: Type["EntityTable"],
        file_path: Union[str, Path],
        entity_class: Type[E],
        data_store: Optional[DataStore] = None,
        **kwargs,
    ) -> "EntityTable":
        """
        Create an EntityTable instance from a file.

        Args:
            file_path: Path to the dataset file
            entity_class: The entity class for validation

        Returns:
            EntityTable instance

        Raises:
            ValidationError: If any row fails validation
            FileNotFoundError: If the file doesn't exist
        """
        data_store = data_store or LocalDataStore()
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        df = read_dataset(data_store, file_path, **kwargs)
        try:
            entities = [entity_class(**row) for row in df.to_dict(orient="records")]
            return cls(entities=entities)
        except ValidationError as e:
            raise ValueError(f"Validation error in input data: {e}")
        except Exception as e:
            raise ValueError(f"Error reading or processing the file: {e}")

    def _check_has_location(self, method_name: str) -> bool:
        """Helper method to check if entities have location data."""
        if not self.entities:
            return False
        if not isinstance(self.entities[0], GigaEntity):
            raise ValueError(
                f"Cannot perform {method_name}: entities of type {type(self.entities[0]).__name__} "
                "do not have location data (latitude/longitude)"
            )
        return True

    def to_dataframe(self) -> pd.DataFrame:
        """Convert the entity table to a pandas DataFrame."""
        return pd.DataFrame([e.model_dump() for e in self.entities])

    def to_geodataframe(self) -> gpd.GeoDataFrame:
        """Convert the entity table to a GeoDataFrame."""
        if not self._check_has_location("to_geodataframe"):
            raise ValueError("Cannot create GeoDataFrame: no entities available")
        df = self.to_dataframe()
        return gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
            crs="EPSG:4326",
        )

    def to_coordinate_vector(self) -> np.ndarray:
        """Transforms the entity table into a numpy vector of coordinates"""
        if not self.entities:
            return np.zeros((0, 2))

        if not self._check_has_location("to_coordinate_vector"):
            return np.zeros((0, 2))

        return np.array([[e.latitude, e.longitude] for e in self.entities])

    def get_lat_array(self) -> np.ndarray:
        """Get an array of latitude values."""
        if not self._check_has_location("get_lat_array"):
            return np.array([])
        return np.array([e.latitude for e in self.entities])

    def get_lon_array(self) -> np.ndarray:
        """Get an array of longitude values."""
        if not self._check_has_location("get_lon_array"):
            return np.array([])
        return np.array([e.longitude for e in self.entities])

    def filter_by_admin1(self, admin1_id_giga: str) -> "EntityTable[E]":
        """Filter entities by primary administrative division."""
        return self.__class__(
            entities=[e for e in self.entities if e.admin1_id_giga == admin1_id_giga]
        )

    def filter_by_admin2(self, admin2_id_giga: str) -> "EntityTable[E]":
        """Filter entities by secondary administrative division."""
        return self.__class__(
            entities=[e for e in self.entities if e.admin2_id_giga == admin2_id_giga]
        )

    def filter_by_polygon(self, polygon: Polygon) -> "EntityTable[E]":
        """Filter entities within a polygon"""
        if not self._check_has_location("filter_by_polygon"):
            return self.__class__(entities=[])

        filtered = [
            e for e in self.entities if polygon.contains(Point(e.longitude, e.latitude))
        ]
        return self.__class__(entities=filtered)

    def filter_by_bounds(
        self, min_lat: float, max_lat: float, min_lon: float, max_lon: float
    ) -> "EntityTable[E]":
        """Filter entities whose coordinates fall within the given bounds."""
        if not self._check_has_location("filter_by_bounds"):
            return self.__class__(entities=[])

        filtered = [
            e
            for e in self.entities
            if min_lat <= e.latitude <= max_lat and min_lon <= e.longitude <= max_lon
        ]
        return self.__class__(entities=filtered)

    def get_nearest_neighbors(
        self, lat: float, lon: float, k: int = 5
    ) -> "EntityTable[E]":
        """Find k nearest neighbors to a point using a cached KDTree."""
        if not self._check_has_location("get_nearest_neighbors"):
            return self.__class__(entities=[])

        if not self._cached_kdtree:
            self._build_kdtree()  # Build the KDTree if not already cached

        if not self._cached_kdtree:  # If still None after building
            return self.__class__(entities=[])

        _, indices = self._cached_kdtree.query([[lat, lon]], k=k)
        return self.__class__(entities=[self.entities[i] for i in indices[0]])

    def _build_kdtree(self):
        """Builds and caches the KDTree."""
        if not self._check_has_location("_build_kdtree"):
            self._cached_kdtree = None
            return
        coords = self.to_coordinate_vector()
        if coords:
            self._cached_kdtree = cKDTree(coords)

    def clear_cache(self):
        """Clears the KDTree cache."""
        self._cached_kdtree = None

    def to_file(
        self,
        file_path: Union[str, Path],
        data_store: Optional[DataStore] = None,
        **kwargs,
    ) -> None:
        """
        Save the entity data to a file.

        Args:
            file_path: Path to save the file
        """
        if not self.entities:
            raise ValueError("Cannot write to a file: no entities available.")

        data_store = data_store or LocalDataStore()

        write_dataset(self.to_dataframe(), data_store, file_path, **kwargs)

    def __len__(self) -> int:
        return len(self.entities)

    def __iter__(self):
        return iter(self.entities)
