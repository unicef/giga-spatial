"""
Module for administrative boundary schema and processing.
Defines the AdminBoundary entity, representing hierarchical divisions like
countries, states, and districts with polygon geometries.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional, List, Set, Union, Dict, ClassVar
from pydantic import Field
from shapely.geometry import Point
import pandas as pd
from pathlib import Path

from gigaspatial.core.io.data_store import DataStore
from .entity import GEO_ENUM_ENTITY_CONFIG, GigaGeoEntity, EntityTable
from gigaspatial.processing.entity_processor import EntityProcessor
from gigaspatial.config import config

logger = config.get_logger("AdminBoundaryManager")


class AdminLevel(int, Enum):
    """Hierarchical administrative division levels."""

    COUNTRY = 0
    STATE = 1  # state / province / region
    DISTRICT = 2  # district / county / department
    SUBDISTRICT = 3  # sub-district / municipality
    WARD = 4  # ward / village / locality


class AdminBoundary(GigaGeoEntity):
    """
    Represents an administrative boundary at any hierarchical level.

    Inherits polygon geometry from GigaGeoEntity. Used for spatial joins,
    country-level filtering, and administrative enrichment workflows.
    """

    model_config = GEO_ENUM_ENTITY_CONFIG

    # Identity
    boundary_id: str = Field(
        ..., max_length=100, description="Unique identifier for the boundary"
    )
    admin_level: AdminLevel = Field(
        ..., description="Hierarchical administrative level (0=country, 1=state, ...)"
    )

    # Names
    name: str = Field(..., max_length=200, description="Primary name in local language")
    name_en: Optional[str] = Field(None, max_length=200, description="Name in English")
    name_int: Optional[str] = Field(
        None, max_length=200, description="International or standardized name"
    )

    # Codes and references
    iso_code: Optional[str] = Field(
        None, max_length=20, description="ISO or local administrative code"
    )
    country_iso: Optional[str] = Field(
        None, max_length=3, description="ISO 3166-1 alpha-3 country code"
    )
    parent_id: Optional[str] = Field(
        None,
        max_length=100,
        description="boundary_id of the parent administrative unit",
    )

    # Attributes
    population: Optional[int] = Field(
        None, ge=0, description="Population count within the boundary"
    )
    population_year: Optional[int] = Field(
        None, ge=1900, le=2100, description="Reference year for population data"
    )
    area_km2: Optional[float] = Field(
        None,
        ge=0,
        description="Area of the boundary in square kilometres as reported by source",
    )
    bbox: Optional[List[float]] = Field(
        None,
        min_length=4,
        max_length=4,
        description="Bounding box [min_lon, min_lat, max_lon, max_lat]",
    )

    @property
    def id(self) -> str:
        """Map `id` to `boundary_id`."""
        return self.boundary_id


# ---------------------------------------------------------------------------
# Processor
# ---------------------------------------------------------------------------


class AdminBoundaryProcessor(EntityProcessor):
    """Cleaning and normalization for AdminBoundary entities."""

    NUMERIC_COLUMNS: ClassVar[List[str]] = [
        "population",
        "population_year",
        "area_km2",
        "admin_level",
    ]

    LOWERCASE_COLUMNS: ClassVar[List[str]] = []

    verbose: bool = False

    def process(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Execute the full processing pipeline for admin boundaries.

        Args:
            df: Raw admin boundary DataFrame.
            **kwargs: Additional processing arguments.

        Returns:
            Processed and normalized DataFrame.
        """
        df = super().process(df, **kwargs)
        df = self._normalize_country_iso(df)
        df = self._parse_geometry(df)
        return df

    def _normalize_country_iso(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Uppercase country ISO codes for consistency.

        Args:
            df: DataFrame to normalize.

        Returns:
            DataFrame with normalized country_iso column.
        """
        if "country_iso" in df.columns:
            df["country_iso"] = df["country_iso"].str.upper()
        return df


# ---------------------------------------------------------------------------
# Table
# ---------------------------------------------------------------------------


class AdminBoundaryTable(EntityTable[AdminBoundary]):
    """Container for AdminBoundary entities with hierarchy and spatial operations."""

    @classmethod
    def from_file(
        cls,
        file_path: Union[str, Path],
        data_store: Optional[DataStore] = None,
        **kwargs,
    ) -> "AdminBoundaryTable":
        """
        Create an AdminBoundaryTable from a file.

        Args:
            file_path: Path to the dataset file.
            data_store: DataStore instance for file access. Defaults to LocalDataStore.
            **kwargs: Additional arguments forwarded to read_dataset.

        Returns:
            AdminBoundaryTable with validated AdminBoundary entities.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file cannot be read or parsed.
        """
        return super().from_file(
            file_path=file_path,
            entity_class=AdminBoundary,
            data_store=data_store,
            processor=AdminBoundaryProcessor,
            **kwargs,
        )

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        entity_class: type = AdminBoundary,
        clean: bool = False,
    ) -> "AdminBoundaryTable":
        """
        Create an AdminBoundaryTable from an existing DataFrame.

        Args:
            df: DataFrame containing admin boundary data.
            entity_class: Entity class to validate against. Defaults to AdminBoundary.
            clean: Whether to apply AdminBoundaryProcessor before validation.

        Returns:
            AdminBoundaryTable with validated AdminBoundary entities.
        """
        if clean:
            df = AdminBoundaryProcessor().process(df)
        return super().from_dataframe(df=df, entity_class=entity_class)

    # ------------------------------------------------------------------
    # Filters
    # ------------------------------------------------------------------

    def filter_by_admin_level(self, level: AdminLevel) -> "AdminBoundaryTable":
        """Filter boundaries by administrative level."""
        return self.__class__(
            entities=[e for e in self.entities if e.admin_level == level.value]
        )

    def filter_by_country(self, country_iso: str) -> "AdminBoundaryTable":
        """Filter boundaries by ISO 3166-1 alpha-3 country code."""
        return self.__class__(
            entities=[e for e in self.entities if e.country_iso == country_iso.upper()]
        )

    def filter_by_parent(self, parent_id: str) -> "AdminBoundaryTable":
        """Return all direct children of a given boundary."""
        return self.__class__(
            entities=[e for e in self.entities if e.parent_id == parent_id]
        )

    def filter_contains_point(self, lat: float, lon: float) -> "AdminBoundaryTable":
        """Return all boundaries whose geometry contains the given point."""
        point = Point(lon, lat)
        return self.__class__(
            entities=[e for e in self.entities if e.geometry.contains(point)]
        )

    # ------------------------------------------------------------------
    # Hierarchy
    # ------------------------------------------------------------------

    def get_children(self, boundary_id: str) -> "AdminBoundaryTable":
        """Return all direct children of a boundary by its ID."""
        return self.filter_by_parent(boundary_id)

    def get_parent(self, boundary_id: str) -> Optional[AdminBoundary]:
        """Return the parent boundary of a given boundary, if present in the table."""
        entity = self.get_by_id(boundary_id)
        if entity is None or entity.parent_id is None:
            return None
        return self.get_by_id(entity.parent_id)

    def get_by_id(self, boundary_id: str) -> Optional[AdminBoundary]:
        """Return a boundary by its ID, or None if not found."""
        for e in self.entities:
            if e.boundary_id == boundary_id:
                return e
        return None

    def to_hierarchy_dict(self) -> Dict[str, List[str]]:
        """
        Build a parent → children mapping of boundary IDs.

        Returns:
            Dict mapping parent boundary_id → list of child boundary_ids.
        """
        hierarchy: Dict[str, List[str]] = {}
        for e in self.entities:
            if e.parent_id:
                hierarchy.setdefault(e.parent_id, []).append(e.boundary_id)
        return hierarchy

    # ------------------------------------------------------------------
    # Aggregations
    # ------------------------------------------------------------------

    def get_countries(self) -> Set[str]:
        """
        Return the set of all unique country ISO codes present in the table.

        Returns:
            Set of unique country ISO-3 strings.
        """
        return {e.country_iso for e in self.entities if e.country_iso is not None}

    def get_admin_levels(self) -> Set[int]:
        """
        Return the set of all unique admin levels present in the table.

        Returns:
            Set of unique administrative level integers.
        """
        return {e.admin_level for e in self.entities}
