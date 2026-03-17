from enum import Enum
from typing import Optional, ClassVar, Union
from pathlib import Path

from pydantic import Field
import pandas as pd

from gigaspatial.config import config
from gigaspatial.core.io.data_store import DataStore
from .entity import GEO_ENTITY_CONFIG, GigaGeoEntity, EntityTable
from gigaspatial.processing.entity_processor import EntityProcessor


logger = config.get_logger(__name__)


class BuildingType(str, Enum):
    """Functional classification of a building."""

    RESIDENTIAL = "residential"
    COMMERCIAL = "commercial"
    INDUSTRIAL = "industrial"
    EDUCATIONAL = "educational"
    HEALTHCARE = "healthcare"
    RELIGIOUS = "religious"
    GOVERNMENTAL = "governmental"
    TRANSPORTATION = "transportation"
    AGRICULTURAL = "agricultural"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class BuildingCondition(str, Enum):
    """Physical condition of a building structure."""

    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    RUIN = "ruin"


class BuildingMaterial(str, Enum):
    """Primary construction material of the building."""

    CONCRETE = "concrete"
    BRICK = "brick"
    WOOD = "wood"
    METAL = "metal"
    MUD = "mud"
    STONE = "stone"
    MIXED = "mixed"


BUILDING_TYPE_ALIASES: dict[BuildingType, list[str]] = {
    BuildingType.RESIDENTIAL: [
        "residential",
        "house",
        "home",
        "apartment",
        "dwelling",
        "hut",
        "bungalow",
        "villa",
    ],
    BuildingType.COMMERCIAL: [
        "commercial",
        "shop",
        "store",
        "office",
        "retail",
        "market",
        "mall",
        "warehouse",
    ],
    BuildingType.INDUSTRIAL: [
        "industrial",
        "factory",
        "plant",
        "workshop",
        "facility",
        "depot",
    ],
    BuildingType.EDUCATIONAL: [
        "educational",
        "school",
        "university",
        "college",
        "kindergarten",
        "training",
    ],
    BuildingType.HEALTHCARE: [
        "healthcare",
        "hospital",
        "clinic",
        "pharmacy",
        "health centre",
        "health_centre",
        "medical",
    ],
    BuildingType.RELIGIOUS: [
        "religious",
        "church",
        "mosque",
        "temple",
        "shrine",
        "chapel",
        "synagogue",
    ],
    BuildingType.GOVERNMENTAL: [
        "governmental",
        "government",
        "public",
        "admin",
        "administration",
        "municipal",
        "civic",
    ],
    BuildingType.TRANSPORTATION: [
        "transportation",
        "station",
        "terminal",
        "airport",
        "port",
        "garage",
        "parking",
    ],
    BuildingType.AGRICULTURAL: [
        "agricultural",
        "farm",
        "barn",
        "silo",
        "stable",
        "greenhouse",
    ],
    BuildingType.MIXED: ["mixed", "multipurpose", "multi_purpose"],
    BuildingType.UNKNOWN: ["unknown", "yes", "other", "building"],
}

BUILDING_TYPE_ALIAS_MAP = {
    alias: building_type.value
    for building_type, aliases in BUILDING_TYPE_ALIASES.items()
    for alias in aliases
}


class BuildingFootprint(GigaGeoEntity):
    """
    Represents a building footprint as a polygon geometry.

    Used for infrastructure proximity analysis, population estimation,
    and spatial enrichment workflows.
    """

    model_config = GEO_ENTITY_CONFIG

    # Identity
    footprint_id: str = Field(
        ..., max_length=100, description="Unique identifier for the building footprint"
    )
    footprint_id_source: Optional[str] = Field(
        None, max_length=100, description="Original identifier in the source dataset"
    )

    # Classification
    building_type: Optional[BuildingType] = Field(
        None, description="Functional classification of the building"
    )
    building_condition: Optional[BuildingCondition] = Field(
        None, description="Physical condition of the building structure"
    )
    building_material: Optional[BuildingMaterial] = Field(
        None, description="Primary construction material of the building"
    )

    # Physical attributes
    height_meters: Optional[float] = Field(
        None, ge=0, description="Height of the building in meters"
    )
    num_floors: Optional[int] = Field(
        None, ge=1, description="Number of floors in the building"
    )
    area_sq_m: Optional[float] = Field(
        None, ge=0, description="Footprint area in square metres as reported by source"
    )
    roof_type: Optional[str] = Field(
        None,
        max_length=50,
        description="Type of roof structure (e.g. flat, gabled, hipped)",
    )

    # Provenance
    source: Optional[str] = Field(
        None,
        max_length=100,
        description="Data source (e.g. OSM, Google, Microsoft, DigitizeAfrica)",
    )
    capture_date: Optional[str] = Field(
        None,
        description="ISO 8601 date when the footprint was captured or last updated",
    )
    confidence: Optional[float] = Field(
        None,
        ge=0,
        le=1,
        description="Model confidence score for AI-derived footprints (0–1)",
    )

    @property
    def id(self) -> str:
        """Map `id` to `footprint_id`."""
        return self.footprint_id


# ---------------------------------------------------------------------------
# Processor
# ---------------------------------------------------------------------------


class BuildingFootprintProcessor(EntityProcessor):
    """Cleaning and normalization for BuildingFootprint entities."""

    # BuildingFootprint has no lat/lon (GigaGeoEntity)
    NUMERIC_COLUMNS: ClassVar[list[str]] = [
        "height_meters",
        "num_floors",
        "area_sq_m",
        "confidence",
    ]

    LOWERCASE_COLUMNS: ClassVar[list[str]] = [
        "building_type",
        "building_condition",
        "building_material",
    ]

    BUILDING_TYPE_ALIAS_MAP: ClassVar[dict[str, str]] = BUILDING_TYPE_ALIAS_MAP

    def process(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = super().process(df, **kwargs)
        df = self._normalize_building_type(df)
        df = self._parse_geometry(df)
        return df

    def _normalize_building_type(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize building_type values to canonical BuildingType enum values."""
        return self._normalize_enum_column(
            df,
            column="building_type",
            alias_map=self.BUILDING_TYPE_ALIAS_MAP,
            valid_values={bt.value for bt in BuildingType},
            required=False,
        )


# ---------------------------------------------------------------------------
# Table
# ---------------------------------------------------------------------------


class BuildingFootprintTable(EntityTable[BuildingFootprint]):
    """Container for BuildingFootprint entities with footprint-specific operations."""

    @classmethod
    def from_file(
        cls,
        file_path: Union[str, Path],
        data_store: Optional[DataStore] = None,
        **kwargs,
    ) -> "BuildingFootprintTable":
        """
        Create a BuildingFootprintTable from a file.

        Args:
            file_path: Path to the dataset file.
            data_store: DataStore instance for file access. Defaults to LocalDataStore.
            **kwargs: Additional arguments forwarded to read_dataset.

        Returns:
            BuildingFootprintTable with validated BuildingFootprint entities.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file cannot be read or parsed.
        """
        return super().from_file(
            file_path=file_path,
            entity_class=BuildingFootprint,
            data_store=data_store,
            processor=BuildingFootprintProcessor,
            **kwargs,
        )

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        entity_class: type = BuildingFootprint,
        clean: bool = False,
    ) -> "BuildingFootprintTable":
        """
        Create a BuildingFootprintTable from an existing DataFrame.

        Args:
            df: DataFrame containing building footprint data.
            entity_class: Entity class to validate against. Defaults to BuildingFootprint.
            clean: Whether to apply BuildingFootprintProcessor before validation.

        Returns:
            BuildingFootprintTable with validated BuildingFootprint entities.
        """
        if clean:
            df = BuildingFootprintProcessor().process(df)
        return super().from_dataframe(df=df, entity_class=entity_class)

    # ------------------------------------------------------------------
    # Filters
    # ------------------------------------------------------------------

    def filter_by_building_type(
        self, building_type: BuildingType
    ) -> "BuildingFootprintTable":
        """Filter footprints by building functional type."""
        return self.__class__(
            entities=[
                e for e in self.entities if e.building_type == building_type.value
            ]
        )

    def filter_by_building_types(
        self, building_types: set[BuildingType]
    ) -> "BuildingFootprintTable":
        """Filter footprints matching any of the given building types."""
        values = {bt.value for bt in building_types}
        return self.__class__(
            entities=[e for e in self.entities if e.building_type in values]
        )

    def filter_by_condition(
        self, condition: BuildingCondition
    ) -> "BuildingFootprintTable":
        """Filter footprints by physical condition."""
        return self.__class__(
            entities=[
                e for e in self.entities if e.building_condition == condition.value
            ]
        )

    def filter_by_min_confidence(
        self, min_confidence: float
    ) -> "BuildingFootprintTable":
        """Filter AI-derived footprints by minimum model confidence score."""
        return self.__class__(
            entities=[
                e
                for e in self.entities
                if e.confidence is not None and e.confidence >= min_confidence
            ]
        )

    def filter_by_min_area(self, min_area_sq_m: float) -> "BuildingFootprintTable":
        """Filter footprints by minimum reported area in square metres."""
        return self.__class__(
            entities=[
                e
                for e in self.entities
                if e.area_sq_m is not None and e.area_sq_m >= min_area_sq_m
            ]
        )

    def filter_contains_point(self, lat: float, lon: float) -> "BuildingFootprintTable":
        """Return all footprints whose geometry contains the given point."""
        from shapely.geometry import Point

        point = Point(lon, lat)
        return self.__class__(
            entities=[e for e in self.entities if e.geometry.contains(point)]
        )

    # ------------------------------------------------------------------
    # Aggregations
    # ------------------------------------------------------------------

    def get_building_types(self) -> set[str]:
        """Return all unique building types present in the table."""
        return {e.building_type for e in self.entities if e.building_type is not None}

    def get_sources(self) -> set[str]:
        """Return all unique data sources present in the table."""
        return {e.source for e in self.entities if e.source is not None}

    def total_footprint_area_sq_m(self) -> float:
        """
        Compute total building footprint area in square metres from actual geometries.

        Uses UTM projection for accuracy. Does not dissolve overlapping footprints
        since buildings should not overlap in a clean dataset.

        Returns:
            Total footprint area in square metres.
        """
        import geopandas as gpd
        from gigaspatial.processing.geo import estimate_utm_crs_with_fallback

        gdf = self.to_geodataframe()
        utm_crs = estimate_utm_crs_with_fallback(gdf)
        return gdf.to_crs(utm_crs).geometry.area.sum()

    def get_footprint_summary(self) -> pd.DataFrame:
        """
        Return a summary DataFrame of footprint counts and area by building type and source.

        Returns:
            DataFrame with columns: building_type, source, count, total_area_sq_m.
        """
        import geopandas as gpd
        from gigaspatial.processing.geo import estimate_utm_crs_with_fallback

        gdf = self.to_geodataframe()
        utm_crs = estimate_utm_crs_with_fallback(gdf)
        gdf["area_sq_m"] = gdf.to_crs(utm_crs).geometry.area

        return (
            gdf.groupby(["building_type", "source"])
            .agg(
                count=("footprint_id", "count"),
                total_area_sq_m=("area_sq_m", "sum"),
            )
            .reset_index()
        )
