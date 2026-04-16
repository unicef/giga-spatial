"""
Module for mobile coverage schema and processing.
Defines the MobileCoverage entity, representing cellular coverage areas
(measured or modeled) with polygon geometries.
"""
from enum import Enum
from typing import Optional, List, Set, Union, Dict, ClassVar
from pydantic import Field
import geopandas as gpd
from shapely.geometry import Point
import pandas as pd
from pathlib import Path

from gigaspatial.core.io.data_store import DataStore
from .shared import DataConfidence, RadioType
from .entity import GEO_ENUM_ENTITY_CONFIG, GigaGeoEntity, EntityTable
from .cell import RADIO_ALIAS_MAP
from gigaspatial.processing.entity_processor import EntityProcessor
from gigaspatial.config import config

logger = config.get_logger("MobileCoverageManager")


class SignalStrength(str, Enum):
    """Enum for signal strength categories."""

    WEAK = "weak"
    VARIABLE = "variable"
    STRONG = "strong"


class CoverageType(str, Enum):
    """Whether coverage data is empirically measured, propagation-modeled, or operator-declared."""

    MEASURED = "measured"
    MODELED = "modeled"
    PREDICTED = "predicted"


SIGNAL_STRENGTH_ALIASES: Dict[SignalStrength, List[str]] = {
    SignalStrength.WEAK: [
        "weak",
        "poor",
        "low",
        "bad",
        "no signal",
        "marginal",
        "-1",
        "1",
    ],
    SignalStrength.VARIABLE: [
        "variable",
        "moderate",
        "fair",
        "medium",
        "average",
        "intermittent",
        "2",
    ],
    SignalStrength.STRONG: [
        "strong",
        "good",
        "excellent",
        "high",
        "full",
        "3",
    ],
}

COVERAGE_TYPE_ALIASES: Dict[CoverageType, List[str]] = {
    CoverageType.MEASURED: [
        "measured",
        "drive test",
        "drive_test",
        "empirical",
        "crowdsourced",
        "field",
        "surveyed",
    ],
    CoverageType.MODELED: [
        "modeled",
        "modelled",
        "propagation",
        "propagation model",
        "simulated",
        "rf model",
        "rf_model",
    ],
    CoverageType.PREDICTED: [
        "predicted",
        "operator declared",
        "operator_declared",
        "regulatory",
        "nominal",
        "theoretical",
        "planned",
    ],
}

# Reverse lookups built once at module level
SIGNAL_STRENGTH_ALIAS_MAP = {
    alias: signal.value
    for signal, aliases in SIGNAL_STRENGTH_ALIASES.items()
    for alias in aliases
}

COVERAGE_TYPE_ALIAS_MAP = {
    alias: coverage_type.value
    for coverage_type, aliases in COVERAGE_TYPE_ALIASES.items()
    for alias in aliases
}


class MobileCoverage(GigaGeoEntity):
    """Represents a cellular coverage area as a polygon or multipolygon geometry."""

    model_config = GEO_ENUM_ENTITY_CONFIG

    # Identity
    mobile_coverage_id: str = Field(
        ..., max_length=50, description="Unique identifier for the mobile coverage area"
    )

    # Classification
    radio_type: RadioType = Field(
        ..., description="Mobile network generation technology"
    )
    coverage_type: Optional[CoverageType] = Field(
        None, description="Whether coverage is measured, modeled, or operator-predicted"
    )
    signal_strength: Optional[SignalStrength] = Field(
        None, description="Categorical signal strength classification"
    )
    operator_name: Optional[str] = Field(
        None, max_length=100, description="Name of the mobile network operator"
    )

    # RF metrics
    average_rssi_dbm: Optional[float] = Field(
        None, ge=-120, le=0, description="Average RSSI in dBm"
    )
    average_rsrp_dbm: Optional[float] = Field(
        None, ge=-140, le=-44, description="Average RSRP in dBm"
    )
    average_sinr_db: Optional[float] = Field(
        None, ge=-20, le=30, description="Average SINR in dB"
    )

    # Provenance
    reported_area_km2: Optional[float] = Field(
        None,
        ge=0,
        description="Coverage area as reported by the source in km². "
        "May differ from area computed from the actual geometry.",
    )
    related_cell_sites: Optional[List[str]] = Field(
        None, description="Cell site IDs contributing to this coverage area"
    )
    measurement_date: Optional[str] = Field(
        None, description="ISO 8601 date when coverage data was collected or modeled"
    )
    data_confidence: Optional[DataConfidence] = Field(
        None, description="Level of confidence in the data source"
    )

    @property
    def id(self) -> str:
        """Map `id` to `mobile_coverage_id`."""
        return self.mobile_coverage_id


class MobileCoverageProcessor(EntityProcessor):
    """Cleaning and normalization for MobileCoverage entities."""

    # MobileCoverage has no lat/lon (GigaGeoEntity), so base numeric
    # columns are excluded — only RF metrics and reported area apply
    NUMERIC_COLUMNS: ClassVar[List[str]] = [
        "average_rssi_dbm",
        "average_rsrp_dbm",
        "average_sinr_db",
        "reported_area_km2",
    ]

    LOWERCASE_COLUMNS: ClassVar[List[str]] = [
        "radio_type",
        "signal_strength",
        "coverage_type",
        "data_confidence",
    ]

    RADIO_ALIAS_MAP: ClassVar[Dict[str, str]] = RADIO_ALIAS_MAP

    SIGNAL_STRENGTH_ALIAS_MAP: ClassVar[Dict[str, str]] = SIGNAL_STRENGTH_ALIAS_MAP

    COVERAGE_TYPE_ALIAS_MAP: ClassVar[Dict[str, str]] = COVERAGE_TYPE_ALIAS_MAP

    def process(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Execute the full processing pipeline for mobile coverage.

        Args:
            df: Raw mobile coverage DataFrame.
            **kwargs: Additional processing arguments.

        Returns:
            Processed and normalized DataFrame.
        """
        df = super().process(df, **kwargs)
        df = self._normalize_radio_type(df)
        df = self._normalize_signal_strength(df)
        df = self._normalize_coverage_type(df)
        df = self._parse_geometry(df)
        return df

    def _normalize_radio_type(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize radio_type values to canonical RadioType enums.

        Args:
            df: DataFrame to normalize.

        Returns:
            DataFrame with normalized radio_type column.
        """
        return self._normalize_enum_column(
            df,
            "radio_type",
            self.RADIO_ALIAS_MAP,
            {rt.value for rt in RadioType},
            required=True,
        )

    def _normalize_signal_strength(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize signal_strength values to canonical SignalStrength enums.

        Args:
            df: DataFrame to normalize.

        Returns:
            DataFrame with normalized signal_strength column.
        """
        return self._normalize_enum_column(
            df,
            "signal_strength",
            self.SIGNAL_STRENGTH_ALIAS_MAP,
            {s.value for s in SignalStrength},
        )

    def _normalize_coverage_type(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize coverage_type values to canonical CoverageType enums.

        Args:
            df: DataFrame to normalize.

        Returns:
            DataFrame with normalized coverage_type column.
        """
        return self._normalize_enum_column(
            df,
            "coverage_type",
            self.COVERAGE_TYPE_ALIAS_MAP,
            {ct.value for ct in CoverageType},
        )


# ---------------------------------------------------------------------------
# Table
# ---------------------------------------------------------------------------


class MobileCoverageTable(EntityTable[MobileCoverage]):
    """Container for MobileCoverage entities with coverage-specific spatial operations."""

    @classmethod
    def from_file(
        cls,
        file_path: Union[str, Path],
        data_store: Optional[DataStore] = None,
        **kwargs,
    ) -> "MobileCoverageTable":
        """
        Create a MobileCoverageTable from a file.

        Args:
            file_path: Path to the dataset file.
            data_store: DataStore instance for file access. Defaults to LocalDataStore.
            **kwargs: Additional arguments forwarded to read_dataset.

        Returns:
            MobileCoverageTable with validated MobileCoverage entities.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file cannot be read or parsed.
        """
        return super().from_file(
            file_path=file_path,
            entity_class=MobileCoverage,
            data_store=data_store,
            processor=MobileCoverageProcessor,
            **kwargs,
        )

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        entity_class: type = MobileCoverage,
        clean: bool = False,
    ) -> "MobileCoverageTable":
        """
        Create a MobileCoverageTable from an existing DataFrame.

        Args:
            df: DataFrame containing mobile coverage data.
            entity_class: Entity class to validate against. Defaults to MobileCoverage.
            clean: Whether to apply MobileCoverageProcessor before validation.
                Defaults to False since DataFrames passed directly are assumed pre-cleaned.

        Returns:
            MobileCoverageTable with validated MobileCoverage entities.
        """
        if clean:
            df = MobileCoverageProcessor().process(df)
        return super().from_dataframe(df=df, entity_class=entity_class)

    # ------------------------------------------------------------------
    # Filters
    # ------------------------------------------------------------------

    def filter_by_radio_type(self, radio_type: RadioType) -> "MobileCoverageTable":
        """Filter coverage areas by radio technology."""
        return self.__class__(
            entities=[e for e in self.entities if e.radio_type == radio_type.value]
        )

    def filter_by_radio_types(
        self, radio_types: Set[RadioType]
    ) -> "MobileCoverageTable":
        """Filter coverage areas matching any of the given radio technologies."""
        values = {rt.value for rt in radio_types}
        return self.__class__(
            entities=[e for e in self.entities if e.radio_type in values]
        )

    def filter_by_operator(self, operator_name: str) -> "MobileCoverageTable":
        """Filter coverage areas by mobile network operator."""
        return self.__class__(
            entities=[e for e in self.entities if e.operator_name == operator_name]
        )

    def filter_by_signal_strength(
        self, signal_strength: SignalStrength
    ) -> "MobileCoverageTable":
        """Filter coverage areas by signal strength category."""
        return self.__class__(
            entities=[
                e for e in self.entities if e.signal_strength == signal_strength.value
            ]
        )

    def filter_by_coverage_type(
        self, coverage_type: CoverageType
    ) -> "MobileCoverageTable":
        """Filter coverage areas by measurement methodology."""
        return self.__class__(
            entities=[
                e for e in self.entities if e.coverage_type == coverage_type.value
            ]
        )

    def filter_by_measurement_date(
        self,
        since: Optional[str] = None,
        until: Optional[str] = None,
    ) -> "MobileCoverageTable":
        """
        Filter coverage areas by measurement date range.

        Args:
            since: ISO 8601 date string. Include records on or after this date.
            until: ISO 8601 date string. Include records on or before this date.
        """
        entities = self.entities
        if since:
            entities = [
                e
                for e in entities
                if e.measurement_date is not None and e.measurement_date >= since
            ]
        if until:
            entities = [
                e
                for e in entities
                if e.measurement_date is not None and e.measurement_date <= until
            ]
        return self.__class__(entities=entities)

    # ------------------------------------------------------------------
    # Spatial operations
    # ------------------------------------------------------------------

    def covers_point(self, lat: float, lon: float) -> "MobileCoverageTable":
        """
        Return all coverage areas whose geometry contains the given point.

        Args:
            lat: Latitude of the point.
            lon: Longitude of the point.

        Returns:
            MobileCoverageTable of coverage polygons that contain the point.
        """
        point = Point(lon, lat)
        return self.__class__(
            entities=[e for e in self.entities if e.geometry.contains(point)]
        )

    def covers_point_by_radio_type(
        self, lat: float, lon: float
    ) -> dict[str, "MobileCoverageTable"]:
        """
        For a given point, return coverage areas grouped by radio type.

        Useful for determining which technologies cover a specific location.

        Args:
            lat: Latitude of the point.
            lon: Longitude of the point.

        Returns:
            Dict mapping radio_type value → MobileCoverageTable.
        """
        covering = self.covers_point(lat, lon)
        groups: dict[str, list[MobileCoverage]] = {}
        for entity in covering.entities:
            groups.setdefault(entity.radio_type, []).append(entity)
        return {
            radio_type: self.__class__(entities=entities)
            for radio_type, entities in groups.items()
        }

    def dissolve_by_operator(self) -> gpd.GeoDataFrame:
        """
        Dissolve coverage geometries by operator and radio type.

        Returns a GeoDataFrame with merged polygons per operator/radio_type pair,
        useful for computing total coverage area without double-counting overlaps.

        Returns:
            GeoDataFrame with columns: operator_name, radio_type, geometry.
        """
        gdf = self.to_geodataframe()
        return gdf.dissolve(by=["operator_name", "radio_type"]).reset_index()[
            ["operator_name", "radio_type", "geometry"]
        ]

    def total_coverage_area_km2(self, dissolve: bool = True) -> float:
        """
        Compute total geographic area covered in square kilometres.

        Args:
            dissolve: If True, dissolve overlapping polygons before computing area
                to avoid double-counting. Defaults to True.

        Returns:
            Total coverage area in km².
        """
        gdf = self.to_geodataframe()

        from gigaspatial.processing.geo import estimate_utm_crs_with_fallback

        utm_crs = estimate_utm_crs_with_fallback(gdf)
        gdf = gdf.to_crs(utm_crs)

        if dissolve:
            gdf = gdf.dissolve().reset_index()

        return gdf.geometry.area.sum() / 1e6

    # ------------------------------------------------------------------
    # Aggregations
    # ------------------------------------------------------------------

    def get_operators(self) -> Set[str]:
        """
        Return the set of all unique operator names present in the table.

        Returns:
            Set of operator name strings.
        """
        return {e.operator_name for e in self.entities if e.operator_name is not None}

    def get_radio_types(self) -> Set[str]:
        """
        Return the set of all unique radio types present in the table.

        Returns:
            Set of radio type strings.
        """
        return {e.radio_type for e in self.entities if e.radio_type is not None}

    def get_coverage_summary(self) -> pd.DataFrame:
        """
        Return a summary DataFrame of coverage counts and area by operator and radio type.

        Returns:
            DataFrame with columns: operator_name, radio_type, polygon_count, total_area_km2.
        """
        gdf = self.to_geodataframe()

        from gigaspatial.processing.geo import estimate_utm_crs_with_fallback

        utm_crs = estimate_utm_crs_with_fallback(gdf)
        gdf["area_km2"] = gdf.to_crs(utm_crs).geometry.area / 1e6

        return (
            gdf.groupby(["operator_name", "radio_type"])
            .agg(
                polygon_count=("mobile_coverage_id", "count"),
                total_area_km2=("area_km2", "sum"),
            )
            .reset_index()
        )
