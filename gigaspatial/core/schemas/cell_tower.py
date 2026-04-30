"""
Module for cell tower schema and processing.
Defines the CellTower entity, representing physical cellular tower infrastructure.
"""

import pandas as pd
from pydantic import Field
from typing import Optional, List, Union, Dict, Set, ClassVar
from enum import Enum
from pathlib import Path

from gigaspatial.core.io.data_store import DataStore
from .shared import DataConfidence, PowerSource, RadioType
from .entity import ENUM_ENTITY_CONFIG, GigaEntity, EntityTable
from gigaspatial.processing.entity_processor import EntityProcessor
from gigaspatial.config import config


logger = config.get_logger("CellTowerManager")


class BackhaulType(str, Enum):
    """Enum for different backhaul types."""

    FIBER = "fiber"
    MICROWAVE = "microwave"
    SATELLITE = "satellite"


BACKHAUL_ALIASES: Dict[BackhaulType, List[str]] = {
    BackhaulType.FIBER: [
        "fibre",
        "fiber optic",
        "fibre optic",
        "optical fiber",
        "optical fibre",
        "fo",
        "fttx",
        "ftth",
        "fttp",
    ],
    BackhaulType.MICROWAVE: [
        "mw",
        "micro wave",
        "micro-wave",
        "microwave link",
        "mw link",
        "point to point",
        "p2p",
        "ptp",
        "lte backhaul",
        "wireless",
    ],
    BackhaulType.SATELLITE: [
        "sat",
        "sattelite",  # common misspelling
        "satelite",  # common misspelling
        "vsat",
        "leo",
        "geo",
        "meo",
        "starlink",
        "geostationary",
    ],
}

BACKHAUL_ALIAS_MAP = {
    alias: backhaul_type.value
    for backhaul_type, aliases in BACKHAUL_ALIASES.items()
    for alias in aliases
}


class CellTower(GigaEntity):
    """Represents a cell tower."""

    model_config = ENUM_ENTITY_CONFIG

    cell_tower_id: str = Field(
        ..., max_length=50, description="Unique identifier for the cell tower"
    )
    cell_tower_id_source: Optional[str] = Field(
        None, max_length=50, description="Original tower identifier in source system"
    )
    tower_height: Optional[float] = Field(
        None, ge=0, description="Total height of the tower structure in meters"
    )
    site_name: Optional[str] = Field(
        None,
        max_length=100,
        description="Common/operational name of the cell tower site",
    )
    operator_name: Optional[str] = Field(
        None, description="Name of the mobile network operator"
    )
    data_confidence: Optional[DataConfidence] = Field(
        None, description="Level of confidence in site data"
    )
    backhaul_type: Optional[BackhaulType] = Field(
        None, description="Type of backhaul connection"
    )
    backhaul_capacity_mbps: Optional[float] = Field(
        None, ge=0, description="Throughput capacity of backhaul in Mbps"
    )
    power_source: Optional[PowerSource] = Field(
        None, description="Type of power source used"
    )
    radio_types: Optional[Set[RadioType]] = Field(
        None,
        description="Set of radio technologies available on this tower (e.g. {4G, 5G})",
    )
    frequency_bands_mhz: Optional[Set[float]] = Field(
        None, description="Set of frequency bands available on this tower in MHz"
    )
    max_antenna_height: Optional[float] = Field(
        None, ge=0, description="Height of the highest antenna on the tower in meters"
    )
    sector_count: Optional[int] = Field(
        None, ge=1, description="Number of sectors on the tower"
    )

    @property
    def id(self) -> str:
        """Map `id` to `cell_tower_id`."""
        return self.cell_tower_id


# ---------------------------------------------------------------------------
# Processor
# ---------------------------------------------------------------------------


class CellTowerProcessor(EntityProcessor):
    """Cell tower-specific cleaning and normalization."""

    NUMERIC_COLUMNS: ClassVar[List[str]] = [
        *EntityProcessor.NUMERIC_COLUMNS,  # inherit base numeric columns
        "tower_height",
        "backhaul_capacity_mbps",
        "max_antenna_height",
        "sector_count",
    ]
    LOWERCASE_COLUMNS: ClassVar[List[str]] = [
        "backhaul_type",
        "power_source",
        "data_confidence",
    ]
    COLUMN_ALIASES: ClassVar[Dict[str, str]] = {
        "op": "operator_name",
        "operator": "operator_name",
        "site": "site_name",
        "height": "tower_height",
    }

    BACKHAUL_ALIAS_MAP: ClassVar[Dict[str, str]] = BACKHAUL_ALIAS_MAP

    def process(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Execute the full processing pipeline for cell towers.

        Args:
            df: Raw cell tower DataFrame.
            **kwargs: Additional processing arguments.

        Returns:
            Processed and normalized DataFrame.
        """
        df = super().process(df, **kwargs)
        df = self._normalize_backhaul(df)
        df = self._normalize_tower_height(df)
        return df

    def _normalize_backhaul(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize backhaul_type values to canonical BackhaulType enums.

        Args:
            df: DataFrame to normalize.

        Returns:
            DataFrame with normalized backhaul_type column.
        """
        return self._normalize_enum_column(
            df,
            column="backhaul_type",
            alias_map=self.BACKHAUL_ALIAS_MAP,
            valid_values={bt.value for bt in BackhaulType},
            required=False,
        )

    def _normalize_tower_height(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Coerce negative tower heights to None pre-validation.
        Prevents entire row rejection due to a single out-of-range value.
        """
        if "tower_height" in df.columns:
            df.loc[df["tower_height"] < 0, "tower_height"] = None
        return df


# ---------------------------------------------------------------------------
# Table
# ---------------------------------------------------------------------------


class CellTowerTable(EntityTable[CellTower]):
    """Container for CellTower entities with tower-specific operations."""

    @classmethod
    def from_file(
        cls,
        file_path: Union[str, Path],
        data_store: Optional[DataStore] = None,
        **kwargs,
    ) -> "CellTowerTable":
        """
        Create a CellTable from a file.

        Args:
            file_path: Path to the dataset file.
            data_store: DataStore instance for file access. Defaults to LocalDataStore.
            **kwargs: Additional arguments forwarded to read_dataset.

        Returns:
            CellTable instance with validated Cell entities.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file cannot be read or parsed.
        """
        return super().from_file(
            file_path=file_path,
            entity_class=CellTower,
            data_store=data_store,
            processor=CellTowerProcessor,
            **kwargs,
        )

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        entity_class=CellTower,
        clean: bool = False,
        **kwargs,
    ) -> "CellTowerTable":
        """
        Create a CellTowerTable from an existing DataFrame.

        Args:
            df: DataFrame containing cell tower data.
            entity_class: Entity class to validate against. Defaults to CellTower.
            clean: Whether to apply CellTowerProcessor before validation.
                Defaults to False since DataFrames passed directly are assumed pre-cleaned.

        Returns:
            CellTowerTable instance with validated CellTower entities.
        """
        return super().from_dataframe(
            df=df,
            entity_class=entity_class,
            clean=clean,
            processor=CellTowerProcessor(),
            **kwargs,
        )

    @classmethod
    def from_files(
        cls,
        file_paths: List[Union[str, Path]],
        data_store: Optional[DataStore] = None,
        **kwargs,
    ) -> "CellTowerTable":
        """
        Load and merge multiple CellTower source files into a single CellTowerTable.

        Args:
            file_paths: List of paths to source files.
            data_store: DataStore instance for file access. Defaults to LocalDataStore.
            **kwargs: Additional arguments forwarded to read_dataset.

        Returns:
            CellTowerTable with merged and validated CellTower entities.
        """
        return super().from_files(
            file_paths=file_paths,
            entity_class=CellTower,
            data_store=data_store,
            processor=CellTowerProcessor,
            **kwargs,
        )

    def filter_by_operator(self, operator_name: str) -> "CellTowerTable":
        """Filter towers by mobile network operator."""
        return self.__class__(
            entities=[e for e in self.entities if e.operator_name == operator_name]
        )

    def filter_by_backhaul(self, backhaul_type: BackhaulType) -> "CellTowerTable":
        """Filter towers by backhaul type."""
        return self.__class__(
            entities=[e for e in self.entities if e.backhaul_type == backhaul_type]
        )

    def enrich_from_cells(self, cell_table: "CellTable") -> "CellTowerTable":
        """
        Populate denormalized cell summary fields on each tower.

        Args:
            cell_table: Table containing cell-level data to aggregate.

        Returns:
            New CellTowerTable with enriched towers.
        """

        cell_by_tower = {}
        for cell in cell_table:
            cell_by_tower.setdefault(cell.cell_tower_id, []).append(cell)

        enriched = []
        for tower in self.entities:
            cells = cell_by_tower.get(tower.cell_tower_id, [])
            if cells:
                tower = tower.model_copy(
                    update={
                        "radio_types": {c.radio_type for c in cells},
                        "frequency_bands_mhz": {
                            c.downlink_frequency_mhz
                            for c in cells
                            if c.downlink_frequency_mhz is not None
                        },
                        "max_antenna_height": max(
                            (
                                c.antenna_height
                                for c in cells
                                if c.antenna_height is not None
                            ),
                            default=None,
                        ),
                        "sector_count": len(
                            {c.sector_id for c in cells if c.sector_id}
                        ),
                    }
                )
            enriched.append(tower)

        return self.__class__(entities=enriched)
