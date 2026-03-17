import pandas as pd
from pydantic import Field
from typing import Optional, List, Union, Dict, Set, ClassVar
from pathlib import Path

from gigaspatial.core.io.data_store import DataStore
from .shared import RadioType
from .entity import ENUM_ENTITY_CONFIG, GigaEntityNoLocation, EntityTable
from gigaspatial.processing.entity_processor import EntityProcessor
from gigaspatial.config import config


logger = config.get_logger("CellManager")

RADIO_ALIASES: Dict[RadioType, List[str]] = {
    RadioType.FIVE_G: [
        "5g",
        "nr",
        "new radio",
        "5g nr",
        "5g sa",
        "5g nsa",
        "nr5g",
    ],
    RadioType.FOUR_G: [
        "4g",
        "lte",
        "lte-a",
        "lte advanced",
        "4g lte",
        "e-utra",
        "fdd-lte",
        "tdd-lte",
    ],
    RadioType.THREE_G: [
        "3g",
        "cdma",
        "umts",
        "wcdma",
        "hspa",
        "hspa+",
        "hsdpa",
        "hsupa",
        "evdo",
        "cdma2000",
    ],
    RadioType.TWO_G: [
        "2g",
        "gsm",
        "gprs",
        "edge",
        "cdmaone",
    ],
}

# Reverse lookup: alias string → RadioType value, built once at module level
RADIO_ALIAS_MAP = {
    alias: radio_type.value
    for radio_type, aliases in RADIO_ALIASES.items()
    for alias in aliases
}


class Cell(GigaEntityNoLocation):
    """Represents a cell in a cellular network."""

    model_config = ENUM_ENTITY_CONFIG

    cell_id: str = Field(
        ..., max_length=50, description="Unique identifier for the cell"
    )
    cell_tower_id: str = Field(
        ..., max_length=50, description="Reference to the parent cell tower"
    )
    radio_type: RadioType = Field(..., description="Radio technology type")

    # Additional optional fields that might be useful
    downlink_frequency_mhz: Optional[float] = Field(
        None, ge=0, description="Downlink frequency used by cell in MHz"
    )
    uplink_frequency_mhz: Optional[float] = Field(
        None, ge=0, description="Uplink frequency used by cell in MHz"
    )
    bandwidth_mhz: Optional[float] = Field(
        None, ge=0, description="Channel bandwidth in MHz"
    )
    eirp_dbm: Optional[float] = Field(
        None, description="Effective Isotropic Radiated Power in dBm"
    )
    antenna_height: Optional[float] = Field(
        None, ge=0, description="Height of antenna on tower in meters"
    )
    azimuth_degrees: Optional[float] = Field(
        None,
        ge=0,
        lt=360,
        description="Horizontal pointing direction of antenna in degrees from north",
    )
    mechanical_tilt_degrees: Optional[float] = Field(
        None, ge=-90, le=90, description="Physical tilt angle of antenna in degrees"
    )
    electrical_tilt_degrees: Optional[float] = Field(
        None, ge=-90, le=90, description="Electrical tilt angle of antenna in degrees"
    )
    antenna_model: Optional[str] = Field(
        None, max_length=100, description="Model identifier of antenna"
    )
    antenna_gain_dbi: Optional[float] = Field(
        None, ge=0, description="Gain of the antenna in dBi"
    )
    horizontal_beamwidth_degrees: Optional[float] = Field(
        None, ge=0, le=360, description="Horizontal beam width in degrees"
    )
    vertical_beamwidth_degrees: Optional[float] = Field(
        None, ge=0, le=360, description="Vertical beam width in degrees"
    )
    sector_id: Optional[str] = Field(
        None,
        max_length=20,
        description="Sector identifier on the parent tower (e.g. A, B, C or 1, 2, 3)",
    )
    is_active: Optional[bool] = Field(
        None, description="Whether cell is currently active"
    )

    @property
    def id(self) -> str:
        """Map `id` to `cell_id`."""
        return self.cell_id


# ---------------------------------------------------------------------------
# Processor
# ---------------------------------------------------------------------------


class CellProcessor(EntityProcessor):
    """Cleaning and normalization for Cell entities."""

    # Cell has no coordinates (GigaEntityNoLocation), so latitude/longitude excluded
    NUMERIC_COLUMNS: ClassVar[List[str]] = [
        "downlink_frequency_mhz",
        "uplink_frequency_mhz",
        "bandwidth_mhz",
        "eirp_dbm",
        "antenna_height",
        "azimuth_degrees",
        "mechanical_tilt_degrees",
        "electrical_tilt_degrees",
        "antenna_gain_dbi",
        "horizontal_beamwidth_degrees",
        "vertical_beamwidth_degrees",
    ]

    LOWERCASE_COLUMNS: ClassVar[List[str]] = ["radio_type"]

    RADIO_ALIAS_MAP: ClassVar[Dict[str, str]] = RADIO_ALIAS_MAP

    def process(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = super().process(df, **kwargs)
        df = self._normalize_radio_type(df)
        return df

    def _normalize_radio_type(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize radio_type values to canonical RadioType enum values."""
        return self._normalize_enum_column(
            df,
            column="radio_type",
            alias_map=self.RADIO_ALIAS_MAP,
            valid_values={rt.value for rt in RadioType},
            required=True,
        )


# ---------------------------------------------------------------------------
# Table
# ---------------------------------------------------------------------------


class CellTable(EntityTable[Cell]):
    """Container for Cell entities with cell-specific operations."""

    @classmethod
    def from_file(
        cls,
        file_path: Union[str, Path],
        data_store: Optional[DataStore] = None,
        **kwargs,
    ) -> "CellTable":
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
            entity_class=Cell,
            data_store=data_store,
            processor=CellProcessor,
            **kwargs,
        )

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, entity_class=Cell) -> "CellTable":
        return super().from_dataframe(df=df, entity_class=entity_class)

    # ------------------------------------------------------------------
    # Filters
    # ------------------------------------------------------------------

    def filter_by_tower(self, cell_tower_id_giga: str) -> "CellTable":
        """Return all cells belonging to a specific tower."""
        return self.__class__(
            entities=[
                e for e in self.entities if e.cell_tower_id_giga == cell_tower_id_giga
            ]
        )

    def filter_by_radio_type(self, radio_type: RadioType) -> "CellTable":
        """Return all cells of a specific radio technology."""
        return self.__class__(
            entities=[e for e in self.entities if e.radio_type == radio_type.value]
        )

    def filter_by_radio_types(self, radio_types: Set[RadioType]) -> "CellTable":
        """Return all cells matching any of the given radio technologies."""
        values = {rt.value for rt in radio_types}
        return self.__class__(
            entities=[e for e in self.entities if e.radio_type in values]
        )

    def filter_active(self) -> "CellTable":
        """Return only active cells (is_active=True)."""
        return self.__class__(
            entities=[e for e in self.entities if e.is_active is True]
        )

    def filter_by_sector(self, sector_id: str) -> "CellTable":
        """Return all cells belonging to a specific sector."""
        return self.__class__(
            entities=[e for e in self.entities if e.sector_id == sector_id]
        )

    # ------------------------------------------------------------------
    # Aggregations
    # ------------------------------------------------------------------

    def get_radio_types(self) -> Set[RadioType]:
        """Return the set of all unique radio types present in the table."""
        return {e.radio_type for e in self.entities if e.radio_type is not None}

    def get_cells_for_tower(self, cell_tower_id_giga: str) -> "CellTable":
        """
        Alias for filter_by_tower — returns all cells for a given tower.
        Useful when calling from CellTowerTable.enrich_from_cells.
        """
        return self.filter_by_tower(cell_tower_id_giga)

    def group_by_tower(self) -> dict[str, "CellTable"]:
        """
        Group cells by their parent tower ID.

        Returns:
            Dict mapping cell_tower_id_giga → CellTable of its cells.
        """
        groups: dict[str, list[Cell]] = {}
        for cell in self.entities:
            groups.setdefault(cell.cell_tower_id_giga, []).append(cell)
        return {
            tower_id: self.__class__(entities=cells)
            for tower_id, cells in groups.items()
        }
