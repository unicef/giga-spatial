"""
Module for wireless access service schema and processing.

Defines the WirelessAccessService entity, representing a wireless access
capability delivered from a WirelessSite. The schema supports mobile,
fixed-wireless, and mixed access services without requiring detailed
cell-, sector-, or radio-installation-level engineering data.

The module includes:
    - WirelessAccessService: Pydantic entity representing an access capability.
    - WirelessAccessServiceProcessor: Cleaning and normalization pipeline.
    - WirelessAccessServiceTable: Typed entity-table container and filters.

A WirelessAccessService can represent a reported service deployment, a
site-level access role, or an aggregation of lower-level source records.
"""

from pathlib import Path
from typing import ClassVar, Dict, List, Optional, Set, Type, Union

import pandas as pd
from pydantic import Field

from gigaspatial.config import config
from gigaspatial.core.io.data_store import DataStore
from gigaspatial.processing.entity_processor import EntityProcessor

from .entity import ENUM_ENTITY_CONFIG, EntityTable, GigaEntityNoLocation
from .shared import (
    DataConfidence,
    InfrastructureStatus,
    WirelessAccessServiceType,
    WirelessAccessTechnology,
    SpectrumType,
    WIRELESS_ACCESS_TECHNOLOGY_ALIAS_MAP,
)


logger = config.get_logger("WirelessAccessServiceManager")


class WirelessAccessService(GigaEntityNoLocation):
    """
    Represents an access capability delivered from a WirelessSite.

    The entity supports mobile and fixed-wireless access without requiring
    cell-, sector-, or radio-installation-level engineering data. It can
    represent a reported access deployment, a service role, or a meaningful
    aggregation of detailed source records at one wireless site.
    """

    model_config = ENUM_ENTITY_CONFIG

    wireless_access_service_id: str = Field(
        ...,
        max_length=50,
        description="Unique identifier for the wireless access service record",
    )
    wireless_site_id: str = Field(
        ...,
        max_length=50,
        description="Identifier of the WirelessSite hosting this access capability",
    )

    access_service_type: WirelessAccessServiceType = Field(
        ...,
        description="Primary access role delivered from the site",
    )
    service_status: Optional[InfrastructureStatus] = Field(
        None,
        description="Operational status of this access capability",
    )

    access_technologies: Optional[Set[WirelessAccessTechnology]] = Field(
        None,
        description=(
            "Technologies used by this access capability, such as 4G, 5G, Wi-Fi, "
            "WiMAX, or a proprietary system"
        ),
    )
    frequency_bands_mhz: Optional[Set[float]] = Field(
        None,
        description=(
            "Reported frequency bands used by this service in MHz; a summary "
            "rather than a per-sector or per-channel allocation"
        ),
    )
    spectrum_type: Optional[SpectrumType] = Field(
        None,
        description="Spectrum licensing category used by the access capability",
    )

    network_provider: Optional[str] = Field(
        None,
        max_length=100,
        description=(
            "Mobile network operator, ISP, or other provider operating this "
            "specific access capability"
        ),
    )

    equipped_capacity_mbps: Optional[float] = Field(
        None,
        ge=0,
        description="Installed aggregate access capacity in Mbps",
    )
    potential_capacity_mbps: Optional[float] = Field(
        None,
        ge=0,
        description=(
            "Maximum plausible aggregate capacity after documented upgrades or "
            "configuration changes, in Mbps"
        ),
    )
    available_capacity_mbps: Optional[float] = Field(
        None,
        ge=0,
        description=(
            "Capacity currently unallocated for additional connections, in Mbps; "
            "valid only with capacity_observed_at"
        ),
    )
    capacity_observed_at: Optional[str] = Field(
        None,
        description="ISO 8601 timestamp when available capacity was measured",
    )

    sector_count: Optional[int] = Field(
        None,
        ge=1,
        description=(
            "Known number of sectors supporting this access capability; "
            "optional site/service summary, not an individual-sector record"
        ),
    )
    antenna_height_agl_meters: Optional[float] = Field(
        None,
        ge=0,
        description=(
            "Known antenna height above ground level (AGL) for this access "
            "capability, in meters; may be a representative or maximum value "
            "when multiple antennas are used."
        ),
    )

    data_confidence: Optional[DataConfidence] = Field(
        None,
        description="Confidence assigned to the source or observed service data",
    )
    last_verified_date: Optional[str] = Field(
        None,
        description="ISO 8601 date the access capability was last verified",
    )

    @property
    def id(self) -> str:
        """Alias for wireless_access_service_id."""
        return self.wireless_access_service_id


class WirelessAccessServiceProcessor(EntityProcessor):
    """Processor for cleaning and normalizing WirelessAccessService data."""

    NUMERIC_COLUMNS: ClassVar[List[str]] = [
        *EntityProcessor.NUMERIC_COLUMNS,
        "equipped_capacity_mbps",
        "potential_capacity_mbps",
        "available_capacity_mbps",
        "sector_count",
        "antenna_height_agl_meters",
    ]

    LOWERCASE_COLUMNS: ClassVar[List[str]] = [
        "access_service_type",
        "service_status",
        "spectrum_type",
        "data_confidence",
    ]

    COLUMN_ALIASES: ClassVar[Dict[str, str]] = {
        "id": "wireless_access_service_id",
        "service_id": "wireless_access_service_id",
        "access_service_id": "wireless_access_service_id",
        "cell_id": "wireless_access_service_id",
        "wireless_site": "wireless_site_id",
        "site_id": "wireless_site_id",
        "cell_tower_id": "wireless_site_id",
        "tower_id": "wireless_site_id",
        "fixed_wireless_site_id": "wireless_site_id",
        "service_type": "access_service_type",
        "access_type": "access_service_type",
        "technology": "access_technologies",
        "technologies": "access_technologies",
        # Source values such as "4G" and "5G" are normalized to detailed
        # WirelessAccessTechnology values where known; raw generation is not retained.
        "radio_type": "access_technologies",
        "radio_types": "access_technologies",
        "radio_technology": "access_technologies",
        "radio_technologies": "access_technologies",
        "frequency_band_mhz": "frequency_bands_mhz",
        "frequency_mhz": "frequency_bands_mhz",
        "band_mhz": "frequency_bands_mhz",
        "frequency_band": "frequency_bands_mhz",
        "frequency_bands": "frequency_bands_mhz",
        "band": "frequency_bands_mhz",
        "operator": "network_provider",
        "operator_name": "network_provider",
        "provider": "network_provider",
        "network_operator": "network_provider",
        "capacity_mbps": "equipped_capacity_mbps",
        "installed_capacity_mbps": "equipped_capacity_mbps",
        "available_capacity": "available_capacity_mbps",
        "antenna_height": "antenna_height_agl_meters",
        "antenna_height_agl": "antenna_height_agl_meters",
        "is_active": "service_status",
        "status": "service_status",
        "potential_capacity": "potential_capacity_mbps",
        "potential_capacity_mbps": "potential_capacity_mbps",
        "available_capacity_mbps": "available_capacity_mbps",
        "capacity_observed_date": "capacity_observed_at",
        "capacity_observed_at": "capacity_observed_at",
        "last_verified": "last_verified_date",
        "verification_date": "last_verified_date",
    }

    ACCESS_SERVICE_TYPE_ALIAS_MAP: ClassVar[Dict[str, str]] = {
        "mobile": WirelessAccessServiceType.MOBILE.value,
        "mobile cellular": WirelessAccessServiceType.MOBILE.value,
        "cellular": WirelessAccessServiceType.MOBILE.value,
        "mobile broadband": WirelessAccessServiceType.MOBILE.value,
        "fwa": WirelessAccessServiceType.FIXED_WIRELESS.value,
        "fixed wireless": WirelessAccessServiceType.FIXED_WIRELESS.value,
        "fixed_wireless": WirelessAccessServiceType.FIXED_WIRELESS.value,
        "wireless broadband": WirelessAccessServiceType.FIXED_WIRELESS.value,
        "mixed": WirelessAccessServiceType.MIXED.value,
        "mobile and fixed wireless": WirelessAccessServiceType.MIXED.value,
        "unknown": WirelessAccessServiceType.UNKNOWN.value,
    }

    STATUS_ALIAS_MAP: ClassVar[Dict[str, str]] = {
        "active": InfrastructureStatus.OPERATIONAL.value,
        "in service": InfrastructureStatus.OPERATIONAL.value,
        "live": InfrastructureStatus.OPERATIONAL.value,
        "under construction": InfrastructureStatus.UNDER_CONSTRUCTION.value,
        "under_construction": InfrastructureStatus.UNDER_CONSTRUCTION.value,
        "retired": InfrastructureStatus.DECOMMISSIONED.value,
        "decommissioned": InfrastructureStatus.DECOMMISSIONED.value,
        "inactive": InfrastructureStatus.INACTIVE.value,
    }

    SPECTRUM_TYPE_ALIAS_MAP: ClassVar[Dict[str, str]] = {
        "licensed": SpectrumType.LICENSED.value,
        "licenced": SpectrumType.LICENSED.value,
        "license": SpectrumType.LICENSED.value,
        "shared": SpectrumType.SHARED.value,
        "shared spectrum": SpectrumType.SHARED.value,
        "unlicensed": SpectrumType.UNLICENSED.value,
        "unlicenced": SpectrumType.UNLICENSED.value,
        "un-licensed": SpectrumType.UNLICENSED.value,
        "licence exempt": SpectrumType.UNLICENSED.value,
        "license exempt": SpectrumType.UNLICENSED.value,
    }

    def process(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Execute the complete WirelessAccessService normalization pipeline.

        Args:
            df: Raw wireless access service DataFrame.
            **kwargs: Additional processing arguments.

        Returns:
            Processed and normalized DataFrame.
        """
        df = super().process(df, **kwargs)
        df = self._normalize_status_from_boolean(df)
        df = self._normalize_enum_columns(df)
        df = self._normalize_access_technologies(df)
        df = self._normalize_frequency_bands_mhz(df)
        df = self._normalize_sector_count(df)
        df = self._coerce_invalid_nonnegative_values(df)
        return df

    def _normalize_enum_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize scalar enum-backed fields to canonical values.

        Args:
            df: DataFrame to process.

        Returns:
            DataFrame with normalized enum values.
        """
        enum_columns = (
            (
                "access_service_type",
                self.ACCESS_SERVICE_TYPE_ALIAS_MAP,
                {item.value for item in WirelessAccessServiceType},
                True,
            ),
            (
                "service_status",
                self.STATUS_ALIAS_MAP,
                {item.value for item in InfrastructureStatus},
                False,
            ),
            (
                "spectrum_type",
                self.SPECTRUM_TYPE_ALIAS_MAP,
                {item.value for item in SpectrumType},
                False,
            ),
        )

        for column, alias_map, valid_values, required in enum_columns:
            df = self._normalize_enum_column(
                df,
                column=column,
                alias_map=alias_map,
                valid_values=valid_values,
                required=required,
            )

        return df

    def _normalize_access_technologies(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Normalize access technologies to canonical enum values."""
        return self._normalize_enum_collection_column(
            df=df,
            column="access_technologies",
            alias_map=WIRELESS_ACCESS_TECHNOLOGY_ALIAS_MAP,
            valid_values={technology.value for technology in WirelessAccessTechnology},
        )

    def _normalize_frequency_bands_mhz(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert frequency-band values to lists of valid MHz measurements.

        Args:
            df: DataFrame to process.

        Returns:
            DataFrame with normalized frequency_bands_mhz values.
        """
        if "frequency_bands_mhz" not in df.columns:
            return df

        def normalize(value: object) -> Optional[List[float]]:
            if value is None or (isinstance(value, float) and pd.isna(value)):
                return None

            if isinstance(value, str):
                values = (
                    value.replace(";", ",")
                    .replace("|", ",")
                    .replace("/", ",")
                    .split(",")
                )
            elif isinstance(value, (list, tuple, set)):
                values = value
            else:
                values = [value]

            normalized: List[float] = []
            for item in values:
                try:
                    numeric_value = float(item)
                except (TypeError, ValueError):
                    logger.warning(
                        "Invalid frequency-band value '%s'; value omitted.",
                        item,
                    )
                    continue

                if numeric_value <= 0:
                    logger.warning(
                        "Non-positive frequency-band value '%s'; value omitted.",
                        item,
                    )
                    continue

                normalized.append(numeric_value)

            return list(dict.fromkeys(normalized)) or None

        df["frequency_bands_mhz"] = df["frequency_bands_mhz"].apply(normalize)
        return df

    def _normalize_status_from_boolean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert legacy is_active values into service_status when needed.

        This supports old Cell-like input datasets after column alias mapping.

        Args:
            df: DataFrame to process.

        Returns:
            DataFrame with normalized service_status values.
        """
        if "service_status" not in df.columns:
            return df

        active_status = InfrastructureStatus.OPERATIONAL.value
        inactive_status = InfrastructureStatus.INACTIVE.value

        boolean_map = {
            True: active_status,
            False: inactive_status,
            "true": active_status,
            "false": inactive_status,
            "yes": active_status,
            "no": inactive_status,
            "1": active_status,
            "0": inactive_status,
            1: active_status,
            0: inactive_status,
        }

        df["service_status"] = df["service_status"].apply(
            lambda value: boolean_map.get(value, value)
        )
        return df

    def _normalize_sector_count(self, df: pd.DataFrame) -> pd.DataFrame:
        """Retain only positive whole-number sector counts."""
        if "sector_count" not in df.columns:
            return df

        values = pd.to_numeric(df["sector_count"], errors="coerce")
        valid = values.notna() & (values >= 1) & (values % 1 == 0)
        df["sector_count"] = values.where(valid).astype("Int64")
        return df

    def _coerce_invalid_nonnegative_values(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Replace invalid negative values in constrained numeric fields with null.

        Args:
            df: DataFrame to process.

        Returns:
            DataFrame with invalid values coerced to null.
        """
        fields = (
            "equipped_capacity_mbps",
            "potential_capacity_mbps",
            "available_capacity_mbps",
            "sector_count",
            "antenna_height_agl_meters",
        )

        for column in fields:
            if column in df.columns:
                df.loc[df[column] < 0, column] = None

        return df


class WirelessAccessServiceTable(EntityTable[WirelessAccessService]):
    """Container for WirelessAccessService entities and service-level operations."""

    @classmethod
    def from_file(
        cls,
        file_path: Union[str, Path],
        data_store: Optional[DataStore] = None,
        clean: bool = True,
        processor: Optional[
            Union[Type[EntityProcessor], EntityProcessor]
        ] = WirelessAccessServiceProcessor,
        **kwargs,
    ) -> "WirelessAccessServiceTable":
        """
        Create a WirelessAccessServiceTable from a source file.

        Args:
            file_path: Path to the source dataset.
            data_store: Optional DataStore used to access the file.
            clean: Whether to process records before validation.
            processor: Processor class or instance used for normalization.
            **kwargs: Additional loading or processing options.

        Returns:
            Validated WirelessAccessServiceTable.
        """
        return super().from_file(
            file_path=file_path,
            entity_class=WirelessAccessService,
            data_store=data_store,
            clean=clean,
            processor=processor,
            **kwargs,
        )

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        entity_class: Type[WirelessAccessService] = WirelessAccessService,
        clean: bool = False,
        processor: Optional[
            Union[Type[EntityProcessor], EntityProcessor]
        ] = WirelessAccessServiceProcessor,
        **kwargs,
    ) -> "WirelessAccessServiceTable":
        """
        Create a WirelessAccessServiceTable from an existing DataFrame.

        Args:
            df: DataFrame containing service records.
            entity_class: Entity schema used for validation.
            clean: Whether to process records before validation.
            processor: Processor class or instance used for normalization.
            **kwargs: Additional processing options.

        Returns:
            Validated WirelessAccessServiceTable.
        """
        return super().from_dataframe(
            df=df,
            entity_class=entity_class,
            clean=clean,
            processor=processor,
            **kwargs,
        )

    @classmethod
    def from_files(
        cls,
        file_paths: List[Union[str, Path]],
        data_store: Optional[DataStore] = None,
        clean: bool = True,
        processor: Optional[
            Union[Type[EntityProcessor], EntityProcessor]
        ] = WirelessAccessServiceProcessor,
        **kwargs,
    ) -> "WirelessAccessServiceTable":
        """
        Load and combine several source files into one table.

        Args:
            file_paths: Paths to source datasets.
            data_store: Optional DataStore used to access files.
            clean: Whether to process records before validation.
            processor: Processor class or instance used for normalization.
            **kwargs: Additional loading or processing options.

        Returns:
            Merged and validated WirelessAccessServiceTable.
        """
        return super().from_files(
            file_paths=file_paths,
            entity_class=WirelessAccessService,
            data_store=data_store,
            clean=clean,
            processor=processor,
            **kwargs,
        )

    def filter_by_wireless_site(
        self,
        wireless_site_id: str,
    ) -> "WirelessAccessServiceTable":
        """
        Return access services hosted by a given wireless site.

        Args:
            wireless_site_id: Canonical WirelessSite identifier.

        Returns:
            Table containing services at the requested site.
        """
        return self.__class__(
            entities=[
                entity
                for entity in self.entities
                if entity.wireless_site_id == wireless_site_id
            ]
        )

    def filter_by_service_type(
        self,
        service_type: Union[WirelessAccessServiceType, str],
    ) -> "WirelessAccessServiceTable":
        """
        Filter access services by service role.

        Args:
            service_type: Access-service enum or its canonical string value.

        Returns:
            Table containing matching services.
        """
        value = (
            WirelessAccessServiceType(service_type).value
            if isinstance(service_type, str)
            else service_type.value
        )
        return self.__class__(
            entities=[
                entity
                for entity in self.entities
                if entity.access_service_type == value
            ]
        )

    def filter_by_status(
        self,
        status: Union[InfrastructureStatus, str],
    ) -> "WirelessAccessServiceTable":
        """
        Filter access services by operational status.

        Args:
            status: InfrastructureStatus enum or canonical string value.

        Returns:
            Table containing matching services.
        """
        value = (
            InfrastructureStatus(status).value
            if isinstance(status, str)
            else status.value
        )
        return self.__class__(
            entities=[
                entity for entity in self.entities if entity.service_status == value
            ]
        )

    def filter_operational(self) -> "WirelessAccessServiceTable":
        """
        Return only operational access services.

        Returns:
            Table containing operational services.
        """
        return self.filter_by_status(InfrastructureStatus.OPERATIONAL)

    def filter_by_technology(
        self,
        technology: Union[WirelessAccessTechnology, str],
    ) -> "WirelessAccessServiceTable":
        """
        Filter services that use a requested access technology.

        Args:
            technology: Access technology enum or canonical string value.

        Returns:
            Table containing matching services.
        """
        value = (
            WirelessAccessTechnology(technology).value
            if isinstance(technology, str)
            else technology.value
        )
        return self.__class__(
            entities=[
                entity
                for entity in self.entities
                if entity.access_technologies and value in entity.access_technologies
            ]
        )

    def filter_by_provider(
        self,
        provider: str,
    ) -> "WirelessAccessServiceTable":
        """
        Filter services by the operating network provider.

        Args:
            provider: Exact provider name to match.

        Returns:
            Table containing matching services.
        """
        return self.__class__(
            entities=[
                entity
                for entity in self.entities
                if entity.network_provider == provider
            ]
        )

    def filter_by_min_capacity(
        self,
        min_mbps: float,
        capacity_type: str = "equipped",
    ) -> "WirelessAccessServiceTable":
        """
        Filter services by a capacity threshold.

        Args:
            min_mbps: Minimum capacity in Mbps.
            capacity_type: One of ``equipped``, ``potential``, or ``available``.

        Returns:
            Table containing matching services.

        Raises:
            ValueError: If min_mbps is negative or capacity_type is unsupported.
        """
        if min_mbps < 0:
            raise ValueError("min_mbps must be greater than or equal to zero.")

        field_map = {
            "equipped": "equipped_capacity_mbps",
            "potential": "potential_capacity_mbps",
            "available": "available_capacity_mbps",
        }

        if capacity_type not in field_map:
            raise ValueError(
                f"Unsupported capacity_type '{capacity_type}'. "
                f"Expected one of: {list(field_map)}."
            )

        field_name = field_map[capacity_type]
        return self.__class__(
            entities=[
                entity
                for entity in self.entities
                if getattr(entity, field_name) is not None
                and getattr(entity, field_name) >= min_mbps
            ]
        )

    def get_wireless_site_ids(self) -> Set[str]:
        """
        Return the wireless-site identifiers referenced by this table.

        Returns:
            Set of WirelessSite identifiers.
        """
        return {entity.wireless_site_id for entity in self.entities}

    def get_network_providers(self) -> Set[str]:
        """
        Return the service operators represented in the table.

        Returns:
            Set of non-null network-provider names.
        """
        return {
            entity.network_provider
            for entity in self.entities
            if entity.network_provider is not None
        }

    def get_access_technologies(self) -> Set[str]:
        """
        Return the access technologies represented in the table.

        Returns:
            Set of canonical WirelessAccessTechnology values.
        """
        return {
            technology
            for entity in self.entities
            if entity.access_technologies
            for technology in entity.access_technologies
        }

    def group_by_wireless_site(self) -> Dict[str, "WirelessAccessServiceTable"]:
        """
        Group access services by their parent WirelessSite identifier.

        Returns:
            Mapping from wireless-site ID to its service table.
        """
        groups: Dict[str, List[WirelessAccessService]] = {}

        for entity in self.entities:
            groups.setdefault(entity.wireless_site_id, []).append(entity)

        return {
            wireless_site_id: self.__class__(entities=entities)
            for wireless_site_id, entities in groups.items()
        }
