"""
Module for wireless site schema and processing.

Defines the WirelessSite entity, representing a physical location that hosts
one or more wireless installations. A wireless site can be a tower, rooftop,
mast, pole, relay, hub, or customer-premises location. It may host mobile and
fixed-wireless equipment concurrently.

This entity represents the shared physical-site inventory. Radio technologies,
spectrum, antenna configuration, sectors, and access-service capacity belong
to service- or installation-level schemas rather than this site-level schema.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import ClassVar, Dict, List, Optional, Set, Type, Union, TYPE_CHECKING

import pandas as pd
from pydantic import Field

from gigaspatial.config import config
from gigaspatial.core.io.data_store import DataStore
from gigaspatial.processing.entity_processor import EntityProcessor

from .entity import ENUM_ENTITY_CONFIG, EntityTable, GigaEntity
from .shared import (
    enum_value,
    DataConfidence,
    InfrastructureStatus,
    PowerSource,
    RadioType,
    RADIO_ALIAS_MAP,
    ACCESS_SERVICE_TYPE_ALIAS_MAP,
    WIRELESS_ACCESS_TECHNOLOGY_ALIAS_MAP,
    WIRELESS_ACCESS_TECHNOLOGY_TO_RADIO_TYPE,
    WirelessAccessServiceType,
    WirelessAccessTechnology,
)

if TYPE_CHECKING:
    from .wireless_access_service import WirelessAccessServiceTable

logger = config.get_logger("WirelessSiteManager")


class BackhaulType(str, Enum):
    """Enum for different backhaul types."""

    FIBER = "fiber"
    MICROWAVE = "microwave"
    CELLULAR = "cellular"
    SATELLITE = "satellite"


BACKHAUL_ALIASES: Dict[BackhaulType, List[str]] = {
    BackhaulType.FIBER: [
        "fiber",
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
        "microwave",
        "mw",
        "micro wave",
        "micro-wave",
        "microwave link",
        "mw link",
        "point to point",
        "p2p",
        "ptp",
    ],
    BackhaulType.CELLULAR: [
        "cellular",
        "cellular backhaul",
        "mobile backhaul",
        "lte backhaul",
        "4g backhaul",
        "5g backhaul",
        "nr backhaul",
        "lte",
        "5g",
    ],
    BackhaulType.SATELLITE: [
        "satellite",
        "sat",
        "sattelite",
        "satelite",
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


class WirelessSiteType(str, Enum):
    """Enum for physical wireless site types."""

    TOWER = "tower"
    ROOFTOP = "rooftop"
    MAST = "mast"
    POLE = "pole"
    RELAY = "relay"
    HUB = "hub"
    CUSTOMER_PREMISES = "customer_premises"


class WirelessSite(GigaEntity):
    """
    Represents a physical site that hosts one or more wireless installations.

    A site may support mobile and fixed-wireless access simultaneously. The
    entity models shared physical infrastructure, power, providers, backhaul,
    and optional upstream transport relationships; it does not model a
    customer-specific connection or an individual radio installation.
    """

    model_config = ENUM_ENTITY_CONFIG

    # Identity
    wireless_site_id: str = Field(
        ...,
        max_length=50,
        description="Unique identifier for the wireless site",
    )
    wireless_site_id_source: Optional[str] = Field(
        None,
        max_length=100,
        description="Original site identifier in the source system",
    )
    site_name: Optional[str] = Field(
        None,
        max_length=150,
        description="Common, operational, or published name of the site",
    )

    # Classification
    site_type: Optional[WirelessSiteType] = Field(
        None,
        description="Physical type of site hosting wireless installations",
    )
    site_status: Optional[InfrastructureStatus] = Field(
        None,
        description="Current operational status of the wireless site",
    )
    access_service_types: Optional[Set[WirelessAccessServiceType]] = Field(
        None,
        description=(
            "Access-service roles known to be supported from the site, aggregated "
            "from linked WirelessAccessService records or reported directly. Does "
            "not confirm serviceability at a particular point of interest."
        ),
    )
    radio_types: Optional[Set[RadioType]] = Field(
        None,
        description=(
            "Observed or reported mobile generations hosted at the site, aggregated "
            "from linked WirelessAccessService records or reported directly. This is "
            "a site-level summary, not a per-service configuration."
        ),
    )
    access_technologies: Optional[Set[WirelessAccessTechnology]] = Field(
        None,
        description=(
            "Observed or reported wireless access technologies hosted at the site, "
            "aggregated from linked WirelessAccessService records or reported "
            "directly. This is a site-level summary, not a per-service configuration."
        ),
    )

    # Physical characteristics
    structure_height_meters: Optional[float] = Field(
        None,
        ge=0,
        description=(
            "Height above ground of the supporting tower, mast, pole, or other "
            "freestanding structure, in meters; not applicable to rooftop-only sites."
        ),
    )

    highest_antenna_height_agl_meters: Optional[float] = Field(
        None,
        ge=0,
        description=(
            "Highest known hosted antenna height above ground level (AGL), in "
            "meters; a site-level summary across hosted installations."
        ),
    )
    elevation_meters: Optional[float] = Field(
        None,
        description="Site elevation above mean sea level in meters",
    )
    power_source: Optional[PowerSource] = Field(
        None,
        description="Primary power source serving the wireless site",
    )
    backup_power_hours: Optional[float] = Field(
        None,
        ge=0,
        description="Estimated backup power autonomy in hours",
    )

    # Transport and backhaul
    backhaul_type: Optional[BackhaulType] = Field(
        None,
        description="Primary known backhaul medium serving the site",
    )
    backhaul_capacity_mbps: Optional[float] = Field(
        None,
        ge=0,
        description="Known aggregate backhaul capacity serving the site in Mbps",
    )
    upstream_transmission_node_id: Optional[str] = Field(
        None,
        max_length=50,
        description=(
            "Identifier of the known upstream TransmissionNode providing transport "
            "or aggregation connectivity"
        ),
    )

    # Ownership and operations
    physical_infrastructure_provider: Optional[str] = Field(
        None,
        max_length=100,
        description=(
            "Organization that owns, hosts, or provides the passive physical "
            "infrastructure or site"
        ),
    )
    network_providers: Optional[List[str]] = Field(
        None,
        description=(
            "Network operators, mobile network operators, or internet service "
            "providers operating active equipment at the site"
        ),
    )

    # Data quality and lifecycle
    installation_date: Optional[str] = Field(
        None,
        description="ISO 8601 date when the site became operational",
    )
    decommission_date: Optional[str] = Field(
        None,
        description="ISO 8601 date when the site was decommissioned",
    )
    last_verified_date: Optional[str] = Field(
        None,
        description="ISO 8601 date when the site information was last verified",
    )
    data_confidence: Optional[DataConfidence] = Field(
        None,
        description="Confidence level assigned to the source or observed data",
    )

    @property
    def id(self) -> str:
        """Alias for wireless_site_id."""
        return self.wireless_site_id


class WirelessSiteProcessor(EntityProcessor):
    """
    Processor for cleaning and normalizing WirelessSite data.

    The processor normalizes source aliases for site types, lifecycle status,
    power, backhaul, and provider lists before Pydantic validation.
    """

    NUMERIC_COLUMNS: ClassVar[List[str]] = [
        *EntityProcessor.NUMERIC_COLUMNS,
        "structure_height_meters",
        "highest_antenna_height_agl_meters",
        "elevation_meters",
        "backup_power_hours",
        "backhaul_capacity_mbps",
    ]

    LOWERCASE_COLUMNS: ClassVar[List[str]] = [
        "site_type",
        "site_status",
        "backhaul_type",
        "power_source",
        "data_confidence",
    ]

    COLUMN_ALIASES: ClassVar[Dict[str, str]] = {
        "id": "wireless_site_id",
        "site_id": "wireless_site_id",
        "tower_id": "wireless_site_id",
        "cell_tower_id": "wireless_site_id",
        "fixed_wireless_site_id": "wireless_site_id",
        "source_id": "wireless_site_id_source",
        "tower_id_source": "wireless_site_id_source",
        "cell_tower_id_source": "wireless_site_id_source",
        "fixed_wireless_site_id_source": "wireless_site_id_source",
        "site": "site_name",
        "tower_name": "site_name",
        "site_category": "site_type",
        "tower_type": "site_type",
        "operator": "network_providers",
        "operator_name": "network_providers",
        "network_operator": "network_providers",
        "provider": "network_providers",
        "backhaul": "backhaul_type",
        "backhaul_capacity": "backhaul_capacity_mbps",
        "transmission_node_id": "upstream_transmission_node_id",
        "height": "structure_height_meters",
        "tower_height": "structure_height_meters",
        "structure_height": "structure_height_meters",
        "max_antenna_height": "highest_antenna_height_agl_meters",
        "highest_antenna_height": "highest_antenna_height_agl_meters",
        "antenna_height": "highest_antenna_height_agl_meters",
        "antenna_height_agl": "highest_antenna_height_agl_meters",
        "service_type": "access_service_types",
        "service_types": "access_service_types",
        "access_service_type": "access_service_types",
        "access_service_types": "access_service_types",
        "technology": "access_technologies",
        "technologies": "access_technologies",
        "radio_technology": "access_technologies",
        "radio_technologies": "access_technologies",
        "radio_type": "radio_types",
        "radio_types": "radio_types",
        "generation": "radio_types",
        "network_generation": "radio_types",
    }

    SITE_TYPE_ALIAS_MAP: ClassVar[Dict[str, str]] = {
        "tower": "tower",
        "cell tower": "tower",
        "tower site": "tower",
        "macro tower": "tower",
        "rooftop": "rooftop",
        "roof top": "rooftop",
        "roof-top": "rooftop",
        "mast": "mast",
        "pole": "pole",
        "utility pole": "pole",
        "street pole": "pole",
        "relay": "relay",
        "repeater": "relay",
        "hub": "hub",
        "pop": "hub",
        "point of presence": "hub",
        "customer premises": "customer_premises",
        "customer premise": "customer_premises",
        "cpe": "customer_premises",
    }

    STATUS_ALIAS_MAP: ClassVar[Dict[str, str]] = {
        "under construction": "underconstruction",
        "under_construction": "underconstruction",
        "in service": "operational",
        "active": "operational",
        "retired": "decommissioned",
    }

    def process(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Execute the complete WirelessSite normalization pipeline."""
        df = super().process(df, **kwargs)
        df = self._normalize_enum_columns(df)
        df = self._normalize_provider_lists(df)
        df = self._normalize_access_service_types(df)
        df = self._normalize_radio_types(df)
        df = self._normalize_access_technologies(df)
        df = self._derive_radio_types(df)
        df = self._coerce_invalid_nonnegative_values(df)
        return df

    def _normalize_enum_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize enum-backed source fields to canonical values.

        Args:
            df: DataFrame to process.

        Returns:
            DataFrame with normalized enum-backed columns.
        """
        enum_columns = (
            (
                "site_type",
                self.SITE_TYPE_ALIAS_MAP,
                {site_type.value for site_type in WirelessSiteType},
            ),
            (
                "site_status",
                self.STATUS_ALIAS_MAP,
                {status.value for status in InfrastructureStatus},
            ),
            (
                "backhaul_type",
                BACKHAUL_ALIAS_MAP,
                {backhaul_type.value for backhaul_type in BackhaulType},
            ),
        )

        for column, alias_map, valid_values in enum_columns:
            df = self._normalize_enum_column(
                df,
                column=column,
                alias_map=alias_map,
                valid_values=valid_values,
                required=False,
            )

        return df

    def _normalize_provider_lists(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert provider fields into de-duplicated lists.

        Args:
            df: DataFrame to process.

        Returns:
            DataFrame with normalized network_providers values.
        """
        if "network_providers" not in df.columns:
            return df

        def normalize(value: object) -> Optional[List[str]]:
            if value is None or (isinstance(value, float) and pd.isna(value)):
                return None

            if isinstance(value, str):
                values = value.replace(";", ",").split(",")
            elif isinstance(value, (list, tuple, set)):
                values = value
            else:
                return None

            normalized = [
                str(item).strip()
                for item in values
                if item is not None and str(item).strip()
            ]
            return list(dict.fromkeys(normalized)) or None

        df["network_providers"] = df["network_providers"].apply(normalize)
        return df

    def _coerce_invalid_nonnegative_values(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Replace negative values for nonnegative fields with null values.

        This prevents an otherwise valid record from failing Pydantic validation
        because of one invalid source measurement.

        Args:
            df: DataFrame to process.

        Returns:
            DataFrame with invalid negative values coerced to null.
        """
        fields = (
            "structure_height_meters",
            "highest_antenna_height_agl_meters",
            "backup_power_hours",
            "backhaul_capacity_mbps",
        )

        for column in fields:
            if column in df.columns:
                df.loc[df[column] < 0, column] = None

        return df

    def _normalize_radio_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize reported 2G–5G values to canonical RadioType values."""
        return self._normalize_enum_collection_column(
            df=df,
            column="radio_types",
            alias_map=RADIO_ALIAS_MAP,
            valid_values={radio_type.value for radio_type in RadioType},
        )

    def _derive_radio_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Derive and merge 2G–5G summaries from detailed access technologies.

        Existing normalized radio_types are preserved. Wi-Fi, WiMAX, proprietary,
        and other technologies do not map to RadioType and are excluded.
        """
        if "access_technologies" not in df.columns:
            return df

        def derive(technologies: object) -> List[str]:
            if not technologies:
                return []

            derived: List[str] = []

            for technology in technologies:
                technology_enum = WirelessAccessTechnology(technology)
                radio_type = WIRELESS_ACCESS_TECHNOLOGY_TO_RADIO_TYPE.get(
                    technology_enum
                )
                if radio_type is not None:
                    derived.append(radio_type.value)

            return list(dict.fromkeys(derived))

        derived_radio_types = df["access_technologies"].apply(derive)

        if "radio_types" not in df.columns:
            df["radio_types"] = derived_radio_types.apply(lambda values: values or None)
            return df

        def merge(
            reported: object,
            derived: List[str],
        ) -> Optional[List[str]]:
            """Merge reported and derived radio types without duplicate values."""
            if reported is None or (isinstance(reported, float) and pd.isna(reported)):
                reported_values: List[str] = []
            else:
                reported_values = list(reported)

            values = [*reported_values, *derived]
            return list(dict.fromkeys(values)) or None

        df["radio_types"] = [
            merge(reported, derived)
            for reported, derived in zip(df["radio_types"], derived_radio_types)
        ]
        return df

    def _normalize_access_service_types(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Normalize site-level wireless access-service roles."""
        return self._normalize_enum_collection_column(
            df=df,
            column="access_service_types",
            alias_map=ACCESS_SERVICE_TYPE_ALIAS_MAP,
            valid_values={
                service_type.value for service_type in WirelessAccessServiceType
            },
        )

    def _normalize_access_technologies(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Normalize site-level detailed wireless access technologies."""
        return self._normalize_enum_collection_column(
            df=df,
            column="access_technologies",
            alias_map=WIRELESS_ACCESS_TECHNOLOGY_ALIAS_MAP,
            valid_values={technology.value for technology in WirelessAccessTechnology},
        )


class WirelessSiteTable(EntityTable[WirelessSite]):
    """Container for WirelessSite entities and site-level operations."""

    @classmethod
    def from_file(
        cls,
        file_path: Union[str, Path],
        data_store: Optional[DataStore] = None,
        clean: bool = True,
        processor: Optional[
            Union[Type[EntityProcessor], EntityProcessor]
        ] = WirelessSiteProcessor,
        **kwargs,
    ) -> "WirelessSiteTable":
        """
        Create a WirelessSiteTable from a file.

        Args:
            file_path: Path to the source dataset.
            data_store: Optional data-store implementation.
            clean: Whether to run the configured processor before validation.
            processor: Processor class or instance to use.
            **kwargs: Additional loading and processing options.

        Returns:
            Validated WirelessSiteTable.
        """
        return super().from_file(
            file_path=file_path,
            entity_class=WirelessSite,
            data_store=data_store,
            clean=clean,
            processor=processor,
            **kwargs,
        )

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        entity_class: Type[WirelessSite] = WirelessSite,
        clean: bool = False,
        processor: Optional[
            Union[Type[EntityProcessor], EntityProcessor]
        ] = WirelessSiteProcessor,
        **kwargs,
    ) -> "WirelessSiteTable":
        """
        Create a WirelessSiteTable from an existing DataFrame.

        Args:
            df: DataFrame containing wireless-site records.
            entity_class: Entity schema used to validate the records.
            clean: Whether to run the configured processor before validation.
            processor: Processor class or instance to use.
            **kwargs: Additional processing options.

        Returns:
            Validated WirelessSiteTable.
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
        ] = WirelessSiteProcessor,
        **kwargs,
    ) -> "WirelessSiteTable":
        """
        Load, process, and merge several source files into one table.

        Args:
            file_paths: Paths to the source datasets.
            data_store: Optional data-store implementation.
            clean: Whether to run the configured processor before validation.
            processor: Processor class or instance to use.
            **kwargs: Additional loading and processing options.

        Returns:
            Merged and validated WirelessSiteTable.
        """
        return super().from_files(
            file_paths=file_paths,
            entity_class=WirelessSite,
            data_store=data_store,
            clean=clean,
            processor=processor,
            **kwargs,
        )

    def filter_by_site_type(
        self,
        site_type: Union[WirelessSiteType, str],
    ) -> "WirelessSiteTable":
        """
        Filter sites by physical type.

        Args:
            site_type: WirelessSiteType or its canonical string value.

        Returns:
            Table containing matching sites.
        """
        value = (
            WirelessSiteType(site_type).value
            if isinstance(site_type, str)
            else site_type.value
        )
        return self.__class__(
            entities=[entity for entity in self.entities if entity.site_type == value]
        )

    def filter_by_status(
        self,
        status: Union[InfrastructureStatus, str],
    ) -> "WirelessSiteTable":
        """
        Filter sites by operational status.

        Args:
            status: InfrastructureStatus or its canonical string value.

        Returns:
            Table containing matching sites.
        """
        value = (
            InfrastructureStatus(status).value
            if isinstance(status, str)
            else status.value
        )
        return self.__class__(
            entities=[entity for entity in self.entities if entity.site_status == value]
        )

    def filter_operational(self) -> "WirelessSiteTable":
        """
        Return operational sites only.

        Returns:
            Table containing operational sites.
        """
        return self.filter_by_status(InfrastructureStatus.OPERATIONAL)

    def filter_by_backhaul_type(
        self,
        backhaul_type: Union[BackhaulType, str],
    ) -> "WirelessSiteTable":
        """
        Filter sites by primary backhaul medium.

        Args:
            backhaul_type: BackhaulType or canonical string value.

        Returns:
            Table containing matching sites.
        """
        value = (
            BackhaulType(backhaul_type).value
            if isinstance(backhaul_type, str)
            else backhaul_type.value
        )
        return self.__class__(
            entities=[
                entity for entity in self.entities if entity.backhaul_type == value
            ]
        )

    def filter_by_network_provider(self, provider: str) -> "WirelessSiteTable":
        """
        Filter sites hosting a given active network provider.

        Args:
            provider: Exact provider name to match.

        Returns:
            Table containing sites where the provider operates equipment.
        """
        return self.__class__(
            entities=[
                entity
                for entity in self.entities
                if entity.network_providers and provider in entity.network_providers
            ]
        )

    def filter_by_access_service_type(
        self,
        service_type: Union[WirelessAccessServiceType, str],
    ) -> "WirelessSiteTable":
        """Return sites supporting a specified wireless access-service role."""
        value = (
            WirelessAccessServiceType(service_type).value
            if isinstance(service_type, str)
            else service_type.value
        )
        return self.__class__(
            entities=[
                entity
                for entity in self.entities
                if entity.access_service_types and value in entity.access_service_types
            ]
        )

    def filter_by_access_technology(
        self,
        technology: Union[WirelessAccessTechnology, str],
    ) -> "WirelessSiteTable":
        """Return sites hosting a specified wireless access technology."""
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

    def filter_by_radio_type(
        self,
        radio_type: Union[RadioType, str],
    ) -> "WirelessSiteTable":
        """Return sites supporting a specified mobile radio generation."""
        value = (
            RadioType(radio_type).value
            if isinstance(radio_type, str)
            else radio_type.value
        )
        return self.__class__(
            entities=[
                entity
                for entity in self.entities
                if entity.radio_types and value in entity.radio_types
            ]
        )

    def filter_by_infrastructure_provider(self, provider: str) -> "WirelessSiteTable":
        """
        Filter sites by passive infrastructure provider.

        Args:
            provider: Exact provider name to match.

        Returns:
            Table containing sites with the matching infrastructure provider.
        """
        return self.__class__(
            entities=[
                entity
                for entity in self.entities
                if entity.physical_infrastructure_provider == provider
            ]
        )

    def filter_by_min_backhaul_capacity(
        self,
        min_mbps: float,
    ) -> "WirelessSiteTable":
        """
        Filter sites with known backhaul capacity at or above a threshold.

        Args:
            min_mbps: Minimum known aggregate backhaul capacity in Mbps.

        Returns:
            Table containing matching sites.

        Raises:
            ValueError: If min_mbps is negative.
        """
        if min_mbps < 0:
            raise ValueError("min_mbps must be greater than or equal to zero.")

        return self.__class__(
            entities=[
                entity
                for entity in self.entities
                if entity.backhaul_capacity_mbps is not None
                and entity.backhaul_capacity_mbps >= min_mbps
            ]
        )

    def get_site_types(self) -> Set[str]:
        """
        Return the canonical site types represented in the table.

        Returns:
            Set of site-type values.
        """
        return {
            entity.site_type for entity in self.entities if entity.site_type is not None
        }

    def get_network_providers(self) -> Set[str]:
        """
        Return all active network providers represented in the table.

        Returns:
            Set of provider names.
        """
        return {
            provider
            for entity in self.entities
            if entity.network_providers
            for provider in entity.network_providers
        }

    def get_infrastructure_providers(self) -> Set[str]:
        """
        Return all passive infrastructure providers represented in the table.

        Returns:
            Set of provider names.
        """
        return {
            entity.physical_infrastructure_provider
            for entity in self.entities
            if entity.physical_infrastructure_provider is not None
        }

    def get_upstream_transmission_node_ids(self) -> Set[str]:
        """
        Return referenced upstream transmission-node identifiers.

        Returns:
            Set of non-null TransmissionNode identifiers.
        """
        return {
            entity.upstream_transmission_node_id
            for entity in self.entities
            if entity.upstream_transmission_node_id is not None
        }

    def enrich_from_wireless_access_services(
        self,
        service_table: "WirelessAccessServiceTable",
    ) -> "WirelessSiteTable":
        """
        Enrich site-level service summaries from linked wireless access services.

        Existing values reported directly on WirelessSite entities are preserved and
        unioned with values derived from WirelessAccessService records. Services
        whose wireless_site_id is absent from this table are ignored.

        Args:
            service_table: Validated wireless access services to aggregate by site.

        Returns:
            New WirelessSiteTable with enriched service, technology, and radio-type
            summary fields.
        """
        summaries: Dict[str, Dict[str, Set[str]]] = {}

        for service in service_table.entities:
            summary = summaries.setdefault(
                service.wireless_site_id,
                {
                    "access_service_types": set(),
                    "access_technologies": set(),
                    "radio_types": set(),
                },
            )

            summary["access_service_types"].add(enum_value(service.access_service_type))

            for technology in service.access_technologies or set():
                technology_value = enum_value(technology)
                summary["access_technologies"].add(technology_value)

                radio_type = WIRELESS_ACCESS_TECHNOLOGY_TO_RADIO_TYPE.get(
                    WirelessAccessTechnology(technology_value)
                )
                if radio_type is not None:
                    summary["radio_types"].add(radio_type.value)

        enriched_entities: List[WirelessSite] = []

        for site in self.entities:
            summary = summaries.get(site.wireless_site_id)

            if summary is None:
                enriched_entities.append(site)
                continue

            access_service_types = (site.access_service_types or set()) | summary[
                "access_service_types"
            ]
            access_technologies = (site.access_technologies or set()) | summary[
                "access_technologies"
            ]
            radio_types = (site.radio_types or set()) | summary["radio_types"]

            enriched_entities.append(
                WirelessSite.model_validate(
                    {
                        **site.model_dump(),
                        "access_service_types": access_service_types or None,
                        "access_technologies": access_technologies or None,
                        "radio_types": radio_types or None,
                    }
                )
            )

        return self.__class__(entities=enriched_entities)
