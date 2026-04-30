"""
Module for transmission node schema and processing.
Defines the TransmissionNode entity, representing physical network infrastructure nodes
like backbone cores, metro sites, and aggregation points.
"""
import pandas as pd
import networkx as nx
from pydantic import Field
from typing import Optional, List, Set, Union, ClassVar
from enum import Enum
from pathlib import Path

from gigaspatial.core.io.data_store import DataStore
from .shared import DataConfidence, PowerSource
from .entity import ENUM_ENTITY_CONFIG, GigaEntity, EntityTable
from gigaspatial.processing.entity_processor import EntityProcessor
from gigaspatial.config import config
from .cell_tower import BackhaulType, BACKHAUL_ALIAS_MAP

logger = config.get_logger("TransmissionNodeManager")


class TransmissionMedium(str, Enum):
    """Enum for transmission medium types."""

    FIBER = "fiber"
    MICROWAVE = "microwave"
    COPPER = "copper"
    COAXIAL = "coaxial"
    SATELLITE = "satellite"  # VSAT backhaul
    FREE_SPACE_OPTICAL = "fso"  # FSO links


class BackhaulTechnology(str, Enum):
    """Enum for backhaul technologies."""

    DWDM = "dwdm"
    SDH = "sdh"
    TDM = "tdm"
    SONET = "sonet"
    CARRIER_ETHERNET = "carrier_ethernet"
    MPLS = "mpls"
    IP_OPTICAL = "ip_optical"
    PACKET_MICROWAVE = "packet_microwave"


class NodeType(str, Enum):
    """Enum for node types."""

    BACKBONE = "backbone"  # Core/national network, highest capacity
    METRO = "metro"  # Metropolitan/regional network
    AGGREGATION = "aggregation"  # Aggregating traffic from access nodes
    ACCESS = "access"  # Direct connection to cell sites


class NodeStatus(str, Enum):
    """Enum for node operational status."""

    PROPOSED = "proposed"
    PLANNED = "planned"
    UNDER_CONSTRUCTION = "underconstruction"
    OPERATIONAL = "operational"
    DECOMMISSIONED = "decommissioned"
    INACTIVE = "inactive"


class TransmissionNode(GigaEntity):
    """Represents a transmission node in the network infrastructure."""

    model_config = ENUM_ENTITY_CONFIG

    # Identity
    transmission_node_id: str = Field(
        ..., max_length=50, description="Unique identifier for the transmission node"
    )
    transmission_node_id_source: Optional[str] = Field(
        None, max_length=50, description="Original node identifier in source system"
    )
    node_name: Optional[str] = Field(
        None, max_length=100, description="Common/operational name of the node"
    )

    # Classification
    node_type: Optional[NodeType] = Field(
        None, description="Hierarchical role of the node in the network"
    )
    node_status: Optional[NodeStatus] = Field(
        None, description="Current operational status of the node"
    )
    transmission_medium: Optional[TransmissionMedium] = Field(
        None, description="Physical transmission medium used"
    )
    backhaul_technologies: Optional[List[BackhaulTechnology]] = Field(
        None, description="Backhaul technologies in use at this node"
    )
    access_technologies: Optional[List[str]] = Field(
        None, description="Access technologies available at this node"
    )
    is_logical_node: Optional[bool] = Field(
        None, description="True if this site hosts active transmission equipment"
    )

    # Ownership
    physical_infrastructure_provider: Optional[str] = Field(
        None, max_length=100, description="Entity providing physical infrastructure"
    )
    network_providers: Optional[List[str]] = Field(
        None, description="Network service providers operating on this node"
    )

    # Capacity
    equipped_capacity_access_mbps: Optional[float] = Field(
        None, ge=0, description="Current equipped access capacity in Mbps"
    )
    potential_capacity_access_mbps: Optional[float] = Field(
        None, ge=0, description="Maximum potential access capacity in Mbps"
    )
    equipped_capacity_backhaul_mbps: Optional[float] = Field(
        None, ge=0, description="Current equipped backhaul capacity in Mbps"
    )
    potential_capacity_backhaul_mbps: Optional[float] = Field(
        None, ge=0, description="Maximum potential backhaul capacity in Mbps"
    )

    # Physical
    elevation_meters: Optional[float] = Field(
        None, description="Elevation above sea level in meters"
    )
    power_source: Optional[PowerSource] = Field(
        None, description="Type of power source used"
    )
    data_confidence: Optional[DataConfidence] = Field(
        None, description="Level of confidence in the data"
    )

    # Temporal
    installation_date: Optional[str] = Field(
        None, description="ISO 8601 date when the node became operational"
    )
    decommission_date: Optional[str] = Field(
        None, description="ISO 8601 date when the node was decommissioned"
    )

    # Topology
    connected_node_ids: Optional[List[str]] = Field(
        None, description="IDs of directly connected transmission nodes"
    )

    @property
    def id(self) -> str:
        """Alias for transmission_node_id."""
        return self.transmission_node_id


# ---------------------------------------------------------------------------
# Processor
# ---------------------------------------------------------------------------


class TransmissionNodeProcessor(EntityProcessor):
    """Processor for cleaning and normalizing transmission node data."""

    NUMERIC_COLUMNS: ClassVar[List[str]] = [
        *EntityProcessor.NUMERIC_COLUMNS,
        "equipped_capacity_access_mbps",
        "potential_capacity_access_mbps",
        "equipped_capacity_backhaul_mbps",
        "potential_capacity_backhaul_mbps",
        "elevation_meters",
    ]

    LOWERCASE_COLUMNS: ClassVar[List[str]] = [
        "transmission_medium",
        "node_type",
        "node_status",
        "power_source",
    ]

    def process(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Execute the full processing pipeline for transmission nodes.

        Args:
            df: Raw transmission node DataFrame.
            **kwargs: Additional processing arguments.

        Returns:
            Processed and normalized DataFrame.
        """
        df = super().process(df, **kwargs)
        df = self._normalize_transmission_medium(df)
        df = self._split_list_columns(df)
        return df

    def _normalize_transmission_medium(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize transmission_medium values to canonical enums.

        Args:
            df: DataFrame to normalize.

        Returns:
            DataFrame with normalized transmission_medium column.
        """
        return self._normalize_enum_column(
            df,
            column="transmission_medium",
            alias_map=BACKHAUL_ALIAS_MAP,
            valid_values={tm.value for tm in TransmissionMedium},
            required=False,
        )

    def _split_list_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Split comma-separated string columns into Python lists.

        Args:
            df: DataFrame to process.

        Returns:
            DataFrame with list columns.
        """
        for col in (
            "network_providers",
            "access_technologies",
            "backhaul_technologies",
        ):
            if col in df.columns:
                df[col] = df[col].apply(
                    lambda v: (
                        [x.strip() for x in v.split(",")] if isinstance(v, str) else v
                    )
                )
        return df


# ---------------------------------------------------------------------------
# Table
# ---------------------------------------------------------------------------


class TransmissionNodeTable(EntityTable[TransmissionNode]):
    """Container for TransmissionNode entities with network-specific operations."""

    @classmethod
    def from_file(
        cls,
        file_path: Union[str, Path],
        data_store: Optional[DataStore] = None,
        **kwargs,
    ) -> "TransmissionNodeTable":
        """
        Create a TransmissionNodeTable from a file.

        Args:
            file_path: Path to the dataset file.
            data_store: DataStore instance for file access. Defaults to LocalDataStore.
            **kwargs: Additional arguments forwarded to read_dataset.

        Returns:
            TransmissionNodeTable instance with validated TransmissionNode entities.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file cannot be read or parsed.
        """
        return super().from_file(
            file_path=file_path,
            entity_class=TransmissionNode,
            data_store=data_store,
            processor=TransmissionNodeProcessor,
            **kwargs,
        )

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        entity_class: type = TransmissionNode,
        clean: bool = False,
        **kwargs,
    ) -> "TransmissionNodeTable":
        """
        Create a TransmissionNodeTable from an existing DataFrame.

        Args:
            df: DataFrame containing transmission node data.
            entity_class: Entity class to validate against. Defaults to TransmissionNode.
            clean: Whether to apply TransmissionNodeProcessor before validation.
                Defaults to False since DataFrames passed directly are assumed pre-cleaned.

        Returns:
            TransmissionNodeTable instance with validated TransmissionNode entities.
        """
        return super().from_dataframe(
            df=df,
            entity_class=entity_class,
            clean=clean,
            processor=TransmissionNodeProcessor(),
            **kwargs,
        )

    @classmethod
    def from_files(
        cls,
        file_paths: List[Union[str, Path]],
        data_store: Optional[DataStore] = None,
        **kwargs,
    ) -> "TransmissionNodeTable":
        """
        Load and merge multiple TransmissionNode source files into a single TransmissionNodeTable.

        Args:
            file_paths: List of paths to source files.
            data_store: DataStore instance for file access. Defaults to LocalDataStore.
            **kwargs: Additional arguments forwarded to read_dataset.

        Returns:
            TransmissionNodeTable with merged and validated TransmissionNode entities.
        """
        return super().from_files(
            file_paths=file_paths,
            entity_class=TransmissionNode,
            data_store=data_store,
            processor=TransmissionNodeProcessor,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Filters
    # ------------------------------------------------------------------

    def filter_by_node_type(self, node_type: NodeType) -> "TransmissionNodeTable":
        """Filter nodes by their hierarchical type in the network."""
        return self.__class__(
            entities=[e for e in self.entities if e.node_type == node_type.value]
        )

    def filter_by_node_types(
        self, node_types: Set[NodeType]
    ) -> "TransmissionNodeTable":
        """Filter nodes matching any of the given node types."""
        values = {nt.value for nt in node_types}
        return self.__class__(
            entities=[e for e in self.entities if e.node_type in values]
        )

    def filter_by_status(self, status: NodeStatus) -> "TransmissionNodeTable":
        """Filter nodes by operational status."""
        return self.__class__(
            entities=[e for e in self.entities if e.node_status == status.value]
        )

    def filter_operational(self) -> "TransmissionNodeTable":
        """Return only operational nodes."""
        return self.filter_by_status(NodeStatus.OPERATIONAL)

    def filter_by_medium(self, medium: TransmissionMedium) -> "TransmissionNodeTable":
        """Filter nodes by transmission medium."""
        return self.__class__(
            entities=[e for e in self.entities if e.transmission_medium == medium.value]
        )

    def filter_by_provider(self, provider: str) -> "TransmissionNodeTable":
        """Filter nodes by physical infrastructure provider."""
        return self.__class__(
            entities=[
                e
                for e in self.entities
                if e.physical_infrastructure_provider == provider
            ]
        )

    def filter_by_backhaul_technology(
        self, technology: BackhaulTechnology
    ) -> "TransmissionNodeTable":
        """Filter nodes that use a specific backhaul technology."""
        value = technology.value
        return self.__class__(
            entities=[
                e
                for e in self.entities
                if e.backhaul_technologies and value in e.backhaul_technologies
            ]
        )

    def filter_by_min_capacity(
        self,
        min_mbps: float,
        capacity_type: str = "equipped_access",
    ) -> "TransmissionNodeTable":
        """
        Filter nodes by minimum capacity threshold.

        Args:
            min_mbps: Minimum capacity in Mbps.
            capacity_type: One of 'equipped_access', 'potential_access',
                'equipped_backhaul', 'potential_backhaul'.
        """
        field_map = {
            "equipped_access": "equipped_capacity_access_mbps",
            "potential_access": "potential_capacity_access_mbps",
            "equipped_backhaul": "equipped_capacity_backhaul_mbps",
            "potential_backhaul": "potential_capacity_backhaul_mbps",
        }
        if capacity_type not in field_map:
            raise ValueError(
                f"Invalid capacity_type '{capacity_type}'. "
                f"Must be one of: {list(field_map.keys())}"
            )
        field = field_map[capacity_type]
        return self.__class__(
            entities=[
                e
                for e in self.entities
                if getattr(e, field) is not None and getattr(e, field) >= min_mbps
            ]
        )

    # ------------------------------------------------------------------
    # Aggregations
    # ------------------------------------------------------------------

    def get_node_types(self) -> Set[str]:
        """
        Return the set of all unique node types present in the table.

        Returns:
            Set of unique node type strings.
        """
        return {e.node_type for e in self.entities if e.node_type is not None}

    def get_providers(self) -> Set[str]:
        """
        Return all unique physical infrastructure providers.

        Returns:
            Set of unique provider name strings.
        """
        return {
            e.physical_infrastructure_provider
            for e in self.entities
            if e.physical_infrastructure_provider is not None
        }

    def get_backbone_nodes(self) -> "TransmissionNodeTable":
        """
        Filter to only backbone (core network) nodes.

        Returns:
            TransmissionNodeTable with only backbone nodes.
        """
        return self.filter_by_node_type(NodeType.BACKBONE)

    def get_access_nodes(self) -> "TransmissionNodeTable":
        """
        Filter to only access-layer nodes.

        Returns:
            TransmissionNodeTable with only access nodes.
        """
        return self.filter_by_node_type(NodeType.ACCESS)

    # ------------------------------------------------------------------
    # Topology
    # ------------------------------------------------------------------

    def to_topology_graph(self) -> nx.Graph:
        """
        Build a NetworkX graph from the connected_node_ids field on each node.

        Uses the explicitly declared node connections rather than spatial proximity.
        Use to_distance_graph() instead for proximity-based graph construction.

        Returns:
            NetworkX Graph where nodes are transmission_node_ids and edges
            represent declared connections between nodes.
        """
        G = nx.Graph()

        for entity in self.entities:
            G.add_node(
                entity.transmission_node_id,
                node_type=entity.node_type,
                node_status=entity.node_status,
                transmission_medium=entity.transmission_medium,
                equipped_capacity_backhaul_mbps=entity.equipped_capacity_backhaul_mbps,
            )

        for entity in self.entities:
            if entity.connected_node_ids:
                for connected_id in entity.connected_node_ids:
                    if G.has_node(connected_id):
                        G.add_edge(entity.transmission_node_id, connected_id)
                    else:
                        logger.warning(
                            "Node '%s' references unknown connected node '%s'. "
                            "Edge skipped.",
                            entity.transmission_node_id,
                            connected_id,
                        )

        logger.debug(
            "Topology graph built: %d nodes, %d edges.",
            G.number_of_nodes(),
            G.number_of_edges(),
        )
        return G
