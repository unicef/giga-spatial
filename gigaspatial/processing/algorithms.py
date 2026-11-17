"""
Deprecated distance-graph wrapper; use :meth:`SpatialNetwork.from_points` or
:meth:`SpatialNetwork.from_point_sets` instead.
"""

from typing import Optional, Tuple, Union
import warnings

import geopandas as gpd
import networkx as nx
import pandas as pd

from gigaspatial.config import config
from gigaspatial.processing.network import SpatialNetwork

LOGGER = config.get_logger("GigaSpatialProcessing")
_RESERVED = ("x", "y", "kind", "bipartite")


def build_distance_graph(
    left_df: Union[pd.DataFrame, gpd.GeoDataFrame],
    right_df: Union[pd.DataFrame, gpd.GeoDataFrame],
    distance_threshold: float,
    max_k: int = 100,
    return_dataframe: bool = False,
    verbose: bool = True,
    exclude_same_index: Optional[bool] = None,
) -> Union[nx.Graph, Tuple[nx.Graph, pd.DataFrame]]:
    """
    Build a graph of spatial matches between two dataframes.

    .. deprecated:: 0.11.1
        Removed in 0.12; use :meth:`SpatialNetwork.from_points` or
        :meth:`SpatialNetwork.from_point_sets` instead.
    """
    warnings.warn(
        "build_distance_graph is deprecated and will be removed in 0.12; "
        "use SpatialNetwork.from_points / from_point_sets instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    if distance_threshold < 0:
        raise ValueError("distance_threshold must be non-negative")
    if max_k <= 0:
        raise ValueError("max_k must be positive")
    if left_df.empty or right_df.empty:
        if verbose:
            LOGGER.warning("Warning: One or both dataframes are empty")
        G = nx.Graph()
        return (G, pd.DataFrame(columns=["left_idx", "right_idx", "distance"])) if return_dataframe else G

    if exclude_same_index is None:
        exclude_same_index = left_df.equals(right_df)

    left_clean = _strip_reserved(left_df)
    if exclude_same_index:
        net = SpatialNetwork.from_points(
            left_clean,
            connect="knn",
            k=max_k,
            max_distance=distance_threshold,
        )
    else:
        right_clean = _strip_reserved(right_df)
        net = SpatialNetwork.from_point_sets(
            left_clean,
            right_clean,
            connect="knn",
            k=max_k,
            max_distance=distance_threshold,
            within="none",
        )

    return _network_to_distance_graph(net, return_dataframe)


def _network_to_distance_graph(
    net: SpatialNetwork, return_dataframe: bool
) -> Union[nx.Graph, Tuple[nx.Graph, pd.DataFrame]]:
    edges = net.edges
    matches_df = pd.DataFrame(
        {
            "left_idx": edges["u"].to_numpy(),
            "right_idx": edges["v"].to_numpy(),
            "distance": edges["length"].to_numpy(),
        }
    )
    G = nx.from_pandas_edgelist(
        matches_df,
        source="left_idx",
        target="right_idx",
        edge_attr="distance",
        create_using=nx.Graph(),
    )
    G.add_nodes_from(net.node_index)
    return (G, matches_df) if return_dataframe else G


def _strip_reserved(df: Union[pd.DataFrame, gpd.GeoDataFrame]) -> Union[pd.DataFrame, gpd.GeoDataFrame]:
    drop = [c for c in _RESERVED if c in df.columns]
    return df.drop(columns=drop) if drop else df
