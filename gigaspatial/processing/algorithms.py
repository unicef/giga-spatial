import sys, os

import numpy as np
from typing import Literal, List, Tuple, Union, Optional
import geopandas as gpd
import pandas as pd
from scipy.spatial import cKDTree
import networkx as nx

from gigaspatial.processing.geo import (
    convert_to_geodataframe,
)
from gigaspatial.config import config

LOGGER = config.get_logger("GigaSpatialProcessing")


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
    Build a graph of spatial matches between two dataframes using KD-tree.

    Args:
        left_df: Left dataframe to match from
        right_df: Right dataframe to match to
        distance_threshold: Maximum distance for matching (in meters)
        max_k: Maximum number of neighbors to consider per point (default: 100)
        return_dataframe: If True, also return the matches DataFrame
        verbose: If True, print statistics about the graph
        exclude_same_index: If True, exclude self-matches. If None, auto-detect based on df equality

    Returns:
        NetworkX Graph, or tuple of (Graph, DataFrame) if return_dataframe=True

    Raises:
        ValueError: If distance_threshold is negative or max_k is not positive
    """

    # Input validation
    if distance_threshold < 0:
        raise ValueError("distance_threshold must be non-negative")

    if max_k <= 0:
        raise ValueError("max_k must be positive")

    if left_df.empty or right_df.empty:
        if verbose:
            LOGGER.warning("Warning: One or both dataframes are empty")
        G = nx.Graph()
        return (G, pd.DataFrame()) if return_dataframe else G

    def get_utm_coordinates(df: Union[pd.DataFrame, gpd.GeoDataFrame]) -> np.ndarray:
        """Extract coordinates as numpy array in UTM projection."""
        if isinstance(df, pd.DataFrame):
            gdf = convert_to_geodataframe(df)
        else:
            gdf = df.copy()

        # More robust UTM CRS estimation
        try:
            gdf_utm = gdf.to_crs(gdf.estimate_utm_crs())
        except Exception as e:
            if verbose:
                LOGGER.warning(
                    f"Warning: UTM CRS estimation failed, using Web Mercator. Error: {e}"
                )
            gdf_utm = gdf.to_crs("EPSG:3857")  # Fallback to Web Mercator

        return gdf_utm.get_coordinates().to_numpy()

    # Auto-detect same dataframe case
    if exclude_same_index is None:
        exclude_same_index = left_df.equals(right_df)
        if verbose and exclude_same_index:
            LOGGER.info("Auto-detected same dataframe - excluding self-matches")

    # Get coordinates
    left_coords = get_utm_coordinates(left_df)
    right_coords = (
        get_utm_coordinates(right_df) if not exclude_same_index else left_coords
    )

    # Build KD-tree and query
    kdtree = cKDTree(right_coords)

    # Use the provided max_k parameter, but don't exceed available points
    k_to_use = min(max_k, len(right_coords))

    if verbose and k_to_use < max_k:
        LOGGER.info(
            f"Note: max_k ({max_k}) reduced to {k_to_use} (number of available points)"
        )

    # Note: Distance calculations here are based on Euclidean distance in UTM projection.
    # This can introduce errors up to ~50 cm for a 50 meter threshold, especially near the poles where distortion increases.
    distances, indices = kdtree.query(
        left_coords, k=k_to_use, distance_upper_bound=distance_threshold
    )

    # Handle single k case (when k_to_use = 1, results are 1D)
    if distances.ndim == 1:
        distances = distances.reshape(-1, 1)
        indices = indices.reshape(-1, 1)

    # Extract valid pairs using vectorized operations
    left_indices = np.arange(len(distances))[:, np.newaxis]
    left_indices = np.broadcast_to(left_indices, distances.shape)
    valid_mask = np.isfinite(distances)

    if exclude_same_index:
        same_index_mask = left_indices == indices
        valid_mask = valid_mask & ~same_index_mask

    valid_left = left_indices[valid_mask]
    valid_right = indices[valid_mask]
    valid_distances = distances[valid_mask]

    # Map back to original indices
    valid_left_indices = left_df.index.values[valid_left]
    valid_right_indices = right_df.index.values[valid_right]

    # Create matches DataFrame
    matches_df = pd.DataFrame(
        {
            "left_idx": valid_left_indices,
            "right_idx": valid_right_indices,
            "distance": valid_distances,
        }
    )

    # Build graph more efficiently
    G = nx.from_pandas_edgelist(
        matches_df,
        source="left_idx",
        target="right_idx",
        edge_attr="distance",
        create_using=nx.Graph(),
    )

    # Add isolated nodes (nodes without any matches within threshold)
    # This ensures all original indices are represented in the graph
    all_left_nodes = set(left_df.index.values)
    all_right_nodes = set(right_df.index.values)

    if not exclude_same_index:
        all_nodes = all_left_nodes | all_right_nodes
    else:
        all_nodes = all_left_nodes  # Same dataframe, so same node set

    # Add nodes that don't have edges
    existing_nodes = set(G.nodes())
    isolated_nodes = all_nodes - existing_nodes
    G.add_nodes_from(isolated_nodes)

    # Print statistics
    if verbose:
        print(
            f"Total potential matches: {len(left_df)} × {len(right_df)} = {len(left_df) * len(right_df):,}"
        )
        print(f"Matches found within {distance_threshold}m: {len(matches_df):,}")
        print(f"Graph nodes: {G.number_of_nodes():,}")
        print(f"Graph edges: {G.number_of_edges():,}")

        components = list(nx.connected_components(G))
        print(f"Connected components: {len(components):,}")

        if len(components) > 1:
            component_sizes = [len(c) for c in components]
            print(f"Largest component size: {max(component_sizes):,}")
            print(
                f"Isolated nodes: {sum(1 for size in component_sizes if size == 1):,}"
            )

        if len(matches_df) > 0:
            print(
                f"Distance stats - min: {matches_df['distance'].min():.1f}m, "
                f"max: {matches_df['distance'].max():.1f}m, "
                f"mean: {matches_df['distance'].mean():.1f}m"
            )

    return (G, matches_df) if return_dataframe else G
