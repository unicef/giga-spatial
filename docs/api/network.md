# Spatial networks

A `SpatialNetwork` holds places as nodes with coordinates, joined by edges with a length in meters or another weight. A road network from [`OSMRoadsFetcher`](fetchers.md#openstreetmap) is one kind. The network is two tables, and every edit returns a new network. The [road networks guide](../user-guide/road-networks.md) shows a full workflow.

```python
from gigaspatial import SpatialNetwork, OSMRoadsFetcher
```

`to_pandana` and `backend="pandana"` need `pip install "giga-spatial[pandana]"`. Why the network is held as tables, and how a saved network is laid out, are in [ADR 0004](../adr/0004-columnar-spatial-network.md) and [ADR 0005](../adr/0005-spatial-network-storage.md).

The methods below are in this order: building, saving, and converting; reading the tables; editing; inserting points; algorithms; and reducing.

::: gigaspatial.processing.network.core.SpatialNetwork
    options:
      show_source: false
      inherited_members: true
      members:
        - from_networkx
        - from_points
        - load
        - save
        - to_networkx
        - to_pandana
        - nodes
        - edges
        - nodes_gdf
        - edges_gdf
        - to_geodataframes
        - node_coordinates
        - node_index
        - degree
        - bounds
        - total_length
        - info
        - nearest_nodes
        - filter_nodes
        - remove_nodes
        - set_node_attribute
        - update_nodes
        - add_nodes
        - filter_edges
        - remove_edges
        - set_edge_attribute
        - update_edges
        - add_weighted_edges
        - clip
        - drop_geometry
        - to_undirected
        - concat
        - insert_points
        - adjacency
        - shortest_path_lengths
        - shortest_paths
        - reachable
        - nearest_by_network
        - count_within
        - connected_components
        - largest_component
        - connect_components
        - minimum_spanning_tree
        - simplify
        - connecting_subnetwork

## Connection planning

`ConnectionPlanner` plans how to connect terminals, such as schools, to roots, such as fibre nodes, along a network at the smallest shared cost, and returns a `ConnectionPlan`. Why it is a rooted Steiner tree and a separate object is in [ADR 0008](../adr/0008-connection-planning.md).

```python
from gigaspatial import ConnectionPlanner

plan = ConnectionPlanner(network, school_ids, roots=fibre_ids, weight="cost").plan()
```

::: gigaspatial.processing.network.planning.ConnectionPlanner
    options:
      show_source: false
      members:
        - plan

::: gigaspatial.processing.network.planning.ConnectionPlan
    options:
      show_source: false
