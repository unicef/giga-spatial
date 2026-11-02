## Run of 29 October 2026

Apple M1 Pro, 8 CPUs, 16 GiB, macOS 15.7; Python 3.10.11, numpy 2.2.6, pandas 2.3.0, scipy 1.15.2, shapely 2.1.2, osmnx 2.0.2, networkx 3.3, pandana 0.7.

### Sub-task 2 — SpatialNetwork core, G_large (125,058 nodes, 332,535 edges)

| operation | s | target |
|---|---|---|
| from_networkx(osmnx graph) | 0.854 | under 2 s |
| SpatialNetwork(nodes, edges) from lean tables, validation included | 0.059 |  |
| RSS after reading the tables (MB) | 325 |  |
| RSS with the network built and the input tables released (MB) | 350 | under 400 MB |
| allocations the network keeps beyond its input tables, tracemalloc (MB) | 16.7 |  |
| peak extra allocations while building, tracemalloc (MB) | 39.2 |  |
| node_coordinates() (ms) | 0.38 | under 5 ms |
| node_coordinates(10,000 ids) (ms) | 0.51 |  |
| to_networkx() | 1.375 | under 3 s |
| nodes_gdf, first access | 0.036 |  |
| edges_gdf, first access (fills straight lines) | 0.208 |  |
| degree() | 0.002 |  |
| degree(directed=False) | 0.014 |  |
| info() | 0.065 |  |
| from_points, 100,000 points, k=5 (297,892 edges) | 0.538 |  |
| from_points, 100,000 points, within 500 m (83,188 edges) | 0.081 |  |

## Earlier run (27 and 28 October 2026)

#### Sub-task 2 — SpatialNetwork core, G_large (125,058 nodes, 332,535 edges)

| operation | s | target |
|---|---|---|
| from_networkx(osmnx graph) | 0.859 | under 2 s |
| SpatialNetwork(nodes, edges) from lean tables, validation included | 0.048 |  |
| RSS after reading the tables (MB) | 353 |  |
| RSS with the network built and the input tables released (MB) | 385 | under 400 MB |
| allocations the network keeps beyond its input tables, tracemalloc (MB) | 16.7 |  |
| peak extra allocations while building, tracemalloc (MB) | 39.2 |  |
| node_coordinates() (ms) | 0.35 | under 5 ms |
| node_coordinates(10,000 ids) (ms) | 0.67 |  |
| to_networkx() | 1.371 | under 3 s |
| nodes_gdf, first access | 0.033 |  |
| edges_gdf, first access (fills straight lines) | 0.194 |  |
| degree() | 0.002 |  |
| degree(directed=False) | 0.014 |  |
| info() | 0.065 |  |
| from_points, 100,000 points, k=5 (297,892 edges) | 0.538 |  |
| from_points, 100,000 points, within 500 m (83,188 edges) | 0.078 |  |
