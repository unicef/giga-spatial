
### Sub-task 2 — SpatialNetwork core, G_large (125,058 nodes, 332,535 edges)

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
