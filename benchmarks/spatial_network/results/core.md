
### Sub-task 2 — SpatialNetwork core, G_large (125,058 nodes, 332,535 edges)

| operation | s | target |
|---|---|---|
| from_networkx(osmnx graph) | 1.042 | under 2 s |
| SpatialNetwork(nodes, edges) from lean tables, validation included | 0.095 |  |
| RSS after reading the tables (MB) | 354 |  |
| RSS with the network built and the input tables released (MB) | 376 | under 400 MB |
| allocations the network keeps beyond its input tables, tracemalloc (MB) | 16.7 |  |
| peak extra allocations while building, tracemalloc (MB) | 39.1 |  |
| node_coordinates() (ms) | 0.37 | under 5 ms |
| node_coordinates(10,000 ids) (ms) | 0.57 |  |
| to_networkx() | 2.503 | under 3 s |
| nodes_gdf, first access | 0.041 |  |
| edges_gdf, first access (fills straight lines) | 0.206 |  |
| degree() | 0.002 |  |
| degree(directed=False) | 0.019 |  |
| info() | 0.062 |  |
| from_points, 100,000 points, k=5 (297,892 edges) | 0.579 |  |
| from_points, 100,000 points, within 500 m (83,188 edges) | 0.088 |  |
