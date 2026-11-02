## Run of 29 October 2026

Apple M1 Pro, 8 CPUs, 16 GiB, macOS 15.7; Python 3.10.11, numpy 2.2.6, pandas 2.3.0, scipy 1.15.2, shapely 2.1.2, osmnx 2.0.2, networkx 3.3, pandana 0.7.

### Sub-task 4 — editing, G_large (125,058 nodes, 332,535 edges), mean of 3

| operation | s | target |
|---|---|---|
| first STRtree on edges (cached per network) | 0.098 |  |
| clip, 4% of the area (index built) (84,210 edges) | 0.066 | under 0.1 s |
| first KD-tree on nodes (cached per network) | 0.032 |  |
| nearest_nodes, 10,000 points (tree built) | 0.030 |  |
| filter_nodes(geometry=area) | 0.022 |  |
| filter_nodes(ids=10,000) | 0.004 |  |
| remove_nodes(ids=10,000) | 0.050 |  |
| filter_edges(mask) | 0.034 |  |
| set_node_attribute, Series for 10,000 nodes | 0.004 |  |
| update_nodes, 10,000 rows by id | 0.003 |  |
| update_edges, 10,000 rows | 0.049 |  |
| remove_edges, 10,000 (u, v, key) | 0.092 |  |
| add_weighted_edges, 10,000 edges (revalidates the network) | 0.341 |  |
| drop_geometry | 0.039 |  |
| to_undirected | 0.105 |  |
| concat of two overlapping halves | 0.355 |  |

## Earlier run (27 and 28 October 2026)

#### Sub-task 4 — editing, G_large (125,058 nodes, 332,535 edges), mean of 3

| operation | s | target |
|---|---|---|
| first STRtree on edges (cached per network) | 0.101 |  |
| clip, 4% of the area (index built) (84,210 edges) | 0.073 | under 0.1 s |
| first KD-tree on nodes (cached per network) | 0.032 |  |
| nearest_nodes, 10,000 points (tree built) | 0.031 |  |
| filter_nodes(geometry=area) | 0.021 |  |
| filter_nodes(ids=10,000) | 0.004 |  |
| remove_nodes(ids=10,000) | 0.053 |  |
| filter_edges(mask) | 0.034 |  |
| set_node_attribute, Series for 10,000 nodes | 0.003 |  |
| update_nodes, 10,000 rows by id | 0.004 |  |
| update_edges, 10,000 rows | 0.054 |  |
| remove_edges, 10,000 (u, v, key) | 0.090 |  |
| add_weighted_edges, 10,000 edges (revalidates the network) | 0.334 |  |
| drop_geometry | 0.040 |  |
| to_undirected | 0.103 |  |
| concat of two overlapping halves | 0.360 |  |
