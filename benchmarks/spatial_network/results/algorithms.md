## Run of 29 October 2026

Apple M1 Pro, 8 CPUs, 16 GiB, macOS 15.7; Python 3.10.11, numpy 2.2.6, pandas 2.3.0, scipy 1.15.2, shapely 2.1.2, osmnx 2.0.2, networkx 3.3, pandana 0.7.

### Sub-task 5 — algorithms, G_large (125,058 nodes, 332,535 edges), mean of 3

| operation | s | target |
|---|---|---|
| adjacency() (first build) | 0.062 | under 0.3 s |
| shortest_path_lengths, 100 x 1,000, scipy | 2.576 | under 3 s |
| shortest_path_lengths, 100 x 1,000, scipy, directed=False | 2.600 |  |
| shortest_paths, 1,000 pairs (10 origins) | 0.342 |  |
| shortest_paths, 1,000 pairs (10 origins), geometry | 1.191 |  |
| reachable, 100 origins, 5 km | 0.080 |  |
| nearest_by_network, 10,000 origins to 100 targets, k=1 | 0.046 |  |
| nearest_by_network, 100 origins to 1,000 targets, k=3 | 2.548 |  |
| count_within, 1,000 origins, 100 targets, 5 km | 0.039 |  |
| connected_components() | 0.006 |  |
| connected_components(strong=True) | 0.005 |  |
| minimum_spanning_tree() | 0.069 |  |
| connect_components() (first; components found) (0 link rows) | 0.100 |  |
| connect_components() after removing 3% of nodes (966 components), 1 left | 0.430 |  |
| to_pandana() (contraction hierarchy build) | 1.431 |  |
| shortest_path_lengths, 100,000 pairs, pandana | 0.348 |  |
| shortest_path_lengths, 100 x 1,000, pandana | 0.241 |  |

## Earlier run (27 and 28 October 2026)

#### Sub-task 5 — algorithms, G_large (125,058 nodes, 332,535 edges), mean of 3

| operation | s | target |
|---|---|---|
| adjacency() (first build) | 0.073 | under 0.3 s |
| shortest_path_lengths, 100 x 1,000, scipy | 2.681 | under 3 s |
| shortest_path_lengths, 100 x 1,000, scipy, directed=False | 2.686 |  |
| shortest_paths, 1,000 pairs (10 origins) | 0.342 |  |
| shortest_paths, 1,000 pairs (10 origins), geometry | 1.202 |  |
| reachable, 100 origins, 5 km | 0.084 |  |
| nearest_by_network, 10,000 origins to 100 targets, k=1 | 0.047 |  |
| nearest_by_network, 100 origins to 1,000 targets, k=3 | 2.718 |  |
| count_within, 1,000 origins, 100 targets, 5 km | 0.040 |  |
| connected_components() | 0.006 |  |
| connected_components(strong=True) | 0.006 |  |
| minimum_spanning_tree() | 0.072 |  |
| connect_components() (first; components found) (0 link rows) | 0.101 |  |
| connect_components() after removing 3% of nodes (966 components), 1 left | 0.475 |  |
| to_pandana() (contraction hierarchy build) | 1.519 |  |
| shortest_path_lengths, 100,000 pairs, pandana | 0.412 |  |
| shortest_path_lengths, 100 x 1,000, pandana | 0.328 |  |
