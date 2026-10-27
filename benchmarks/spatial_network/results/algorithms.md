
### Sub-task 5 — algorithms, G_large (125,058 nodes, 332,535 edges), mean of 3

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
