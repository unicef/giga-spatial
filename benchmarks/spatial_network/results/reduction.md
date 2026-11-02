
### Sub-task 6 — simplify, G_large_raw (2,224,948 nodes, 4,526,967 edges)

| operation | s | target |
|---|---|---|
| SpatialNetwork.from_networkx (not part of simplify) | 49.17 |  |
| SpatialNetwork.simplify (125,058 nodes, 332,535 edges) | 9.10 | faster than osmnx |
| ox.simplify_graph (125,058 nodes, 332,535 edges) | 196.68 |  |
| speed-up over osmnx | 22x |  |

- same nodes as osmnx: True; same edge lengths: True; total length 106,220,098.3 m vs osmnx 106,220,098.3 m

### Sub-task 6 — connecting_subnetwork, G_large (125,058 nodes, 332,535 edges), mean of 3

| operation | s |
|---|---|
| steiner, 100 terminals (156 edges, 1,537 km) | 0.101 |
| steiner, 1,000 terminals (1,489 edges, 4,070 km) | 0.169 |
| nearest source, 1,000 terminals, 20 sources (5,802 km) | 0.169 |
| all_pairs, 50 terminals, 20 sources (6,453 km) | 1.593 |
| simplify of the already simplified network | 0.708 |
