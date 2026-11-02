## Run of 30 October 2026: connecting_subnetwork through ConnectionPlanner

Apple M1 Pro, 8 CPUs, 16 GiB, macOS 15.7; Python 3.14.6, numpy 2.5.2, pandas 3.0.3, scipy 1.18.0, shapely 2.1.2, networkx 3.6.1. `sources=` is now `roots=`, and with roots the default method is the rooted Steiner tree; the old behaviour is `method="nearest_root"`.

### connecting_subnetwork, G_large (125,058 nodes, 332,535 edges), mean of 3

| operation | s |
|---|---|
| steiner, 100 terminals (156 edges, 1,537 km) | 0.078 |
| steiner, 1,000 terminals (1,489 edges, 4,070 km) | 0.140 |
| rooted steiner, 1,000 terminals, 20 roots (4,037 km) | 0.193 |
| nearest root, 1,000 terminals, 20 roots (5,802 km) | 0.168 |
| all_pairs, 50 terminals, 20 roots (6,453 km) | 0.948 |
| simplify of the already simplified network | 0.871 |

## Run of 29 October 2026

Apple M1 Pro, 8 CPUs, 16 GiB, macOS 15.7; Python 3.10.11, numpy 2.2.6, pandas 2.3.0, scipy 1.15.2, shapely 2.1.2, osmnx 2.0.2, networkx 3.3, pandana 0.7.

### Sub-task 6 — simplify, G_large_raw (2,224,948 nodes, 4,526,967 edges)

| operation | s | target |
|---|---|---|
| SpatialNetwork.from_networkx (not part of simplify) | 12.94 |  |
| SpatialNetwork.simplify (125,058 nodes, 332,535 edges) | 7.13 | faster than osmnx |
| ox.simplify_graph (125,058 nodes, 332,535 edges) | 147.54 |  |
| speed-up over osmnx | 21x |  |

- same nodes as osmnx: True; same edge lengths: True; total length 106,220,098.3 m vs osmnx 106,220,098.3 m

### Sub-task 6 — connecting_subnetwork, G_large (125,058 nodes, 332,535 edges), mean of 3

| operation | s |
|---|---|
| steiner, 100 terminals (156 edges, 1,537 km) | 0.108 |
| steiner, 1,000 terminals (1,489 edges, 4,070 km) | 0.178 |
| nearest source, 1,000 terminals, 20 sources (5,802 km) | 0.178 |
| all_pairs, 50 terminals, 20 sources (6,453 km) | 1.538 |
| simplify of the already simplified network | 0.710 |

## Earlier run (27 and 28 October 2026)

#### Sub-task 6 — simplify, G_large_raw (2,224,948 nodes, 4,526,967 edges)

| operation | s | target |
|---|---|---|
| SpatialNetwork.from_networkx (not part of simplify) | 49.17 |  |
| SpatialNetwork.simplify (125,058 nodes, 332,535 edges) | 9.10 | faster than osmnx |
| ox.simplify_graph (125,058 nodes, 332,535 edges) | 196.68 |  |
| speed-up over osmnx | 22x |  |

- same nodes as osmnx: True; same edge lengths: True; total length 106,220,098.3 m vs osmnx 106,220,098.3 m

#### Sub-task 6 — connecting_subnetwork, G_large (125,058 nodes, 332,535 edges), mean of 3

| operation | s |
|---|---|
| steiner, 100 terminals (156 edges, 1,537 km) | 0.101 |
| steiner, 1,000 terminals (1,489 edges, 4,070 km) | 0.169 |
| nearest source, 1,000 terminals, 20 sources (5,802 km) | 0.169 |
| all_pairs, 50 terminals, 20 sources (6,453 km) | 1.593 |
| simplify of the already simplified network | 0.708 |
