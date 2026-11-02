## Run of 29 October 2026

Apple M1 Pro, 8 CPUs, 16 GiB, macOS 15.7; Python 3.10.11, numpy 2.2.6, pandas 2.3.0, scipy 1.15.2, shapely 2.1.2, osmnx 2.0.2, networkx 3.3, pandana 0.7.

### Sub-task 7 — insert_points, G_large (125,058 nodes, 332,535 edges)

| operation | s | target |
|---|---|---|
After the fix for points on one-way roads (which had been pooled onto the edge table's last row):

| operation | s | target |
|---|---|---|
| road edges projected to UTM and indexed (first insertion only) | 1.36 |  |
| 1,000 points, chain=False (3,564 split pieces, 0 point links) | 0.66 |  |
| 1,000 points, chain=True (3,560 split pieces, 2 point links) | 0.64 |  |
| 10,000 points, chain=False (34,558 split pieces, 0 point links) | 1.83 | under 10 s |
| 10,000 points, chain=True (34,160 split pieces, 382 point links) | 1.77 |  |
| 1,000 points, chain=True, snap_to_node=0: total connection length (m) | 68,187 |  |

Before the fix, earlier the same day:

| operation | s | target |
|---|---|---|
| road edges projected to UTM and indexed (first insertion only) | 1.30 |  |
| 1,000 points, chain=False (3,561 split pieces, 0 point links) | 0.63 |  |
| 1,000 points, chain=True (3,557 split pieces, 2 point links) | 0.61 |  |
| 10,000 points, chain=False (34,467 split pieces, 0 point links) | 1.70 | under 10 s |
| 10,000 points, chain=True (34,071 split pieces, 382 point links) | 1.71 |  |
| 1,000 points, chain=True, snap_to_node=0: total connection length (m) | 68,187 |  |

## Earlier run (27 and 28 October 2026)

#### Sub-task 7 — insert_points, G_large (125,058 nodes, 332,535 edges)

| operation | s | target |
|---|---|---|
| road edges projected to UTM and indexed (first insertion only) | 1.33 |  |
| 1,000 points, chain=False (3,561 split pieces, 0 point links) | 0.75 |  |
| 1,000 points, chain=True (3,557 split pieces, 2 point links) | 0.65 |  |
| 10,000 points, chain=False (34,467 split pieces, 0 point links) | 1.75 | under 10 s |
| 10,000 points, chain=True (34,071 split pieces, 382 point links) | 1.78 |  |
| 1,000 points, chain=True, snap_to_node=0: total connection length (m) | 68,187 |  |

#### Sub-task 7 — the original merge_points_to_graph on the same points (one run each)

| operation | s | result |
|---|---|---|
| merge_points_to_graph, 1,000 points (legacy) | 291.7 | total connection length 67,985 m (projected meters; insert_points 68,187 m by haversine) |
| merge_points_to_graph, 2,000 points (legacy) | 542.8 | total connection length 137,882 m |

The old function takes about 0.27 s per point and grows faster than linearly, so 10,000 points would take more than 45 minutes; insert_points takes 1.8 s.
