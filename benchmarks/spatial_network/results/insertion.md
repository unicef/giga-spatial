
### Sub-task 7 — insert_points, G_large (125,058 nodes, 332,535 edges)

| operation | s | target |
|---|---|---|
| road edges projected to UTM and indexed (first insertion only) | 1.33 |  |
| 1,000 points, chain=False (3,561 split pieces, 0 point links) | 0.75 |  |
| 1,000 points, chain=True (3,557 split pieces, 2 point links) | 0.65 |  |
| 10,000 points, chain=False (34,467 split pieces, 0 point links) | 1.75 | under 10 s |
| 10,000 points, chain=True (34,071 split pieces, 382 point links) | 1.78 |  |
| 1,000 points, chain=True, snap_to_node=0: total connection length (m) | 68,187 |  |

### Sub-task 7 — the original merge_points_to_graph on the same points (one run each)

| operation | s | result |
|---|---|---|
| merge_points_to_graph, 1,000 points (legacy) | 291.7 | total connection length 67,985 m (projected meters; insert_points 68,187 m by haversine) |
| merge_points_to_graph, 2,000 points (legacy) | 542.8 | total connection length 137,882 m |

The old function takes about 0.27 s per point and grows faster than linearly, so 10,000 points would take more than 45 minutes; insert_points takes 1.8 s.
