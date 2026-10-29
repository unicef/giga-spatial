## Run of 29 October 2026

Apple M1 Pro, 8 CPUs, 16 GiB, macOS 15.7; Python 3.10.11, numpy 2.2.6, pandas 2.3.0, scipy 1.15.2, shapely 2.1.2, osmnx 2.0.2, networkx 3.3, pandana 0.7.

### Sub-task 8 — networkx conversion after the speed-up

| operation | s | target |
|---|---|---|
| from_networkx, G_large (332,535 edges) | 0.880 | under 2 s |
| to_networkx, G_large | 1.351 | under 3 s |
| from_networkx, G_large_raw (4,526,967 edges) | 14.625 |  |
| to_networkx, G_large_raw | 22.436 |  |

## Earlier run (27 and 28 October 2026)

#### Sub-task 8 — networkx conversion after the speed-up

| operation | s | target |
|---|---|---|
| from_networkx, G_large (332,535 edges) | 0.813 | under 2 s |
| to_networkx, G_large | 1.406 | under 3 s |
| from_networkx, G_large_raw (4,526,967 edges) | 14.511 |  |
| to_networkx, G_large_raw | 23.284 |  |
