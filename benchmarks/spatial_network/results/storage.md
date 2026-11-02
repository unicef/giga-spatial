## Run of 29 October 2026

Apple M1 Pro, 8 CPUs, 16 GiB, macOS 15.7; Python 3.10.11, numpy 2.2.6, pandas 2.3.0, scipy 1.15.2, shapely 2.1.2, osmnx 2.0.2, networkx 3.3, pandana 0.7.

### Sub-task 3 — save and load, G_large (125,058 nodes, 332,535 edges), mean of 3

| operation | s | target |
|---|---|---|
| normalise raw osmnx edge attributes | 0.521 |  |
| save, raw osmnx attributes (normalisation included) | 1.201 | under 1 s |
| folder size (MB) | 64.9 |  |
| load, all columns | 0.467 | under 0.5 s |
| load, required columns and geometry | 0.481 |  |
| load, topology only (u, v, key, length, kind) | 0.022 |  |
| save, a loaded (already normalised) network | 0.618 | under 1 s |

## Earlier run (27 and 28 October 2026)

#### Sub-task 3 — save and load, G_large (125,058 nodes, 332,535 edges), mean of 3

| operation | s | target |
|---|---|---|
| normalise raw osmnx edge attributes | 0.526 |  |
| save, raw osmnx attributes (normalisation included) | 1.190 | under 1 s |
| folder size (MB) | 64.9 |  |
| load, all columns | 0.479 | under 0.5 s |
| load, required columns and geometry | 0.495 |  |
| load, topology only (u, v, key, length, kind) | 0.022 |  |
| save, a loaded (already normalised) network | 0.750 | under 1 s |

#### Sub-task 3 — save and load, G_large_raw (2,224,948 nodes, 4,526,967 edges), mean of 1

| operation | s | target |
|---|---|---|
| normalise raw osmnx edge attributes | 2.065 |  |
| save, raw osmnx attributes (normalisation included) | 3.207 | under 1 s |
| folder size (MB) | 80.2 |  |
| load, all columns | 0.791 | under 0.5 s |
| load, required columns and geometry | 0.497 |  |
| load, topology only (u, v, key, length, kind) | 0.468 |  |
| save, a loaded (already normalised) network | 1.362 | under 1 s |
