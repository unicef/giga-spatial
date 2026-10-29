## Run of 29 October 2026

Apple M1 Pro, 8 CPUs, 16 GiB, macOS 15.7; Python 3.14.6, numpy 2.5.2, pandas 3.0.3, scipy 1.18.0, shapely 2.1.2, pyosmium 4.3.2. Rwanda's Geofabrik extract of 26 October 2026 (65 MiB).

### Reading the extract (`read_strategies.py`, one process per strategy)

| strategy | s | peak MB | read |
|---|---|---|---|
| two reads: ways, then the referenced nodes through an ID filter | 12.2 | 2,351 | 4,062,782 references, 3,801,395 nodes; first read 4.0 s |
| two reads: ways, then the ways again with the location index (`flex_mem`) and WKB | 7.5 | 1,047 | 4,062,782 references and coordinates; first read 4.1 s |
| one read with the location index (`flex_mem`) and WKB | 7.3 | 1,042 | 4,062,782 references and coordinates |
| one read, disk-backed index (`sparse_file_array`, on the laptop's SSD) | 7.2 | 846 | the same |
| one read, `sparse_mem_array` | 7.3 | 1,025 | the same |

### Building the network (`build_rwanda.py drive_service walk`, one process)

Peak memory accumulates across the four builds; a simplified `drive_service` build alone in a fresh process takes 13.5 s and peaks at 2,493 MB.

| network | nodes | edges | s | peak MB so far |
|---|---|---|---|---|
| drive_service, raw | 2,270,376 | 4,615,754 | 8.4 | 1,930 |
| drive_service, simplified | 129,073 | 338,550 | 13.8 | 2,964 |
| walk, raw | 3,799,166 | 7,770,848 | 13.7 | 3,245 |
| walk, simplified | 262,964 | 698,444 | 24.4 | 4,227 |

### Loading for a source (`load_rwanda.py`)

| load | nodes | edges | s |
|---|---|---|---|
| four school points, first load (builds and saves the raw network) | 6,076 | 15,044 | 9.95 |
| four school points | 6,076 | 15,044 | 1.06 |
| a 30 by 22 km box around Kigali | 23,724 | 61,615 | 1.31 |
| the country, first load (simplifies the stored raw network) | 125,058 | 332,535 | 7.16 |
| the country | 125,058 | 332,535 | 0.43 |

### The route through osmnx (`../spatial_network/step0.py`, 27 October 2026, Python 3.10.11, osmnx 2.0.2)

| step | s |
|---|---|
| pyosmium: filter to `drive_service` ways and write OSM XML (314 MB) | 22.5 |
| `ox.graph_from_xml(simplify=False)` | 177.8 |
| `ox.graph_from_xml(simplify=True)`, the simplified network | 277.6 |
| total for the simplified network | 300.2 |
