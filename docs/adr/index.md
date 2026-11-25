# Design decisions

These records explain decisions that are hard to reverse or surprising without context: what was decided, why, and which alternatives were rejected. Read them before proposing to change one of these areas. A new decision gets the next number.

| Record | Decision |
| --- | --- |
| [0001](0001-search-key-and-search-geometry.md) | Handlers separate the search key, which finds the files to fetch, from the search geometry, which crops the loaded data. |
| [0002](0002-handler-base-in-the-api-reference.md) | The handler base is documented in the API reference, but it is not a stable API for other packages. |
| [0003](0003-public-import-paths.md) | Users import from the top level or a public subpackage, which load lazily. Every other module path is internal. |
| [0004](0004-columnar-spatial-network.md) | A spatial network is held as node and edge tables; networkx and pandana are exports, and edits return a new network. |
| [0005](0005-spatial-network-storage.md) | A saved spatial network is a folder with a nodes Parquet file, an edges GeoParquet file, and a JSON metadata file. |
| [0006](0006-osm-extract-storage.md) | OSM extracts are stored by date, and the network built from each is saved uncropped beside it; cropping happens at load. |
| [0007](0007-osm-extract-choice.md) | OSM extracts are chosen by polygon, with ISO codes only as a fast path and a curated list of multi-country extracts. |
| [0008](0008-connection-planning.md) | Connecting points to an existing network is a rooted Steiner tree with the existing network as one virtual root, solved by `ConnectionPlanner`. |
| [0009](0009-point-sets-and-distance-table.md) | Networks between two point sets are built with `from_point_sets`, tagged with `bipartite`, and queried as long cost tables with `distance_table`; `build_distance_graph` is deprecated. |
| [0010](0010-view-generator-base-and-contract.md) | Point-of-interest and zonal view generators share an internal `BaseViewGenerator` and a uniform `self.view` return and overwrite contract. |
| [0011](0011-fetcher-and-handler-by-artifact.md) | A handler persists files and loads them for a source. A fetcher returns its result in memory and imposes no data-store layout. |
| [0012](0012-tif-processor-working-file-and-derivations.md) | A `TifProcessor` reads one working file through one owner, copies non-local stored files locally once, and derives new processors; `to_crs` replaces `reproject_to`. |
