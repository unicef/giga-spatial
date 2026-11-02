# A saved spatial network is a folder of Parquet files

`SpatialNetwork.save` writes a folder to the data store: `nodes.parquet`, `edges.parquet` as GeoParquet with WKB geometry and zstd compression, and `network.json` with whether the network is directed, its CRS, the default weight, the fetch parameters, and the library version. On the unsimplified Rwanda drive network (4.5 million edges) this is 73 MB and loads in 0.18 s, against 1.4 GB and 437 s for GraphML and 532 MB and 25 s for pickle. List-valued OSM attributes are normalised before saving: `osmid` becomes a list of integers on every edge, and text attributes are joined with `;`, because GeoParquet cannot store a column that mixes lists and single values.

## Considered options

- **GraphML or pickle of a networkx graph.** Rejected: the sizes and load times above, and pickle cannot be read safely from an untrusted data store or by other tools.
- **Feather (Arrow IPC).** Rejected: it was no faster than Parquet on these networks, and geopandas, DuckDB, and QGIS do not read it as widely. The library's `write_dataset` already writes Parquet.
- **One zip file holding both tables.** Rejected: it read no faster than a folder, and it stops users from reading `edges.parquet` on its own.
- **GeoPackage with two layers.** Rejected: 167 MB and 0.47 s on the simplified network, against 64 MB and 0.17 s for Parquet, and it stringifies list attributes.
