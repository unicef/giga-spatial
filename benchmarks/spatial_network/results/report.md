# giga-spatial road network benchmarks

Machine: Apple M1 Pro, 8 CPUs, 17.2 GB RAM, macOS (darwin 24.6). Python 3.10.11 (py310 env), pyosmium 4.3.2 and igraph 1.0.0 in /tmp/gs_bench/pylib.

## Step 0 — data

- PBF size: 68.4 MB; filtered XML: 314.1 MB
- pyosmium pass 1 (collect refs of 81,489 drive_service ways, 2,270,376 node refs): 2.13 s
- pyosmium pass 2 (write 2,270,376 nodes + ways to OSM XML): 20.41 s; total filter 22.54 s
- osmnx 2.0.2, networkx 3.3
- ox.graph_from_xml(simplify=False): 177.81 s -> G_large_raw 2,224,948 nodes, 4,526,967 edges
- ox.graph_from_xml(simplify=True): 277.62 s -> G_large 125,058 nodes, 332,535 edges; crs=epsg:4326
- G_small = truncate_graph_bbox(Kigali 29.98,-2.03,30.2,-1.89): 2.08 s -> 18,547 nodes, 47,499 edges
- sample edge attrs: ['highway', 'lanes', 'length', 'oneway', 'osmid', 'ref', 'reversed']
- sample node attrs: ['x', 'y']
- **PBF route total (filter + graph_from_xml simplify=True): 300.2 s**
- Added node street_count via ox.stats.count_streets_per_node (graph_from_xml in osmnx 2.0.2 does not set it) and re-pickled.

### Part A — G_small (18,547 nodes, 47,499 edges), read = mean of 3

- input prep: ox.graph_to_gdfs 0.56 s (not included in GeoParquet write times)
- input prep: lean table extraction (4 variants) 0.54 s (not included in lean write times)
  GeoParquet snappy (raw gdfs): write FAILED ArrowInvalid('cannot mix list and non-list, non-null values', 'Conversion failed for column osmid with type object')
Columns: write s (from the stated input), read s per reader (mean), on-disk MB, peak RSS increase during first read in a fresh process (MB, ru_maxrss), RSS retained after read (MB).

Readers: graphml=ox.load_graphml; gpkg=gpd.read_file x2 (pyogrio), gpkg_arrow=same with use_arrow=True; geoparquet=gpd.read_parquet x2; geoparquet_nx=+ox.graph_from_gdfs; lean_arrow=pq.read_table x2; lean_pandas=+to_pandas (WKB left as bytes); lean_geo=+GeoSeries.from_wkb; lean_gpq=pq nodes + gpd.read_parquet edges; zip_bytes/dir_bytes=open(...,'rb') -> BytesIO -> pq.read_table -> pandas.

| format | input | write s | read s | MB | peak MB | retained MB |
|---|---|---|---|---|---|---|
| GraphML | nx graph | 1.784 | graphml: 1.607 | 21.84 | 325 | 309 |
| pickle p5 | nx graph | 0.270 | pickle: 0.200 | 10.64 | 79 | 83 |
| pickle p5 + gzip(6) | nx graph | 0.495 | pickle_gz: 0.223 | 4.35 | 80 | 84 |
| GeoPackage (ox, directed) | nx graph | 1.633 | gpkg: 0.206<br>gpkg_arrow: 0.073 | 16.15 | 82 | 86 |
| GeoParquet snappy (raw gdfs) | gdfs | FAIL: ArrowInvalid('cannot mix list and non-list, non-null values', 'Conversion failed for column osmid with type object') |  |  |  |  |
| GeoParquet snappy (lists stringified) | gdfs | 0.239 | geoparquet: 0.114<br>geoparquet_nx: 0.624 | 5.38 | 82 | 86 |
| GeoParquet zstd (lists stringified) | gdfs | 0.250 | geoparquet: 0.069<br>geoparquet_nx: 0.627 | 4.53 | 85 | 89 |
| Lean pq zstd f32, osmid int64, WKB | lean tbl | 0.027 | lean_pandas: 0.014<br>lean_arrow: 0.009<br>lean_geo: 0.025 | 3.92 | 50 | 54 |
| Lean pq zstd f64, osmid int64, WKB | lean tbl | 0.025 | lean_pandas: 0.012<br>lean_arrow: 0.009 | 4.05 | 46 | 50 |
| Lean pq zstd f32, osmid list<int64>, WKB | lean tbl | 0.025 | lean_pandas: 0.020<br>lean_arrow: 0.009 | 3.93 | 54 | 58 |
| Lean pq zstd f32, no geometry | lean tbl | 0.015 | lean_pandas: 0.004<br>lean_arrow: 0.002 | 1.02 | 17 | 21 |
| Lean pq snappy f32, WKB | lean tbl | 0.020 | lean_pandas: 0.011<br>lean_arrow: 0.006 | 4.63 | 44 | 48 |
| Lean GeoParquet zstd (WKB, gpd) | lean tbl | 0.068 | lean_gpq: 0.037 | 3.93 | 56 | 60 |
| Lean GeoParquet zstd (GeoArrow native) | lean tbl | 0.068 | lean_gpq: 0.037 | 3.63 | 58 | 62 |
| Feather/IPC zstd lean f32 WKB | lean tbl | 0.011 | feather_pandas: 0.012<br>feather_arrow: 0.006 | 3.79 | 26 | 30 |
| Feather/IPC uncompressed lean f32 WKB | lean tbl | 0.002 | feather_pandas: 0.006<br>feather_arrow: 0.001 | 7.04 | 21 | 25 |
| Pkg (b) zip(stored) of lean parquet, via file obj | lean tbl | 0.034 | zip_bytes: 0.016 | 3.92 | 43 | 47 |
| Pkg (c) dir of lean parquet, via file obj | (same files) | 0.000 | dir_bytes: 0.017 | 3.92 | 51 | 55 |
| CSR npz_compressed (topology only) | lean tbl | 0.072 | csr: 0.005 | 0.55 | 0 | 2 |


#### List-valued attribute round trip (G_small)

| format | osmid | highway | name | lanes | maxspeed | reversed | ref |
|---|---|---|---|---|---|---|---|
| GraphML | faithful; scalars: same type | faithful; scalars: same type | faithful; scalars: same type | n/a; scalars: same type | faithful; scalars: same type | faithful; scalars: same type | faithful; scalars: same type |
| pickle p5 | faithful; scalars: same type | faithful; scalars: same type | faithful; scalars: same type | n/a; scalars: same type | faithful; scalars: same type | faithful; scalars: same type | faithful; scalars: same type |
| GeoPackage (ox, directed) | stringified; scalars: int->str | stringified; scalars: same type | stringified; scalars: same type | n/a; scalars: same type | stringified; scalars: same type | stringified; scalars: bool->str | stringified; scalars: same type |
| GeoParquet snappy (lists stringified) | stringified; scalars: int->str | stringified; scalars: same type | stringified; scalars: same type | n/a; scalars: same type | stringified; scalars: same type | stringified; scalars: bool->str | stringified; scalars: same type |
| Lean pq zstd f32, osmid int64, WKB | first elem only; scalars: same type | ';'-joined; scalars: same type | ';'-joined; scalars: same type | n/a; scalars: missing | column missing; scalars: missing | column missing; scalars: missing | column missing; scalars: missing |
| Lean pq zstd f32, osmid list<int64>, WKB | faithful; scalars: int->list | ';'-joined; scalars: same type | ';'-joined; scalars: same type | n/a; scalars: missing | column missing; scalars: missing | column missing; scalars: missing | column missing; scalars: missing |


### Part A — G_large (125,058 nodes, 332,535 edges), read = mean of 3

- input prep: ox.graph_to_gdfs 4.16 s (not included in GeoParquet write times)
- input prep: lean table extraction (4 variants) 4.47 s (not included in lean write times)
  GeoParquet snappy (raw gdfs): write FAILED ArrowInvalid('cannot mix list and non-list, non-null values', 'Conversion failed for column osmid with type object')
Columns: write s (from the stated input), read s per reader (mean), on-disk MB, peak RSS increase during first read in a fresh process (MB, ru_maxrss), RSS retained after read (MB).

Readers: graphml=ox.load_graphml; gpkg=gpd.read_file x2 (pyogrio), gpkg_arrow=same with use_arrow=True; geoparquet=gpd.read_parquet x2; geoparquet_nx=+ox.graph_from_gdfs; lean_arrow=pq.read_table x2; lean_pandas=+to_pandas (WKB left as bytes); lean_geo=+GeoSeries.from_wkb; lean_gpq=pq nodes + gpd.read_parquet edges; zip_bytes/dir_bytes=open(...,'rb') -> BytesIO -> pq.read_table -> pandas.

| format | input | write s | read s | MB | peak MB | retained MB |
|---|---|---|---|---|---|---|
| GraphML | nx graph | 12.595 | graphml: 11.966 | 216.08 | 2286 | 2182 |
| pickle p5 | nx graph | 2.412 | pickle: 1.887 | 120.94 | 750 | 754 |
| pickle p5 + gzip(6) | nx graph | 5.364 | pickle_gz: 2.247 | 61.18 | 752 | 756 |
| GeoPackage (ox, directed) | nx graph | 8.013 | gpkg: 2.349<br>gpkg_arrow: 0.474 | 166.85 | 604 | 608 |
| GeoParquet snappy (raw gdfs) | gdfs | FAIL: ArrowInvalid('cannot mix list and non-list, non-null values', 'Conversion failed for column osmid with type object') |  |  |  |  |
| GeoParquet snappy (lists stringified) | gdfs | 1.789 | geoparquet: 0.844<br>geoparquet_nx: 4.843 | 84.88 | 641 | 556 |
| GeoParquet zstd (lists stringified) | gdfs | 1.903 | geoparquet: 0.879<br>geoparquet_nx: 4.888 | 67.42 | 632 | 552 |
| Lean pq zstd f32, osmid int64, WKB | lean tbl | 0.256 | lean_pandas: 0.170<br>lean_arrow: 0.112<br>lean_geo: 0.290 | 63.84 | 376 | 235 |
| Lean pq zstd f64, osmid int64, WKB | lean tbl | 0.255 | lean_pandas: 0.165<br>lean_arrow: 0.107 | 64.73 | 375 | 234 |
| Lean pq zstd f32, osmid list<int64>, WKB | lean tbl | 0.285 | lean_pandas: 0.216<br>lean_arrow: 0.106 | 63.93 | 392 | 250 |
| Lean pq zstd f32, no geometry | lean tbl | 0.078 | lean_pandas: 0.009<br>lean_arrow: 0.006 | 7.27 | 63 | 67 |
| Lean pq snappy f32, WKB | lean tbl | 0.129 | lean_pandas: 0.114<br>lean_arrow: 0.051 | 80.64 | 365 | 224 |
| Lean GeoParquet zstd (WKB, gpd) | lean tbl | 0.622 | lean_gpq: 0.272 | 63.85 | 457 | 348 |
| Lean GeoParquet zstd (GeoArrow native) | lean tbl | 0.603 | lean_gpq: 0.313 | 64.46 | 617 | 484 |
| Feather/IPC zstd lean f32 WKB | lean tbl | 0.148 | feather_pandas: 0.139<br>feather_arrow: 0.078 | 62.41 | 320 | 258 |
| Feather/IPC uncompressed lean f32 WKB | lean tbl | 0.031 | feather_pandas: 0.076<br>feather_arrow: 0.015 | 95.77 | 301 | 239 |
| Pkg (b) zip(stored) of lean parquet, via file obj | lean tbl | 0.309 | zip_bytes: 0.180 | 63.84 | 464 | 323 |
| Pkg (c) dir of lean parquet, via file obj | (same files) | 0.000 | dir_bytes: 0.171 | 63.84 | 487 | 346 |
| CSR npz_compressed (topology only) | lean tbl | 0.408 | csr: 0.024 | 3.77 | 3 | 7 |


#### List-valued attribute round trip (G_large)

| format | osmid | highway | name | lanes | maxspeed | reversed | ref |
|---|---|---|---|---|---|---|---|
| GraphML | faithful; scalars: same type | faithful; scalars: same type | faithful; scalars: same type | n/a; scalars: same type | faithful; scalars: same type | faithful; scalars: same type | faithful; scalars: same type |
| pickle p5 | faithful; scalars: same type | faithful; scalars: same type | faithful; scalars: same type | n/a; scalars: same type | faithful; scalars: same type | faithful; scalars: same type | faithful; scalars: same type |
| GeoPackage (ox, directed) | stringified; scalars: int->str | stringified; scalars: same type | stringified; scalars: same type | n/a; scalars: same type | stringified; scalars: same type | stringified; scalars: bool->str | stringified; scalars: same type |
| GeoParquet snappy (lists stringified) | stringified; scalars: int->str | stringified; scalars: same type | stringified; scalars: same type | n/a; scalars: same type | stringified; scalars: same type | stringified; scalars: bool->str | stringified; scalars: same type |
| Lean pq zstd f32, osmid int64, WKB | first elem only; scalars: same type | ';'-joined; scalars: same type | ';'-joined; scalars: same type | n/a; scalars: missing | column missing; scalars: missing | column missing; scalars: missing | column missing; scalars: missing |
| Lean pq zstd f32, osmid list<int64>, WKB | faithful; scalars: int->list | ';'-joined; scalars: same type | ';'-joined; scalars: same type | n/a; scalars: missing | column missing; scalars: missing | column missing; scalars: missing | column missing; scalars: missing |


### Part A — G_large_raw (2,224,948 nodes, 4,526,967 edges), read = mean of 1

- input prep: ox.graph_to_gdfs 82.42 s (not included in GeoParquet write times)
- input prep: lean table extraction (4 variants) 122.51 s (not included in lean write times)
Columns: write s (from the stated input), read s per reader (mean), on-disk MB, peak RSS increase during first read in a fresh process (MB, ru_maxrss), RSS retained after read (MB).

Readers: graphml=ox.load_graphml; gpkg=gpd.read_file x2 (pyogrio), gpkg_arrow=same with use_arrow=True; geoparquet=gpd.read_parquet x2; geoparquet_nx=+ox.graph_from_gdfs; lean_arrow=pq.read_table x2; lean_pandas=+to_pandas (WKB left as bytes); lean_geo=+GeoSeries.from_wkb; lean_gpq=pq nodes + gpd.read_parquet edges; zip_bytes/dir_bytes=open(...,'rb') -> BytesIO -> pq.read_table -> pandas.

| format | input | write s | read s | MB | peak MB | retained MB |
|---|---|---|---|---|---|---|
| GraphML | nx graph | 186.629 | graphml: 436.858 | 1389.08 | 3254 | 2967 |
| pickle p5 | nx graph | 14.506 | pickle: 25.298 | 532.49 | 2968 | 2342 |
| Lean pq zstd f32, osmid int64, WKB | lean tbl | 1.247 | lean_pandas: 0.184<br>lean_arrow: 0.049<br>lean_geo: 0.283 | 73.16 | 681 | 685 |
| Lean pq zstd f32, no geometry | lean tbl | 0.768 | lean_pandas: 0.122<br>lean_arrow: 0.050 | 73.16 | 614 | 618 |
| Feather/IPC zstd lean f32 WKB | lean tbl | 0.171 | feather_pandas: 0.108<br>feather_arrow: 0.073 | 63.87 | 539 | 543 |
| Feather/IPC uncompressed lean f32 WKB | lean tbl | 0.148 | feather_pandas: 0.076<br>feather_arrow: 0.040 | 249.77 | 536 | 540 |
| CSR npz_compressed (topology only) | lean tbl | 7.044 | csr: 0.358 | 52.78 | 100 | 104 |


#### List-valued attribute round trip (G_large_raw)

| format | osmid | highway | name | lanes | maxspeed | reversed | ref |
|---|---|---|---|---|---|---|---|
| GraphML | n/a; scalars: same type | n/a; scalars: same type | n/a; scalars: same type | n/a; scalars: same type | n/a; scalars: same type | n/a; scalars: same type | n/a; scalars: same type |
| pickle p5 | n/a; scalars: same type | n/a; scalars: same type | n/a; scalars: same type | n/a; scalars: same type | n/a; scalars: same type | n/a; scalars: same type | n/a; scalars: same type |
| Lean pq zstd f32, osmid int64, WKB | n/a; scalars: same type | n/a; scalars: same type | n/a; scalars: same type | n/a; scalars: missing | n/a; scalars: missing | n/a; scalars: missing | n/a; scalars: missing |


## Part C — packaging facts (PyPI JSON / GitHub API)

| package | latest | uploaded | requires_python | wheels (cp → platforms) | sdist | requires_dist |
|---|---|---|---|---|---|---|
| pandana | 0.8 | 2026-08-24 | >=3.10 | cp310: macos-arm64,linux-x86_64,win-amd64<br>cp311: macos-arm64,linux-x86_64,win-amd64<br>cp312: macos-arm64,linux-x86_64,win-amd64<br>cp313: macos-arm64,linux-x86_64,win-amd64<br>cp314: macos-arm64,linux-x86_64,win-amd64 | yes | numpy>=1.26; pandas>=2.2; requests>=2.0; scikit-learn>=1.5; tables>=3.10 |
| osmnx | 2.1.1 | 2026-07-21 | >=3.11 | cp310: any<br>cp311: any<br>cp312: any<br>cp313: any<br>cp314: any | yes | geopandas>=1.0.1; networkx>=2.5; numpy>=1.24; pandas>=1.5; requests>=2.30; shapely>=2.0; osmnx[entropy,neighbors,raster,visualization]; extra == "all"; scipy>=1.10; extra == "entropy"; scikit-learn>=1.2; extra == "neighbors"; scipy>=1.10; extra == "neighbors"; rasterio>=1.4; extra == "raster"; rio-vrt>=0.3; extra == "raster"; matplotlib>=3.6; extra == "visualization" |
| igraph | 1.0.0 | 2025-10-23 | >=3.9 | cp310: macos-arm64,linux-x86_64,win-amd64<br>cp311: macos-arm64,linux-x86_64,win-amd64<br>cp312: macos-arm64,linux-x86_64,win-amd64<br>cp313: macos-arm64,linux-x86_64,win-amd64<br>cp314: macos-arm64,linux-x86_64,win-amd64 | yes | texttable>=1.6.2; cairocffi>=1.2.0; extra == "cairo"; matplotlib>=3.6.0; platform_python_implementation != "PyPy" and extra == "matplotlib"; plotly>=5.3.0; extra == "plotly"; cairocffi>=1.2.0; extra == "plotting"; cairocffi>=1.2.0; extra == "test"; networkx>=2.5; extra == "test"; pytest>=7.0.1; extra == "test"; pytest-timeout>=2.1.0; extra == "test"; numpy>=1.19.0; platform_python_implementation ! |
| pyrosm | 0.25.0 | 2026-10-25 | >=3.11 | cp310: -<br>cp311: macos-arm64,linux-x86_64,win-amd64<br>cp312: macos-arm64,linux-x86_64,win-amd64<br>cp313: macos-arm64,linux-x86_64,win-amd64<br>cp314: macos-arm64,linux-x86_64,win-amd64 | yes | python-rapidjson; setuptools>=18.0; geopandas>=0.12.0; shapely>=2.1; protobuf>=6.33.5; certifi; tqdm>=4.66; ipywidgets>=8; extra == "notebook"; pyyaml; extra == "recipes"; pyarrow; extra == "recipes" |
| quackosm | 0.19.0 | 2026-09-28 | >=3.10 | cp310: any<br>cp311: any<br>cp312: any<br>cp313: any<br>cp314: any | yes | geopandas>=0.6; shapely>=2; pyarrow>=16.0.0; duckdb>=1.1.2; geoarrow-pyarrow>=0.1.2; typeguard>=3.0.0; psutil>=5.6.2; pooch>=1.6.0; tqdm>=4.42.0; beautifulsoup4; requests; rich>=12.0.0; geopy>=2.0.0; numpy>=1.26.0; rq_geo_toolkit>=2026.9.0; anyascii>=0.3.3; osmfinder>=1.2.0; typer>=0.9.0; extra == "cli"; click>=8.2.1; extra == "cli"; s2sphere>=0.2.5; extra == "cli" |
| osmium | 4.3.2 | 2026-10-18 | >=3.8 | cp310: macos-arm64,linux-x86_64,win-amd64<br>cp311: macos-arm64,linux-x86_64,win-amd64<br>cp312: macos-arm64,linux-x86_64,win-amd64<br>cp313: macos-arm64,linux-x86_64,win-amd64<br>cp314: macos-arm64,linux-x86_64,win-amd64 | yes | requests; pytest; extra == "tests"; pytest-httpserver; extra == "tests"; pytest-run-parallel; extra == "tests"; werkzeug; extra == "tests"; shapely; extra == "tests"; mkdocs; extra == "docs"; mkdocs-material; extra == "docs"; mkdocstrings; extra == "docs"; mkdocstrings-python; extra == "docs"; mkdocs-autorefs; extra == "docs"; mkdocs-gen-files; extra == "docs"; mkdocs-jupyter; extra == "docs"; arg |

- UDST/pandana: pushed_at 2026-10-01T14:49:23Z, updated_at 2026-10-23T13:40:22Z, archived=False, open issues 35, stars 431
- last commits: 2026-10-01 Bound disconnected shortest-path warnings (#211); 2026-09-02 Merge pull request #210 from UDST/dependabot/github_actions/; 2026-08-27 Bump the actions group with 2 updates; 2026-08-25 Docs fixes after v0.8 (#209); 2026-08-24 Version 0.8 (#208)
- last releases: v0.8 (2026-08-24); v0.8rc1 (2026-08-21); v0.7 (2026-08-20)

### B1 — memory footprint, G_large (fresh process each)

RSS = resident memory retained after load (MB); deep = pandas memory_usage(deep=True) / Arrow nbytes (shapely geometries are counted only as pointers by pandas); peak = max RSS increase during load.

| representation | RSS MB | deep/nbytes MB | peak MB |
|---|---|---|---|
| nx MultiDiGraph (pickle load), with geometry | 731 |  | 776 |
| nx MultiDiGraph, no geometry | 462 |  | 462 |
| ox.graph_to_gdfs output (all attrs, geometry filled) | 1052 | 203 | 1109 |
| lean pandas + GeoSeries (decoded shapely) | 393 | 17 | 439 |
| lean pandas, geometry as WKB bytes | 229 | 106 | 373 |
| lean pandas, no geometry | 68 | 15 | 68 |
| lean Arrow tables, WKB | 173 | 96 | 293 |
| lean Arrow tables, no geometry | 57 | 16 | 57 |
| scipy CSR (float64 weights, int32 idx) | 119 | 4 | 119 |


### B2–B5 — conversion costs, G_large (125,058 nodes, 332,535 edges)

#### B2 — nx → tables (s)

| operation | s |
|---|---|
| ox.graph_to_gdfs (nodes+edges) | 3.991 |
| ox.graph_to_gdfs nodes only | 0.579 |
| ox.graph_to_gdfs edges only | 2.961 |
| ox.graph_to_gdfs edges only, fill_edge_geometry=False | 1.515 |
| nx.to_pandas_edgelist | 2.193 |
| pd.DataFrame.from_dict(G.nodes, orient='index') | 0.149 |
| hand-vectorized lean extraction (np.fromiter per column), no geom | 0.505 |
| hand-vectorized lean extraction + WKB geometry | 0.927 |
| generic: pd.DataFrame([dict(u=,v=,key=,**d) for edges]) all attrs | 0.772 |

#### B3 — tables → nx (s)

| operation | s | edges |
|---|---|---|
| add_nodes_from(zip(ids, to_dict('records'))) + add_edges_from(zip(u,v,k,records)), 5 attrs + geometry | 2.517 | 332,535 |
| same, no geometry | 1.896 |  |
| topology + length only (generator of small dicts) | 0.792 |  |
| ox.graph_from_gdfs(graph_to_gdfs output) | 4.495 | 332,535 |

#### B4 — node coordinate array (ms, mean of 5)

| method | ms |
|---|---|
| [list(map(G.nodes[n].get, ['y','x'])) for n in G.nodes()] | 42.66 |
| np.array([(d['y'], d['x']) for _, d in G.nodes(data=True)]) | 42.57 |
| np.fromiter x2 over G.nodes(data=...) | 23.05 |
| df[['y','x']].to_numpy() (lean nodes DataFrame) | 0.61 |

#### B5 — sub-region edges, box 29.856,-2.138,30.264,-1.762 (4% of Rwanda bbox area) (s, mean)

Note: sindex/cx select edges whose geometry intersects the box; node-bbox variants select edges with either endpoint inside; subgraph keeps edges with both endpoints inside.

| method | s | edges |
|---|---|---|
| edges_gdf.sindex.query(box, 'intersects') (+ sindex build 0.088 s, once) | 0.029 | 84,110 |
| edges_gdf.sindex.query(box) bbox-only | 0.016 | 84,114 |
| lean tables: node-coord bbox mask + np.isin(u|v) on edges | 0.037 | 84,098 |
| lean Arrow tables: same with pyarrow.compute | 0.013 | 84,098 |
| nx: node filter + G.subgraph + graph_to_gdfs(edges) | 1.633 | 83,612 |
| full ox.graph_to_gdfs(edges) + .cx[box] | 2.985 | 84,110 |


### B6–B8 — routing engines, G_large (125,058 nodes, 332,535 edges)

#### B7 — scipy.sparse.csgraph (s)

| operation | s | note |
|---|---|---|
| build CSR from tables (searchsorted id->pos, dedupe parallel edges keep-min) | 0.137 | nnz=329,759 |
| connected_components strong | 0.005 | 11 comps, largest 125,047 |
| connected_components weak | 0.005 | 1 comps |
| dijkstra 100 sources, full (100 x n matrix) | 2.411 | 24.1 ms/source |
| dijkstra 100 sources, limit=20000 m | 0.138 | 6673 nodes reached/source |
| 100x1000 many-to-many (dijkstra + index columns) | 2.412 | 24.12 µs/pair |
| dijkstra 100 sources, return_predecessors=True | 2.407 | +-0% vs no predecessors |
| minimum_spanning_tree (symmetrize from tables + MST) | 0.314 | total 29,146 km, 125,057 edges |

- networkx single_source_dijkstra_path_length, 10 sources: 2.92 s (292 ms/source → ~29.2 s for 100 sources) vs scipy 24.1 ms/source → scipy ≈ 12x faster
- check networkx vs scipy distances (source 0, 1000 targets): max |diff| = 0.0027 m
#### B6 — pandana 0.7 (s)

| operation | s | note |
|---|---|---|
| pandana.Network(...) from lean tables, twoway=False (osmid int64 index) | 1.418 | includes CH contraction |
| shortest_path_lengths 10k OD pairs | 0.032 | 3.2 µs/pair |
| shortest_paths 1k OD pairs (node lists) | 0.027 | 0.03 ms/pair |
| 100x1000 = 100k pairs via shortest_path_lengths | 0.192 | 1.92 µs/pair (scipy: see B7) |
| check vs scipy 100x1000 |  | max |diff| 0.474 m, median 0.1481 m (pandana uses float32-ish/mm ints) |
| nodes_in_range 10 sources, 20 km | 0.049 |  |

- string node ids: OK (build 1.3 s; lengths match int-id run: True)
- non-contiguous int64 osmids (min 271317577, max 14321556707, > 2^32): OK (used above)
- query with unknown node id: raises ValueError: Buffer dtype mismatch, expected 'long' but got 'double'

- user function on G_large: DataFrame.from_dict(G.nodes) 0.19 s + nx.to_pandas_edgelist 2.90 s = conversion 3.09 s; pandana.Network (default twoway=True, so every edge also added reversed) CH build 1.31 s

#### B8 — igraph 1.0.0

- Graph(n, edges=list, edge_attrs) build: 0.28 s; distances(100 sources, 1000 targets, weights='length'): 1.33 s (13.34 µs/pair); max |diff| vs scipy 0.0000 m
- igraph distances 100 sources → all nodes: 1.94 s (19.4 ms/source)
