# OSM roads handler fixtures

`make_fixtures.py` builds the test fixtures of the OSM roads handler in `tests/data/osm_roads/`, following section 10 of the [design](../../docs/design/osm-roads-handler.md). It is run by hand, not by pytest.

| Step | Writes | From |
|---|---|---|
| `index` | `geofabrik_index_subset.json`: the extracts the tests use, with their polygons. `geofabrik_index_nogeom.json`: every extract, without polygons, for the test of the curated multi-country list. | Geofabrik's `index-v1.json`, at `$GS_GEOFABRIK_INDEX` (default `/tmp/gs_geofabrik/index-v1.json`; downloaded when missing) |
| `pbf` | `kigali.osm.pbf`: every highway way touching central Kigali, with all of its nodes and tags, and Rwanda's replication timestamp in its header so it stands in for an extract; `kigali.json` records its source and data date | `$GS_BENCH_DIR/rwanda.osm.pbf`, from `../spatial_network/step0.py` |
| `osmnx` | `kigali_osmnx/<network type>-<simplified or raw>/`: osmnx's own network for `kigali.osm.pbf`, saved with `SpatialNetwork.save` | `kigali.osm.pbf` |

osmnx's `graph_from_xml` builds every way in a file, so the `osmnx` step first keeps the ways that pass the filter osmnx sends to Overpass for that network type (`osmnx._overpass._get_network_filter`), evaluated with Overpass's rules: a bare key must be present, and a `!~` pattern is an unanchored regular expression that a way without the key passes. `street_count` comes from the unsimplified graph, as in `graph_from_polygon`, and `ox.simplify_graph` runs with its defaults.

The script needs osmnx and pyosmium. When pyosmium is not installed, put a copy on the path:

```bash
pip install --target /tmp/gs_geofabrik/pylib osmium
PYTHONPATH=/tmp/gs_geofabrik/pylib:../.. python make_fixtures.py
```

`build_rwanda.py` times `build_road_network` on Rwanda's whole extract, raw and simplified, for the network types given (`drive_service` by default); the target is under 30 s. Results are in section 7 of the design.

```bash
PYTHONPATH=/tmp/gs_geofabrik/pylib:../.. python build_rwanda.py drive_service walk
```
