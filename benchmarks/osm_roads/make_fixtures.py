"""Build the OSM roads handler's test fixtures in tests/data/osm_roads/.

Usage: python make_fixtures.py [index] [pbf] [osmnx]   (all three by default)

- index: from Geofabrik's index-v1.json, a subset with geometries for the
  extracts the tests use, and the whole index without geometries for the test
  of the curated multi-country list.
- pbf: kigali.osm.pbf, every highway way touching the central Kigali box of the
  spatial network fixtures, with all of its nodes and tags, cut from the
  Rwanda extract with pyosmium.
- osmnx: osmnx's own networks for kigali.osm.pbf, one folder per network type,
  saved with SpatialNetwork.save. graph_from_xml does not filter by network
  type, so ways are first filtered with the filter string osmnx sends to
  Overpass for that type (osmnx._overpass._get_network_filter), evaluated here
  with Overpass semantics.

Inputs: $GS_BENCH_DIR/rwanda.osm.pbf (step0.py downloads it) and
$GS_GEOFABRIK_INDEX (default /tmp/gs_geofabrik/index-v1.json; downloaded when
missing). Needs osmnx and osmium; put osmium on PYTHONPATH if it is not installed.
"""
import datetime
import json
import os
import re
import sys
import urllib.request
from pathlib import Path

ROOT = os.environ.get("GS_BENCH_DIR", "/tmp/gs_bench")
INDEX = os.environ.get("GS_GEOFABRIK_INDEX", "/tmp/gs_geofabrik/index-v1.json")
REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "tests" / "data" / "osm_roads"
BOX = (30.045, -1.965, 30.085, -1.935)  # west, south, east, north: the spatial network fixtures' box

SUBSET = [
    # Rwanda, its neighbours, and its continent
    "rwanda", "burundi", "uganda", "tanzania", "congo-democratic-republic", "kenya", "africa",
    # a country with sub-region extracts
    "brazil", "centro-oeste", "nordeste", "norte", "sudeste", "sul", "south-america",
    # extracts holding several countries
    "israel-and-palestine", "malaysia-singapore-brunei", "senegal-and-gambia", "haiti-and-domrep",
    "gcc-states", "ireland-and-northern-ireland", "south-africa-and-lesotho",
    # a mis-coded territory next to the country whose code it carries
    "vanuatu", "polynesie-francaise",
]
NETWORKS = [("drive_service", True), ("drive_service", False), ("drive", True), ("walk", True), ("all", True)]


def index():
    if not os.path.exists(INDEX):
        os.makedirs(os.path.dirname(INDEX), exist_ok=True)
        urllib.request.urlretrieve("https://download.geofabrik.de/index-v1.json", INDEX)
    with open(INDEX) as f:
        full = json.load(f)
    by_id = {feature["properties"]["id"]: feature for feature in full["features"]}
    missing = [i for i in SUBSET if i not in by_id]
    if missing:
        raise SystemExit(f"Not in the index: {missing}")
    OUT.mkdir(parents=True, exist_ok=True)
    snapshot = datetime.datetime.fromtimestamp(os.path.getmtime(INDEX), datetime.timezone.utc).date().isoformat()
    subset = {"type": "FeatureCollection", "snapshot": snapshot, "features": [by_id[i] for i in SUBSET]}
    with open(OUT / "geofabrik_index_subset.json", "w") as f:
        json.dump(subset, f, separators=(",", ":"))
    bare = {
        "type": "FeatureCollection",
        "snapshot": snapshot,
        "features": [{"type": "Feature", "properties": feature["properties"], "geometry": None} for feature in full["features"]],
    }
    with open(OUT / "geofabrik_index_nogeom.json", "w") as f:
        json.dump(bare, f, separators=(",", ":"))
    for name in ("geofabrik_index_subset.json", "geofabrik_index_nogeom.json"):
        print(f"{name}: {(OUT / name).stat().st_size / 1e3:.0f} KB")


def pbf():
    import osmium

    source = f"{ROOT}/rwanda.osm.pbf"
    west, south, east, north = BOX
    out = OUT / "kigali.osm.pbf"
    cut = OUT / "kigali.cut.osm.pbf"
    OUT.mkdir(parents=True, exist_ok=True)
    writer = osmium.BackReferenceWriter(str(cut), ref_src=source, overwrite=True, remove_tags=False)
    ways = 0
    entities = osmium.osm.NODE | osmium.osm.WAY
    processor = osmium.FileProcessor(source, entities).with_locations().with_filter(osmium.filter.KeyFilter("highway"))
    for way in processor:
        if not way.is_way():
            continue
        if any(west <= n.lon <= east and south <= n.lat <= north for n in way.nodes if n.location.valid()):
            writer.add_way(way)
            ways += 1
    writer.close()
    # BackReferenceWriter writes no header; the handler reads an extract's date from it.
    reader = osmium.io.Reader(source, osmium.osm.NOTHING)
    stamp = reader.header().get("osmosis_replication_timestamp")
    reader.close()
    header = osmium.io.Header()
    header.set("osmosis_replication_timestamp", stamp)
    header.add_box(osmium.osm.Box(osmium.osm.Location(west, south), osmium.osm.Location(east, north)))
    with osmium.SimpleWriter(str(out), header=header, overwrite=True) as final:
        for entity in osmium.FileProcessor(str(cut)):
            final.add(entity)
    cut.unlink()
    with open(OUT / "kigali.json", "w") as f:
        json.dump({"source": os.path.basename(source), "replication_timestamp": stamp, "bbox": list(BOX), "highway_ways": ways}, f, indent=2)
    print(f"kigali.osm.pbf: {ways:,} highway ways, {out.stat().st_size / 1e3:.0f} KB, data as of {stamp}")


_CONDITION = re.compile(r'\["([^"]+)"(?:(!~|~|!=|=)"([^"]*)")?\]')


def overpass_filter(text):
    """A test for a way's tags from an Overpass way filter such as osmnx's network filters."""
    conditions = _CONDITION.findall(text)
    if "".join(f'["{k}"' + (f'{op}"{v}"' if op else "") + "]" for k, op, v in conditions) != text:
        raise ValueError(f"Unsupported filter: {text}")

    def passes(tags):
        for key, op, value in conditions:
            present = key in tags
            if op == "":
                ok = present
            elif op == "!~":
                ok = not present or re.search(value, tags[key]) is None
            elif op == "~":
                ok = present and re.search(value, tags[key]) is not None
            elif op == "=":
                ok = present and tags[key] == value
            else:
                ok = not present or tags[key] != value
            if not ok:
                return False
        return True

    return passes


def osmnx_networks():
    import osmium
    import osmnx as ox
    from osmnx import _overpass

    sys.path.insert(0, str(REPO))
    from gigaspatial.processing.network import SpatialNetwork

    source = OUT / "kigali.osm.pbf"
    # pyosmium objects are only valid while the file is read, so copy what is needed.
    ways = [(w.id, [n.ref for n in w.nodes], dict(w.tags)) for w in osmium.FileProcessor(str(source), osmium.osm.WAY)]
    nodes = {
        n.id: (n.location.lat, n.location.lon, dict(n.tags))
        for n in osmium.FileProcessor(str(source), osmium.osm.NODE)
    }
    folder = OUT / "kigali_osmnx"
    for network_type, simplified in NETWORKS:
        passes = overpass_filter(_overpass._get_network_filter(network_type))
        kept = [w for w in ways if passes(w[2])]
        refs = sorted({r for _, node_refs, _ in kept for r in node_refs})
        xml = OUT / f"_{network_type}.osm"
        _write_osm_xml(xml, refs, nodes, kept)
        bidirectional = network_type in ox.settings.bidirectional_network_types
        raw = ox.graph_from_xml(str(xml), bidirectional=bidirectional, simplify=False, retain_all=True)
        raw.graph["simplified"] = False
        for node, count in ox.stats.count_streets_per_node(raw).items():
            raw.nodes[node]["street_count"] = count
        graph = ox.simplify_graph(raw) if simplified else raw
        name = f"{network_type}-{'simplified' if simplified else 'raw'}"
        SpatialNetwork.from_networkx(graph).save(folder / name)
        xml.unlink()
        size = sum(p.stat().st_size for p in (folder / name).iterdir()) / 1e3
        print(f"{name}: {len(kept):,} ways, {graph.number_of_nodes():,} nodes, {graph.number_of_edges():,} edges, {size:.0f} KB")


def _write_osm_xml(path, refs, nodes, kept):
    """OSM XML for osmnx's graph_from_xml: the kept ways and only the nodes they use."""
    from xml.sax.saxutils import quoteattr

    with open(path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n<osm version="0.6" generator="make_fixtures">\n')
        for ref in refs:
            lat, lon, tags = nodes[ref]
            f.write(f'  <node id="{ref}" lat="{lat:.7f}" lon="{lon:.7f}">')
            f.write("".join(f"<tag k={quoteattr(k)} v={quoteattr(v)}/>" for k, v in tags.items()))
            f.write("</node>\n")
        for way_id, node_refs, tags in kept:
            f.write(f'  <way id="{way_id}">')
            f.write("".join(f'<nd ref="{r}"/>' for r in node_refs))
            f.write("".join(f"<tag k={quoteattr(k)} v={quoteattr(v)}/>" for k, v in tags.items()))
            f.write("</way>\n")
        f.write("</osm>\n")


if __name__ == "__main__":
    steps = sys.argv[1:] or ["index", "pbf", "osmnx"]
    for step in steps:
        {"index": index, "pbf": pbf, "osmnx": osmnx_networks}[step]()
