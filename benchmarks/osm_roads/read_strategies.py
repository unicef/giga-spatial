"""
Compare ways of reading every highway way of an extract with its node coordinates, using pyosmium.

    PYTHONPATH=. python benchmarks/osm_roads/read_strategies.py [extract]

Reads ``$GS_BENCH_DIR/rwanda.osm.pbf`` (``/tmp/gs_bench`` by default). Each
strategy runs in its own process, so peak memory is its own:

- ``two-reads-ids``: read the ways and their node references, then read only
  the referenced nodes through an ID filter, in Python.
- ``two-reads-index``: read the ways and their node references, then read the
  ways again with pyosmium's node location index, writing each way's line as
  WKB in C++.
- ``one-read``: read nodes and ways once with the location index, taking the
  references, tags, and WKB line of each way together. This is what
  ``gigaspatial.handlers.osm.pbf`` does.
- ``one-read:<index>``: the same with another pyosmium location index, such
  as the disk-backed ``sparse_file_array`` the handler uses above 1 GiB.
"""

import os
import resource
import subprocess
import sys
import tempfile
import time

PBF = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else os.path.join(
    os.environ.get("GS_BENCH_DIR", "/tmp/gs_bench"), "rwanda.osm.pbf"
)
KEPT = ("access", "area", "bridge", "est_width", "highway", "junction", "landuse", "lanes", "maxspeed", "name",
        "oneway", "ref", "service", "tunnel", "width")


def peak_mb() -> float:
    scale = 1e6 if sys.platform == "darwin" else 1e3
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / scale


def read_ways(osmium):
    ids, refs, tags = [], [], []
    for way in osmium.FileProcessor(PBF, osmium.osm.WAY).with_filter(osmium.filter.KeyFilter("highway")):
        ids.append(way.id)
        refs.extend(n.ref for n in way.nodes)
        tags.append({k: v for k, v in way.tags if k in KEPT})
    return ids, refs


def run(strategy: str) -> None:
    import numpy as np
    import osmium
    import shapely

    started = time.perf_counter()
    if strategy == "two-reads-ids":
        _, refs = read_ways(osmium)
        first = time.perf_counter() - started
        wanted = np.unique(np.array(refs, dtype=np.int64))
        coordinates = [
            (n.id, n.location.lon, n.location.lat)
            for n in osmium.FileProcessor(PBF, osmium.osm.NODE).with_filter(osmium.filter.IdFilter(wanted.tolist()))
        ]
        detail = f"{len(refs):,} references, {len(coordinates):,} nodes; first read {first:.1f} s"
    elif strategy == "two-reads-index":
        _, refs = read_ways(osmium)
        first = time.perf_counter() - started
        factory = osmium.geom.WKBFactory()
        processor = (
            osmium.FileProcessor(PBF, osmium.osm.NODE | osmium.osm.WAY)
            .with_locations("flex_mem")
            .with_filter(osmium.filter.EntityFilter(osmium.osm.WAY))
            .with_filter(osmium.filter.KeyFilter("highway"))
        )
        lines = [factory.create_linestring(way.nodes, osmium.geom.ALL) for way in processor]
        coordinates = shapely.get_coordinates(shapely.from_wkb(lines))
        detail = f"{len(refs):,} references, {len(coordinates):,} coordinates; first read {first:.1f} s"
    else:
        index = strategy.split(":", 1)[1] if ":" in strategy else "flex_mem"
        if index == "sparse_file_array":
            index = f"sparse_file_array,{tempfile.mkdtemp()}/nodes.idx"
        factory = osmium.geom.WKBFactory()
        refs, lines = [], []
        processor = (
            osmium.FileProcessor(PBF, osmium.osm.NODE | osmium.osm.WAY)
            .with_locations(index)
            .with_filter(osmium.filter.KeyFilter("highway").enable_for(osmium.osm.WAY))
            .with_filter(osmium.filter.KeyFilter("highway", "junction", "railway", "ref").enable_for(osmium.osm.NODE))
        )
        for entity in processor:
            if entity.is_node():
                continue
            refs.extend(n.ref for n in entity.nodes)
            {k: v for k, v in entity.tags if k in KEPT}
            lines.append(factory.create_linestring(entity.nodes, osmium.geom.ALL))
        coordinates = shapely.get_coordinates(shapely.from_wkb(lines))
        detail = f"{len(refs):,} references, {len(coordinates):,} coordinates"
    print(f"| {strategy} | {time.perf_counter() - started:.1f} | {peak_mb():,.0f} | {detail} |")


if __name__ == "__main__":
    if "--run" in sys.argv:
        run(sys.argv[sys.argv.index("--run") + 1])
    else:
        print(f"Extract: {PBF} ({os.path.getsize(PBF) / 2**20:.0f} MiB)\n")
        print("| strategy | s | peak MB | read |\n|---|---|---|---|", flush=True)
        strategies = ("two-reads-ids", "two-reads-index", "one-read", "one-read:sparse_file_array", "one-read:sparse_mem_array")
        for strategy in strategies:
            subprocess.run([sys.executable, "-W", "ignore", __file__, PBF, "--run", strategy], check=True)
