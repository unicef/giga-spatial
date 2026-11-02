"""
Time loading Rwanda's road network for sources, from its stored extract.

    PYTHONPATH=. python benchmarks/osm_roads/load_rwanda.py

Copies ``$GS_BENCH_DIR/rwanda.osm.pbf`` (``/tmp/gs_bench`` by default) and
Geofabrik's index (``$GS_GEOFABRIK_INDEX``) into a temporary data store, then
times the first load (which builds and saves the networks) and later ones.
"""

import os
import shutil
import tempfile
import time
from pathlib import Path

from shapely.geometry import box

from gigaspatial.core.io.local_data_store import LocalDataStore
from gigaspatial.handlers.osm.roads import OSMRoadsHandler

PBF = Path(os.environ.get("GS_BENCH_DIR", "/tmp/gs_bench")) / "rwanda.osm.pbf"
INDEX = Path(os.environ.get("GS_GEOFABRIK_INDEX", "/tmp/gs_geofabrik/index-v1.json"))
SCHOOLS = [(-1.95, 30.06), (-1.94, 30.09), (-1.97, 30.10), (-2.0, 30.05)]

with tempfile.TemporaryDirectory() as root:
    base = Path(root) / "osm/geofabrik"
    (base / "2026-10-26/africa").mkdir(parents=True)
    shutil.copy(PBF, base / "2026-10-26/africa/rwanda.osm.pbf")
    shutil.copy(INDEX, base / "index-v1.json")
    handler = OSMRoadsHandler(data_store=LocalDataStore(root), base_path=Path("osm/geofabrik"))

    for label, source in [
        ("schools, first load (builds raw)", SCHOOLS),
        ("schools", SCHOOLS),
        ("Kigali box", box(29.98, -2.05, 30.25, -1.85)),
        ("country, first load (simplifies)", "RWA"),
        ("country", "RWA"),
    ]:
        started = time.perf_counter()
        network = handler.reader.load(source)
        print(f"{label}: {len(network.nodes):,} nodes, {len(network.edges):,} edges, {time.perf_counter() - started:.2f} s")
