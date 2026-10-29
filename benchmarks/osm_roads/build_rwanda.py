"""
Time building Rwanda's road network from its Geofabrik extract (target: under 30 s).

    PYTHONPATH=. python benchmarks/osm_roads/build_rwanda.py [network_type ...]

Reads ``$GS_BENCH_DIR/rwanda.osm.pbf`` (``/tmp/gs_bench`` by default).
"""

import os
import resource
import sys
import time

from gigaspatial.handlers.osm.pbf import build_road_network

PBF = os.path.join(os.environ.get("GS_BENCH_DIR", "/tmp/gs_bench"), "rwanda.osm.pbf")


def peak_mb() -> float:
    scale = 1e6 if sys.platform == "darwin" else 1e3
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / scale


for network_type in sys.argv[1:] or ["drive_service"]:
    for simplify in (False, True):
        started = time.perf_counter()
        network = build_road_network(PBF, network_type=network_type, simplify=simplify)
        seconds = time.perf_counter() - started
        variant = "simplified" if simplify else "raw"
        print(
            f"{network_type}-{variant}: {len(network.nodes):,} nodes, {len(network.edges):,} edges, "
            f"{seconds:.1f} s, peak memory {peak_mb():,.0f} MB"
        )
