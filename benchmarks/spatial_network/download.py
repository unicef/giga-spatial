"""Download test graphs with osmnx and cache as pickles in /tmp/gs_bench."""
import json
import os
import pickle
import sys
import time

import osmnx as ox

ox.settings.use_cache = True
ox.settings.cache_folder = "/tmp/gs_bench/cache"
ox.settings.log_console = False
ox.settings.requests_timeout = 600

OUT = "/tmp/gs_bench"
META = os.path.join(OUT, "download_meta.json")

JOBS = {
    "G_small": dict(place="Kigali, Rwanda", simplify=True),
    "G_small_raw": dict(place="Kigali, Rwanda", simplify=False),
    "G_large": dict(place="Rwanda", simplify=True),
}


def load_meta():
    if os.path.exists(META):
        with open(META) as f:
            return json.load(f)
    return {}


def main(names):
    meta = load_meta()
    for name in names:
        path = os.path.join(OUT, f"{name}.pkl")
        if os.path.exists(path):
            print(name, "exists, skipping")
            continue
        job = JOBS[name]
        for attempt in range(2):
            try:
                t0 = time.perf_counter()
                G = ox.graph_from_place(
                    job["place"], network_type="drive_service", simplify=job["simplify"]
                )
                dt = time.perf_counter() - t0
                break
            except Exception as e:
                print(name, "attempt", attempt, "failed:", repr(e))
                if attempt == 1:
                    raise
                time.sleep(30)
        with open(path, "wb") as f:
            pickle.dump(G, f, protocol=5)
        meta[name] = dict(
            place=job["place"],
            simplify=job["simplify"],
            nodes=G.number_of_nodes(),
            edges=G.number_of_edges(),
            wall_s=round(dt, 1),
            attempts=attempt + 1,
        )
        with open(META, "w") as f:
            json.dump(meta, f, indent=2)
        print(name, meta[name], flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or list(JOBS))
