"""Build the Kigali test fixture in the saved-network layout of ADR 0005.

Reads G_small.pkl from step0.py, crops it to central Kigali, normalises the OSM
attributes as section 8 of docs/design/spatial-network.md describes, and writes
tests/data/spatial_network/kigali/{nodes.parquet, edges.parquet, network.json}.

Usage: GS_BENCH_DIR=/tmp/gs_bench python make_fixture.py
"""
import datetime
import json
import os
import pickle
import sys

import geopandas as gpd
import numpy as np
import osmnx as ox
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = os.environ.get("GS_BENCH_DIR", "/tmp/gs_bench")
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(REPO, "tests", "data", "spatial_network", "kigali")
BOX = (30.045, -1.965, 30.085, -1.935)  # west, south, east, north: central Kigali

SPECIAL = {"osmid", "oneway", "reversed", "length", "geometry"}


def _as_list(value):
    return value if isinstance(value, list) else [value]


def _join(value):
    if isinstance(value, list):
        return ";".join(dict.fromkeys(str(x) for x in value))
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return str(value)


def main():
    with open(f"{ROOT}/G_small.pkl", "rb") as f:
        G = pickle.load(f)
    G = ox.truncate.truncate_graph_bbox(G, BOX, truncate_by_edge=False)

    ids = np.fromiter(G.nodes, dtype=np.int64, count=G.number_of_nodes())
    data = G.nodes
    nodes = pd.DataFrame(
        {
            "node_id": ids,
            "x": np.fromiter((data[n]["x"] for n in ids), np.float64, len(ids)),
            "y": np.fromiter((data[n]["y"] for n in ids), np.float64, len(ids)),
            "kind": pd.Categorical(["road"] * len(ids), categories=["road", "point", "split"]),
            "street_count": np.fromiter(
                (data[n].get("street_count", 0) for n in ids), np.int16, len(ids)
            ),
        }
    )

    rows = list(G.edges(keys=True, data=True))
    attrs = [r[3] for r in rows]
    text_columns = sorted({k for a in attrs for k in a} - SPECIAL)
    columns = {
        "u": np.fromiter((r[0] for r in rows), np.int64, len(rows)),
        "v": np.fromiter((r[1] for r in rows), np.int64, len(rows)),
        "key": np.fromiter((r[2] for r in rows), np.int16, len(rows)),
        "length": np.fromiter((a["length"] for a in attrs), np.float64, len(rows)),
        "kind": pd.Categorical(
            ["road"] * len(rows),
            categories=["road", "split", "connection", "point_link", "component_link", "added"],
        ),
        "osmid": [[int(x) for x in _as_list(a["osmid"])] for a in attrs],
        "oneway": [all(_as_list(a.get("oneway", False))) for a in attrs],
        "reversed": [any(_as_list(a.get("reversed", False))) for a in attrs],
    }
    for name in text_columns:
        columns[name] = pd.Categorical([_join(a.get(name)) for a in attrs])
    edges = gpd.GeoDataFrame(
        columns, geometry=[a.get("geometry") for a in attrs], crs="EPSG:4326"
    )

    os.makedirs(OUT, exist_ok=True)
    pq.write_table(
        pa.Table.from_pandas(nodes, preserve_index=False),
        os.path.join(OUT, "nodes.parquet"),
        compression="zstd",
    )
    edges.to_parquet(os.path.join(OUT, "edges.parquet"), compression="zstd", index=False)
    meta = {
        "format": "gigaspatial.spatial_network",
        "format_version": 1,
        "directed": True,
        "crs": "EPSG:4326",
        "weight": "length",
        "metadata": {
            "source": "Geofabrik africa/rwanda-latest.osm.pbf, built by step0.py",
            "network_type": "drive_service (step0.py filter)",
            "simplify": True,
            "bbox": list(BOX),
        },
        "library_version": None,
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    }
    with open(os.path.join(OUT, "network.json"), "w") as f:
        json.dump(meta, f, indent=2)

    size = sum(os.path.getsize(os.path.join(OUT, p)) for p in os.listdir(OUT)) / 1e3
    print(f"{len(nodes):,} nodes, {len(edges):,} edges, {size:.0f} KB in {OUT}")
    print(f"text columns: {text_columns}")


if __name__ == "__main__":
    sys.exit(main())
