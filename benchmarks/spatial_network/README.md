# Spatial network benchmarks

These scripts produced the numbers in [ADR 0004](../../docs/adr/0004-columnar-spatial-network.md), [ADR 0005](../../docs/adr/0005-spatial-network-storage.md), and section 9 of the [design](../../docs/design/spatial-network.md). They are run by hand, not by pytest. The last full run is in [`results/report.md`](results/report.md).

## Setup

The scripts need osmnx, pandana, psutil, and pyosmium (`osmium`) on top of the library's dependencies. igraph is optional. Data goes to `GS_BENCH_DIR`, which defaults to `/tmp/gs_bench`; about 1.5 GB is written there. To keep it in the repo, use `benchmarks/spatial_network/data`, which git ignores. If the packages are not in your environment, install them into the data folder instead:

```bash
export GS_BENCH_DIR=/tmp/gs_bench
pip install --target "$GS_BENCH_DIR/pylib" osmium igraph
```

The scripts add `$GS_BENCH_DIR/pylib` to the import path when it exists.

## Run

Run from this folder, in order. Each script appends its tables to `$GS_BENCH_DIR/out/report.md`.

| Script | What it does |
|---|---|
| `step0.py` | Downloads the Geofabrik Rwanda extract, filters it to drive and service roads with pyosmium, and builds three osmnx networks: `G_large_raw` (unsimplified Rwanda), `G_large` (simplified Rwanda), and `G_small` (simplified Kigali box). It does not use Overpass. |
| `partA.py run <graph> [reps]` | Storage formats: write and read time, size, memory, and how list-valued attributes survive. |
| `partB.py b1 <graph>`, `conv <graph>`, `algo <graph>` | Memory of each representation, conversion costs, and routing with scipy, pandana, networkx, and igraph. |
| `partC.py` | Package versions and wheel coverage from PyPI, and pandana's maintenance activity from GitHub. Needs the network. |
| `core.py run <graph>` | `SpatialNetwork` construction, memory, accessors, `to_networkx`, and `from_points` against the section 9 targets. Needs `partA.py` to have written `fmt/<graph>/lean_f64`. Last run in [`results/core.md`](results/core.md). |
| `storage.py <graph> [reps]` | `SpatialNetwork.save` and `load` against the section 9 targets, starting from raw osmnx attributes. Last run in [`results/storage.md`](results/storage.md). |
| `q2_micro.py` | A synthetic road grid comparing tables and networkx, without OSM data. Set `SIDE` for the grid size. |
| `make_fixture.py` | Builds the Kigali test fixture in `tests/data/spatial_network/kigali/` from `G_small`. |

Example:

```bash
python -u -W ignore step0.py
python -u -W ignore partA.py run G_large 3
python -u -W ignore partB.py algo G_large
```
