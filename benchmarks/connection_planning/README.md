# Connection planning benchmarks

Benchmarks of `ConnectionPlanner`, for the [connection planning design](../../docs/design/connection-planning.md) and Part 4 of the [design and algorithm decisions](../../docs/design/spatial-networks-decisions.md). Results are in `results.md`, and SteinLib's instance by instance in `steinlib.md`.

Run them from the repository root with `PYTHONPATH=.`, so they use the repository's code and not an installed release:

```bash
PYTHONPATH=. python benchmarks/connection_planning/rollout.py 100 1000 10000
```

| Script | What it measures | Needs |
|---|---|---|
| `rollout.py [terminals ...]` | `method="shortest_path"` against a naive version, and the time targets | `G_large.pkl` from `benchmarks/spatial_network/step0.py` |
| `improve.py [terminals ...]` | `improve()` from each method's plan, and the time of each pass | `G_large.pkl` |
| `bound.py [terminals ...]` | `lower_bound()`, and the gap it gives plans | `G_large.pkl` |
| `steinlib.py [set ...]` | Every method, `improve()`, and the bound on SteinLib's sets B, C, D, and E, against their proven optima | the internet the first time, for steinlib.zib.de |
| `exact.py [terminals ...]` | The same on Kigali's roads, against exact optima from scipy's `milp` (HiGHS); also writes the instances as STP files | the Kigali test fixture |
| `exact.py --rwanda [terminals ...]` | Writes Rwanda's instances of `bound.py` as STP files, for an exact solver such as SCIP-Jack | `G_large.pkl` |

`G_large.pkl` and the downloads go to `$GS_BENCH_DIR` (`/tmp/gs_bench` by default), as for the spatial network benchmarks.
