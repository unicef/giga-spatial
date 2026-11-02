## Run of 30 October 2026

Apple M1 Pro, 8 CPUs, 16 GiB, macOS 15.7; Python 3.14.6, numpy 2.5.2, pandas 3.0.3, scipy 1.18.0, networkx 3.6.1. Rwanda's simplified drive network (`G_large`, 125,058 nodes, 332,535 edges), 20 random roots, random terminals with seed 0.

### Sub-task 3: rollout order (`rollout.py 100 1000 10000`)

| terminals | planner s | rollout | rooted Steiner | naive s | same cost |
|---|---|---|---|---|---|
| 100 | 0.36 | 1,163 km | 1,194 km | 1.61 | yes |
| 1,000 | 1.08 | 3,845 km | 3,942 km | 19.53 | yes |
| 10,000 | 4.40 | 10,760 km | 10,954 km |  |  |

"planner s" includes assembling the plan. The naive version runs one full Dijkstra from the connected set per terminal, and is run up to 1,000 terminals.

An earlier version updated distances with scipy's Dijkstra limited to the largest distance of a waiting terminal. It took 0.64 s, 5.42 s, and 58.4 s for the same three sizes: each update searched about 40,000 nodes (455 million in the 10,000-terminal run), where a Dijkstra that stops wherever a distance does not fall updates 6.2 million nodes in the whole run.

### Sub-task 4: improve() (`improve.py 100 1000`)

| terminals | method | plan | improved | saved | moves | passes | longest pass s | total s |
|---|---|---|---|---|---|---|---|---|
| 100 | steiner | 1,194 km (0.12 s) | 1,150 km | 3.7% | 78 | 5 | 1.4 | 6.0 |
| 100 | shortest_path | 1,163 km (0.36 s) | 1,149 km | 1.2% | 38 | 4 | 1.1 | 4.4 |
| 100 | nearest_root | 1,541 km (0.07 s) | 1,150 km | 25.3% | 132 | 5 | 1.3 | 5.9 |
| 1,000 | steiner | 3,942 km (0.16 s) | 3,755 km | 4.7% | 560 | 5 | 5.8 | 26.6 |
| 1,000 | shortest_path | 3,845 km (1.04 s) | 3,759 km | 2.2% | 331 | 4 | 5.8 | 22.3 |
| 1,000 | nearest_root | 5,510 km (0.18 s) | 3,759 km | 31.8% | 1,051 | 5 | 9.4 | 31.6 |

Passes run until neither move improves the plan; `time_limit` stops earlier. The last pass of each run found nothing, so it only confirms the plan. An earlier version searched for key-vertex elimination's links from every node left in the plan; with 1,000 terminals from `steiner` it took 60 s for four passes, against about 5 s per pass now, to within 0.01% of the same cost on Kigali.

### Sub-task 5: lower_bound() (`bound.py 100 1000 10000`)

| terminals | bound s | bound | steiner (gap) | improved, 120 s at most (gap) |
|---|---|---|---|---|
| 100 | 1.7 | 1,127 km | 1,194 km (6.0%) | 1,150 km (2.1%) |
| 1,000 | 5.0 | 3,703 km | 3,942 km (6.5%) | 3,755 km (1.4%) |
| 10,000 | 40.5 | 10,530 km | 10,954 km (4.0%) | 10,591 km (0.6%) |

The gap says how far above the optimum a plan can be at most.

The ascent follows Wong's rule: every step, the terminal whose region has the fewest entering arcs raises the bound by one arc. The first version gave each terminal one whole turn until it reached the root, nearest the root first. It was valid but much weaker: against SteinLib's proven optima (sub-task 6), its bound was on average 7.0% below the optimum on set B and 9.7% on set C, at worst 45%, against 0.05% and 0.42%, at worst 3.1%, for Wong's rule. On Rwanda it gave gaps of 5.5%, 10.9%, and 12.0% for the improved plans above, in 2.1, 7.0, and 31.6 s.

Wong's rule recomputing each region from scratch every step took 41 s for 100 terminals on Rwanda, where few far-apart terminals grow large regions. Regions only grow, since reduced costs only fall, so each terminal keeps its region and entering arcs, and a step checks only those arcs: the same bounds on every B and C instance, in 1.5 s.

A single-commodity flow model of one Kigali instance in scipy's `milp` (HiGHS) did not finish in 600 s: its best plan was 29,413 m against the planner's improved 28,079 m, and its proven lower bound 20,997 m against the first version's dual ascent at 26,092 m.

### Sub-task 6: SteinLib test sets against their proven optima (`steinlib.py B C D E`)

| set | instances | nodes | steiner | + improve | shortest_path | + improve | worst improved | bound below optimum | worst bound | slowest s |
|---|---|---|---|---|---|---|---|---|---|---|
| B | 18 | 50-100 | 3.44% (5) | 0.60% (14) | 2.57% (4) | 0.41% (14) | 3.94% | 0.05% (17) | 0.82% | 0.0 |
| C | 20 | 500-500 | 6.23% (1) | 1.87% (7) | 3.32% (4) | 1.85% (6) | 9.09% | 0.42% (13) | 3.12% | 0.4 |
| D | 20 | 1,000-1,000 | 7.90% (1) | 1.97% (6) | 3.87% (3) | 1.37% (8) | 8.97% | 0.38% (14) | 4.35% | 1.9 |
| E | 20 | 2,500-2,500 | 9.85% (0) | 2.20% (3) | 4.69% (1) | 1.39% (4) | 8.82% | 0.30% (8) | 2.48% | 17.9 |

Plans: mean percentage above the optimum, and (in brackets) the instances solved optimally. Bound: mean percentage below the optimum, and the instances where it equals it. Slowest: the longest steiner plan, improve, and bound of one instance.

Every optimum in these sets is proven. Instance by instance in `steinlib.md`. The worst improved plan is 9% above its optimum (on C and D), so on these dense random graphs the deferred moves, vertex insertion and elimination, would still help; on Kigali's roads below the improved plans' worst is 0.86%.

### Sub-task 6: exact optima on Kigali (`exact.py 30 60 --seeds 5`)

Kigali's largest component (1,195 nodes), 3 random roots, solved exactly by scipy's `milp` (HiGHS) with the multi-commodity flow model.

| terminals | seed | optimum | HiGHS s | steiner | + improve | shortest_path | + improve | bound below optimum |
|---|---|---|---|---|---|---|---|---|
| 30 | 0 | 20,299 m | 5.4 | 7.08% | 0.38% | 5.73% | 0.38% | 0.64% |
| 30 | 1 | 18,091 m | 23.9 | 3.49% | 0.00% | 0.65% | 0.00% | 0.58% |
| 30 | 2 | 22,771 m | 71.3 | 8.77% | 0.04% | 5.90% | 0.04% | 1.10% |
| 30 | 3 | 21,564 m | 9.9 | 3.24% | 0.49% | 1.21% | 0.49% | 0.89% |
| 30 | 4 | 22,424 m | 3.7 | 3.00% | 0.32% | 0.86% | 0.32% | 0.52% |
| 60 | 0 | 27,988 m | 13.5 | 3.77% | 0.32% | 0.65% | 0.32% | 0.20% |
| 60 | 1 | 28,210 m | 21.7 | 5.34% | 0.04% | 1.36% | 0.03% | 0.07% |
| 60 | 2 | 31,027 m | 11.5 | 4.66% | 0.02% | 2.18% | 0.02% | 0.82% |
| 60 | 3 | 28,000 m | 110.5 | 4.01% | 0.69% | 1.09% | 0.86% | 0.77% |
| 60 | 4 | 29,046 m | 87.2 | 4.70% | 0.15% | 1.84% | 0.70% | 0.16% |

Plans: percentage above the optimum. Bound: percentage below it. The instances are also written in SteinLib's STP format to `$GS_BENCH_DIR/stp`, with the virtual root as the last node, for an exact Steiner tree solver such as SCIP-Jack. `exact.py --rwanda 100 1000 10000` writes Rwanda's instances of `bound.py` the same way. The weaker single-commodity flow model of sub-task 5 did not solve one of these in 600 s.

**The default method.** Without `improve()`, `shortest_path` was 0.65 to 5.9% above the optimum on Kigali and `steiner` 3.0 to 8.8%; on SteinLib 2.6 to 4.7% and 3.4 to 9.9% on average. After `improve()`, both end within about 0.1% of each other. `shortest_path` is the default of `plan()` and `connecting_subnetwork()`; on Rwanda it takes 1.1 s for 1,000 terminals and 4.4 s for 10,000, against 0.2 s for `steiner`.
