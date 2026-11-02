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
