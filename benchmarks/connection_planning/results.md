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
