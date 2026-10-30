# Spatial networks: concepts and methods

`SpatialNetwork` is the library's model of a network laid out on the ground: nodes with coordinates, joined by edges that carry a length in metres. Road networks from OpenStreetMap are the main kind, but points joined to their nearest neighbours, or any network you build from your own tables, work the same way.

This note explains how it works, method by method, and what each method is good for. It is written for data scientists and statisticians who know Python, pandas, geopandas, and the basics of graphs. Where a method's result depends on how its algorithm works, the note explains the algorithm and why we chose it. The evidence behind those choices, with the benchmarks, is in [design and algorithm decisions](../../design/spatial-networks-decisions.md). For a ten-minute start, read the [road networks guide](../../user-guide/road-networks.md) first.

## Why a spatial network

Most of Giga's questions about schools are network questions. How many metres of fibre would connect these schools to the nearest fibre node, along roads? Which schools have a clinic within 5 km by road? How much longer is the road to the nearest secondary school than the straight line, and does that explain attendance? Straight-line distance answers none of them well: a river, a ridge, or a missing bridge can turn 2 km into 15.

These questions share a few building blocks: get the roads, put the schools on them, measure along them, and keep the part of the network that matters. `SpatialNetwork` provides those blocks and leaves the application to you.

## What it is like to use

A few principles run through every method:

- **Tables, not graph objects.** A network is a table of nodes and a table of edges, which you can inspect, filter, and join like any pandas table. networkx and pandana are export formats, available when you need them.
- **Metres.** Every length is in metres, measured on the sphere the same way osmnx measures roads, so lengths the library adds are consistent with the roads they join.
- **Directed, like OpenStreetMap.** A road network follows one-way restrictions, and a two-way road is two edges. Methods that model cables or walking, where direction does not matter, take `directed=False` or work on the undirected network themselves.
- **Every edit returns a new network.** Nothing changes in place, so a network you computed a result on stays the network you computed it on.
- **Many at once.** Methods take many points, nodes, or edges in one call. A Python loop that edits one edge at a time is the slow way to use the library.

## Contents

1. Why a spatial network: this page.
2. [Getting a network](getting-a-network.md): roads from OpenStreetMap, for an area or a whole country, and networks from your own tables or points. It also sets up the example used throughout.
3. [The data model](data-model.md): the node and edge tables, kinds, keys, lengths, and weights.
4. [Editing](editing.md): selecting, clipping, attributes, weights, combining networks.
5. [Inserting points](inserting-points.md): joining schools and other points to the roads.
6. [Reduction](reduction.md): simplifying a network, and keeping only the roads that connect a set of points.
7. [Algorithms](algorithms.md): shortest paths, service areas, nearest facilities, counts within a distance, components, spanning trees.
8. [Storage and interoperability](storage.md): saving, loading, and converting.
9. [Worked use cases](use-cases.md): fibre planning and accessibility, end to end.
10. [Performance and pitfalls](performance.md).

## Use cases

Five applications come back throughout the note. Each method's section says which it serves.

| Use case | Question | Main methods |
|---|---|---|
| Fibre backbone planning | Which roads would carry fibre from the existing fibre to these schools, what would it cost, and in which order? | `insert_points`, `ConnectionPlanner`, `connecting_subnetwork`, custom weights |
| Last-mile cost | How far is each school from the nearest fibre node, by road? | `insert_points`, `nearest_by_network` |
| Accessibility | Which schools have a clinic within 5 km by road, and what area does a facility serve? | `count_within`, `nearest_by_network`, `reachable` |
| Network distance as a model feature | Road distances, and how much longer they are than straight lines, as covariates or outcomes | `shortest_path_lengths` |
| Network diagnostics | Which parts of the network are cut off, and which schools cannot be reached? | `connected_components`, `connect_components` |

## The example

Every snippet in the note runs on the same example: Rwanda's roads from OpenStreetMap, with 200 schools, 5 fibre nodes, and 10 clinics placed at random around Kigali with a fixed seed. The points are made up, so anyone can run the note without data access or credentials. Snippets build on each other, page after page, and every output shown was produced by running them in order (`docs/scripts/run_concept_examples.py` reruns them all).

The timings shown come from one laptop (an Apple M1 Pro), and are a guide to how costs compare rather than a promise.
