import warnings
from itertools import combinations

import networkx as nx
import numpy as np
import pandas as pd
import pytest
import shapely
from networkx.algorithms.approximation import steiner_tree
from scipy.sparse.csgraph import dijkstra

from gigaspatial.processing.network import ConnectionPlan, ConnectionPlanner, SpatialNetwork
from gigaspatial.processing.network.geodesy import haversine, line_lengths
from gigaspatial.processing.network.planning import BRIDGES_PER_COMPONENT, _bridge_candidates, _pair_weights, _Tree


def nodes_on_a_line(ids):
    return pd.DataFrame({"x": 30.0 + 0.01 * np.arange(len(ids)), "y": [-1.95] * len(ids)}, index=ids)


@pytest.fixture
def chain():
    """R1 -10- A -1- B -10.5- R2: B is nearer to R2, but cheaper to reach through A."""
    edges = pd.DataFrame(
        {"u": ["R1", "A", "B"], "v": ["A", "B", "R2"], "length": [10.0, 1.0, 10.5], "cost": [10.0, 50.0, 10.5]}
    )
    return SpatialNetwork(nodes_on_a_line(["R1", "A", "B", "R2"]), edges, directed=False)


@pytest.fixture(scope="module")
def connected(kigali):
    return kigali.largest_component()


@pytest.fixture(scope="module")
def chosen(connected):
    rng = np.random.default_rng(1)
    ids = connected.node_index
    return list(ids[rng.choice(len(ids), 30, replace=False)]), list(ids[rng.choice(len(ids), 3, replace=False)])


def with_virtual_root(network, roots):
    G = nx.Graph(network.to_undirected().to_networkx(directed=False))
    G.add_edges_from((r, "virtual", {"length": 0.0}) for r in roots)
    return G


def plan_trees(plan):
    """The plan's edges as a networkx forest, without its isolated nodes."""
    G = nx.Graph()
    G.add_edges_from(plan.edges[["u", "v"]].itertuples(index=False, name=None))
    return G


# ----- the rooted Steiner tree


def test_the_methods_on_a_chain(chain):
    planner = ConnectionPlanner(chain, ["A", "B"], roots=["R1", "R2"])
    rooted, nearest, pairs = (planner.plan(m) for m in ("steiner", "nearest_root", "all_pairs"))
    assert (rooted.cost, nearest.cost, pairs.cost) == (11.0, 20.5, 21.5)
    assert ConnectionPlanner(chain, ["A", "B", "R1", "R2"]).plan().cost == 21.5
    assert rooted.terminals.to_dict("index") == {
        "A": {"status": "connected", "root": "R1", "distance": 10.0, "bridged": False},
        "B": {"status": "connected", "root": "R1", "distance": 11.0, "bridged": False},
    }
    assert nearest.terminals.loc["B", "root"] == "R2" and nearest.terminals.loc["B", "distance"] == 10.5


def test_rooted_tree_matches_networkx_on_the_virtual_root_graph(connected, chosen):
    terminals, roots = chosen
    plan = ConnectionPlanner(connected, terminals, roots=roots).plan()
    G = with_virtual_root(connected, roots)
    networkx_best = min(
        steiner_tree(G, terminals + ["virtual"], weight="length", method=m).size(weight="length")
        for m in ("mehlhorn", "kou")
    )
    # networkx 3.6 gives the same tree; older versions' approximations can be longer.
    assert plan.cost <= networkx_best * (1 + 1e-9)
    assert plan.cost < ConnectionPlanner(connected, terminals + roots).plan().cost
    assert plan.cost < ConnectionPlanner(connected, terminals, roots=roots).plan("nearest_root").cost


def test_a_rooted_plan_is_a_forest_whose_every_tree_reaches_a_root(connected, chosen):
    terminals, roots = chosen
    plan = ConnectionPlanner(connected, terminals, roots=roots).plan()
    forest = plan_trees(plan)
    assert nx.is_forest(forest)
    for tree in nx.connected_components(forest):
        assert tree & set(roots)
    degree = dict(forest.degree())
    assert {n for n, d in degree.items() if d == 1} <= set(terminals) | set(roots)
    assert set(terminals) <= set(forest) | set(roots)


def test_distances_follow_the_plan(connected, chosen):
    terminals, roots = chosen
    plan = ConnectionPlanner(connected, terminals, roots=roots).plan()
    along = plan.network.nearest_by_network(terminals, roots)
    assert plan.terminals["distance"].to_numpy() == pytest.approx(along["distance"].to_numpy())
    assert (plan.terminals["root"].to_numpy() == along["target"].to_numpy()).all()
    assert (plan.terminals["status"] == "connected").all()


# ----- against brute force


def random_grid(seed, rows=3, columns=4):
    rng = np.random.default_rng(seed)
    G = nx.grid_2d_graph(rows, columns)
    ids = {node: f"n{i}" for i, node in enumerate(G)}
    nodes = pd.DataFrame(
        {"x": [30.0 + 0.01 * c for _, c in G], "y": [-1.95 + 0.01 * r for r, _ in G]}, index=list(ids.values())
    )
    pairs = list(G.edges)
    lengths = rng.choice([0.0, 1.0, 2.0, 3.0, 5.0, 8.0], size=len(pairs))
    edges = pd.DataFrame({"u": [ids[a] for a, _ in pairs], "v": [ids[b] for _, b in pairs], "length": lengths})
    network = SpatialNetwork(nodes, edges, directed=False)
    picked = list(rng.choice(list(ids.values()), 6, replace=False))
    return network, picked[:4], picked[4:]


def optimum(network, terminals, roots):
    """The smallest Steiner tree, over every set of Steiner nodes."""
    G = with_virtual_root(network, roots) if roots else nx.Graph(network.to_networkx(directed=False))
    required = set(terminals) | ({"virtual"} if roots else set())
    others = [n for n in G if n not in required]
    best = np.inf
    for k in range(len(others) + 1):
        for extra in combinations(others, k):
            H = G.subgraph(required | set(extra))
            if nx.is_connected(H):
                best = min(best, nx.minimum_spanning_tree(H, weight="length").size(weight="length"))
    return best


@pytest.mark.parametrize("seed", range(8))
def test_plans_against_the_optimum(seed):
    network, terminals, roots = random_grid(seed)
    rooted = optimum(network, terminals, roots)
    planner = ConnectionPlanner(network, terminals, roots=roots)
    steiner = planner.plan().cost
    assert rooted - 1e-9 <= steiner <= 2 * rooted + 1e-9
    assert rooted - 1e-9 <= planner.plan("shortest_path").cost <= 2 * rooted + 1e-9
    for method in ("nearest_root", "all_pairs"):
        assert planner.plan(method).cost >= rooted - 1e-9
    unrooted = optimum(network, terminals, [])
    assert unrooted - 1e-9 <= ConnectionPlanner(network, terminals).plan().cost <= 2 * unrooted + 1e-9


# ----- terminals and edge cases


def test_a_terminal_that_is_a_root(chain):
    plan = ConnectionPlanner(chain, ["R1", "B"], roots=["R1", "R2"]).plan()
    assert plan.terminals.loc["R1"].to_dict() == {"status": "root", "root": "R1", "distance": 0.0, "bridged": False}
    assert plan.terminals.loc["B", "distance"] == 10.5
    assert plan.cost == 10.5 and plan.metadata["status"] == {"connected": 1, "root": 1, "unreachable": 0}


def test_terminals_that_cannot_reach_a_root(kigali, connected, chosen):
    terminals, roots = chosen
    island = kigali.connected_components().idxmax()
    plan = ConnectionPlanner(kigali, terminals[:5] + [island], roots=roots).plan()
    row = plan.terminals.loc[island]
    assert row["status"] == "unreachable" and row["root"] is None and np.isnan(row["distance"])
    assert island in plan.network.node_index and plan.network.degree()[island] == 0
    assert island not in set(plan.edges["u"]) | set(plan.edges["v"])
    assert plan.cost == pytest.approx(ConnectionPlanner(kigali, terminals[:5], roots=roots).plan().cost)


def test_without_roots_the_part_with_most_terminals_is_joined(kigali, chosen):
    terminals, _ = chosen
    island = kigali.connected_components().idxmax()
    plan = ConnectionPlanner(kigali, terminals[:5] + [island]).plan()
    assert plan.terminals.loc[island, "status"] == "unreachable"
    assert (plan.terminals.drop(island)["status"] == "connected").all()
    assert plan.terminals["root"].isna().all() and plan.terminals["distance"].isna().all()


def test_one_terminal_without_roots_is_an_empty_plan(connected, chosen):
    plan = ConnectionPlanner(connected, chosen[0][0]).plan()
    assert plan.cost == 0.0 and len(plan.edges) == 0 and len(plan.network) == 1
    assert plan.terminals["status"].tolist() == ["connected"]


def test_repeated_terminals_count_once(chain):
    plan = ConnectionPlanner(chain, ["B", "A", "B"], roots=["R1"]).plan()
    assert list(plan.terminals.index) == ["B", "A"] and plan.cost == 11.0


def test_directions_are_ignored():
    edges = pd.DataFrame({"u": ["A", "B"], "v": ["R", "A"], "length": [4.0, 3.0]})
    network = SpatialNetwork(nodes_on_a_line(["R", "A", "B"]), edges, directed=True)
    plan = ConnectionPlanner(network, ["B"], roots=["R"]).plan()
    assert plan.cost == 7.0 and not plan.network.directed


def test_zero_weights_are_allowed():
    edges = pd.DataFrame({"u": ["R", "A"], "v": ["A", "B"], "length": [0.0, 2.0]})
    network = SpatialNetwork(nodes_on_a_line(["R", "A", "B"]), edges, directed=False)
    plan = ConnectionPlanner(network, ["A", "B"], roots=["R"]).plan()
    assert plan.cost == 2.0 and len(plan.edges) == 2
    assert plan.terminals["distance"].tolist() == [0.0, 2.0]


# ----- weights, edges, and errors


def test_a_cost_weight_changes_the_plan(chain):
    plan = ConnectionPlanner(chain, ["A", "B"], roots=["R1", "R2"], weight="cost").plan()
    assert plan.cost == 20.5 and plan.metadata["weight"] == "cost"
    assert plan.terminals["root"].tolist() == ["R1", "R2"]
    assert set(plan.network.edges.columns) >= {"length", "cost"}


def test_edges_are_the_uncontracted_plan(connected, chosen):
    terminals, roots = chosen
    plan = ConnectionPlanner(connected, terminals, roots=roots).plan()
    assert plan.edges.crs == "EPSG:4326" and plan.edges.geometry.notna().all()
    assert len(plan.edges) > len(plan.network.edges)
    assert plan.edges["length"].sum() == pytest.approx(plan.network.total_length())
    assert plan.cost == pytest.approx(plan.edges["length"].sum())
    assert plan.metadata["method"] == "steiner" and set(plan.metadata["seconds"]) == {"plan", "assemble"}
    assert isinstance(plan, ConnectionPlan) and "terminals=30 of 30 connected" in repr(plan)


def test_connecting_subnetwork_is_the_plans_network(connected, chosen):
    terminals, roots = chosen
    for method in ("steiner", "shortest_path", "nearest_root", "all_pairs"):
        plan = ConnectionPlanner(connected, terminals, roots=roots).plan(method)
        sub = connected.connecting_subnetwork(terminals, roots=roots, method=method)
        assert set(sub.node_index) == set(plan.network.node_index)
        assert sub.total_length() == pytest.approx(plan.network.total_length())


def test_input_errors(chain):
    with pytest.raises(KeyError):
        ConnectionPlanner(chain, ["A", "nowhere"])
    with pytest.raises(KeyError):
        ConnectionPlanner(chain, ["A"], roots=["nowhere"])
    planner = ConnectionPlanner(chain, ["A", "B"])
    with pytest.raises(ValueError, match="method"):
        planner.plan("kou")
    with pytest.raises(ValueError, match="needs roots"):
        planner.plan("nearest_root")
    negative = SpatialNetwork(chain.nodes, chain.edges.assign(cost=[1.0, -1.0, 1.0]), directed=False)
    with pytest.raises(ValueError, match="negative"):
        ConnectionPlanner(negative, ["A"], weight="cost")


# ----- bridges


@pytest.fixture
def two_parts():
    """R -1- A and B -1- C -10- R2, with no road between A and B, which are about 1,112 m apart."""
    edges = pd.DataFrame({"u": ["R", "B", "C"], "v": ["A", "C", "R2"], "length": [1.0, 1.0, 10.0]})
    return SpatialNetwork(nodes_on_a_line(["R", "A", "B", "C", "R2"]), edges, directed=False)


def gap(network, a, b):
    lat, lon = network.node_coordinates([a, b]).T
    return haversine(lat[0], lon[0], lat[1], lon[1])


def test_a_bridge_connects_a_part_without_roots(two_parts):
    ab = gap(two_parts, "A", "B")
    without = ConnectionPlanner(two_parts, ["A", "B"], roots=["R"]).plan()
    assert without.terminals.loc["B", "status"] == "unreachable" and without.metadata["bridges"]["offered"] == 0
    plan = ConnectionPlanner(two_parts, ["A", "B"], roots=["R"], bridge_cost=0.001).plan()
    assert plan.cost == pytest.approx(1 + 0.001 * ab)
    assert plan.terminals["bridged"].tolist() == [False, True]
    assert plan.terminals.loc["B", "distance"] == pytest.approx(1 + 0.001 * ab)
    bridge = plan.edges[plan.edges["kind"] == "bridge"]
    assert len(bridge) == 1 and bridge["length"].iloc[0] == pytest.approx(ab)
    assert {bridge["u"].iloc[0], bridge["v"].iloc[0]} == {"A", "B"}
    assert (plan.network.edges["kind"] == "bridge").sum() == 1
    assert plan.metadata["bridges"] == {"offered": 1, "used": 1} and plan.metadata["bridge_cost"] == 0.001


def test_a_bridge_is_used_only_when_cheaper(two_parts):
    ab = gap(two_parts, "A", "B")
    cheap = ConnectionPlanner(two_parts, ["A", "C"], roots=["R", "R2"], bridge_cost=0.001).plan()
    assert cheap.cost == pytest.approx(1 + 0.001 * ab + 1) and cheap.terminals.loc["C", "root"] == "R"
    costly = ConnectionPlanner(two_parts, ["A", "C"], roots=["R", "R2"], bridge_cost=0.01).plan()
    assert costly.cost == pytest.approx(11.0) and costly.terminals.loc["C", "root"] == "R2"
    assert costly.metadata["bridges"]["used"] == 0 and not costly.terminals["bridged"].any()


def test_bridges_respect_the_longest_link(two_parts):
    plan = ConnectionPlanner(two_parts, ["A", "B"], roots=["R"], bridge_cost=0.001, bridge_max_distance=1000).plan()
    assert plan.metadata["bridges"]["offered"] == 0 and plan.terminals.loc["B", "status"] == "unreachable"


def test_bridges_without_roots(two_parts):
    plan = ConnectionPlanner(two_parts, ["A", "C"], bridge_cost=0.001).plan()
    assert (plan.terminals["status"] == "connected").all()
    assert plan.terminals["bridged"].tolist() == [False, True]


def test_bridges_on_a_connected_network_change_nothing(chain):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        plan = ConnectionPlanner(chain, ["A", "B"], roots=["R1", "R2"], bridge_cost=0.001).plan()
    assert plan.cost == 11.0 and plan.metadata["bridges"] == {"offered": 0, "used": 0}


def test_bridges_in_a_cost_weight(two_parts):
    network = SpatialNetwork(two_parts.nodes, two_parts.edges.assign(cost=[2.0, 2.0, 20.0]), directed=False)
    plan = ConnectionPlanner(network, ["A", "B"], roots=["R"], weight="cost", bridge_cost=0.01).plan()
    bridge = plan.edges[plan.edges["kind"] == "bridge"].iloc[0]
    assert bridge["cost"] == pytest.approx(0.01 * bridge["length"])
    assert plan.cost == pytest.approx(plan.edges["cost"].sum())


def test_bridge_options_are_checked(chain):
    with pytest.raises(ValueError, match="bridge_cost"):
        ConnectionPlanner(chain, ["A"], bridge_cost=-1)
    with pytest.raises(ValueError, match="bridge_max_distance"):
        ConnectionPlanner(chain, ["A"], bridge_cost=1, bridge_max_distance=0)


def test_bridges_join_kigalis_islands(kigali, chosen):
    terminals, _ = chosen
    labels = kigali.connected_components()
    islands = list(labels[labels > 0].groupby(labels[labels > 0]).head(1).index)
    roots = list(labels[labels == 0].index[:3])
    plan = ConnectionPlanner(kigali, terminals + islands, roots=roots, bridge_cost=1.0).plan()
    assert (plan.terminals["status"] == "connected").all()
    assert set(plan.terminals.index[plan.terminals["bridged"]]) == set(islands)
    bridges = plan.edges[plan.edges["kind"] == "bridge"]
    assert line_lengths(bridges.geometry) == pytest.approx(bridges["length"].to_numpy())
    assert shapely.get_num_coordinates(bridges.geometry.to_numpy()).tolist() == [2] * len(bridges)


def test_bridge_candidates_are_the_closest_pairs(kigali):
    labels = kigali.connected_components().to_numpy()
    candidates = _bridge_candidates(kigali, labels, 5000.0)
    lat, lon = kigali.node_coordinates().T
    distance = haversine(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
    count = labels.max() + 1
    closest = np.full((count, count), np.inf)
    np.minimum.at(closest, (labels[:, None].repeat(len(labels), 1), labels[None, :].repeat(len(labels), 0)), distance)
    found = {}
    for a, b, length in zip(*candidates):
        pair = tuple(sorted((labels[a], labels[b])))
        assert pair[0] != pair[1] and pair not in found
        found[pair] = length
    for pair, length in found.items():
        assert length == pytest.approx(closest[pair])
    np.fill_diagonal(closest, np.inf)
    for part in range(count):
        nearest = int(np.argmin(closest[part]))
        assert tuple(sorted((part, nearest))) in found
    largest = np.argsort(closest[0])[:BRIDGES_PER_COMPONENT]
    assert {tuple(sorted((0, int(p)))) for p in largest} <= set(found)


# ----- the rollout order


def naive_rollout(network, terminals, roots):
    """Takahashi and Matsuyama with a full Dijkstra from the connected set at every step."""
    graph = network._graph(network.weight, False)
    position = {t: i for i, t in enumerate(network.node_index)}
    connected = np.zeros(graph.n, dtype=bool)
    waiting = list(terminals)
    order, step, cost = {}, 0, 0.0
    if roots:
        connected[[position[r] for r in roots]] = True
    else:
        connected[position[waiting[0]]] = True
        order[waiting.pop(0)] = 1
        step = 1
    while waiting:
        distance, predecessors = dijkstra(
            graph.matrix, directed=True, indices=np.flatnonzero(connected), min_only=True, return_predecessors=True
        )[:2]
        best = min(waiting, key=lambda t: (distance[position[t]], waiting.index(t)))
        if not np.isfinite(distance[position[best]]):
            break
        cost += distance[position[best]]
        node = position[best]
        while not connected[node]:
            connected[node] = True
            node = predecessors[node]
        step += 1
        order[best] = step
        waiting = [t for t in waiting if not connected[position[t]]]
    return order, cost


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("rooted", [True, False])
def test_the_rollout_matches_the_naive_version(seed, rooted):
    rng = np.random.default_rng(seed)
    G = nx.grid_2d_graph(6, 6)
    ids = {node: f"n{i}" for i, node in enumerate(G)}
    nodes = pd.DataFrame(
        {"x": [30.0 + 0.01 * c for _, c in G], "y": [-1.95 + 0.01 * r for r, _ in G]}, index=list(ids.values())
    )
    edges = pd.DataFrame(
        {
            "u": [ids[a] for a, _ in G.edges],
            "v": [ids[b] for _, b in G.edges],
            "length": rng.uniform(1, 10, G.number_of_edges()),
        }
    )
    network = SpatialNetwork(nodes, edges, directed=False)
    picked = list(rng.choice(list(ids.values()), 12, replace=False))
    terminals, roots = picked[:9], (picked[9:] if rooted else [])
    plan = ConnectionPlanner(network, terminals, roots=roots).plan("shortest_path")
    order, cost = naive_rollout(network, terminals, roots)
    assert plan.cost == pytest.approx(cost)
    assert plan.terminals["order"].dropna().astype(int).to_dict() == order
    assert plan.terminals["marginal_cost"].sum() == pytest.approx(plan.cost)


def test_the_rollout_on_a_chain(chain):
    plan = ConnectionPlanner(chain, ["B", "A", "R2"], roots=["R1", "R2"]).plan("shortest_path")
    assert plan.cost == 11.0
    assert plan.terminals[["order", "marginal_cost"]].to_dict("index") == {
        "B": {"order": 2, "marginal_cost": 1.0},
        "A": {"order": 1, "marginal_cost": 10.0},
        "R2": {"order": 0, "marginal_cost": 0.0},
    }


def test_ties_go_to_the_terminal_listed_first():
    edges = pd.DataFrame({"u": ["R", "A"], "v": ["A", "B"], "length": [1.0, 0.0]})
    network = SpatialNetwork(nodes_on_a_line(["R", "A", "B"]), edges, directed=False)
    plan = ConnectionPlanner(network, ["B", "A"], roots=["R"]).plan("shortest_path")
    assert plan.terminals["order"].tolist() == [2, 1]
    assert plan.terminals["marginal_cost"].tolist() == [0.0, 1.0]
    flipped = ConnectionPlanner(network, ["A", "B"], roots=["R"]).plan("shortest_path")
    assert flipped.terminals["order"].tolist() == [1, 2]


def test_the_rollout_without_roots_starts_from_the_first_terminal(chain):
    plan = ConnectionPlanner(chain, ["B", "R1", "A"]).plan("shortest_path")
    assert plan.terminals["order"].tolist() == [1, 3, 2]
    assert plan.terminals["marginal_cost"].tolist() == [0.0, 10.0, 1.0] and plan.cost == 11.0


def test_unreachable_terminals_have_no_order(kigali, chosen):
    terminals, roots = chosen
    island = kigali.connected_components().idxmax()
    plan = ConnectionPlanner(kigali, terminals[:5] + [island], roots=roots).plan("shortest_path")
    assert plan.terminals.loc[island, "order"] is pd.NA and np.isnan(plan.terminals.loc[island, "marginal_cost"])
    assert sorted(plan.terminals["order"].dropna().astype(int)) == [1, 2, 3, 4, 5]


def test_the_rollout_uses_bridges(two_parts):
    plan = ConnectionPlanner(two_parts, ["A", "B"], roots=["R"], bridge_cost=0.001).plan("shortest_path")
    assert plan.terminals["order"].tolist() == [1, 2] and plan.terminals["bridged"].tolist() == [False, True]
    assert plan.metadata["bridges"]["used"] == 1


def test_the_rollout_on_kigali(connected, chosen):
    terminals, roots = chosen
    planner = ConnectionPlanner(connected, terminals, roots=roots)
    plan = planner.plan("shortest_path")
    order, cost = naive_rollout(connected.to_undirected(), terminals, roots)
    assert plan.cost == pytest.approx(cost)
    assert nx.is_forest(plan_trees(plan))
    assert plan.terminals["marginal_cost"].sum() == pytest.approx(plan.cost)
    assert "order" not in planner.plan().terminals.columns


# ----- local search


def tree_weight(tree):
    a, b = tree.edges()
    return float(_pair_weights(tree.graph, a, b).sum())


def star():
    """A hub H 10 from each of A, B, and C, which roads of 4 also join: A-B and B-C."""
    edges = pd.DataFrame(
        {"u": ["H", "H", "H", "A", "B"], "v": ["A", "B", "C", "B", "C"], "length": [10.0, 10.0, 10.0, 4.0, 4.0]}
    )
    return SpatialNetwork(nodes_on_a_line(["H", "A", "B", "C"]), edges, directed=False)


def star_tree():
    network = star()
    graph = network._graph("length", False)
    h, a, b, c = network._positions(["H", "A", "B", "C"])
    return _Tree(graph, np.array([h, h, h]), np.array([a, b, c]), np.array([a, b, c])), (h, a, b, c)


def test_key_vertex_elimination_replaces_a_hub():
    tree, (h, a, b, c) = star_tree()
    assert tree_weight(tree) == 30.0
    assert tree.eliminate(h)
    assert tree_weight(tree) == 8.0 and h not in tree.adjacent


def test_key_path_exchange_replaces_a_spoke():
    tree, (h, a, b, c) = star_tree()
    assert tree.exchange([a, h])
    assert tree_weight(tree) == 24.0


def test_moves_that_do_not_save_are_refused(chain):
    graph = chain._graph("length", False)
    r1, a, b, r2 = chain._positions(["R1", "A", "B", "R2"])
    tree = _Tree(graph, np.array([r1, a]), np.array([a, b]), np.array([r1, b]))
    assert not tree.exchange([r1, a, b])
    assert tree_weight(tree) == 11.0


def test_improve_fixes_the_nearest_root_plan(chain):
    planner = ConnectionPlanner(chain, ["A", "B"], roots=["R1", "R2"])
    plan = planner.plan("nearest_root")
    better = planner.improve(plan)
    assert (plan.cost, better.cost) == (20.5, 11.0)
    assert better.metadata["method"] == "nearest_root"
    assert better.metadata["improved"]["cost_before"] == 20.5 and better.metadata["improved"]["moves"] >= 1
    assert better.terminals["root"].tolist() == ["R1", "R1"]


@pytest.mark.parametrize("seed", range(8))
def test_improve_is_never_worse_and_never_below_the_optimum(seed):
    network, terminals, roots = random_grid(seed)
    for some in (roots, []):
        best = optimum(network, terminals, some)
        planner = ConnectionPlanner(network, terminals, roots=some)
        for method in ("steiner", "shortest_path", "all_pairs") + (("nearest_root",) if some else ()):
            plan = planner.plan(method)
            better = planner.improve(plan)
            assert best - 1e-9 <= better.cost <= plan.cost + 1e-9


def test_improve_on_kigali(connected, chosen):
    terminals, roots = chosen
    planner = ConnectionPlanner(connected, terminals, roots=roots)
    plan = planner.plan()
    better = planner.improve(plan)
    assert better.cost < plan.cost
    forest = plan_trees(better)
    assert nx.is_forest(forest)
    for tree in nx.connected_components(forest):
        assert tree & set(roots)
    degree = dict(forest.degree())
    assert {n for n, d in degree.items() if d == 1} <= set(terminals) | set(roots)
    assert (better.terminals["status"] == "connected").all()
    assert better.terminals["distance"].to_numpy() == pytest.approx(
        better.network.nearest_by_network(terminals, roots)["distance"].to_numpy()
    )
    assert len(better.metadata["improved"]["pass_seconds"]) >= 1


def test_improve_without_roots(connected, chosen):
    terminals, _ = chosen
    planner = ConnectionPlanner(connected, terminals)
    plan = planner.plan()
    better = planner.improve(plan)
    assert better.cost <= plan.cost and nx.is_tree(plan_trees(better))


def test_improve_keeps_bridges_and_unreachable_terminals(two_parts):
    planner = ConnectionPlanner(two_parts, ["A", "B"], roots=["R"], bridge_cost=0.001)
    better = planner.improve(planner.plan("all_pairs"))
    assert better.terminals["bridged"].tolist() == [False, True] and better.metadata["bridges"]["used"] == 1
    alone = ConnectionPlanner(two_parts, ["A", "B"], roots=["R"])
    kept = alone.improve(alone.plan())
    assert kept.terminals["status"].tolist() == ["connected", "unreachable"] and kept.cost == 1.0


def test_improve_drops_the_rollout_columns(connected, chosen):
    terminals, roots = chosen
    planner = ConnectionPlanner(connected, terminals, roots=roots)
    better = planner.improve(planner.plan("shortest_path"))
    assert "order" not in better.terminals.columns and better.metadata["method"] == "shortest_path"


def test_improve_stops_at_the_time_limit(connected, chosen):
    terminals, roots = chosen
    planner = ConnectionPlanner(connected, terminals, roots=roots)
    plan = planner.plan("all_pairs")
    stopped = planner.improve(plan, time_limit=0)
    assert stopped.metadata["improved"]["moves"] == 0 and stopped.cost <= plan.cost


def test_improve_needs_the_planners_own_plan(chain):
    plan = ConnectionPlanner(chain, ["A"], roots=["R1"]).plan()
    with pytest.raises(ValueError, match="other terminals"):
        ConnectionPlanner(chain, ["B"], roots=["R1"]).improve(plan)


# ----- lower bound


def test_the_bound_on_a_chain_is_exact(chain):
    planner = ConnectionPlanner(chain, ["A", "B"], roots=["R1", "R2"])
    plan = planner.plan()
    assert planner.lower_bound(plan) == 11.0
    assert (plan.bound, plan.gap) == (11.0, 0.0) and "gap=0.0%" in repr(plan)
    assert plan.metadata["bound"]["complete"]
    assert ConnectionPlanner(chain, ["A", "B", "R1", "R2"]).lower_bound() == 21.5


@pytest.mark.parametrize("seed", range(12))
def test_the_bound_never_exceeds_the_optimum(seed):
    network, terminals, roots = random_grid(seed)
    for some in (roots, []):
        best = optimum(network, terminals, some)
        planner = ConnectionPlanner(network, terminals, roots=some)
        bound = planner.lower_bound()
        assert 0 <= bound <= best + 1e-9
        plan = planner.improve(planner.plan())
        planner.lower_bound(plan)
        assert plan.gap >= -1e-12


def test_the_bound_is_exact_on_a_tree():
    rng = np.random.default_rng(3)
    G = nx.random_labeled_tree(40, seed=3)
    nodes = pd.DataFrame(
        {"x": 30.0 + rng.uniform(0, 0.1, 40), "y": -1.95 + rng.uniform(0, 0.1, 40)}, index=[f"n{i}" for i in G]
    )
    edges = pd.DataFrame(
        {"u": [f"n{a}" for a, _ in G.edges], "v": [f"n{b}" for _, b in G.edges], "length": rng.uniform(1, 10, 39)}
    )
    network = SpatialNetwork(nodes, edges, directed=False)
    planner = ConnectionPlanner(network, [f"n{i}" for i in range(0, 40, 4)], roots=["n1", "n2"])
    assert planner.lower_bound() == pytest.approx(planner.plan().cost)


def test_the_bound_on_kigali(connected, chosen):
    terminals, roots = chosen
    planner = ConnectionPlanner(connected, terminals, roots=roots)
    plan = planner.improve(planner.plan())
    bound = planner.lower_bound(plan)
    assert 0 < bound <= plan.cost and 0 <= plan.gap < 0.15
    assert planner.lower_bound() == bound


def test_unreachable_terminals_are_left_out_of_the_bound(kigali, chosen):
    terminals, roots = chosen
    island = kigali.connected_components().idxmax()
    with_island = ConnectionPlanner(kigali, terminals[:5] + [island], roots=roots).lower_bound()
    assert with_island == pytest.approx(ConnectionPlanner(kigali, terminals[:5], roots=roots).lower_bound())


def test_the_bound_counts_bridges(two_parts):
    planner = ConnectionPlanner(two_parts, ["A", "B"], roots=["R"], bridge_cost=0.001)
    plan = planner.plan()
    assert planner.lower_bound(plan) == pytest.approx(plan.cost)


def test_a_bound_stopped_by_the_time_limit_is_valid_and_recomputed(connected, chosen):
    terminals, roots = chosen
    planner = ConnectionPlanner(connected, terminals, roots=roots)
    plan = planner.plan()
    early = planner.lower_bound(plan, time_limit=0)
    assert early == 0.0 and not plan.metadata["bound"]["complete"] and plan.gap == np.inf
    full = planner.lower_bound(plan)
    assert full > 0 and plan.metadata["bound"]["complete"]


def test_the_bound_without_terminals(chain):
    island = SpatialNetwork(chain.nodes, chain.edges.iloc[:1], directed=False)
    assert ConnectionPlanner(island, ["R2"], roots=["R1"]).lower_bound() == 0.0
