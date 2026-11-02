from itertools import combinations

import networkx as nx
import numpy as np
import pandas as pd
import pytest
from networkx.algorithms.approximation import steiner_tree

from gigaspatial.processing.network import ConnectionPlan, ConnectionPlanner, SpatialNetwork


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
        "A": {"status": "connected", "root": "R1", "distance": 10.0},
        "B": {"status": "connected", "root": "R1", "distance": 11.0},
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
    for method in ("nearest_root", "all_pairs"):
        assert planner.plan(method).cost >= rooted - 1e-9
    unrooted = optimum(network, terminals, [])
    assert unrooted - 1e-9 <= ConnectionPlanner(network, terminals).plan().cost <= 2 * unrooted + 1e-9


# ----- terminals and edge cases


def test_a_terminal_that_is_a_root(chain):
    plan = ConnectionPlanner(chain, ["R1", "B"], roots=["R1", "R2"]).plan()
    assert plan.terminals.loc["R1"].to_dict() == {"status": "root", "root": "R1", "distance": 0.0}
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
    for method in ("steiner", "nearest_root", "all_pairs"):
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
