"""Sub-task 6: the planner on SteinLib's test sets B, C, D, and E, against their proven optima.

Usage: steinlib.py [set ...]
Downloads the sets and their optima from steinlib.zib.de into $GS_BENCH_DIR/steinlib (/tmp/gs_bench by default)
the first time. Writes every instance's result to steinlib.md beside this script.

The instances are random sparse graphs with integer weights, without coordinates or roots, so they test the
unrooted Steiner tree. Every optimum in these sets is proven.
"""
import os, re, sys, tarfile, time, urllib.request, warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "spatial_network"))
from common import ROOT, log, table  # noqa

from gigaspatial.processing.network import ConnectionPlanner, SpatialNetwork

SITE = "https://steinlib.zib.de"
CACHE = os.path.join(ROOT, "steinlib")
HERE = os.path.dirname(os.path.abspath(__file__))


def fetch(name):
    path = os.path.join(CACHE, name.replace("?", "_"))
    if not os.path.exists(path):
        os.makedirs(CACHE, exist_ok=True)
        url = f"{SITE}/showset.php?{name[:-5]}" if name.endswith(".html") else f"{SITE}/download/{name}"
        with urllib.request.urlopen(url, timeout=120) as response, open(path, "wb") as out:
            out.write(response.read())
    return path


def optima(testset):
    """Instance name to optimum, from the set's page."""
    text = open(fetch(f"{testset}.html"), encoding="latin-1").read()
    found = {}
    for row in re.findall(r"<tr[^>]*>(.*?)</tr>", text, flags=re.S | re.I):
        cells = [re.sub(r"<[^>]+>|&nbsp;", "", c).strip() for c in re.findall(r"<td[^>]*>(.*?)</td>", row, re.S)]
        if len(cells) == 6 and cells[5].isdigit():
            found[cells[0].lower()] = int(cells[5])
    return found


def instances(testset):
    """(name, network, terminals) for every instance of the set."""
    with tarfile.open(fetch(f"{testset}.tgz")) as archive:
        for member in sorted(archive.getmembers(), key=lambda m: m.name):
            if not member.name.endswith(".stp"):
                continue
            text = archive.extractfile(member).read().decode("latin-1")
            edges = re.findall(r"^E\s+(\d+)\s+(\d+)\s+(\d+)", text, flags=re.M)
            terminals = [int(t) for t in re.findall(r"^T\s+(\d+)", text, flags=re.M)]
            nodes = int(re.search(r"^Nodes\s+(\d+)", text, flags=re.M).group(1))
            u, v, w = (np.array(c, dtype=np.int64) for c in zip(*edges))
            table_ = pd.DataFrame({"x": np.zeros(nodes), "y": np.zeros(nodes)}, index=np.arange(1, nodes + 1))
            network = SpatialNetwork(table_, pd.DataFrame({"u": u, "v": v, "length": w.astype(float)}), directed=False)
            yield os.path.basename(member.name)[:-4].lower(), network, terminals


def main(sets):
    rows, summary = [], []
    for testset in sets:
        best = optima(testset)
        results = []
        for name, network, terminals in instances(testset):
            planner = ConnectionPlanner(network, terminals)
            row = {"set": testset, "instance": name, "nodes": len(network), "edges": len(network.edges),
                   "terminals": len(terminals), "optimum": best[name]}
            for method in ("steiner", "shortest_path"):
                started = time.perf_counter()
                plan = planner.plan(method)
                row[f"{method}"] = plan.cost
                row[f"{method}_s"] = time.perf_counter() - started
                started = time.perf_counter()
                row[f"{method}_improved"] = planner.improve(plan).cost
                row[f"{method}_improve_s"] = time.perf_counter() - started
            started = time.perf_counter()
            row["bound"] = planner.lower_bound(time_limit=None)
            row["bound_s"] = time.perf_counter() - started
            results.append(row)
        rows += results
        frame = pd.DataFrame(results)
        gap = lambda column: frame[column] / frame["optimum"] - 1  # noqa: E731
        summary.append(
            [
                testset,
                f"{len(frame)}",
                f"{frame['nodes'].min():,}-{frame['nodes'].max():,}",
                f"{100 * gap('steiner').mean():.2f}% ({(gap('steiner') < 1e-9).sum()})",
                f"{100 * gap('steiner_improved').mean():.2f}% ({(gap('steiner_improved') < 1e-9).sum()})",
                f"{100 * gap('shortest_path').mean():.2f}% ({(gap('shortest_path') < 1e-9).sum()})",
                f"{100 * gap('shortest_path_improved').mean():.2f}% ({(gap('shortest_path_improved') < 1e-9).sum()})",
                f"{100 * max(gap('steiner_improved').max(), gap('shortest_path_improved').max()):.2f}%",
                f"{100 * (1 - frame['bound'] / frame['optimum']).mean():.2f}% "
                f"({(frame['bound'] > frame['optimum'] - 1e-9).sum()})",
                f"{100 * (1 - frame['bound'] / frame['optimum']).max():.2f}%",
                f"{frame[['steiner_s', 'steiner_improve_s', 'bound_s']].sum(axis=1).max():.1f}",
            ]
        )
    log("\n### Sub-task 6: SteinLib test sets against their proven optima\n")
    table(
        ["set", "instances", "nodes", "steiner", "+ improve", "shortest_path", "+ improve", "worst improved",
         "bound below optimum", "worst bound", "slowest s"],
        summary,
    )
    log("Plans: mean percentage above the optimum, and (in brackets) the instances solved optimally. Bound: mean "
        "percentage below the optimum, and the instances where it equals it. Slowest: the longest steiner plan, "
        "improve, and bound of one instance.\n")
    write_instances(pd.DataFrame(rows), os.path.join(HERE, "steinlib.md"))


def write_instances(frame, path):
    """One row per instance: the optimum, each plan's percentage above it, the bound's below it, and seconds."""
    above = lambda column: (100 * (frame[column] / frame["optimum"] - 1)).map("{:.2f}".format)  # noqa: E731
    columns = {
        "set": frame["set"],
        "instance": frame["instance"],
        "nodes": frame["nodes"],
        "edges": frame["edges"],
        "terminals": frame["terminals"],
        "optimum": frame["optimum"],
        "steiner %": above("steiner"),
        "+ improve %": above("steiner_improved"),
        "shortest_path %": above("shortest_path"),
        "+ improve % ": above("shortest_path_improved"),
        "bound below %": (100 * (1 - frame["bound"] / frame["optimum"])).map("{:.2f}".format),
        "steiner + improve + bound s": frame[["steiner_s", "steiner_improve_s", "bound_s"]].sum(axis=1).map("{:.2f}".format),
    }
    table_ = pd.DataFrame(columns)
    lines = ["# SteinLib sets B, C, D, and E, instance by instance", "",
             "Written by `steinlib.py`. Plans: percentage above the proven optimum. Bound: percentage below it.", "",
             "| " + " | ".join(table_.columns) + " |", "|" + "|".join("---" for _ in table_.columns) + "|"]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in table_.itertuples(index=False)]
    with open(path, "w") as out:
        out.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main(sys.argv[1:] or ["B", "C", "D", "E"])
