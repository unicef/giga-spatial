import time, os, pickle, re, sys
import osmium

PBF = "/tmp/gs_bench/rwanda.osm.pbf"
XML = "/tmp/gs_bench/rwanda_drive.osm"
REPORT = "/tmp/gs_bench/out/report.md"

EXCL_HW = set("abandoned|bridleway|bus_guideway|construction|corridor|cycleway|elevator|escalator|footway|no|path|pedestrian|planned|platform|proposed|raceway|razed|steps|track".split("|"))
EXCL_SERVICE = set("emergency_access|parking|parking_aisle|private".split("|"))


def keep(tags):
    hw = tags.get("highway")
    if hw is None or hw in EXCL_HW:
        return False
    if tags.get("area") == "yes":  # osmnx drive_service also excludes area=yes
        return False
    if tags.get("motor_vehicle") == "no" or tags.get("motorcar") == "no":
        return False
    if tags.get("access") == "private":
        return False
    if tags.get("service") in EXCL_SERVICE:
        return False
    return True


def log(s):
    print(s, flush=True)
    with open(REPORT, "a") as f:
        f.write(s + "\n")


if __name__ == "__main__":
    t0 = time.perf_counter()
    refs = set()
    nways = 0
    for w in osmium.FileProcessor(PBF, osmium.osm.WAY).with_filter(osmium.filter.KeyFilter("highway")):
        if keep(w.tags):
            nways += 1
            refs.update(n.ref for n in w.nodes)
    t1 = time.perf_counter()
    if os.path.exists(XML):
        os.remove(XML)
    wr = osmium.SimpleWriter(XML)
    nn = 0
    for o in osmium.FileProcessor(PBF, osmium.osm.NODE | osmium.osm.WAY):
        if o.is_node():
            if o.id in refs:
                wr.add_node(o)
                nn += 1
        elif o.is_way():
            if "highway" in o.tags and keep(o.tags):
                wr.add_way(o)
    wr.close()
    t2 = time.perf_counter()
    log("\n## Step 0 — data\n")
    log(f"- PBF size: {os.path.getsize(PBF)/1e6:.1f} MB; filtered XML: {os.path.getsize(XML)/1e6:.1f} MB")
    log(f"- pyosmium pass 1 (collect refs of {nways:,} drive_service ways, {len(refs):,} node refs): {t1-t0:.2f} s")
    log(f"- pyosmium pass 2 (write {nn:,} nodes + ways to OSM XML): {t2-t1:.2f} s; total filter {t2-t0:.2f} s")

    import osmnx as ox, networkx as nx
    log(f"- osmnx {ox.__version__}, networkx {nx.__version__}")
    t = time.perf_counter()
    Graw = ox.graph_from_xml(XML, simplify=False, retain_all=False)
    traw = time.perf_counter() - t
    log(f"- ox.graph_from_xml(simplify=False): {traw:.2f} s -> G_large_raw {Graw.number_of_nodes():,} nodes, {Graw.number_of_edges():,} edges")
    with open("/tmp/gs_bench/G_large_raw.pkl", "wb") as f:
        pickle.dump(Graw, f, protocol=5)
    del Graw
    t = time.perf_counter()
    G = ox.graph_from_xml(XML, simplify=True, retain_all=False)
    ts = time.perf_counter() - t
    log(f"- ox.graph_from_xml(simplify=True): {ts:.2f} s -> G_large {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges; crs={G.graph.get('crs')}")
    with open("/tmp/gs_bench/G_large.pkl", "wb") as f:
        pickle.dump(G, f, protocol=5)
    # Kigali box
    W, S, E, N = 29.98, -2.03, 30.20, -1.89
    t = time.perf_counter()
    Gs = ox.truncate.truncate_graph_bbox(G, (W, S, E, N), truncate_by_edge=False)
    tt = time.perf_counter() - t
    log(f"- G_small = truncate_graph_bbox(Kigali {W},{S},{E},{N}): {tt:.2f} s -> {Gs.number_of_nodes():,} nodes, {Gs.number_of_edges():,} edges")
    with open("/tmp/gs_bench/G_small.pkl", "wb") as f:
        pickle.dump(Gs, f, protocol=5)
    e0 = next(iter(G.edges(keys=True, data=True)))
    log(f"- sample edge attrs: {sorted(e0[3].keys())}")
    log(f"- sample node attrs: {sorted(next(iter(G.nodes(data=True)))[1].keys())}")
    log(f"- **PBF route total (filter + graph_from_xml simplify=True): {t2-t0+ts:.1f} s**")
