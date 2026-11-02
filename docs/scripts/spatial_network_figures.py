"""
Draw the figures of the spatial networks concept note into ``docs/assets/spatial-network/``.

    python docs/scripts/spatial_network_figures.py [--workdir DIR]

The figures use the objects the note's own code makes, by running it first
(see ``run_concept_examples.py``), so they always show what the pages
compute. Static maps are PNGs on a light basemap, which contextily
downloads (Esri's World Gray Canvas, which needs no API key); two of them
also get an interactive HTML version made with folium.
Needs matplotlib, contextily, and folium, and the internet for the tiles.
"""

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shapely  # noqa: E402
from shapely.geometry import box  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "assets" / "spatial-network"
ROAD = "#9a9a9a"
POINT_COLOURS = {"school": "#1f77b4", "fibre": "#d62728", "clinic": "#2ca02c"}
WEB_MERCATOR = "EPSG:3857"


def tiles():
    """A light grey basemap that needs no API key (CARTO's now does)."""
    import xyzservices.providers as providers

    return providers.Esri.WorldGrayCanvas


def basemap(ax):
    import contextily

    contextily.add_basemap(ax, crs=WEB_MERCATOR, source=tiles(), attribution_size=6, zorder=0)


def interactive_map():
    import folium

    provider = tiles()
    return folium.Map(location=[-1.955, 30.09], zoom_start=12, tiles=provider.build_url(), attr=provider.html_attribution)


def finish(ax, title):
    ax.set_title(title, fontsize=10)
    ax.set_axis_off()


def frame(ax, area):
    west, south, east, north = gpd.GeoSeries([area], crs="EPSG:4326").to_crs(WEB_MERCATOR).total_bounds
    ax.set_xlim(west, east)
    ax.set_ylim(south, north)


def lines(network):
    return network.edges_gdf.to_crs(WEB_MERCATOR)


def points(network, ids):
    return network.nodes_gdf.loc[list(ids)].to_crs(WEB_MERCATOR)


def insertion_figure(ns):
    """Three schools beside one road, before and after insertion."""
    roads = ns["roads"]
    local = roads.clip(box(30.0560, -1.9570, 30.0680, -1.9470))
    edges = local.edges_gdf.to_crs(local.edges_gdf.estimate_utm_crs())
    inside = edges[edges.within(gpd.GeoSeries([box(30.0585, -1.9545, 30.0655, -1.9495)], crs="EPSG:4326").to_crs(edges.crs).iloc[0])]
    longest = inside.loc[inside.length.idxmax()]
    utm = edges.crs
    line = longest.geometry
    offsets = [(0.25, 45), (0.5, -35), (0.75, 55)]
    made_up = []
    for i, (fraction, side) in enumerate(offsets):
        foot = line.interpolate(fraction, normalized=True)
        ahead = line.interpolate(min(fraction + 0.01, 1), normalized=True)
        dx, dy = ahead.x - foot.x, ahead.y - foot.y
        norm = np.hypot(dx, dy)
        made_up.append(shapely.Point(foot.x - dy / norm * side, foot.y + dx / norm * side))
    pts = gpd.GeoDataFrame({"id": [f"school-{c}" for c in "abc"]}, geometry=made_up, crs=utm).to_crs("EPSG:4326")
    inserted = local.insert_points(pts, id_column="id")
    shown = gpd.GeoSeries([line, *made_up], crs=utm).union_all().buffer(80)
    window = gpd.GeoSeries([shown.envelope], crs=utm).to_crs("EPSG:4326").iloc[0]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, net, title in ((axes[0], local, "Before: schools beside a road"), (axes[1], inserted, "After: the road split at their feet")):
        e = lines(net)
        e[e["kind"] == "road"].plot(ax=ax, color=ROAD, linewidth=1.5, zorder=2)
        if (e["kind"] == "split").any():
            split = e[e["kind"] == "split"]
            split.plot(ax=ax, color=["#ff7f0e", "#9467bd"] * (len(split) // 2 + 1), linewidth=3, zorder=3)
            e[e["kind"] == "connection"].plot(ax=ax, color="#1f77b4", linewidth=1.5, linestyle="--", zorder=3)
            splits = net.nodes_gdf[net.nodes["kind"] == "split"].to_crs(WEB_MERCATOR)
            splits.plot(ax=ax, color="black", markersize=25, marker="x", zorder=4)
        pts.to_crs(WEB_MERCATOR).plot(ax=ax, color=POINT_COLOURS["school"], markersize=60, zorder=5)
        frame(ax, window)
        basemap(ax)
        finish(ax, title)
    fig.text(0.5, 0.06, "Orange and purple: the split pieces. Crosses: split nodes. Dashed: connections.", ha="center", fontsize=8)
    fig.savefig(OUT / "insertion.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def simplify_figure(ns):
    raw, simple = ns["raw_kigali"], ns["simple_kigali"]
    window = box(30.080, -1.960, 30.092, -1.950)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, net, title in ((axes[0], raw, "Raw: a node at every bend"), (axes[1], simple, "Simplified: nodes only at junctions and ends")):
        part = net.clip(window)
        lines(part).plot(ax=ax, color="#555555", linewidth=1, zorder=2)
        part.nodes_gdf.to_crs(WEB_MERCATOR).plot(ax=ax, color="#d62728", markersize=4, zorder=3)
        frame(ax, window)
        basemap(ax)
        finish(ax, f"{title} ({len(part.nodes):,} nodes)")
    fig.savefig(OUT / "simplify.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def subnetwork_results(ns):
    network, school_ids, fibre_ids = ns["network"], ns["school_ids"], ns["fibre_ids"]
    return {
        "Steiner tree joining the schools": network.connecting_subnetwork(school_ids),
        "Each school to its nearest fibre node": network.connecting_subnetwork(school_ids, sources=fibre_ids),
        "All pairs, 30 schools and the fibre nodes": network.connecting_subnetwork(
            school_ids[:30], sources=fibre_ids, method="all_pairs"
        ),
    }


def connecting_figure(ns, results):
    kigali = ns["kigali"]
    background = lines(ns["kigali_roads"].filter_edges(mask=ns["kigali_roads"].edges["highway"].astype(str).str.contains("trunk|primary|secondary|tertiary").to_numpy()))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    for ax, (title, result) in zip(axes, results.items()):
        background.plot(ax=ax, color="#cfcfcf", linewidth=0.5, zorder=1)
        lines(result).plot(ax=ax, color="#ff7f0e", linewidth=1.6, zorder=2)
        schools = [s for s in ns["school_ids"] if s in set(result.node_index)]
        points(result, schools).plot(ax=ax, color=POINT_COLOURS["school"], markersize=8, zorder=3)
        fibres = [f for f in ns["fibre_ids"] if f in set(result.node_index)]
        if fibres:
            points(result, fibres).plot(ax=ax, color=POINT_COLOURS["fibre"], marker="*", markersize=120, zorder=4)
        frame(ax, kigali.intersection(box(29.99, -2.03, 30.20, -1.89)))
        basemap(ax)
        finish(ax, f"{title}\n{result.total_length() / 1000:.0f} km")
    fig.savefig(OUT / "connecting-subnetworks.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    import folium

    m = interactive_map()
    colours = ["#ff7f0e", "#9467bd", "#8c564b"]
    for colour, (title, result) in zip(colours, results.items()):
        layer = folium.FeatureGroup(name=f"{title} ({result.total_length() / 1000:.0f} km)", show=title.startswith("Steiner"))
        e = result.edges_gdf[["length", "geometry"]].copy()
        e["geometry"] = shapely.set_precision(e.geometry.simplify(0.00002), 0.000001)
        e["length"] = e["length"].round(0)
        folium.GeoJson(e, style_function=lambda _, c=colour: {"color": c, "weight": 3}, tooltip=folium.GeoJsonTooltip(["length"])).add_to(layer)
        layer.add_to(m)
    add_points(m, ns)
    folium.LayerControl(collapsed=False).add_to(m)
    m.save(OUT / "connecting-subnetworks.html")


def add_points(m, ns):
    import folium

    network = ns["network"]
    for kind, ids in (("school", ns["school_ids"]), ("fibre", ns["fibre_ids"])):
        layer = folium.FeatureGroup(name=f"{kind} points")
        for node, row in network.nodes.loc[ids, ["x", "y"]].iterrows():
            folium.CircleMarker(
                [row["y"], row["x"]], radius=4 if kind == "school" else 8, color=POINT_COLOURS[kind],
                fill=True, fill_opacity=0.9, tooltip=str(node),
            ).add_to(layer)
        layer.add_to(m)


def service_area_figure(ns):
    network, clinic = ns["network"], ns["clinic_ids"][0]
    reach = network.reachable([clinic], 5000)
    distance = reach.set_index("node_id")["distance"]
    e = lines(network.clip(box(29.98, -2.05, 30.25, -1.85)))
    inside = e["u"].isin(distance.index) & e["v"].isin(distance.index)
    reached = e[inside].copy()
    reached["distance"] = (distance.reindex(reached["u"]).to_numpy() + distance.reindex(reached["v"]).to_numpy()) / 2
    fig, ax = plt.subplots(figsize=(7, 7))
    e[~inside].plot(ax=ax, color="#d0d0d0", linewidth=0.4, zorder=1)
    reached.plot(ax=ax, column="distance", cmap="viridis_r", linewidth=1.2, legend=True,
                 legend_kwds={"label": "metres by road from the clinic", "shrink": 0.6}, zorder=2)
    points(network, [clinic]).plot(ax=ax, color=POINT_COLOURS["clinic"], marker="P", markersize=200, edgecolor="white", zorder=3)
    west, south, east, north = reached.total_bounds
    pad = 800
    ax.set_xlim(west - pad, east + pad)
    ax.set_ylim(south - pad, north + pad)
    basemap(ax)
    finish(ax, f"Roads within 5 km of {clinic}")
    fig.savefig(OUT / "service-area.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fibre_plan_figure(ns):
    plan = ns["plans"]["cheapest routes to a fibre node"]
    e = lines(plan)
    background = lines(ns["kigali_roads"].filter_edges(mask=ns["kigali_roads"].edges["highway"].astype(str).str.contains("trunk|primary|secondary|tertiary").to_numpy()))
    fig, ax = plt.subplots(figsize=(8, 7))
    background.plot(ax=ax, color="#cfcfcf", linewidth=0.5, zorder=1)
    e.plot(ax=ax, column="rate", cmap="plasma", linewidth=2, legend=True,
           legend_kwds={"label": "illustrative cost per metre", "shrink": 0.6}, zorder=2)
    schools = points(plan, [s for s in ns["school_ids"] if s in set(plan.node_index)])
    schools.plot(ax=ax, color=POINT_COLOURS["school"], markersize=8, zorder=3)
    points(plan, ns["fibre_ids"]).plot(ax=ax, color=POINT_COLOURS["fibre"], marker="*", markersize=160, zorder=4)
    frame(ax, box(29.99, -2.03, 30.20, -1.89))
    basemap(ax)
    cost = (plan.edges["rate"] * plan.edges["length"]).sum()
    finish(ax, f"Cheapest routes from {len(schools)} schools to 5 fibre nodes: {plan.total_length() / 1000:.0f} km, {cost / 1e6:.1f} million (illustrative)")
    fig.savefig(OUT / "fibre-plan.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    import branca.colormap as cm
    import folium

    m = interactive_map()
    colours = cm.linear.plasma.scale(float(plan.edges["rate"].min()), float(plan.edges["rate"].max()))
    colours.caption = "illustrative cost per metre"
    web = plan.edges_gdf[["rate", "length", "geometry"]].copy()
    web["geometry"] = shapely.set_precision(web.geometry.simplify(0.00002), 0.000001)
    web[["rate", "length"]] = web[["rate", "length"]].round(1)
    folium.GeoJson(
        web,
        style_function=lambda feature: {"color": colours(feature["properties"]["rate"]), "weight": 4},
        tooltip=folium.GeoJsonTooltip(["rate", "length"]),
        name="cheapest plan",
    ).add_to(m)
    colours.add_to(m)
    add_points(m, ns)
    folium.LayerControl(collapsed=False).add_to(m)
    m.save(OUT / "fibre-plan.html")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", default="/tmp/gs_concepts")
    args = parser.parse_args()
    sys.path.insert(0, str(HERE))
    from run_concept_examples import example_namespace

    ns = example_namespace(args.workdir)
    OUT.mkdir(parents=True, exist_ok=True)
    insertion_figure(ns)
    simplify_figure(ns)
    connecting_figure(ns, subnetwork_results(ns))
    service_area_figure(ns)
    fibre_plan_figure(ns)
    for path in sorted(OUT.iterdir()):
        print(f"{path.name}: {path.stat().st_size / 1e3:,.0f} kB")


if __name__ == "__main__":
    main()
