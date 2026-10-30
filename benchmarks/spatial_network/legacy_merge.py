"""The original merge_points_to_graph from the design conversation.

Unchanged except for the progress bar, and ox.project_gdf, which osmnx 2 moved to ox.projection.project_gdf.

Kept only as the baseline for insertion.py; it is not part of the library.
"""
import uuid

import geopandas as gpd
import networkx as nx
import osmnx as ox
import pandas as pd
from shapely.geometry import LineString
from shapely.ops import snap, split


def pb(items):
    return items


def merge_points_to_graph(graph: nx.Graph, points_df: gpd.GeoDataFrame, progress_bar=False):
    graph = ox.project_graph(graph.to_undirected())
    nodes_df, lines_df = ox.graph_to_gdfs(graph)
    original_crs = points_df.crs
    points_df = ox.projection.project_gdf(points_df, to_crs=graph.graph['crs'])

    def _nearest_line(point, lines):
        return lines.loc[lines.geometry.distance(point).idxmin()]

    def _segments(curve):
        segs = [LineString([curve.coords[i], curve.coords[i + 1]]) for i in range(len(curve.coords) - 1)]
        return gpd.GeoSeries(segs)

    def _nearest_point(point, line):
        return line.interpolate(line.project(point))

    def add_edges_to_graph(graph, u, v, geom, length, name):
        edge_keys = []
        for _u, _v, _geom, _length, _name in zip(u, v, geom, length, name):
            edge_key = graph.add_edge(_u, _v, geometry=_geom, length=_length, name=_name)
            edge_keys.append(edge_key)

        edge_df = pd.DataFrame(
            dict(
                length=length,
                geometry=geom,
                u=u,
                v=v,
                key=edge_keys,
            )
        ).set_index(['u', 'v', 'key'])
        return edge_df

    points_df = points_df.sjoin_nearest(lines_df.reset_index()[['u', 'v', 'key', 'geometry']], distance_col='dist').sort_values('dist').drop_duplicates(subset='coordinate_id', keep='first').drop(columns='index_right')
    points_df = points_df.sjoin_nearest(nodes_df.reset_index()[['osmid', 'geometry']], distance_col='node_dist').drop_duplicates(subset='coordinate_id', keep='first').drop(columns='index_right').reset_index(drop=True)

    for point in pb(list(points_df.itertuples())):
        point_geom = point.geometry

        if point.Index > 0:
            nearest_previously_merged_point = points_df.loc[points_df.loc[range(point.Index)].geometry.distance(point_geom).idxmin()]
            if nearest_previously_merged_point.geometry.distance(point_geom) < point.dist:
                nearest_point = nearest_previously_merged_point.geometry
                point_merge_line_geom = LineString([point_geom, nearest_point])
                point_merge_line_length = point_merge_line_geom.length

                graph.add_node(point.coordinate_id, x=point.geometry.x, y=point.geometry.y, lon=point.lon, lat=point.lat, ref='gigaNode')

                _ = add_edges_to_graph(
                    graph=graph,
                    u=[point.coordinate_id],
                    v=[nearest_previously_merged_point.coordinate_id],
                    geom=[point_merge_line_geom],
                    length=[point_merge_line_length],
                    name=['point_merge_edge']
                )

                continue

        if point.node_dist <= point.dist:
            nearest_point = nodes_df.loc[point.osmid].geometry
            point_merge_line_geom = LineString([point_geom, nearest_point])
            point_merge_line_length = point_merge_line_geom.length

            graph.add_node(point.coordinate_id, x=point.geometry.x, y=point.geometry.y, lon=point.lon, lat=point.lat, ref='gigaNode')

            _ = add_edges_to_graph(
                graph=graph,
                u=[point.coordinate_id],
                v=[point.osmid],
                geom=[point_merge_line_geom],
                length=[point_merge_line_length],
                name=['point_merge_edge']
            )

            continue

        try:
            nearest_line = lines_df.loc[point.u, point.v, point.key]
        except:
            nearest_line = _nearest_line(point_geom, lines_df)
        nearest_line_u, nearest_line_v, nearest_line_key = nearest_line.name
        u_node, v_node = nodes_df.loc[[nearest_line_u, nearest_line_v]].to_records()

        if nearest_line.geometry.coords[0] == (u_node.x, u_node.y):
            line_source = u_node
            line_target = v_node
        else:
            line_source = v_node
            line_target = u_node

        if len(nearest_line.geometry.coords) > 2:
            segs = _segments(nearest_line.geometry)
            nearest_line_segment = _nearest_line(point_geom, segs)
        else:
            nearest_line_segment = nearest_line.geometry

        nearest_point = _nearest_point(point_geom, nearest_line_segment)

        point_merge_line_geom = LineString([point_geom, nearest_point])

        point_merge_line_length = point_merge_line_geom.length

        graph.add_node(point.coordinate_id, x=point.geometry.x, y=point.geometry.y, lon=point.lon, lat=point.lat, ref='gigaNode')

        if nearest_line.geometry.intersects(nearest_point):
            project_dist = nearest_line.geometry.project(nearest_point)

            if project_dist == 0:
                _ = add_edges_to_graph(
                    graph=graph,
                    u=[point.coordinate_id],
                    v=[line_source.osmid],
                    geom=[point_merge_line_geom],
                    length=[point_merge_line_length],
                    name=['point_merge_edge']
                )

                continue
            elif project_dist == nearest_line.geometry.length:
                _ = add_edges_to_graph(
                    graph=graph,
                    u=[point.coordinate_id],
                    v=[line_target.osmid],
                    geom=[point_merge_line_geom],
                    length=[point_merge_line_length],
                    name=['point_merge_edge']
                )

                continue
            else:
                snap_line = nearest_line.geometry
        else:
            snap_line = snap(nearest_line.geometry, nearest_point, 1e-5)

        new_node = f"{graph.number_of_nodes() + 1} - {str(uuid.uuid4()).split('-')[0]}"
        graph.add_node(new_node, x=nearest_point.x, y=nearest_point.y, lat=nearest_point.y, lon=nearest_point.x, ref='splitNode')

        line1, line2 = split(snap_line, nearest_point).geoms

        graph.remove_edge(nearest_line_u, nearest_line_v, nearest_line_key)
        lines_df.drop(index=nearest_line.name, inplace=True)

        new_edge_data = add_edges_to_graph(
            graph=graph,
            u=[line_source.osmid, new_node, point.coordinate_id],
            v=[new_node, line_target.osmid, new_node],
            geom=[line1, line2, point_merge_line_geom],
            length=[line1.length, line2.length, point_merge_line_length],
            name=['split_edge', 'split_edge', 'point_merge_edge']
        )

        new_node_data = pd.DataFrame(
            dict(
                x=[nearest_point.x],
                y=[nearest_point.y],
                geometry=[nearest_point],
                osmid=[new_node],
            ),
        ).set_index('osmid')

        lines_df = pd.concat([lines_df, new_edge_data])
        nodes_df = pd.concat([nodes_df, new_node_data])

    return ox.project_graph(graph, to_crs=original_crs).to_undirected()
