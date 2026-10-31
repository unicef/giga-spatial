import json
import shutil
import types
from pathlib import Path

import numpy as np
import pytest
import shapely
from shapely.geometry import box

import gigaspatial.handlers.osm.roads as roads
from gigaspatial.core.io.data_store import DataStore
from gigaspatial.core.io.local_data_store import LocalDataStore
from gigaspatial.handlers.osm.pbf import way_filter
from gigaspatial.handlers.osm.roads import OSMRoadsHandler, OSMRoadsReadOptions
from gigaspatial.processing.network import SpatialNetwork

osmium = pytest.importorskip("osmium")

DATA = Path(__file__).resolve().parents[2] / "data" / "osm_roads"
KIGALI = DATA / "kigali.osm.pbf"
DATE = "2026-10-26"
RWANDA = f"osm/geofabrik/{DATE}/africa/rwanda.osm.pbf"
AREA = box(30.055, -1.96, 30.075, -1.94)


def store_extract(root: Path, extract_path: str, source: Path = KIGALI) -> Path:
    target = root / "osm/geofabrik" / DATE / f"{extract_path}.osm.pbf"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(source, target)
    return target


@pytest.fixture
def handler(store, geofabrik, tmp_path):
    store_extract(tmp_path, "africa/rwanda")
    return OSMRoadsHandler(data_store=store, base_path=Path("osm/geofabrik"))


@pytest.fixture
def builds(monkeypatch):
    """Counts reads of an extract."""
    calls = []
    real = roads.build_road_network

    def counted(path, **kwargs):
        calls.append(Path(path).name)
        return real(path, **kwargs)

    monkeypatch.setattr(roads, "build_road_network", counted)
    return calls


@pytest.fixture
def no_boundaries(monkeypatch):
    from gigaspatial.handlers import boundaries

    def refuse(*args, **kwargs):
        raise AssertionError("a country with its own extract needs no boundary")

    monkeypatch.setattr(boundaries.AdminBoundaries, "create", staticmethod(refuse))


def osmnx_reference(tmp_path, area, truncate_by_edge=False, simplify=True, retain_all=False):
    """What osmnx's graph_from_polygon returns for the Kigali extract: its steps after the download, replayed."""
    ox = pytest.importorskip("osmnx")
    passes, _ = way_filter("drive_service")
    kept = {w.id for w in osmium.FileProcessor(str(KIGALI), osmium.osm.WAY) if passes(w.tags.get)}
    refs = {n.ref for w in osmium.FileProcessor(str(KIGALI), osmium.osm.WAY) if w.id in kept for n in w.nodes}
    xml = tmp_path / "reference.osm"
    with osmium.SimpleWriter(str(xml)) as writer:
        for entity in osmium.FileProcessor(str(KIGALI)):
            if (entity.is_node() and entity.id in refs) or (entity.is_way() and entity.id in kept):
                writer.add(entity)
    # osmnx 2.0.2's graph_from_polygon, after it has downloaded the ways in the widened area.
    area_utm, utm = ox.projection.project_geometry(area)
    widened, _ = ox.projection.project_geometry(area_utm.buffer(500), crs=utm, to_latlong=True)
    widened_graph = ox.graph_from_xml(xml, simplify=False, retain_all=True)
    widened_graph = ox.truncate.truncate_graph_polygon(widened_graph, widened, truncate_by_edge=truncate_by_edge)
    if not retain_all:
        widened_graph = ox.truncate.largest_component(widened_graph)
    if simplify:
        widened_graph = ox.simplify_graph(widened_graph)
    graph = ox.truncate.truncate_graph_polygon(widened_graph, area, truncate_by_edge=truncate_by_edge)
    if not retain_all:
        graph = ox.truncate.largest_component(graph)
    for node, count in ox.stats.count_streets_per_node(widened_graph, nodes=graph.nodes).items():
        graph.nodes[node]["street_count"] = count
    SpatialNetwork.from_networkx(graph).save(tmp_path / "reference")
    return SpatialNetwork.load(tmp_path / "reference")


# ----- cropping, as osmnx's graph_from_polygon


@pytest.mark.parametrize(
    "truncate_by_edge, simplify, retain_all",
    [(False, True, False), (True, True, False), (False, False, False), (True, True, True)],
)
def test_crop_matches_osmnx(handler, tmp_path, same_network, truncate_by_edge, simplify, retain_all):
    theirs = osmnx_reference(tmp_path, AREA, truncate_by_edge, simplify, retain_all)
    ours = handler.reader.load(AREA, truncate_by_edge=truncate_by_edge, simplify=simplify, retain_all=retain_all)
    same_network(ours, theirs, exact_lengths=not simplify)


def test_crop_keeps_the_largest_part_inside_the_area(handler):
    network = handler.reader.load(AREA)
    nodes = network.nodes
    assert shapely.intersects_xy(AREA, nodes["x"].to_numpy(), nodes["y"].to_numpy()).all()
    assert set(network.connected_components()) == {0}
    assert network.metadata["load"] == {
        "retain_all": False,
        "truncate_by_edge": False,
        "cropped": True,
        "area_bounds": list(AREA.bounds),
    }
    assert network.metadata["extract"]["id"] == "rwanda" and network.metadata["build"]["simplified"]
    widened = handler.reader.load(AREA, truncate_by_edge=True)
    assert len(widened.nodes) > len(network.nodes)


def test_points_are_widened_as_the_fetcher_does(handler):
    near = handler.reader.load([(-1.95, 30.065)], buffer=300)
    far = handler.reader.load([(-1.95, 30.065)], buffer=900)
    assert 0 < len(near.nodes) < len(far.nodes)
    assert handler.config.source_area([(-1.95, 30.065)], roads.OSMRoadsResolveOptions(buffer=300)).contains(
        shapely.Point(30.065, -1.95)
    )


def test_an_area_without_roads(handler):
    with pytest.raises(ValueError, match="No drive_service roads"):
        handler.reader.load(box(29.5, -2.5, 29.51, -2.49))


def test_crop_to_geometry_takes_the_area_crs(handler):
    network = handler.reader.stored_network(RWANDA, OSMRoadsReadOptions(), simplified=False)
    from pyproj import Transformer

    project = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True).transform
    mercator = shapely.transform(AREA, lambda xy: np.column_stack(project(xy[:, 0], xy[:, 1])))
    cropped = handler.reader.crop_to_geometry(network, mercator, crs="EPSG:3857")
    assert set(cropped.node_index) == set(handler.reader.crop_to_geometry(network, AREA).node_index)


# ----- whole extracts


def test_a_country_with_its_own_extract_is_not_cropped(handler, no_boundaries):
    network = handler.reader.load("RWA")
    whole = handler.reader.stored_network(RWANDA, OSMRoadsReadOptions(), simplified=True).largest_component()
    assert set(network.node_index) == set(whole.node_index)
    assert network.metadata["load"]["cropped"] is False


def test_a_country_inside_a_multi_country_extract_is_cropped(store, geofabrik, tmp_path, monkeypatch):
    from gigaspatial.handlers import boundaries

    store_extract(tmp_path, "asia/israel-and-palestine")
    monkeypatch.setattr(
        boundaries.AdminBoundaries, "create", staticmethod(lambda **kwargs: types.SimpleNamespace(to_geoms=lambda: [AREA]))
    )
    handler = OSMRoadsHandler(data_store=store, base_path=Path("osm/geofabrik"))
    network = handler.reader.load("PSE", extracts=["israel-and-palestine"])
    assert network.metadata["load"]["cropped"]
    assert shapely.intersects_xy(AREA, network.nodes["x"].to_numpy(), network.nodes["y"].to_numpy()).all()


def test_crop_to_source_false_uses_whole_extracts(handler):
    whole = handler.reader.load(AREA, crop_to_source=False)
    assert set(whole.node_index) == set(handler.reader.load("RWA").node_index)


# ----- network folders


def test_networks_are_built_once_and_reused(handler, builds, tmp_path):
    handler.reader.load(AREA)
    handler.reader.load("RWA")
    handler.reader.load(AREA, simplify=False)
    assert builds == ["rwanda.osm.pbf"]
    folders = tmp_path / "osm/geofabrik" / DATE / "africa/rwanda.networks"
    assert sorted(p.name for p in folders.iterdir()) == ["drive_service-raw", "drive_service-simplified"]

    fresh = OSMRoadsHandler(data_store=handler.data_store, base_path=Path("osm/geofabrik"))
    fresh.reader.load("RWA")
    assert builds == ["rwanda.osm.pbf"]


def test_each_variant_has_its_folder(handler, tmp_path):
    handler.reader.load(AREA, network_type="walk")
    handler.reader.load(AREA, tags={"highway": ["primary", "secondary"]})
    folders = sorted(p.name for p in (tmp_path / "osm/geofabrik" / DATE / "africa/rwanda.networks").iterdir())
    assert folders[0].startswith("tags-") and folders[0].endswith("-raw") and folders[1] == "walk-raw"
    assert handler.reader.network_folder(RWANDA, OSMRoadsReadOptions(tags={"highway": ["secondary", "primary"]}), False).name == folders[0]


def test_networks_of_an_older_build_are_rebuilt(handler, builds, tmp_path):
    handler.reader.load(AREA)
    manifest = tmp_path / "osm/geofabrik" / DATE / "africa/rwanda.networks/drive_service-raw/network.json"
    saved = json.loads(manifest.read_text())
    saved["metadata"]["build"]["version"] = 0
    manifest.write_text(json.dumps(saved))
    handler.reader.load(AREA)
    assert builds == ["rwanda.osm.pbf", "rwanda.osm.pbf"]


# ----- several extracts


def write_halves(west: Path, east: Path, split: float = 30.065):
    """Kigali cut in two at a meridian, as two neighbouring extracts are: each with every way that enters it."""
    nodes = {n.id: (n.location.lon, n.location.lat, dict(n.tags)) for n in osmium.FileProcessor(str(KIGALI), osmium.osm.NODE)}
    ways = [(w.id, [n.ref for n in w.nodes], dict(w.tags)) for w in osmium.FileProcessor(str(KIGALI), osmium.osm.WAY)]
    for path, side in ((west, lambda lon: lon < split), (east, lambda lon: lon >= split)):
        kept = [w for w in ways if any(side(nodes[r][0]) for r in w[1])]
        refs = sorted({r for w in kept for r in w[1]})
        path.parent.mkdir(parents=True, exist_ok=True)
        with osmium.SimpleWriter(str(path)) as writer:
            for r in refs:
                writer.add_node(osmium.osm.mutable.Node(id=r, location=nodes[r][:2], tags=nodes[r][2]))
            for way_id, way_refs, tags in kept:
                writer.add_way(osmium.osm.mutable.Way(id=way_id, nodes=way_refs, tags=tags))


@pytest.mark.parametrize("order", [["rwanda", "uganda"], ["uganda", "rwanda"]])
def test_extracts_combine_as_one_file(handler, store, tmp_path, same_network, order):
    whole_area = handler.reader.load(AREA)
    whole = handler.reader.load("RWA")
    split_root = tmp_path / "split"
    write_halves(split_root / "osm/geofabrik" / DATE / "africa/rwanda.osm.pbf", split_root / "osm/geofabrik" / DATE / "africa/uganda.osm.pbf")
    shutil.copy(tmp_path / "osm/geofabrik/index-v1.json", split_root / "osm/geofabrik/index-v1.json")
    split = OSMRoadsHandler(data_store=LocalDataStore(split_root), base_path=Path("osm/geofabrik"))

    same_network(split.reader.load(AREA, extracts=order), whole_area, exact_lengths=False)
    combined = split.reader.load(AREA, extracts=order, crop_to_source=False)
    same_network(combined, whole, exact_lengths=False)
    assert [e["id"] for e in combined.metadata["extracts"]] == order


# ----- path sources


def test_path_sources(handler, tmp_path):
    from_path = handler.reader.load(RWANDA)
    assert set(from_path.node_index) == set(handler.reader.load("RWA").node_index)
    every_part = handler.reader.load(RWANDA, retain_all=True)
    assert len(every_part.nodes) > len(from_path.nodes)

    folder = tmp_path / "osm/geofabrik" / DATE / "africa/rwanda.networks/drive_service-raw"
    as_saved = handler.reader.load(folder)
    assert len(as_saved.nodes) == len(SpatialNetwork.load(folder).nodes) and not as_saved.metadata["build"]["simplified"]


# ----- remote data stores


class RemoteStore(DataStore):
    """A data store that is not a LocalDataStore, so extracts must be copied to a local file to be read."""

    def __init__(self, inner: LocalDataStore):
        super().__init__()
        self.inner = inner
        self.opened = []

    def read_file(self, path, encoding=None):
        return self.inner.read_file(path, encoding)

    def write_file(self, path, data, encoding=None):
        return self.inner.write_file(path, data, encoding)

    def file_exists(self, path):
        return self.inner.file_exists(path)

    def list_files(self, path):
        return self.inner.list_files(path)

    def list_directories(self, path):
        return self.inner.list_directories(path)

    def walk(self, top):
        return self.inner.walk(top)

    def open(self, file, mode="r", encoding=None):
        self.opened.append(str(file))
        return self.inner.open(file, mode, encoding) if encoding else self.inner.open(file, mode)

    def is_file(self, path):
        return self.inner.is_file(path)

    def is_dir(self, path):
        return self.inner.is_dir(path)

    def remove(self, path):
        return self.inner.remove(path)

    def rmdir(self, dir):
        return self.inner.rmdir(dir)


def test_remote_stores(geofabrik, tmp_path, builds):
    store_extract(tmp_path, "africa/rwanda")
    remote = RemoteStore(LocalDataStore(tmp_path))
    handler = OSMRoadsHandler(data_store=remote, base_path=Path("osm/geofabrik"))
    network = handler.reader.load(AREA)
    assert len(network.nodes) > 0 and RWANDA in remote.opened
    assert (tmp_path / "osm/geofabrik" / DATE / "africa/rwanda.networks/drive_service-raw/network.json").is_file()
    handler.reader.load(AREA)
    assert len(builds) == 1
