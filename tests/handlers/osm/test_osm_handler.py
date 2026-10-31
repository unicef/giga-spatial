import shutil
from pathlib import Path

import pytest
from shapely.geometry import box

import gigaspatial
import gigaspatial.handlers as handlers
from gigaspatial.handlers.osm import roads

osmium = pytest.importorskip("osmium")

DATA = Path(__file__).resolve().parents[2] / "data" / "osm_roads"
DATE = "2026-10-26"
AREA = box(30.055, -1.96, 30.075, -1.94)
NETWORKS = f"osm/geofabrik/{DATE}/africa/rwanda.networks"


def test_public_imports():
    assert gigaspatial.OSMRoadsHandler is roads.OSMRoadsHandler
    for name in ("OSMRoadsConfig", "OSMRoadsDownloader", "OSMRoadsReader", "OSMRoadsHandler"):
        assert getattr(handlers, name) is getattr(roads, name)


@pytest.fixture
def handler(store, geofabrik):
    return roads.OSMRoadsHandler(data_store=store, base_path=Path("osm/geofabrik"))


@pytest.fixture
def stored(tmp_path):
    target = tmp_path / "osm/geofabrik" / DATE / "africa/rwanda.osm.pbf"
    target.parent.mkdir(parents=True)
    shutil.copy(DATA / "kigali.osm.pbf", target)
    return target


def test_load_data_downloads_builds_and_crops(handler, geofabrik, tmp_path):
    geofabrik.latest["africa/rwanda"] = DATE
    geofabrik.serve("africa/rwanda", DATE)
    network = handler.load_data(AREA)
    assert (tmp_path / "osm/geofabrik" / DATE / "africa/rwanda.osm.pbf").is_file()
    assert network.metadata["load"]["cropped"] and network.metadata["extract"]["date"] == DATE
    assert (tmp_path / NETWORKS / "drive_service-raw/network.json").is_file()


def test_crop_to_source_is_the_default(handler, stored):
    assert handler.load_data(AREA).metadata["load"]["cropped"]
    assert handler.download_and_load(AREA).metadata["load"]["cropped"]
    assert not handler.load_data(AREA, crop_to_source=False).metadata["load"]["cropped"]


def test_options_reach_their_components(handler, stored):
    walk = handler.load_data(AREA, network_type="walk", buffer=10, simplify=False)
    assert walk.metadata["build"]["network_type"] == "walk" and not walk.metadata["build"]["simplified"]
    with pytest.raises(TypeError):
        handler.load_data(AREA, network="walk")


def test_network_folders_are_path_sources(handler, stored, tmp_path):
    handler.load_data("RWA")
    folder = tmp_path / NETWORKS / "drive_service-simplified"
    assert handler.ensure_data_available(folder)
    network = handler.load_data(folder)
    assert len(network.nodes) == len(roads.SpatialNetwork.load(folder).nodes)
    assert handler.load_data(str(stored), retain_all=True).metadata["build"]["simplified"]
    with pytest.raises(RuntimeError):
        handler.load_data(tmp_path / NETWORKS / "walk-simplified")


def test_available_data_info(handler, stored):
    info = handler.get_available_data_info("RWA")
    assert info["available_data_units"] == 1 and info["all_available"]
