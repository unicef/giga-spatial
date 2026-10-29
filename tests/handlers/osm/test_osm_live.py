from pathlib import Path

import pytest

from gigaspatial.core.io.local_data_store import LocalDataStore
from gigaspatial.handlers.osm.pbf import replication_timestamp
from gigaspatial.handlers.osm.roads import OSMRoadsHandler


@pytest.mark.live
def test_live_rwanda(tmp_path):
    """Downloads Geofabrik's index and Rwanda's extract (about 65 MiB), then builds and crops its roads."""
    pytest.importorskip("osmium")
    handler = OSMRoadsHandler(data_store=LocalDataStore(tmp_path), base_path=Path("osm/geofabrik"))
    (unit,) = handler.config.get_relevant_data_units("RWA")
    assert unit["id"] == "rwanda" and unit["url"].endswith(".osm.pbf") and "-latest" not in unit["url"]

    network = handler.load_data([(-1.9441, 30.0619)], buffer=1000)
    stored = handler.config.get_data_unit_path(unit)
    assert (tmp_path / stored).stat().st_size > 30 * 2**20
    assert replication_timestamp(stored, handler.data_store).date().isoformat() == unit["date"]
    assert len(network.edges) > 100 and network.metadata["extract"]["id"] == "rwanda"
    assert network.metadata["load"]["cropped"] and network.total_length() > 10_000
