import json
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest

from gigaspatial.processing.network import SpatialNetwork

KIGALI = Path(__file__).resolve().parents[1] / "data" / "spatial_network" / "kigali"


@pytest.fixture(scope="session")
def kigali() -> SpatialNetwork:
    """Central Kigali drive network: 1,209 nodes, 2,802 directed edges."""
    meta = json.loads((KIGALI / "network.json").read_text())
    return SpatialNetwork(
        pd.read_parquet(KIGALI / "nodes.parquet"),
        gpd.read_parquet(KIGALI / "edges.parquet"),
        directed=meta["directed"],
        weight=meta["weight"],
        metadata=meta["metadata"],
    )
