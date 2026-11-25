import json
import multiprocessing
import signal
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import geopandas as gpd
import pandas as pd
import pytest

from _tif_fakes import InMemoryDataStore, StoredRasters
from gigaspatial.core.io.local_data_store import LocalDataStore
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


@pytest.fixture
def local_rasters(tmp_path) -> StoredRasters:
    """Synthetic rasters on disk under tmp_path, read through a LocalDataStore rooted there."""
    return StoredRasters(LocalDataStore(base_path=tmp_path), tmp_path)


@pytest.fixture
def memory_rasters(tmp_path) -> StoredRasters:
    """Synthetic rasters seeded as bytes into an in-memory, non-local data store."""
    return StoredRasters(InMemoryDataStore(), tmp_path)


@pytest.fixture(params=["local", "memory"])
def rasters(request) -> StoredRasters:
    """Each test runs once per data store kind."""
    return request.getfixturevalue(f"{request.param}_rasters")


@pytest.fixture(params=["spawn", "fork"])
def start_method(request):
    """Runs the test under the given multiprocessing start method, then restores the original."""
    if request.param not in multiprocessing.get_all_start_methods():
        pytest.skip(f"start method {request.param!r} is not available on this platform")
    original = multiprocessing.get_start_method(allow_none=True)
    multiprocessing.set_start_method(request.param, force=True)
    yield request.param
    multiprocessing.set_start_method(original, force=True)


def _fake_virtual_memory(monkeypatch, available: int):
    import psutil

    total = 16 * 1024**3
    fake = SimpleNamespace(
        total=total,
        available=available,
        used=total - available,
        free=available,
        percent=100.0 * (total - available) / total,
    )
    monkeypatch.setattr(psutil, "virtual_memory", lambda: fake)


@pytest.fixture
def low_memory(monkeypatch):
    """Reports one byte of available system memory to anything asking psutil."""
    _fake_virtual_memory(monkeypatch, available=1)


@pytest.fixture
def ample_memory(monkeypatch):
    """Reports 8 GiB of 16 GiB system memory available to anything asking psutil."""
    _fake_virtual_memory(monkeypatch, available=8 * 1024**3)


@pytest.fixture
def deadline():
    """Fails a block that runs too long; a pool whose worker setup raises respawns workers forever."""
    if not hasattr(signal, "setitimer"):
        pytest.skip("needs SIGALRM")

    @contextmanager
    def _deadline(seconds: float):
        def _expire(signum, frame):
            raise TimeoutError(f"did not finish within {seconds}s")

        previous = signal.signal(signal.SIGALRM, _expire)
        signal.setitimer(signal.ITIMER_REAL, seconds)
        try:
            yield
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)

    return _deadline
