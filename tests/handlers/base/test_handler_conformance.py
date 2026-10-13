"""
Structural rules every handler config must follow.

These walk the real handler classes rather than run them, so they need no
data or network access.
"""

import importlib
from pathlib import Path

import pytest

from gigaspatial.handlers.base import BaseHandlerConfig

HANDLER_MODULES = [
    "gigaspatial.handlers.ghsl",
    "gigaspatial.handlers.google_open_buildings",
    "gigaspatial.handlers.google_ms_combined_buildings",
    "gigaspatial.handlers.microsoft_global_buildings",
    "gigaspatial.handlers.worldpop",
    "gigaspatial.handlers.ookla_speedtest",
    "gigaspatial.handlers.opencellid",
    "gigaspatial.handlers.hdx",
    "gigaspatial.handlers.rwi",
    "gigaspatial.handlers.nasa.srtm",
    "gigaspatial.handlers.wri.aqueduct_flood",
]


def _handler_configs():
    for name in HANDLER_MODULES:
        try:
            importlib.import_module(name)
        except ImportError:
            continue

    found, stack = [], list(BaseHandlerConfig.__subclasses__())
    while stack:
        cls = stack.pop()
        stack.extend(cls.__subclasses__())
        if cls.__module__.startswith("gigaspatial.handlers."):
            found.append(cls)
    return sorted(set(found), key=lambda c: c.__qualname__)


HANDLER_CONFIGS = _handler_configs()


def test_handler_configs_were_found():
    assert len(HANDLER_CONFIGS) >= 8


@pytest.mark.parametrize("cls", HANDLER_CONFIGS, ids=lambda c: c.__name__)
def test_search_geometry_is_owned_by_the_base(cls):
    assert "extract_search_geometry" not in vars(cls)


@pytest.mark.parametrize("cls", HANDLER_CONFIGS, ids=lambda c: c.__name__)
def test_no_handler_uses_the_old_lookup_name(cls):
    assert not hasattr(cls, "get_relevant_data_units_by_geometry")


@pytest.mark.parametrize("cls", HANDLER_CONFIGS, ids=lambda c: c.__name__)
def test_resolution_is_customised_only_through_hooks(cls):
    assert "get_relevant_data_units" not in vars(cls)


CACHE_INTERNALS = ("_units", "_unit_cache", "_cache_key", "_source_key")


@pytest.mark.parametrize("module_name", HANDLER_MODULES)
def test_handlers_do_not_touch_the_unit_cache(module_name):
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        pytest.skip(f"{module_name} dependencies are not installed")
    source = Path(module.__file__).read_text()

    assert not [name for name in CACHE_INTERNALS if f".{name}" in source]
