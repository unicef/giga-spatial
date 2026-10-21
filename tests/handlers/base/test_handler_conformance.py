"""
Structural rules every handler config must follow.

These walk the real handler classes rather than run them, so they need no
data or network access.
"""

import dataclasses
import importlib
import inspect
from pathlib import Path

import pytest

from gigaspatial.handlers.base import (
    BaseHandler,
    BaseHandlerConfig,
    BaseHandlerDownloader,
    BaseHandlerReader,
    DownloadOptions,
    ReadOptions,
    ResolveOptions,
)

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


def _handler_subclasses(*bases):
    for name in HANDLER_MODULES:
        try:
            importlib.import_module(name)
        except ImportError:
            continue

    found, stack = [], [sub for base in bases for sub in base.__subclasses__()]
    while stack:
        cls = stack.pop()
        stack.extend(cls.__subclasses__())
        if cls.__module__.startswith("gigaspatial.handlers."):
            found.append(cls)
    return sorted(set(found), key=lambda c: c.__qualname__)


HANDLER_CONFIGS = _handler_subclasses(BaseHandlerConfig)
HANDLER_COMPONENTS = _handler_subclasses(
    BaseHandlerConfig, BaseHandlerDownloader, BaseHandlerReader, BaseHandler
)

OPTION_INTERNALS = (
    "_resolve_units",
    "_stored_paths_of",
    "_stored_paths",
    "_search_geometry",
    "_download_units",
    "_resolve_paths",
    "_load",
    "_crop_to_source",
    "_ensure_available",
    "_read",
    "_missing_units",
)


@pytest.mark.parametrize("cls", HANDLER_COMPONENTS, ids=lambda c: c.__qualname__)
def test_options_are_sorted_only_by_the_base(cls):
    assert not [name for name in OPTION_INTERNALS if name in vars(cls)]


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


@pytest.mark.parametrize("cls", HANDLER_CONFIGS, ids=lambda c: c.__name__)
def test_stored_paths_are_customised_per_unit(cls):
    assert "get_data_unit_paths" not in vars(cls)


CACHE_INTERNALS = ("_units", "_unit_cache", "_cache_key", "_source_key")


@pytest.mark.parametrize("module_name", HANDLER_MODULES)
def test_handlers_do_not_touch_the_unit_cache(module_name):
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        pytest.skip(f"{module_name} dependencies are not installed")
    source = Path(module.__file__).read_text()

    assert not [name for name in CACHE_INTERNALS if f".{name}" in source]


OPTION_HOOKS = {
    BaseHandlerConfig: (
        ResolveOptions,
        (
            "get_search_key",
            "get_relevant_data_units_by_key",
            "get_data_unit_path",
            "get_stored_paths",
        ),
    ),
    BaseHandlerDownloader: (DownloadOptions, ("download_data_unit",)),
    BaseHandlerReader: (ReadOptions, ("load_from_paths",)),
}
OPTION_COMPONENTS = _handler_subclasses(*OPTION_HOOKS)


def _role(cls):
    return next(base for base in OPTION_HOOKS if issubclass(cls, base))


@pytest.mark.parametrize("cls", OPTION_COMPONENTS, ids=lambda c: c.__qualname__)
def test_component_declares_frozen_options(cls):
    options_base, _ = OPTION_HOOKS[_role(cls)]

    assert issubclass(cls.options_type, options_base)
    assert cls.options_type.__dataclass_params__.frozen


@pytest.mark.parametrize("cls", OPTION_COMPONENTS, ids=lambda c: c.__qualname__)
def test_option_hooks_take_no_raw_kwargs(cls):
    _, hooks = OPTION_HOOKS[_role(cls)]
    leaky = [
        name
        for name in hooks
        if name in vars(cls)
        and any(
            p.kind is inspect.Parameter.VAR_KEYWORD
            for p in inspect.signature(getattr(cls, name)).parameters.values()
        )
    ]

    assert not leaky


@pytest.mark.parametrize("cls", HANDLER_CONFIGS, ids=lambda c: c.__qualname__)
def test_default_resolve_options_are_hashable(cls):
    hash(cls.options_type())


HANDLER_PARTS = _handler_subclasses(BaseHandlerDownloader, BaseHandlerReader)


@pytest.mark.parametrize("cls", HANDLER_PARTS, ids=lambda c: c.__qualname__)
def test_component_declares_its_config_type(cls):
    assert cls.config_type is not None
    assert issubclass(cls.config_type, BaseHandlerConfig)
    assert cls.config_type.__module__ == cls.__module__
