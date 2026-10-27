import ast
import importlib
import subprocess
import sys
from pathlib import Path

import pytest

PUBLIC_PACKAGES = [
    "gigaspatial",
    "gigaspatial.handlers",
    "gigaspatial.fetchers",
    "gigaspatial.image_downloaders",
    "gigaspatial.generators",
    "gigaspatial.grid",
    "gigaspatial.processing",
    "gigaspatial.processing.elevation",
    "gigaspatial.core.io",
    "gigaspatial.core.http",
    "gigaspatial.core.schemas",
]

LAZY_PACKAGES = [
    "gigaspatial",
    "gigaspatial.handlers",
    "gigaspatial.fetchers",
    "gigaspatial.processing.elevation",
]


@pytest.mark.parametrize("package", PUBLIC_PACKAGES)
def test_every_public_name_resolves(package):
    module = importlib.import_module(package)
    for name in module.__all__:
        assert getattr(module, name) is not None, f"{package}.{name}"


@pytest.mark.parametrize("package", LAZY_PACKAGES)
def test_type_checking_imports_match_lazy_exports(package):
    module = importlib.import_module(package)
    tree = ast.parse(Path(module.__file__).read_text())
    type_checking_names = set()
    for node in tree.body:
        if isinstance(node, ast.If) and getattr(node.test, "id", None) == "TYPE_CHECKING":
            for stmt in node.body:
                type_checking_names.update(alias.name for alias in stmt.names)
    assert type_checking_names == set(module.__all__) - {"__version__"}


def test_unknown_attribute_raises_attribute_error():
    import gigaspatial

    with pytest.raises(AttributeError):
        gigaspatial.NotAThing


@pytest.mark.parametrize("alias", ["gigaspatial.io", "gigaspatial.elevation", "gigaspatial.algorithms"])
def test_removed_aliases_are_gone(alias):
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(alias)


def test_bare_import_does_not_load_optional_or_heavy_dependencies():
    code = (
        "import sys, gigaspatial; "
        "heavy = ['ee', 'geemap', 'snowflake', 'google.cloud.bigquery', "
        "'azure.storage.blob', 'duckdb', 'geopandas', 'rasterio', 'osmnx', 'pandana']; "
        "print(','.join(m for m in heavy if m in sys.modules))"
    )
    result = subprocess.run(
        [sys.executable, "-W", "ignore", "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == ""
