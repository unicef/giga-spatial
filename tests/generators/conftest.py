from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds
from shapely.geometry import box

from gigaspatial.core.io.local_data_store import LocalDataStore
from gigaspatial.handlers.ghsl import GHSLDataConfig, GHSLDataHandler
from gigaspatial.handlers.worldpop import WPPopulationConfig, WPPopulationHandler


DEFAULT_RASTER_DATA = np.arange(1, 17, dtype=np.float32).reshape(4, 4)
DEFAULT_BOUNDS = (0.0, 0.0, 0.04, 0.04)


def _unexpected_ctor(name: str):
    """Return a failing stub that guards against default handler instantiation."""

    def _fail(*args, **kwargs):
        raise AssertionError(
            f"Default {name} constructor should not be called when handler= is provided"
        )

    return _fail


@pytest.fixture
def default_raster_data() -> np.ndarray:
    """Deterministic 4x4 float32 test raster."""
    return DEFAULT_RASTER_DATA.copy()


@pytest.fixture
def unexpected_ctor():
    """Factory returning a failing constructor guard for handler injection tests."""
    return _unexpected_ctor


@pytest.fixture
def write_raster():
    """Fixture returning helper to write deterministic single-band GeoTIFF files."""
    return write_raster_tif


def write_raster_tif(
    path: Path,
    data: np.ndarray = DEFAULT_RASTER_DATA,
    bounds: Tuple[float, float, float, float] = DEFAULT_BOUNDS,
    crs: str = "EPSG:4326",
    nodata: float = -99999.0,
) -> Path:
    """Write a deterministic single-band GeoTIFF to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = data.shape
    transform = from_bounds(*bounds, width=width, height=height)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=str(data.dtype),
        crs=crs,
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(data, 1)
    return path


def _seed_ghsl_tile_cache(
    ghsl_base_path: Path,
    tile_bounds_4326: Tuple[float, float, float, float] = (-1.0, -1.0, 5.0, 5.0),
    tile_id: str = "R1_C1",
) -> None:
    """Pre-seed both EPSG:4326 and ESRI:54009 GHSL tile caches to keep tests offline."""
    cache_dir = ghsl_base_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    tiles_4326 = gpd.GeoDataFrame(
        {"tile_id": [tile_id]},
        geometry=[box(*tile_bounds_4326)],
        crs="EPSG:4326",
    )
    tiles_4326.to_file(cache_dir / "tiles_4326.geojson", driver="GeoJSON")
    tiles_54009 = tiles_4326.to_crs("ESRI:54009")
    tiles_54009.to_file(cache_dir / "tiles_54009.geojson", driver="GeoJSON")


@pytest.fixture(autouse=True)
def _block_live_http(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail immediately if any test in tests/generators attempts a live HTTP request."""
    import requests

    def _no_network(*args, **kwargs):
        raise RuntimeError("Generator tests must run completely offline without live HTTP calls")

    monkeypatch.setattr(requests.sessions.Session, "request", _no_network)
    monkeypatch.setattr(requests, "get", _no_network)


@pytest.fixture
def make_worldpop_handler(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Callable[..., WPPopulationHandler]:
    """Factory for offline WPPopulationHandler backed by LocalDataStore."""

    def _factory(
        *,
        project: str = "pop",
        release: str = "GR1",
        year: int = 2020,
        resolution: int = 1000,
        constrained: bool = False,
        un_adjusted: bool = False,
        school_age: bool = False,
        under_18: bool = False,
        dug_level: str = "L1",
        country_rasters: Optional[Dict[str, List[Tuple[str, np.ndarray, Tuple[float, float, float, float]]]]] = None,
        store: Optional[LocalDataStore] = None,
        subdir: str = "wp",
    ) -> WPPopulationHandler:
        store = store or LocalDataStore(base_path=tmp_path)
        wp_base = tmp_path / subdir
        config = WPPopulationConfig(
            release=release,
            project=project,
            year=year,
            resolution=resolution,
            un_adjusted=un_adjusted,
            constrained=constrained,
            school_age=school_age,
            under_18=under_18,
            dug_level=dug_level,
            base_path=wp_base,
            data_store=store,
            n_workers=1,
        )

        rasters_by_iso = country_rasters or {
            "RWA": [
                (
                    "https://data.worldpop.org/GIS/Population/rwa_ppp_2020_UNadj.tif",
                    DEFAULT_RASTER_DATA,
                    DEFAULT_BOUNDS,
                )
            ]
        }

        url_map: Dict[str, List[str]] = {}
        for iso3, specs in rasters_by_iso.items():
            urls: List[str] = []
            for url, array, bounds in specs:
                write_raster_tif(
                    config.get_data_unit_path(url),
                    data=array,
                    bounds=bounds,
                    crs="EPSG:4326",
                )
                urls.append(url)
            url_map[iso3] = urls

        def _fake_search(api_dataset_type, category, search_key, search_year):
            if search_key not in url_map:
                return []
            return [{"files": url_map[search_key]}]

        monkeypatch.setattr(config.client, "search_datasets", _fake_search)
        return WPPopulationHandler(config=config, data_store=store)

    return _factory


@pytest.fixture
def make_ghsl_handler(tmp_path: Path) -> Callable[..., GHSLDataHandler]:
    """Factory for offline GHSLDataHandler backed by LocalDataStore and cached tiles."""

    def _factory(
        *,
        product: str = "GHS_BUILT_S",
        year: int = 2020,
        resolution: int = 100,
        coord_system: int = 4326,
        data: np.ndarray = DEFAULT_RASTER_DATA,
        bounds_4326: Tuple[float, float, float, float] = DEFAULT_BOUNDS,
        tile_id: str = "R1_C1",
        store: Optional[LocalDataStore] = None,
        subdir: str = "ghsl",
    ) -> GHSLDataHandler:
        store = store or LocalDataStore(base_path=tmp_path)
        ghsl_base = tmp_path / subdir
        _seed_ghsl_tile_cache(ghsl_base, tile_id=tile_id)

        config = GHSLDataConfig(
            product=product,
            year=year,
            resolution=resolution,
            coord_system=coord_system,
            base_path=ghsl_base,
            data_store=store,
            n_workers=1,
        )

        raster_path = config.get_data_unit_path(tile_id)
        if config.crs == "ESRI:54009":
            poly_54009 = (
                gpd.GeoSeries([box(*bounds_4326)], crs="EPSG:4326")
                .to_crs("ESRI:54009")
                .iloc[0]
            )
            write_bounds = tuple(poly_54009.bounds)
        else:
            write_bounds = bounds_4326

        write_raster_tif(raster_path, data=data, bounds=write_bounds, crs=config.crs)
        return GHSLDataHandler(
            product=product,
            year=year,
            resolution=resolution,
            config=config,
            data_store=store,
        )

    return _factory
