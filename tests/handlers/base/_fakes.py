"""
In-memory fake handler built on the four base classes.

The grid is four 1x1 degree tiles covering lon/lat [0, 2] x [0, 2]. Each tile's
stored file is a CSV with two points: one near the tile's lower-left corner
(``offset 0.1``) and one near its upper-right corner (``offset 0.9``).

Lives in a regular module, not ``conftest.py``, so ``multiprocessing`` workers
can import the classes by name.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List

import pandas as pd
from shapely.geometry import box

from gigaspatial.handlers.base import (
    BaseHandler,
    BaseHandlerConfig,
    BaseHandlerDownloader,
    BaseHandlerReader,
)

TILES = {
    f"t{x}{y}": box(x, y, x + 1, y + 1) for x in range(2) for y in range(2)
}


def tile_rows(tile_id: str) -> pd.DataFrame:
    minx, miny, _, _ = TILES[tile_id].bounds
    return pd.DataFrame(
        {
            "tile_id": [tile_id, tile_id],
            "latitude": [miny + 0.1, miny + 0.9],
            "longitude": [minx + 0.1, minx + 0.9],
        }
    )


@dataclass
class FakeConfig(BaseHandlerConfig):
    n_workers: int = 1

    def __post_init__(self):
        super().__post_init__()
        self.lookups = 0

    def get_relevant_data_units_by_key(self, search_key, **kwargs) -> List[str]:
        self.lookups += 1
        return [tid for tid, geom in TILES.items() if geom.intersects(search_key)]

    def get_data_unit_path(self, unit: str, **kwargs) -> Path:
        return self.base_path / f"{unit}.csv"


class FakeDownloader(BaseHandlerDownloader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.downloaded = []

    def download_data_unit(self, unit: str, **kwargs) -> str:
        path = str(self.config.get_data_unit_path(unit))
        self.data_store.write_file(path, tile_rows(unit).to_csv(index=False))
        self.downloaded.append(unit)
        return path


class FakeReader(BaseHandlerReader):
    def load_from_paths(self, source_data_path, **kwargs) -> pd.DataFrame:
        return self._load_tabular_data(file_paths=source_data_path, show_progress=False)


class FakeHandler(BaseHandler):
    def __init__(self, base_path: Path, n_workers: int = 1, **kwargs):
        self._base_path = base_path
        self._n_workers = n_workers
        super().__init__(**kwargs)

    def create_config(self, data_store, logger, **kwargs) -> FakeConfig:
        return FakeConfig(
            base_path=self._base_path,
            n_workers=self._n_workers,
            data_store=data_store,
            logger=logger,
        )

    def create_downloader(self, config, data_store, logger, **kwargs):
        return FakeDownloader(config=config, data_store=data_store, logger=logger)

    def create_reader(self, config, data_store, logger, **kwargs):
        return FakeReader(config=config, data_store=data_store, logger=logger)
