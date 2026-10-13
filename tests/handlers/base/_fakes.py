"""
In-memory fake handler built on the four base classes.

The grid is four 1x1 degree tiles covering lon/lat [0, 2] x [0, 2]. Each tile's
stored file is a CSV with two points: one near the tile's lower-left corner
(``offset 0.1``) and one near its upper-right corner (``offset 0.9``).

The fake's components take typed options. ``tile`` (resolve) keeps only the
named tile, ``columns`` (read) selects columns, and ``show_progress`` is both a
download and a read option.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
from shapely.geometry import box

from gigaspatial.handlers.base import (
    BaseHandler,
    BaseHandlerConfig,
    BaseHandlerDownloader,
    BaseHandlerReader,
    DownloadOptions,
    ReadOptions,
    ResolveOptions,
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


@dataclass(frozen=True)
class FakeResolveOptions(ResolveOptions):
    tile: Optional[str] = None


@dataclass(frozen=True)
class FakeReadOptions(ReadOptions):
    columns: Optional[Tuple[str, ...]] = None
    show_progress: bool = False


@dataclass
class FakeConfig(BaseHandlerConfig):
    options_type = FakeResolveOptions

    n_workers: int = 1

    def __post_init__(self):
        super().__post_init__()
        self.lookups = 0

    def get_relevant_data_units_by_key(self, search_key, options) -> List[str]:
        self.lookups += 1
        return [
            tid
            for tid, geom in TILES.items()
            if geom.intersects(search_key) and options.tile in (None, tid)
        ]

    def get_data_unit_path(self, unit: str, options=None) -> Path:
        return self.base_path / f"{unit}.csv"


class FakeDownloader(BaseHandlerDownloader):
    options_type = DownloadOptions

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.downloaded = []
        self.received = []

    def download_data_unit(self, unit: str, options) -> str:
        path = str(self.config.get_data_unit_path(unit))
        self.data_store.write_file(path, tile_rows(unit).to_csv(index=False))
        self.downloaded.append(unit)
        self.received.append(options)
        return path


class FakeReader(BaseHandlerReader):
    options_type = FakeReadOptions

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.received = []

    def load_from_paths(self, source_data_path, options) -> pd.DataFrame:
        self.received.append(options)
        data = self._load_tabular_data(
            file_paths=source_data_path, show_progress=options.show_progress
        )
        return data[list(options.columns)] if options.columns else data


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
