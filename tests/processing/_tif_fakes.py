"""
Synthetic GeoTIFFs and an in-memory data store for the TifProcessor tests.

This is a plain module rather than a conftest so that worker processes started
with ``spawn`` can unpickle the data store.
"""

import io
import posixpath
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import rasterio
from rasterio.transform import from_bounds

from gigaspatial.core.io.data_store import DataStore


DEFAULT_RASTER_DATA = np.arange(1, 17, dtype=np.float32).reshape(4, 4)
DEFAULT_BOUNDS = (0.0, 0.0, 0.04, 0.04)
DEFAULT_NODATA = -99999.0
EAST_BOUNDS = (0.04, 0.0, 0.08, 0.04)


def float32_bands(*bands) -> np.ndarray:
    return np.stack(bands).astype(np.float32)


BAND = DEFAULT_RASTER_DATA
RGB = float32_bands(BAND, BAND + 100, BAND + 200)
RGBA = float32_bands(BAND, BAND + 100, BAND + 200, np.full((4, 4), 255))
TWO_BANDS = float32_bands(BAND, BAND * 10)
BANDS_FOR_MODE = {"single": BAND, "rgb": RGB, "rgba": RGBA, "multi": TWO_BANDS}


def write_raster_tif(
    path: Path,
    data: np.ndarray = DEFAULT_RASTER_DATA,
    bounds: Tuple[float, float, float, float] = DEFAULT_BOUNDS,
    crs: str = "EPSG:4326",
    nodata: Optional[float] = DEFAULT_NODATA,
    descriptions: Optional[Sequence[Optional[str]]] = None,
) -> Path:
    """Write a deterministic GeoTIFF; ``data`` is (rows, cols) or (bands, rows, cols)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stack = data[np.newaxis] if data.ndim == 2 else data
    count, height, width = stack.shape
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=count,
        dtype=str(stack.dtype),
        crs=crs,
        transform=from_bounds(*bounds, width=width, height=height),
        nodata=nodata,
    ) as dst:
        dst.write(stack)
        for band, description in enumerate(descriptions or [], start=1):
            if description is not None:
                dst.set_band_description(band, description)
    return path


class StoredRasters:
    """Writes synthetic rasters into a data store and returns the path to open them by."""

    def __init__(self, store: DataStore, tmp_path: Path):
        self.store = store
        self._tmp_path = tmp_path

    def put(self, name: str, data: np.ndarray = DEFAULT_RASTER_DATA, **kwargs) -> str:
        if not isinstance(self.store, InMemoryDataStore):
            return str(
                write_raster_tif(self._tmp_path / "stored" / name, data, **kwargs)
            )
        staged = write_raster_tif(self._tmp_path / "staging" / name, data, **kwargs)
        self.store.write_file(f"stored/{name}", staged.read_bytes())
        return f"stored/{name}"

    def output(self, name: str) -> str:
        """A path in the store to save a raster to."""
        if isinstance(self.store, InMemoryDataStore):
            return f"saved/{name}"
        return str(self._tmp_path / "saved" / name)


class _WriteOnClose(io.BytesIO):
    def __init__(self, commit, text: bool, encoding: str):
        super().__init__()
        self._commit = commit
        self._text = text
        self._encoding = encoding

    def write(self, data):
        if self._text:
            data = data.encode(self._encoding)
        return super().write(data)

    def close(self):
        if not self.closed:
            self._commit(self.getvalue())
        super().close()


class InMemoryDataStore(DataStore):
    """A non-local data store: files are bytes in a dict, directories are implied by paths."""

    def __init__(self):
        self.files = {}

    @staticmethod
    def _key(path) -> str:
        key = posixpath.normpath(str(path).replace("\\", "/"))
        return "" if key == "." else key.lstrip("/")

    def _require(self, path) -> str:
        key = self._key(path)
        if key not in self.files:
            raise FileNotFoundError(f"No such file in memory store: {path}")
        return key

    def read_file(self, path, encoding=None):
        data = self.files[self._require(path)]
        return data.decode(encoding) if encoding else data

    def write_file(self, path, data, encoding=None):
        if isinstance(data, str):
            data = data.encode(encoding or self.DEFAULT_ENCODING)
        elif hasattr(data, "read"):
            data = data.read()
        elif not isinstance(data, bytes):
            raise ValueError("Unsupported data type.")
        self.files[self._key(path)] = bytes(data)

    def file_exists(self, path):
        return self._key(path) in self.files

    def _children(self, path):
        prefix = self._key(path)
        prefix = f"{prefix}/" if prefix else ""
        files, dirs = [], set()
        for key in self.files:
            if key.startswith(prefix):
                head, _, rest = key[len(prefix) :].partition("/")
                if rest:
                    dirs.add(head)
                else:
                    files.append(head)
        return sorted(files), sorted(dirs)

    def list_files(self, path):
        files, _ = self._children(path)
        return [posixpath.join(self._key(path), name) for name in files]

    def walk(self, top):
        files, dirs = self._children(top)
        yield self._key(top), dirs, files
        for name in dirs:
            yield from self.walk(posixpath.join(self._key(top), name))

    def open(self, file, mode="r", encoding=None):
        encoding = encoding or self.DEFAULT_ENCODING
        if mode in ("r", "rb"):
            data = self.files[self._require(file)]
            return (
                io.BytesIO(data) if mode == "rb" else io.StringIO(data.decode(encoding))
            )
        if mode in ("w", "wb"):
            key = self._key(file)
            return _WriteOnClose(
                lambda value: self.files.__setitem__(key, value), mode == "w", encoding
            )
        raise ValueError(f"Unsupported mode: {mode}")

    def is_file(self, path):
        return self.file_exists(path)

    def is_dir(self, path):
        files, dirs = self._children(path)
        return bool(files or dirs) and not self.is_file(path)

    def remove(self, path):
        del self.files[self._require(path)]

    def rmdir(self, dir):
        prefix = f"{self._key(dir)}/"
        for key in [k for k in self.files if k.startswith(prefix)]:
            del self.files[key]

    def copy_file(self, src, dst):
        self.files[self._key(dst)] = self.files[self._require(src)]
