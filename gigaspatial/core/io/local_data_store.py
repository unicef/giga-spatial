from pathlib import Path
import os
from typing import Any, List, Generator, Tuple, Union, IO

from .data_store import DataStore
from gigaspatial.utils.logging import get_logger


class LocalDataStore(DataStore):
    """Implementation for local filesystem storage."""

    def __init__(self, base_path: Union[str, Path] = ""):
        super().__init__()
        self.base_path = Path(base_path).resolve()

    def _resolve_path(self, path: str) -> Path:
        """Resolve path relative to base directory."""
        return self.base_path / path

    def read_file(self, path: str) -> bytes:
        full_path = self._resolve_path(path)
        with open(full_path, "rb") as f:
            return f.read()

    def write_file(self, path: str, data: Union[bytes, str]) -> None:
        full_path = self._resolve_path(path)
        self.makedirs(str(full_path.parent), exist_ok=True)

        if isinstance(data, str):
            mode = "w"
            encoding = "utf-8"
        else:
            mode = "wb"
            encoding = None

        with open(full_path, mode, encoding=encoding) as f:
            f.write(data)

    def file_exists(self, path: str) -> bool:
        return self._resolve_path(path).is_file()

    def list_files(self, path: str) -> List[str]:
        full_path = self._resolve_path(path)
        return [
            str(f.relative_to(self.base_path))
            for f in full_path.iterdir()
            if f.is_file()
        ]

    def walk(self, top: str) -> Generator[Tuple[str, List[str], List[str]], None, None]:
        full_path = self._resolve_path(top)
        for root, dirs, files in os.walk(full_path):
            rel_root = str(Path(root).relative_to(self.base_path))
            yield rel_root, dirs, files

    def open(self, path: str, mode: str = "r") -> IO:
        full_path = self._resolve_path(path)
        self.makedirs(str(full_path.parent), exist_ok=True)
        return open(full_path, mode)

    def is_file(self, path: str) -> bool:
        return self._resolve_path(path).is_file()

    def is_dir(self, path: str) -> bool:
        return self._resolve_path(path).is_dir()

    def remove(self, path: str) -> None:
        full_path = self._resolve_path(path)
        if full_path.is_file():
            os.remove(full_path)

    def rmdir(self, directory: str) -> None:
        full_path = self._resolve_path(directory)
        if full_path.is_dir():
            os.rmdir(full_path)

    def makedirs(self, path: str, exist_ok: bool = False) -> None:
        full_path = self._resolve_path(path)
        full_path.mkdir(parents=True, exist_ok=exist_ok)

    def exists(self, path: str) -> bool:
        return self._resolve_path(path).exists()
