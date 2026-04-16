"""
Module for local filesystem DataStore implementation.
Provides access to local files and directories using a consistent DataStore interface.
"""
from pathlib import Path
import os
import shutil
from typing import Any, List, Generator, Tuple, Union, IO
from os import PathLike
from typing import Union

from .data_store import DataStore

Pathish = Union[str, PathLike[str]]


class LocalDataStore(DataStore):
    """Implementation for local filesystem storage."""

    def __init__(self, base_path: Union[str, Path] = ""):
        """
        Initialize the local data store.

        Args:
            base_path: Base directory for relative paths. Defaults to current directory.
        """
        super().__init__()
        self.base_path = Path(base_path).resolve()

    def _resolve_path(self, path: Pathish) -> Path:
        path_obj = Path(path)

        # If absolute, return as-is
        if path_obj.is_absolute():
            return path_obj.resolve()

        # Otherwise, resolve relative to base_path
        return (self.base_path / path_obj).resolve()

    def read_file(self, path: str) -> bytes:
        """
        Read contents of a file as bytes.

        Args:
            path: Path to the file.

        Returns:
            File contents in bytes.
        """
        full_path = self._resolve_path(path)
        with open(full_path, "rb") as f:
            return f.read()

    def write_file(self, path: str, data: Union[bytes, str]) -> None:
        """
        Write data (string or bytes) to a file.
        Automatically creates parent directories if they don't exist.

        Args:
            path: Path where to write.
            data: Data to write (str or bytes).
        """
        full_path = self._resolve_path(path)
        self.mkdir(str(full_path.parent), exist_ok=True)

        if isinstance(data, str):
            mode = "w"
            encoding = "utf-8"
        else:
            mode = "wb"
            encoding = None

        with open(full_path, mode, encoding=encoding) as f:
            f.write(data)

    def file_exists(self, path: str) -> bool:
        """Checks if file exists at path."""
        return self._resolve_path(path).is_file()

    def list_files(self, path: str) -> List[str]:
        """
        List all files in a directory, returning relative paths from base_path.

        Args:
            path: Directory to list.

        Returns:
            List of relative file paths.
        """
        full_path = self._resolve_path(path)
        return [
            str(f.relative_to(self.base_path))
            for f in full_path.iterdir()
            if f.is_file()
        ]

    def walk(self, top: str) -> Generator[Tuple[str, List[str], List[str]], None, None]:
        """
        Walk through directory tree.

        Args:
            top: Starting directory.

        Yields:
            Tuple of (relative_root, dirnames, filenames).
        """
        full_path = self._resolve_path(top)
        for root, dirs, files in os.walk(full_path):
            rel_root = str(Path(root).relative_to(self.base_path))
            yield rel_root, dirs, files

    def list_directories(self, path: str) -> List[str]:
        """
        List immediate subdirectories in path.

        Args:
            path: Directory to list.

        Returns:
            List of subdirectory names.
        """
        full_path = self._resolve_path(path)

        if not full_path.exists():
            return []

        if not full_path.is_dir():
            return []

        return [d.name for d in full_path.iterdir() if d.is_dir()]

    def open(self, path: str, mode: str = "r") -> IO:
        """
        Open a file-like object using the local filesystem.

        Args:
            path: File path.
            mode: Open mode.

        Returns:
            File descriptor.
        """
        full_path = self._resolve_path(path)
        self.mkdir(str(full_path.parent), exist_ok=True)
        return open(full_path, mode)

    def is_file(self, path: str) -> bool:
        """Checks if path is a file."""
        return self._resolve_path(path).is_file()

    def is_dir(self, path: str) -> bool:
        """Checks if path is a directory."""
        return self._resolve_path(path).is_dir()

    def remove(self, path: str) -> None:
        """Removes a file."""
        full_path = self._resolve_path(path)
        if full_path.is_file():
            os.remove(full_path)

    def copy_file(self, src: str, dst: str) -> None:
        """Copy a file from src to dst."""
        src_path = self._resolve_path(src)
        dst_path = self._resolve_path(dst)
        self.mkdir(str(dst_path.parent), exist_ok=True)
        shutil.copy2(src_path, dst_path)

    def rmdir(self, directory: str) -> None:
        """Removes a directory."""
        full_path = self._resolve_path(directory)
        if full_path.is_dir():
            os.rmdir(full_path)

    def mkdir(self, path: str, exist_ok: bool = False) -> None:
        """Creates a directory and its parents."""
        full_path = self._resolve_path(path)
        full_path.mkdir(parents=True, exist_ok=exist_ok)

    def exists(self, path: str) -> bool:
        """Checks if path exists."""
        return self._resolve_path(path).exists()
