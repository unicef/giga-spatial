"""
Module for defining the abstract DataStore interface.
This interface provides a consistent API for reading and writing files across different
storage backends like local disk, ADLS, or Snowflake.
"""
from abc import ABC, abstractmethod
from typing import Any, List, Generator, Union, Optional, IO


class DataStore(ABC):
    """
    Abstract base class defining the interface for data store implementations.
    """

    #: Default text encoding used across all backends when the caller doesn't specify one.
    DEFAULT_ENCODING = "utf-8"

    @abstractmethod
    def read_file(self, path: str, encoding: Optional[str] = None) -> Union[str, bytes]:
        """
        Read contents of a file from the data store.

        Args:
            path: Path to the file to read
            encoding: Text encoding to decode with. If None, returns raw bytes.

        Returns:
            File contents as bytes, or as str if `encoding` is given.

        Raises:
            IOError: If file cannot be read
        """
        pass

    @abstractmethod
    def write_file(self, path: str, data: Any, encoding: Optional[str] = None) -> None:
        """
        Write data to a file in the data store.

        Args:
            path: Path where to write the file
            data: Data to write (str, bytes, or a readable binary stream)
            encoding: Text encoding used when `data` is a str.
                Defaults to DEFAULT_ENCODING ("utf-8") if not given.

        Raises:
            IOError: If file cannot be written
        """
        pass

    @abstractmethod
    def file_exists(self, path: str) -> bool:
        pass

    @abstractmethod
    def list_files(self, path: str) -> List[str]:
        pass

    @abstractmethod
    def walk(self, top: str) -> Generator:
        pass

    @abstractmethod
    def open(self, file: str, mode: str = "r", encoding: Optional[str] = None) -> IO:
        """
        Context manager for file operations.

        Args:
            file: Path to the file
            mode: File mode ('r', 'w', 'rb', 'wb')
            encoding: Text encoding for text modes ('r'/'w'); ignored for binary
                modes. Defaults to DEFAULT_ENCODING ("utf-8") if not given.

        Yields:
            File-like object

        Raises:
            IOError: If file cannot be opened
        """
        pass

    @abstractmethod
    def is_file(self, path: str) -> bool:
        pass

    @abstractmethod
    def is_dir(self, path: str) -> bool:
        pass

    @abstractmethod
    def remove(self, path: str) -> None:
        pass

    @abstractmethod
    def rmdir(self, dir: str) -> None:
        pass
