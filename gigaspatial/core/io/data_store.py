from abc import ABC, abstractmethod
from typing import Any, List, Generator, Union


class DataStore(ABC):
    """
    Abstract base class defining the interface for data store implementations.
    This class serves as a parent for both local and cloud-based storage solutions.
    """

    @abstractmethod
    def read_file(self, path: str) -> Any:
        """
        Read contents of a file from the data store.

        Args:
            path: Path to the file to read

        Returns:
            Contents of the file

        Raises:
            IOError: If file cannot be read
        """
        pass

    @abstractmethod
    def write_file(self, path: str, data: Any) -> None:
        """
        Write data to a file in the data store.

        Args:
            path: Path where to write the file
            data: Data to write to the file

        Raises:
            IOError: If file cannot be written
        """
        pass

    @abstractmethod
    def file_exists(self, path: str) -> bool:
        """
        Check if a file exists in the data store.

        Args:
            path: Path to check

        Returns:
            True if file exists, False otherwise
        """
        pass

    @abstractmethod
    def list_files(self, path: str) -> List[str]:
        """
        List all files in a directory.

        Args:
            path: Directory path to list

        Returns:
            List of file paths in the directory
        """
        pass

    @abstractmethod
    def walk(self, top: str) -> Generator:
        """
        Walk through directory tree, similar to os.walk().

        Args:
            top: Starting directory for the walk

        Returns:
            Generator yielding tuples of (dirpath, dirnames, filenames)
        """
        pass

    @abstractmethod
    def open(self, file: str, mode: str = "r") -> Union[str, bytes]:
        """
        Context manager for file operations.

        Args:
            file: Path to the file
            mode: File mode ('r', 'w', 'rb', 'wb')

        Yields:
            File-like object

        Raises:
            IOError: If file cannot be opened
        """
        pass

    @abstractmethod
    def is_file(self, path: str) -> bool:
        """
        Check if path points to a file.

        Args:
            path: Path to check

        Returns:
            True if path is a file, False otherwise
        """
        pass

    @abstractmethod
    def is_dir(self, path: str) -> bool:
        """
        Check if path points to a directory.

        Args:
            path: Path to check

        Returns:
            True if path is a directory, False otherwise
        """
        pass

    @abstractmethod
    def remove(self, path: str) -> None:
        """
        Remove a file.

        Args:
            path: Path to the file to remove

        Raises:
            IOError: If file cannot be removed
        """
        pass

    @abstractmethod
    def rmdir(self, dir: str) -> None:
        """
        Remove a directory and all its contents.

        Args:
            dir: Path to the directory to remove

        Raises:
            IOError: If directory cannot be removed
        """
        pass
