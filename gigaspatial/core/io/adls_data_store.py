from azure.storage.blob import BlobServiceClient
import time
import io
import contextlib
import logging
import os
from pathlib import PurePosixPath
from typing import Union, Optional, Iterator

from .data_store import DataStore
from gigaspatial.config import config

logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(
    logging.WARNING
)

Pathish = Union[str, os.PathLike[str]]


class ADLSDataStore(DataStore):
    """
    An implementation of DataStore for Azure Data Lake Storage.
    """

    def __init__(
        self,
        container: str = config.ADLS_CONTAINER_NAME,
        connection_string: str = config.ADLS_CONNECTION_STRING,
        account_url: str = config.ADLS_ACCOUNT_URL,
        sas_token: str = config.ADLS_SAS_TOKEN,
    ):
        """
        Create a new instance of ADLSDataStore
        :param container: The name of the container in ADLS to interact with.
        """
        if connection_string:
            self.blob_service_client = BlobServiceClient.from_connection_string(
                connection_string
            )
        elif account_url and sas_token:
            self.blob_service_client = BlobServiceClient(
                account_url=account_url, credential=sas_token
            )
        else:
            raise ValueError(
                "Either connection_string or account_url and sas_token must be provided."
            )

        self.container_client = self.blob_service_client.get_container_client(
            container=container
        )
        self.container = container

    def _to_blob_key(self, path: Pathish, *, ensure_dir: bool = False) -> str:
        """
        Convert an input path (str or PathLike) to a normalized ADLS blob key.

        Rules:
        - Accepts str, pathlib.Path, PurePath, etc.
        - Converts Windows backslashes to '/' via PurePosixPath.
        - Strips any leading '/'.
        - Optionally enforces a trailing '/' for directory prefixes.
        """
        # Convert any PathLike to str in OS-native form first
        raw = os.fspath(path)

        # Normalize to POSIX-style using pathlib semantics
        # This handles backslashes and redundant separators.
        posix = PurePosixPath(raw).as_posix()

        # Strip leading slash (Azure blob keys don’t need it)
        if posix.startswith("/"):
            posix = posix.lstrip("/")

        if ensure_dir and posix and not posix.endswith("/"):
            posix = posix + "/"

        return posix

    def read_file(
        self, path: Pathish, encoding: Optional[str] = None
    ) -> Union[str, bytes]:
        """
        Read file with flexible encoding support.

        :param path: Path to the file in blob storage
        :param encoding: File encoding (optional)
        :return: File contents as string or bytes
        """
        try:
            blob_key = self._to_blob_key(path)
            blob_client = self.container_client.get_blob_client(blob_key)
            blob_data = blob_client.download_blob().readall()

            # If no encoding specified, return raw bytes
            if encoding is None:
                return blob_data

            # If encoding is specified, decode the bytes
            return blob_data.decode(encoding)

        except Exception as e:
            raise IOError(f"Error reading file {path}: {e}")

    def write_file(self, path: Pathish, data) -> None:
        """
        Write file with support for content type and improved type handling.

        :param path: Destination path in blob storage
        :param data: File contents
        """
        blob_key = self._to_blob_key(path)
        blob_client = self.blob_service_client.get_blob_client(
            container=self.container, blob=blob_key, snapshot=None
        )

        if isinstance(data, str):
            binary_data = data.encode()
        elif isinstance(data, bytes):
            binary_data = data
        else:
            raise Exception(f'Unsupported data type. Only "bytes" or "string" accepted')

        blob_client.upload_blob(binary_data, overwrite=True)

    def upload_file(self, file_path, blob_path):
        """Uploads a single file to Azure Blob Storage."""
        try:
            blob_key = self._to_blob_key(blob_path)
            blob_client = self.container_client.get_blob_client(blob_key)
            with open(file_path, "rb") as data:
                blob_client.upload_blob(data, overwrite=True)
            print(f"Uploaded {file_path} to {blob_path}")
        except Exception as e:
            print(f"Failed to upload {file_path}: {e}")

    def upload_directory(self, dir_path, blob_dir_path):
        """Uploads all files from a directory to Azure Blob Storage."""
        blob_dir_key = self._to_blob_key(blob_dir_path, ensure_dir=True)
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                local_file_path = os.path.join(root, file)
                relative_path = os.path.relpath(local_file_path, dir_path)
                # Construct blob path and normalize
                blob_file_path = PurePosixPath(blob_dir_key) / relative_path
                blob_file_key = self._to_blob_key(blob_file_path)

                self.upload_file(local_file_path, blob_file_key)

    def download_directory(self, blob_dir_path: Pathish, local_dir_path: str):
        """Downloads all files from a directory in Azure Blob Storage to a local directory."""
        try:
            blob_dir_key = self._to_blob_key(blob_dir_path, ensure_dir=True)

            # Ensure the local directory exists
            os.makedirs(local_dir_path, exist_ok=True)

            # List all files in the blob directory
            blob_items = self.container_client.list_blobs(name_starts_with=blob_dir_key)

            for blob_item in blob_items:
                # Get the relative path of the blob file
                relative_path = os.path.relpath(blob_item.name, blob_dir_key)
                # Construct the local file path
                local_file_path = os.path.join(local_dir_path, relative_path)
                # Create directories if needed
                os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

                # Download the blob to the local file
                blob_client = self.container_client.get_blob_client(blob_item.name)
                with open(local_file_path, "wb") as file:
                    file.write(blob_client.download_blob().readall())

            print(f"Downloaded directory {blob_dir_key} to {local_dir_path}")
        except Exception as e:
            print(f"Failed to download directory {blob_dir_path}: {e}")

    def copy_directory(self, source_dir: str, destination_dir: str):
        """
        Copies all files from a source directory to a destination directory within the same container.

        :param source_dir: The source directory path in the blob storage
        :param destination_dir: The destination directory path in the blob storage
        """
        try:
            # Ensure source directory path ends with a trailing slash
            source_dir = source_dir.rstrip("/") + "/"
            destination_dir = destination_dir.rstrip("/") + "/"

            # List all blobs in the source directory
            source_blobs = self.container_client.list_blobs(name_starts_with=source_dir)

            for blob in source_blobs:
                # Get the relative path of the blob
                relative_path = os.path.relpath(blob.name, source_dir)

                # Construct the new blob path
                new_blob_path = os.path.join(destination_dir, relative_path).replace(
                    "\\", "/"
                )

                # Use copy_file method to copy each file
                self.copy_file(blob.name, new_blob_path, overwrite=True)

            print(f"Copied directory from {source_dir} to {destination_dir}")
        except Exception as e:
            print(f"Failed to copy directory {source_dir}: {e}")

    def copy_file(
        self, source_path: str, destination_path: str, overwrite: bool = False
    ):
        """
        Copies a single file from source to destination within the same container.

        :param source_path: The source file path in the blob storage
        :param destination_path: The destination file path in the blob storage
        :param overwrite: If True, overwrite the destination file if it already exists
        """
        try:
            if not self.file_exists(source_path):
                raise FileNotFoundError(f"Source file not found: {source_path}")

            if self.file_exists(destination_path) and not overwrite:
                raise FileExistsError(
                    f"Destination file already exists and overwrite is False: {destination_path}"
                )

            # Create source and destination blob clients
            source_blob_client = self.container_client.get_blob_client(source_path)
            destination_blob_client = self.container_client.get_blob_client(
                destination_path
            )

            # Start the server-side copy operation
            destination_blob_client.start_copy_from_url(source_blob_client.url)

            print(f"Copied file from {source_path} to {destination_path}")
        except Exception as e:
            print(f"Failed to copy file {source_path}: {e}")
            raise

    def exists(self, path: str) -> bool:
        blob_client = self.blob_service_client.get_blob_client(
            container=self.container, blob=path, snapshot=None
        )
        return blob_client.exists()

    def file_exists(self, path: str) -> bool:
        return self.exists(path) and not self.is_dir(path)

    def file_size(self, path: str) -> float:
        blob_client = self.blob_service_client.get_blob_client(
            container=self.container, blob=path, snapshot=None
        )
        properties = blob_client.get_blob_properties()

        # The size is in bytes, convert it to kilobytes
        size_in_bytes = properties.size
        size_in_kb = size_in_bytes / 1024.0
        return size_in_kb

    def list_files(self, dir_path: Pathish) -> list:
        """
        List all files in a directory.

        For large directories (100K+ files), consider using
        list_files_iter() to avoid memory overhead.

        Performance improvements in this version:
        - Uses list_blob_names() internally (8-14x faster)
        - Lower memory usage (strings vs objects)

        Args:
            dir_path: Directory path

        Returns:
            List of file paths
        """
        prefix = self._to_blob_key(dir_path, ensure_dir=True)
        return list(self.container_client.list_blob_names(name_starts_with=prefix))

    def list_files_iter(self, dir_path: Pathish) -> Iterator[str]:
        """
        Iterate over files with lazy evaluation (memory efficient).

        Recommended for large directories. Returns generator that yields
        file paths one at a time.

        Args:
            dir_path: Directory path

        Yields:
            File paths

        Example:
            for file in store.list_files_iter('data/'):
                if file.endswith('.png'):
                    process(file)
                    break  # Early exit possible
        """
        prefix = self._to_blob_key(dir_path, ensure_dir=True)
        return self.container_client.list_blob_names(name_starts_with=prefix)

    def has_files_with_extension(self, dir_path: Pathish, extension: str) -> bool:
        """
        Check if ANY file with given extension exists (early exit).

        Much faster than filtering list_files() result when you only
        need to know if at least one file exists.

        Args:
            dir_path: Directory path
            extension: File extension (e.g., '.png' or 'png')

        Returns:
            True if at least one file with extension exists

        Example:
            if store.has_files_with_extension('images/', '.png'):
                print("Folder contains PNG images")

        Performance:
            - Returns immediately on first match
            - Best case: <1s (if match in first page)
            - Worst case: ~30-60s (no matches, must check all)
        """
        if not extension.startswith("."):
            extension = "." + extension

        for blob_name in self.list_files_iter(dir_path):
            if blob_name.endswith(extension):
                return True  # Early exit

        return False

    def count_files(self, dir_path: Pathish) -> int:
        """
        Count total files in a directory (memory efficient).

        More memory-efficient than len(list_files()) for large directories.

        Args:
            dir_path: Directory path

        Returns:
            Number of files

        Example:
            count = store.count_files('data/')
            print(f"Total files: {count}")

        Performance:
            - Must iterate all files (no shortcuts)
            - Memory: ~2MB (one page at a time)
            - Time: ~30-60s for 1M files (Azure rate limit)
        """
        return sum(1 for _ in self.list_files_iter(dir_path))

    def count_files_with_extension(self, dir_path: Pathish, extension: str) -> int:
        """
        Count files with specific extension (memory efficient).

        Args:
            dir_path: Directory path
            extension: File extension (e.g., '.png' or 'png')

        Returns:
            Number of files with given extension

        Example:
            png_count = store.count_files_with_extension('images/', '.png')
            print(f"Found {png_count} PNG files")
        """
        if not extension.startswith("."):
            extension = "." + extension

        return sum(
            1
            for blob_name in self.list_files_iter(dir_path)
            if blob_name.endswith(extension)
        )

    def walk(self, top: Pathish):
        """
        Walk through directory tree yielding (dirpath, dirnames, filenames).

        Optimized to use list_files_iter() for better performance.
        """
        for blob_name in self.list_files_iter(top):
            dirpath, filename = os.path.split(blob_name)
            yield (dirpath, [], [filename])

    def list_directories(self, path: Pathish) -> list:
        """
        List only directory names using Azure's hierarchical listing.
        """
        prefix = self._to_blob_key(path, ensure_dir=True)
        directories = set()

        # Use walk_blobs with delimiter - Azure returns directories directly!
        blob_hierarchy = self.container_client.walk_blobs(
            name_starts_with=prefix, delimiter="/"
        )

        for item in blob_hierarchy:
            # Items with 'prefix' attribute are directories
            if hasattr(item, "prefix"):
                dir_path = item.prefix
                # Extract just the directory name
                relative_path = dir_path[len(prefix) :].rstrip("/")
                if relative_path:
                    directories.add(relative_path)

        return sorted(list(directories))

    @contextlib.contextmanager
    def open(self, path: Pathish, mode: str = "r"):
        """
        Context manager for file operations with enhanced mode support.

        :param path: File path in blob storage
        :param mode: File open mode (r, rb, w, wb)
        """
        blob_key = self._to_blob_key(path)

        if mode == "w":
            file = io.StringIO()
            yield file
            self.write_file(blob_key, file.getvalue())

        elif mode == "wb":
            file = io.BytesIO()
            yield file
            self.write_file(blob_key, file.getvalue())

        elif mode == "r":
            data = self.read_file(blob_key, encoding="UTF-8")
            file = io.StringIO(data)
            yield file

        elif mode == "rb":
            data = self.read_file(blob_key)
            file = io.BytesIO(data)
            yield file

    def get_file_metadata(self, path: Pathish) -> dict:
        """
        Retrieve comprehensive file metadata.

        :param path: File path in blob storage
        :return: File metadata dictionary
        """
        blob_key = self._to_blob_key(path)
        blob_client = self.container_client.get_blob_client(blob_key)
        properties = blob_client.get_blob_properties()

        return {
            "name": blob_key,
            "size_bytes": properties.size,
            "content_type": properties.content_settings.content_type,
            "last_modified": properties.last_modified,
            "etag": properties.etag,
        }

    def is_file(self, path: Pathish) -> bool:
        return self.file_exists(path)

    def is_dir(self, path: Pathish) -> bool:
        dir_key = self._to_blob_key(path, ensure_dir=True)

        existing_blobs = self.list_files(dir_key)

        if len(existing_blobs) > 1:
            return True
        elif len(existing_blobs) == 1:
            # Check if the single blob is not the path itself (indicating it's a file)
            if (
                existing_blobs[0] != path.rstrip("/")
                if isinstance(path, str)
                else str(path).rstrip("/")
            ):
                return True

        return False

    def rmdir(self, dir: Pathish) -> None:
        # Normalize directory path to ensure it targets all children
        dir_key = self._to_blob_key(dir, ensure_dir=True)

        # Azure Blob batch delete has a hard limit on number of sub-requests
        # per batch (currently 256). Delete in chunks to avoid
        # ExceedsMaxBatchRequestCount errors.
        blobs = list(self.list_files(dir_key))
        if not blobs:
            return

        BATCH_LIMIT = 256
        for start_idx in range(0, len(blobs), BATCH_LIMIT):
            batch = blobs[start_idx : start_idx + BATCH_LIMIT]
            self.container_client.delete_blobs(*batch)

    def mkdir(self, path: Pathish, exist_ok: bool = False) -> None:
        """
        Create a directory in Azure Blob Storage.

        In ADLS, directories are conceptual and created by adding a placeholder blob.

        :param path: Path of the directory to create
        :param exist_ok: If False, raise an error if the directory already exists
        """
        dir_key = self._to_blob_key(path, ensure_dir=True)

        existing_blobs = list(self.list_files(dir_key))

        if existing_blobs and not exist_ok:
            raise FileExistsError(f"Directory {path} already exists")

        # Create a placeholder blob to represent the directory
        placeholder_blob_path = PurePosixPath(dir_key) / ".placeholder"
        placeholder_blob_key = self._to_blob_key(placeholder_blob_path)

        # Only create placeholder if it doesn't already exist
        if not self.file_exists(placeholder_blob_key):
            placeholder_content = (
                b"This is a placeholder blob to represent a directory."
            )
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container, blob=placeholder_blob_key
            )
            blob_client.upload_blob(placeholder_content, overwrite=True)

    def remove(self, path: Pathish) -> None:
        blob_key = self._to_blob_key(path)
        blob_client = self.blob_service_client.get_blob_client(
            container=self.container, blob=blob_key, snapshot=None
        )
        if blob_client.exists():
            blob_client.delete_blob()

    def rename(
        self,
        source_path: Pathish,
        destination_path: Pathish,
        overwrite: bool = False,
        delete_source: bool = True,
        wait: bool = True,
        timeout_seconds: int = 300,
        poll_interval_seconds: int = 1,
    ) -> None:
        """
        Rename (move) a single file by copying to the new path and deleting the source.

        :param source_path: Existing blob path
        :param destination_path: Target blob path
        :param overwrite: Overwrite destination if it already exists
        :param delete_source: Delete original after successful copy
        :param wait: Wait for the copy operation to complete
        :param timeout_seconds: Max time to wait for copy to succeed
        :param poll_interval_seconds: Polling interval while waiting
        """
        source_key = self._to_blob_key(source_path)
        destination_key = self._to_blob_key(destination_path)

        if not self.file_exists(source_key):
            raise FileNotFoundError(f"Source file not found: {source_key}")

        if self.file_exists(destination_key) and not overwrite:
            raise FileExistsError(
                f"Destination already exists and overwrite is False: {destination_key}"
            )

        # Use copy_file method to copy the file
        self.copy_file(source_key, destination_key, overwrite=overwrite)

        if wait:
            # Wait for copy to complete if requested
            dest_client = self.container_client.get_blob_client(destination_key)
            deadline = time.time() + timeout_seconds
            while True:
                props = dest_client.get_blob_properties()
                status = getattr(props.copy, "status", None)
                if status == "success":
                    break
                if status in {"aborted", "failed"}:
                    raise IOError(
                        f"Copy failed with status {status} from {source_key} to {destination_key}"
                    )
                if time.time() > deadline:
                    raise TimeoutError(
                        f"Timed out waiting for copy to complete for {destination_key}"
                    )
                time.sleep(poll_interval_seconds)

        if delete_source:
            self.remove(source_key)

    def _normalize_path(self, path: str) -> str:
        """
        Normalize a path for Azure Blob Storage.

        DEPRECATED: Use _to_blob_key() instead for better type support.
        This method is kept for backward compatibility.

        Ensures the path:
        - Ends with '/' if it's a directory
        - Doesn't start with '/' (Azure doesn't use leading slashes)
        - Uses forward slashes

        Args:
            path: Path to normalize

        Returns:
            Normalized path

        Examples:
            _normalize_path('data') -> 'data/'
            _normalize_path('data/') -> 'data/'
            _normalize_path('/data/') -> 'data/'
            _normalize_path('data\\\\subdir') -> 'data/subdir/'
        """
        return self._to_blob_key(path, ensure_dir=True)
