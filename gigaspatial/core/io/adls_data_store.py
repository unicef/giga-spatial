from azure.storage.blob import BlobServiceClient
import io
import contextlib
import logging
import os
from typing import Union, Optional

from .data_store import DataStore
from gigaspatial.config import config

logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(
    logging.WARNING
)


class ADLSDataStore(DataStore):
    """
    An implementation of DataStore for Azure Data Lake Storage.
    """

    def __init__(
        self,
        container: str = config.ADLS_CONTAINER_NAME,
        connection_string: str = config.ADLS_CONNECTION_STRING,
    ):
        """
        Create a new instance of ADLSDataStore
        :param container: The name of the container in ADLS to interact with.
        """
        self.blob_service_client = BlobServiceClient.from_connection_string(
            connection_string
        )
        self.container_client = self.blob_service_client.get_container_client(
            container=container
        )
        self.container = container

    def read_file(self, path: str, encoding: Optional[str] = None) -> Union[str, bytes]:
        """
        Read file with flexible encoding support.

        :param path: Path to the file in blob storage
        :param encoding: File encoding (optional)
        :return: File contents as string or bytes
        """
        try:
            blob_client = self.container_client.get_blob_client(path)
            blob_data = blob_client.download_blob().readall()

            # If no encoding specified, return raw bytes
            if encoding is None:
                return blob_data

            # If encoding is specified, decode the bytes
            return blob_data.decode(encoding)

        except Exception as e:
            raise IOError(f"Error reading file {path}: {e}")

    def write_file(self, path: str, data) -> None:
        """
        Write file with support for content type and improved type handling.

        :param path: Destination path in blob storage
        :param data: File contents
        """
        blob_client = self.blob_service_client.get_blob_client(
            container=self.container, blob=path, snapshot=None
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
            blob_client = self.container_client.get_blob_client(blob_path)
            with open(file_path, "rb") as data:
                blob_client.upload_blob(data, overwrite=True)
            print(f"Uploaded {file_path} to {blob_path}")
        except Exception as e:
            print(f"Failed to upload {file_path}: {e}")

    def upload_directory(self, dir_path, blob_dir_path):
        """Uploads all files from a directory to Azure Blob Storage."""
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                local_file_path = os.path.join(root, file)
                relative_path = os.path.relpath(local_file_path, dir_path)
                blob_file_path = os.path.join(blob_dir_path, relative_path).replace(
                    "\\", "/"
                )

                self.upload_file(local_file_path, blob_file_path)

    def download_directory(self, blob_dir_path: str, local_dir_path: str):
        """Downloads all files from a directory in Azure Blob Storage to a local directory."""
        try:
            # Ensure the local directory exists
            os.makedirs(local_dir_path, exist_ok=True)

            # List all files in the blob directory
            blob_items = self.container_client.list_blobs(
                name_starts_with=blob_dir_path
            )

            for blob_item in blob_items:
                # Get the relative path of the blob file
                relative_path = os.path.relpath(blob_item.name, blob_dir_path)
                # Construct the local file path
                local_file_path = os.path.join(local_dir_path, relative_path)
                # Create directories if needed
                os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

                # Download the blob to the local file
                blob_client = self.container_client.get_blob_client(blob_item.name)
                with open(local_file_path, "wb") as file:
                    file.write(blob_client.download_blob().readall())

            print(f"Downloaded directory {blob_dir_path} to {local_dir_path}")
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

                # Create a source blob client
                source_blob_client = self.container_client.get_blob_client(blob.name)

                # Create a destination blob client
                destination_blob_client = self.container_client.get_blob_client(
                    new_blob_path
                )

                # Start the copy operation
                destination_blob_client.start_copy_from_url(source_blob_client.url)

            print(f"Copied directory from {source_dir} to {destination_dir}")
        except Exception as e:
            print(f"Failed to copy directory {source_dir}: {e}")

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

    def list_files(self, path: str):
        blob_items = self.container_client.list_blobs(name_starts_with=path)
        return [item["name"] for item in blob_items]

    def walk(self, top: str):
        top = top.rstrip("/") + "/"
        blob_items = self.container_client.list_blobs(name_starts_with=top)
        blobs = [item["name"] for item in blob_items]
        for blob in blobs:
            dirpath, filename = os.path.split(blob)
            yield (dirpath, [], [filename])

    def list_directories(self, path: str) -> list:
        """List only directory names (not files) from a given path in ADLS."""
        search_path = path.rstrip("/") + "/" if path else ""

        blob_items = self.container_client.list_blobs(name_starts_with=search_path)

        directories = set()

        for blob_item in blob_items:
            # Get the relative path from the search path
            relative_path = blob_item.name[len(search_path) :]

            # Skip if it's empty (shouldn't happen but just in case)
            if not relative_path:
                continue

            # If there's a "/" in the relative path, it means there's a subdirectory
            if "/" in relative_path:
                # Get the first directory name
                dir_name = relative_path.split("/")[0]
                directories.add(dir_name)

        return sorted(list(directories))

    @contextlib.contextmanager
    def open(self, path: str, mode: str = "r"):
        """
        Context manager for file operations with enhanced mode support.

        :param path: File path in blob storage
        :param mode: File open mode (r, rb, w, wb)
        """
        if mode == "w":
            file = io.StringIO()
            yield file
            self.write_file(path, file.getvalue())

        elif mode == "wb":
            file = io.BytesIO()
            yield file
            self.write_file(path, file.getvalue())

        elif mode == "r":
            data = self.read_file(path, encoding="UTF-8")
            file = io.StringIO(data)
            yield file

        elif mode == "rb":
            data = self.read_file(path)
            file = io.BytesIO(data)
            yield file

    def get_file_metadata(self, path: str) -> dict:
        """
        Retrieve comprehensive file metadata.

        :param path: File path in blob storage
        :return: File metadata dictionary
        """
        blob_client = self.container_client.get_blob_client(path)
        properties = blob_client.get_blob_properties()

        return {
            "name": path,
            "size_bytes": properties.size,
            "content_type": properties.content_settings.content_type,
            "last_modified": properties.last_modified,
            "etag": properties.etag,
        }

    def is_file(self, path: str) -> bool:
        return self.file_exists(path)

    def is_dir(self, path: str) -> bool:
        dir_path = path.rstrip("/") + "/"

        existing_blobs = self.list_files(dir_path)

        if len(existing_blobs) > 1:
            return True
        elif len(existing_blobs) == 1:
            if existing_blobs[0] != path.rstrip("/"):
                return True

        return False

    def rmdir(self, dir: str) -> None:
        blobs = self.list_files(dir)
        self.container_client.delete_blobs(*blobs)

    def mkdir(self, path: str, exist_ok: bool = False) -> None:
        """
        Create a directory in Azure Blob Storage.

        In ADLS, directories are conceptual and created by adding a placeholder blob.

        :param path: Path of the directory to create
        :param exist_ok: If False, raise an error if the directory already exists
        """
        dir_path = path.rstrip("/") + "/"

        existing_blobs = list(self.list_files(dir_path))

        if existing_blobs and not exist_ok:
            raise FileExistsError(f"Directory {path} already exists")

        # Create a placeholder blob to represent the directory
        placeholder_blob_path = os.path.join(dir_path, ".placeholder")

        # Only create placeholder if it doesn't already exist
        if not self.file_exists(placeholder_blob_path):
            placeholder_content = (
                b"This is a placeholder blob to represent a directory."
            )
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container, blob=placeholder_blob_path
            )
            blob_client.upload_blob(placeholder_content, overwrite=True)

    def remove(self, path: str) -> None:
        blob_client = self.blob_service_client.get_blob_client(
            container=self.container, blob=path, snapshot=None
        )
        if blob_client.exists():
            blob_client.delete_blob()
