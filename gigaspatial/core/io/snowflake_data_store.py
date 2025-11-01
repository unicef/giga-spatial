import snowflake.connector
from snowflake.connector import DictCursor
import tempfile
import os
import io
import contextlib
import logging
from typing import Union, Optional, List, Generator, Tuple
from pathlib import Path

from .data_store import DataStore
from gigaspatial.config import config

logging.getLogger("snowflake.connector").setLevel(logging.WARNING)


class SnowflakeDataStore(DataStore):
    """
    An implementation of DataStore for Snowflake internal stages.
    Uses Snowflake stages for file storage and retrieval.
    """

    def __init__(
        self,
        account: str = config.SNOWFLAKE_ACCOUNT,
        user: str = config.SNOWFLAKE_USER,
        password: str = config.SNOWFLAKE_PASSWORD,
        warehouse: str = config.SNOWFLAKE_WAREHOUSE,
        database: str = config.SNOWFLAKE_DATABASE,
        schema: str = config.SNOWFLAKE_SCHEMA,
        stage_name: str = config.SNOWFLAKE_STAGE_NAME,
    ):
        """
        Create a new instance of SnowflakeDataStore.

        :param account: Snowflake account identifier
        :param user: Snowflake username
        :param password: Snowflake password
        :param warehouse: Snowflake warehouse name
        :param database: Snowflake database name
        :param schema: Snowflake schema name
        :param stage_name: Name of the Snowflake stage to use for file storage
        """
        if not all([account, user, password, warehouse, database, schema, stage_name]):
            raise ValueError(
                "Snowflake connection parameters (account, user, password, warehouse, "
                "database, schema, stage_name) must be provided via config or constructor."
            )

        self.account = account
        self.user = user
        self.password = password
        self.warehouse = warehouse
        self.database = database
        self.schema = schema
        self.stage_name = stage_name

        # Create connection
        self.connection = self._create_connection()
        self.logger = config.get_logger(self.__class__.__name__)

        # Temporary directory for file operations
        self._temp_dir = tempfile.mkdtemp()

    def _create_connection(self):
        """Create and return a Snowflake connection."""
        conn_params = {
            "account": self.account,
            "user": self.user,
            "password": self.password,
            "warehouse": self.warehouse,
            "database": self.database,
            "schema": self.schema,
        }

        connection = snowflake.connector.connect(**conn_params)
        
        # Explicitly set the database and schema context
        # This ensures the session knows which database/schema to use
        cursor = connection.cursor()
        try:
            # Use database first
            cursor.execute(f'USE DATABASE "{self.database}"')
            # Then use schema (don't need to specify database again)
            cursor.execute(f'USE SCHEMA "{self.schema}"')
            cursor.close()
        except Exception as e:
            cursor.close()
            connection.close()
            error_msg = (
                f"Failed to set database/schema context: {e}\n"
                f"Make sure the database '{self.database}' and schema '{self.schema}' exist.\n"
                f"You may need to run the setup_snowflake_test.sql script first.\n"
                f"Current config - Database: {self.database}, Schema: {self.schema}, Stage: {self.stage_name}"
            )
            raise IOError(error_msg)
        
        return connection

    def _ensure_connection(self):
        """Ensure the connection is active, reconnect if needed."""
        try:
            self.connection.cursor().execute("SELECT 1")
        except Exception:
            self.connection = self._create_connection()

    def _get_stage_path(self, path: str) -> str:
        """Convert a file path to a Snowflake stage path."""
        # Remove leading/trailing slashes and normalize
        path = path.strip("/")
        # Stage paths use forward slashes and @stage_name/ prefix
        return f"@{self.stage_name}/{path}"

    def _normalize_path(self, path: str) -> str:
        """Normalize path for Snowflake stage operations."""
        return path.strip("/").replace("\\", "/")

    def read_file(self, path: str, encoding: Optional[str] = None) -> Union[str, bytes]:
        """
        Read file from Snowflake stage.

        :param path: Path to the file in the stage
        :param encoding: File encoding (optional)
        :return: File contents as string or bytes
        """
        self._ensure_connection()
        cursor = self.connection.cursor(DictCursor)

        try:
            normalized_path = self._normalize_path(path)
            stage_path = self._get_stage_path(normalized_path)

            # Create temporary directory for download
            temp_download_dir = os.path.join(self._temp_dir, "downloads")
            os.makedirs(temp_download_dir, exist_ok=True)

            # Download file from stage using GET command
            # GET command: GET <stage_path> file://<local_path>
            temp_dir_normalized = temp_download_dir.replace("\\", "/")
            if not temp_dir_normalized.endswith("/"):
                temp_dir_normalized += "/"
            
            get_command = f"GET {stage_path} 'file://{temp_dir_normalized}'"
            cursor.execute(get_command)

            # Find the downloaded file (Snowflake may add prefixes/suffixes or preserve structure)
            downloaded_files = []
            for root, dirs, files in os.walk(temp_download_dir):
                for f in files:
                    file_path = os.path.join(root, f)
                    # Check if this file matches our expected filename
                    if os.path.basename(normalized_path) in f or normalized_path.endswith(f):
                        downloaded_files.append(file_path)

            if not downloaded_files:
                raise FileNotFoundError(f"File not found in stage: {path}")

            # Read the first matching file
            downloaded_path = downloaded_files[0]
            with open(downloaded_path, "rb") as f:
                data = f.read()

            # Clean up
            os.remove(downloaded_path)
            # Clean up empty directories
            try:
                if os.path.exists(temp_download_dir) and not os.listdir(temp_download_dir):
                    os.rmdir(temp_download_dir)
            except OSError:
                pass

            # Decode if encoding is specified
            if encoding:
                return data.decode(encoding)
            return data

        except Exception as e:
            raise IOError(f"Error reading file {path} from Snowflake stage: {e}")
        finally:
            cursor.close()

    def write_file(self, path: str, data: Union[bytes, str]) -> None:
        """
        Write file to Snowflake stage.

        :param path: Destination path in the stage
        :param data: File contents
        """
        self._ensure_connection()
        cursor = self.connection.cursor()

        try:
            # Convert to bytes if string
            if isinstance(data, str):
                binary_data = data.encode("utf-8")
            elif isinstance(data, bytes):
                binary_data = data
            else:
                raise ValueError('Unsupported data type. Only "bytes" or "string" accepted')

            normalized_path = self._normalize_path(path)

            # Write to temporary file first
            # Use the full path structure for the temp file to preserve directory structure
            temp_file_path = os.path.join(self._temp_dir, normalized_path)
            os.makedirs(os.path.dirname(temp_file_path), exist_ok=True)

            with open(temp_file_path, "wb") as f:
                f.write(binary_data)

            # Upload to stage using PUT command
            # Snowflake PUT requires the local file path and the target stage path
            # Convert Windows paths to Unix-style for Snowflake
            temp_file_normalized = os.path.abspath(temp_file_path).replace("\\", "/")
            
            # PUT command: PUT 'file://<absolute_local_path>' @stage_name/<path>
            # The file will be stored at the specified path in the stage
            stage_target = f"@{self.stage_name}/"
            if "/" in normalized_path:
                # Include directory structure in stage path
                dir_path = os.path.dirname(normalized_path)
                stage_target = f"@{self.stage_name}/{dir_path}/"
            
            # Snowflake PUT syntax: PUT 'file://<path>' @stage/path
            put_command = f"PUT 'file://{temp_file_normalized}' {stage_target} OVERWRITE=TRUE AUTO_COMPRESS=FALSE"
            cursor.execute(put_command)

            # Clean up temp file
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
                # Clean up empty directories if they were created
                try:
                    temp_dir = os.path.dirname(temp_file_path)
                    if temp_dir != self._temp_dir and os.path.exists(temp_dir):
                        os.rmdir(temp_dir)
                except OSError:
                    pass  # Directory not empty or other error, ignore

        except Exception as e:
            raise IOError(f"Error writing file {path} to Snowflake stage: {e}")
        finally:
            cursor.close()

    def upload_file(self, file_path: str, stage_path: str):
        """
        Uploads a single file from local filesystem to Snowflake stage.

        :param file_path: Local file path
        :param stage_path: Destination path in the stage
        """
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Local file not found: {file_path}")

            # Read the file
            with open(file_path, "rb") as f:
                data = f.read()

            # Write to stage using write_file
            self.write_file(stage_path, data)
            self.logger.info(f"Uploaded {file_path} to {stage_path}")
        except Exception as e:
            self.logger.error(f"Failed to upload {file_path}: {e}")
            raise

    def upload_directory(self, dir_path: str, stage_dir_path: str):
        """
        Uploads all files from a local directory to Snowflake stage.

        :param dir_path: Local directory path
        :param stage_dir_path: Destination directory path in the stage
        """
        if not os.path.isdir(dir_path):
            raise NotADirectoryError(f"Local directory not found: {dir_path}")

        for root, dirs, files in os.walk(dir_path):
            for file in files:
                local_file_path = os.path.join(root, file)
                relative_path = os.path.relpath(local_file_path, dir_path)
                # Normalize path separators for stage
                stage_file_path = os.path.join(stage_dir_path, relative_path).replace("\\", "/")

                self.upload_file(local_file_path, stage_file_path)

    def download_directory(self, stage_dir_path: str, local_dir_path: str):
        """
        Downloads all files from a Snowflake stage directory to a local directory.

        :param stage_dir_path: Source directory path in the stage
        :param local_dir_path: Destination local directory path
        """
        try:
            # Ensure the local directory exists
            os.makedirs(local_dir_path, exist_ok=True)

            # List all files in the stage directory
            files = self.list_files(stage_dir_path)

            for file_path in files:
                # Get the relative path from the stage directory
                if stage_dir_path:
                    if file_path.startswith(stage_dir_path):
                        relative_path = file_path[len(stage_dir_path):].lstrip("/")
                    else:
                        # If file_path doesn't start with stage_dir_path, use it as is
                        relative_path = os.path.basename(file_path)
                else:
                    relative_path = file_path

                # Construct the local file path
                local_file_path = os.path.join(local_dir_path, relative_path)
                # Create directories if needed
                os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

                # Download the file
                data = self.read_file(file_path)
                with open(local_file_path, "wb") as f:
                    if isinstance(data, str):
                        f.write(data.encode("utf-8"))
                    else:
                        f.write(data)

            self.logger.info(f"Downloaded directory {stage_dir_path} to {local_dir_path}")
        except Exception as e:
            self.logger.error(f"Failed to download directory {stage_dir_path}: {e}")
            raise

    def copy_directory(self, source_dir: str, destination_dir: str):
        """
        Copies all files from a source directory to a destination directory within the stage.

        :param source_dir: Source directory path in the stage
        :param destination_dir: Destination directory path in the stage
        """
        try:
            # Normalize directory paths
            source_dir = source_dir.rstrip("/")
            destination_dir = destination_dir.rstrip("/")

            # List all files in the source directory
            files = self.list_files(source_dir)

            for file_path in files:
                # Get relative path from source directory
                if source_dir:
                    if file_path.startswith(source_dir):
                        relative_path = file_path[len(source_dir):].lstrip("/")
                    else:
                        relative_path = os.path.basename(file_path)
                else:
                    relative_path = file_path

                # Construct the destination file path
                if destination_dir:
                    dest_file_path = f"{destination_dir}/{relative_path}".replace("//", "/")
                else:
                    dest_file_path = relative_path

                # Copy each file
                self.copy_file(file_path, dest_file_path, overwrite=True)

            self.logger.info(f"Copied directory from {source_dir} to {destination_dir}")
        except Exception as e:
            self.logger.error(f"Failed to copy directory {source_dir}: {e}")
            raise

    def copy_file(
            self, source_path: str, destination_path: str, overwrite: bool = False
    ):
        """
        Copies a single file within the Snowflake stage.

        :param source_path: Source file path in the stage
        :param destination_path: Destination file path in the stage
        :param overwrite: If True, overwrite the destination file if it already exists
        """
        try:
            if not self.file_exists(source_path):
                raise FileNotFoundError(f"Source file not found: {source_path}")

            if self.file_exists(destination_path) and not overwrite:
                raise FileExistsError(
                    f"Destination file already exists and overwrite is False: {destination_path}"
                )

            # Read from source and write to destination
            data = self.read_file(source_path)
            self.write_file(destination_path, data)

            self.logger.info(f"Copied file from {source_path} to {destination_path}")
        except Exception as e:
            self.logger.error(f"Failed to copy file {source_path}: {e}")
            raise

    def exists(self, path: str) -> bool:
        """Check if a path exists (file or directory)."""
        return self.file_exists(path) or self.is_dir(path)

    def file_exists(self, path: str) -> bool:
        """
        Check if a file exists in the Snowflake stage.

        :param path: Path to check
        :return: True if file exists, False otherwise
        """
        self._ensure_connection()
        cursor = self.connection.cursor(DictCursor)

        try:
            normalized_path = self._normalize_path(path)
            stage_path = self._get_stage_path(normalized_path)

            # List files in stage with the given path pattern
            list_command = f"LIST {stage_path}"
            cursor.execute(list_command)
            results = cursor.fetchall()

            # Check if exact file exists
            for result in results:
                if result["name"].endswith(normalized_path) or result["name"] == stage_path:
                    return True

            return False

        except Exception as e:
            self.logger.warning(f"Error checking file existence {path}: {e}")
            return False
        finally:
            cursor.close()

    def file_size(self, path: str) -> float:
        """
        Get the size of a file in kilobytes.

        :param path: File path in the stage
        :return: File size in kilobytes
        """
        self._ensure_connection()
        cursor = self.connection.cursor(DictCursor)

        try:
            normalized_path = self._normalize_path(path)
            stage_path = self._get_stage_path(normalized_path)

            # LIST command returns file metadata including size
            list_command = f"LIST {stage_path}"
            cursor.execute(list_command)
            results = cursor.fetchall()

            # Find the matching file and get its size
            for result in results:
                file_path = result["name"]
                if normalized_path in file_path.lower() or file_path.endswith(normalized_path):
                    # Size is in bytes, convert to kilobytes
                    size_bytes = result.get("size", 0)
                    size_kb = size_bytes / 1024.0
                    return size_kb

            raise FileNotFoundError(f"File not found: {path}")
        except Exception as e:
            self.logger.error(f"Error getting file size for {path}: {e}")
            raise
        finally:
            cursor.close()

    def list_files(self, path: str) -> List[str]:
        """
        List all files in a directory within the Snowflake stage.

        :param path: Directory path to list
        :return: List of file paths
        """
        self._ensure_connection()
        cursor = self.connection.cursor(DictCursor)

        try:
            normalized_path = self._normalize_path(path)
            stage_path = self._get_stage_path(normalized_path)

            # List files in stage
            list_command = f"LIST {stage_path}"
            cursor.execute(list_command)
            results = cursor.fetchall()

            # Extract file paths relative to the base stage path
            files = []
            for result in results:
                file_path = result["name"]
                # Snowflake LIST returns names in lowercase without @ symbol
                # Remove stage prefix to get relative path
                # Check both @stage_name/ and lowercase stage_name/ formats
                stage_prefixes = [
                    f"@{self.stage_name}/",
                    f"{self.stage_name.lower()}/",
                    f"@{self.stage_name.lower()}/",
                ]
                
                for prefix in stage_prefixes:
                    if file_path.startswith(prefix):
                        relative_path = file_path[len(prefix):]
                        files.append(relative_path)
                        break
                else:
                    # If no prefix matches, try to extract path after stage name
                    # Sometimes stage name might be in different case
                    stage_name_lower = self.stage_name.lower()
                    if stage_name_lower in file_path.lower():
                        # Find the position after the stage name
                        idx = file_path.lower().find(stage_name_lower)
                        if idx != -1:
                            # Get everything after stage name and '/'
                            after_stage = file_path[idx + len(stage_name_lower):].lstrip("/")
                            if after_stage.startswith(normalized_path):
                                relative_path = after_stage
                                files.append(relative_path)

            return files

        except Exception as e:
            self.logger.warning(f"Error listing files in {path}: {e}")
            return []
        finally:
            cursor.close()

    def walk(self, top: str) -> Generator[Tuple[str, List[str], List[str]], None, None]:
        """
        Walk through directory tree in Snowflake stage, similar to os.walk().

        :param top: Starting directory for the walk
        :return: Generator yielding tuples of (dirpath, dirnames, filenames)
        """
        try:
            normalized_top = self._normalize_path(top)
            
            # Use list_files to get all files (it handles path parsing correctly)
            all_files = self.list_files(normalized_top)
            
            # Organize into directory structure
            dirs = {}
            
            for file_path in all_files:
                # Ensure we're working with paths relative to the top
                if normalized_top and not file_path.startswith(normalized_top):
                    continue
                
                # Get relative path from top
                if normalized_top and file_path.startswith(normalized_top):
                    relative_path = file_path[len(normalized_top):].lstrip("/")
                else:
                    relative_path = file_path
                
                if not relative_path:
                    continue
                
                # Get directory and filename
                if "/" in relative_path:
                    dir_path, filename = os.path.split(relative_path)
                    full_dir_path = f"{normalized_top}/{dir_path}" if normalized_top else dir_path
                    if full_dir_path not in dirs:
                        dirs[full_dir_path] = []
                    dirs[full_dir_path].append(filename)
                else:
                    # File in root of the top directory
                    if normalized_top not in dirs:
                        dirs[normalized_top] = []
                    dirs[normalized_top].append(relative_path)

            # Yield results in os.walk format
            for dir_path, files in dirs.items():
                # Extract subdirectories (simplified - Snowflake stages are flat)
                subdirs = []
                yield (dir_path, subdirs, files)
                
        except Exception as e:
            self.logger.warning(f"Error walking directory {top}: {e}")
            yield (top, [], [])

    def list_directories(self, path: str) -> List[str]:
        """
        List only directory names (not files) from a given path in the stage.

        :param path: Directory path to list
        :return: List of directory names
        """
        normalized_path = self._normalize_path(path)
        files = self.list_files(normalized_path)

        directories = set()

        for file_path in files:
            # Get relative path from the search path
            if normalized_path:
                if file_path.startswith(normalized_path):
                    relative_path = file_path[len(normalized_path):].lstrip("/")
                else:
                    continue
            else:
                relative_path = file_path

            # Skip if empty
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
        Context manager for file operations.

        :param path: File path in Snowflake stage
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

        else:
            raise ValueError(f"Unsupported mode: {mode}")

    def get_file_metadata(self, path: str) -> dict:
        """
        Retrieve comprehensive file metadata from Snowflake stage.

        :param path: File path in the stage
        :return: File metadata dictionary
        """
        self._ensure_connection()
        cursor = self.connection.cursor(DictCursor)

        try:
            normalized_path = self._normalize_path(path)
            stage_path = self._get_stage_path(normalized_path)

            # LIST command returns file metadata
            list_command = f"LIST {stage_path}"
            cursor.execute(list_command)
            results = cursor.fetchall()

            # Find the matching file
            for result in results:
                file_path = result["name"]
                if normalized_path in file_path.lower() or file_path.endswith(normalized_path):
                    return {
                        "name": path,
                        "size_bytes": result.get("size", 0),
                        "last_modified": result.get("last_modified"),
                        "md5": result.get("md5"),
                    }

            raise FileNotFoundError(f"File not found: {path}")
        except Exception as e:
            self.logger.error(f"Error getting file metadata for {path}: {e}")
            raise
        finally:
            cursor.close()

    def is_file(self, path: str) -> bool:
        """Check if path points to a file."""
        return self.file_exists(path)

    def is_dir(self, path: str) -> bool:
        """Check if path points to a directory."""
        # First check if it's actually a file (exact match)
        if self.file_exists(path):
            return False
        
        # In Snowflake stages, directories are conceptual
        # Check if there are files with this path prefix
        normalized_path = self._normalize_path(path)
        files = self.list_files(normalized_path)
        
        # Filter out files that are exact matches (they're files, not directories)
        exact_match = any(f == normalized_path or f == path for f in files)
        if exact_match:
            return False
            
        return len(files) > 0

    def rmdir(self, dir: str) -> None:
        """
        Remove a directory and all its contents from the Snowflake stage.

        :param dir: Path to the directory to remove
        """
        self._ensure_connection()
        cursor = self.connection.cursor()

        try:
            normalized_dir = self._normalize_path(dir)
            stage_path = self._get_stage_path(normalized_dir)

            # Remove all files in the directory
            remove_command = f"REMOVE {stage_path}"
            cursor.execute(remove_command)

        except Exception as e:
            raise IOError(f"Error removing directory {dir}: {e}")
        finally:
            cursor.close()

    def mkdir(self, path: str, exist_ok: bool = False) -> None:
        """
        Create a directory in Snowflake stage.

        In Snowflake stages, directories are created implicitly when files are uploaded.
        This method creates a placeholder file if the directory doesn't exist.

        :param path: Path of the directory to create
        :param exist_ok: If False, raise an error if the directory already exists
        """
        # Check if directory already exists
        if self.is_dir(path) and not exist_ok:
            raise FileExistsError(f"Directory {path} already exists")

        # Create a placeholder file to ensure directory exists
        placeholder_path = os.path.join(path, ".placeholder").replace("\\", "/")
        if not self.file_exists(placeholder_path):
            self.write_file(placeholder_path, b"Placeholder file for directory")

    def remove(self, path: str) -> None:
        """
        Remove a file from the Snowflake stage.

        :param path: Path to the file to remove
        """
        self._ensure_connection()
        cursor = self.connection.cursor()

        try:
            normalized_path = self._normalize_path(path)
            stage_path = self._get_stage_path(normalized_path)

            remove_command = f"REMOVE {stage_path}"
            cursor.execute(remove_command)

        except Exception as e:
            raise IOError(f"Error removing file {path}: {e}")
        finally:
            cursor.close()

    def rename(
        self,
        source_path: str,
        destination_path: str,
        overwrite: bool = False,
        delete_source: bool = True,
    ) -> None:
        """
        Rename (move) a single file by copying to the new path and deleting the source.
        
        :param source_path: Existing file path in the stage
        :param destination_path: Target file path in the stage
        :param overwrite: Overwrite destination if it already exists
        :param delete_source: Delete original after successful copy
        """
        if not self.file_exists(source_path):
            raise FileNotFoundError(f"Source file not found: {source_path}")
        
        if self.file_exists(destination_path) and not overwrite:
            raise FileExistsError(
                f"Destination already exists and overwrite is False: {destination_path}"
            )
        
        # Copy file to new location
        self.copy_file(source_path, destination_path, overwrite=overwrite)
        
        # Delete source if requested
        if delete_source:
            self.remove(source_path)

    def close(self):
        """Close the Snowflake connection."""
        if self.connection:
            self.connection.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

