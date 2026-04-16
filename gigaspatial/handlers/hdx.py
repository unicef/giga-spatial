"""
Humantarian Data Exchange (HDX) dataset handler.

This module provides specialized handlers for the HDX platform.
It supports searching for datasets, filtering specific resources (e.g., by country
or file format), and downloading/loading diverse humanitarian datasets (CSV,
GeoJSON, Excel, etc.) into vectorized or tabular formats.
"""
import logging
from tqdm import tqdm
from pathlib import Path
from typing import List, Optional, Union, Dict, Any, Iterable
import tempfile

import geopandas as gpd
from pydantic import Field, ConfigDict
from pydantic.dataclasses import dataclass
from shapely.geometry.base import BaseGeometry
from shapely.geometry import Point
import pycountry

from hdx.api.configuration import Configuration
from hdx.data.dataset import Dataset
from hdx.data.resource import Resource

from gigaspatial.core.io.data_store import DataStore
from gigaspatial.core.io.readers import read_dataset
from gigaspatial.config import config as global_config
from gigaspatial.handlers.base import (
    BaseHandlerConfig,
    BaseHandlerDownloader,
    BaseHandlerReader,
    BaseHandler,
)


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class HDXConfig(BaseHandlerConfig):
    """
    Configuration for HDX dataset access.

    Manages dataset identification, HDX API authentication (user agent),
    and resource filtering logic.

    Attributes:
        dataset_name: Unique identifier for the HDX dataset.
        base_path: Root directory for local dataset storage.
        user_agent: Token identifying the application to the HDX API.
        hdx_site: Target HDX environment ('prod' or 'test').
    """

    # User configuration
    dataset_name: str = Field(
        default=..., description="Name of the HDX dataset to download"
    )

    # Optional configuration with defaults
    base_path: Path = Field(default=global_config.get_path("hdx", "bronze"))
    n_workers: int = (
        1  # HDX data is not parallelizable beyond single file, so just iterate.
    )
    user_agent: str = Field(
        default="gigaspatial", description="User agent for HDX API requests"
    )
    hdx_site: str = Field(default="prod", description="HDX site to use (prod or test)")

    # Internal state
    _hdx_configured: bool = Field(default=False, init=False)
    dataset: Optional[Dataset] = Field(default=None, init=False)

    @staticmethod
    def search_datasets(
        query: str,
        rows: int = None,
        sort: str = "relevance asc, metadata_modified desc",
        hdx_site: str = "prod",
        user_agent: str = "gigaspatial",
    ) -> List[Dict]:
        """
        Search for datasets on the HDX platform.

        Args:
            query: Search keywords or CKAN query string.
            rows: Max number of results to return.
            sort: Field and direction to sort by.
            hdx_site: HDX environment to query.
            user_agent: API collector identifier.

        Returns:
            A list of matching dataset metadata dictionaries.
        """
        try:
            Configuration.create(
                hdx_site=hdx_site,
                user_agent=user_agent,
                hdx_read_only=True,
            )
        except:
            pass

        try:
            results = Dataset.search_in_hdx(query=query, rows=rows, sort=sort)

            return results
        except Exception as e:
            logging.error(f"Error searching HDX datasets: {str(e)}")
            raise

    def __post_init__(self):
        super().__post_init__()
        try:
            Configuration.read()
            self._hdx_configured = True
        except Exception:
            self._hdx_configured = False
        self.configure_hdx()
        self.dataset = self.fetch_dataset()

    @property
    def output_dir_path(self) -> Path:
        """Path to save the downloaded HDX dataset"""
        return self.base_path / self.dataset_name

    def configure_hdx(self):
        """Configure HDX API if not already configured"""
        if not self._hdx_configured:
            try:
                Configuration.create(
                    hdx_site=self.hdx_site,
                    user_agent=self.user_agent,
                    hdx_read_only=True,
                )
                self._hdx_configured = True
            except Exception as e:
                self.logger.error(f"Error configuring HDX API: {str(e)}")
                raise

    def fetch_dataset(self) -> Dataset:
        """Get the HDX dataset"""
        try:
            self.logger.info(f"Fetching HDX dataset: {self.dataset_name}")
            dataset = Dataset.read_from_hdx(self.dataset_name)
            if not dataset:
                raise ValueError(
                    f"Dataset '{self.dataset_name}' not found on HDX. "
                    "Please verify the dataset name or use search_datasets() "
                    "to find available datasets."
                )
            return dataset
        except Exception as e:
            self.logger.error(f"Error fetching HDX dataset: {str(e)}")
            raise

    def _match_pattern(self, value: str, pattern: str, token_match: bool = False) -> bool:
        """Check if a value matches a pattern"""
        if isinstance(pattern, str):
            if token_match:
                import re

                # Match as a token delimited by common separators or start/end of string
                regex = rf"(^|[._/-]){re.escape(pattern)}([._/-]|$)"
                return bool(re.search(regex, value, re.IGNORECASE))
            return pattern.lower() in value.lower()
        return value == pattern

    def _get_patterns_for_value(self, value: Any) -> List[str]:
        """Generate patterns for a given value or list of values"""
        if isinstance(value, list):
            patterns = []
            for v in value:
                patterns.extend(self._get_patterns_for_value(v))
            return patterns

        if not isinstance(value, str):
            return [value]

        patterns = []
        value = value.lower()

        # Add exact match
        patterns.append(value)

        # Add common variations
        patterns.extend(
            [
                f"/{value}_",  # URL path with prefix
                f"/{value}.",  # URL path with extension
                f"_{value}_",  # Filename with value in middle
                f"_{value}.",  # Filename with value at end
            ]
        )

        # If value contains spaces, generate additional patterns
        if " " in value:
            # Generate patterns for space-less version
            no_space = value.replace(" ", "")
            patterns.extend(self._get_patterns_for_value(no_space))

            # Generate patterns for hyphenated version
            hyphenated = value.replace(" ", "-")
            patterns.extend(self._get_patterns_for_value(hyphenated))

        return patterns

    def get_dataset_resources(
        self,
        filter: Optional[Dict[str, Any]] = None,
        exact_match: bool = False,
        token_match: bool = False,
    ) -> List[Resource]:
        """
        Retrieve resource metadata matching specific criteria.

        Args:
            filter: Key-value pairs to match against resource metadata.
            exact_match: If True, requires strict equality for filter values.
            token_match: If True, pattern must match a distinct metadata component.

        Returns:
            A list of HDX Resource objects.
        """
        try:
            resources = self.dataset.get_resources()

            # Apply resource filter if specified
            if filter:
                filtered_resources = []
                for res in resources:
                    match = True
                    for key, value in filter.items():
                        if key not in res.data:
                            match = False
                            break

                        if exact_match:
                            # For exact matching, check if value matches or is in list of values
                            if isinstance(value, list):
                                if res.data[key] not in value:
                                    match = False
                                    break
                            elif res.data[key] != value:
                                match = False
                                break
                        else:
                            # For pattern matching, generate patterns for value(s)
                            patterns = self._get_patterns_for_value(value)
                            if not any(
                                self._match_pattern(
                                    str(res.data[key]),
                                    pattern,
                                    token_match=token_match,
                                )
                                for pattern in patterns
                            ):
                                match = False
                                break

                    if match:
                        filtered_resources.append(res)
                resources = filtered_resources

            return resources
        except Exception as e:
            self.logger.error(f"Error getting dataset resources: {str(e)}")
            raise

    def get_relevant_data_units_by_geometry(
        self, geometry: Union[BaseGeometry, gpd.GeoDataFrame], **kwargs
    ) -> List[Resource]:
        return self.get_dataset_resources(geometry, **kwargs)

    def get_data_unit_path(self, unit: Resource, **kwargs) -> Path:
        """
        Resolve an HDX resource to its local storage path.

        Args:
            unit: HDX Resource object.
            **kwargs: Additional resolution context.

        Returns:
            Absolute local path for the resource file.
        """
        try:
            filename = unit.data["name"]
        except:
            filename = unit.get("download_url").split("/")[-1]

        return self.output_dir_path / filename

    def list_resources(self) -> List[str]:
        """List all resources in the dataset directory using the data_store."""
        dataset_folder = str(self.output_dir_path)
        # Check if the dataset directory exists in the data_store
        if not (
            self.data_store.is_dir(dataset_folder)
            or self.data_store.file_exists(dataset_folder)
        ):
            raise FileNotFoundError(
                f"HDX dataset not found at {dataset_folder}. "
                "Download the data first using HDXDownloader."
            )
        return self.data_store.list_files(dataset_folder)

    def extract_search_geometry(self, source, **kwargs):
        """
        Identify filter criteria from a geographic or structural source.

        Args:
            source: Country name/code, or a direct filter dictionary.
            **kwargs: Resolution parameters (e.g., 'key' to filter on).

        Returns:
            A dictionary for resource filtering.
        """
        if isinstance(source, str):
            country = pycountry.countries.lookup(source)
            values = [country.alpha_3, country.alpha_2, country.name]
            key = kwargs.get(
                "key", "url"
            )  # The key to filter on in the resource data. Defaults to `url`
            return {key: values}
        elif isinstance(source, dict):
            return source
        else:
            raise ValueError(
                f"Unsupported source type: {type(source)}"
                "Please use country-based (str) filtering or direct resource (dict) filtering instead."
            )

    def __repr__(self) -> str:
        return (
            f"HDXConfig(\n"
            f"  dataset_name='{self.dataset_name}'\n"
            f"  base_path='{self.base_path}'\n"
            f"  hdx_site='{self.hdx_site}'\n"
            f"  user_agent='{self.user_agent}'\n"
            f")"
        )


class HDXDownloader(BaseHandlerDownloader):
    """
    Downloader for HDX datasets.

    Handles the acquisition of files from the HDX platform, utilizing
    temporary storage and DataStore abstraction for final persistence.
    """

    def __init__(
        self,
        config: Union[HDXConfig, dict],
        data_store: Optional[DataStore] = None,
        logger: Optional[logging.Logger] = None,
    ):
        config = config if isinstance(config, HDXConfig) else HDXConfig(**config)
        super().__init__(config=config, data_store=data_store, logger=logger)

    def download_data_unit(self, resource: Resource, **kwargs) -> str:
        """
        Download a specific HDX resource.

        Args:
            resource: HDX Resource object to download.
            **kwargs: Additional parameters.

        Returns:
            Local path to the downloaded resource on success, or None on failure.
        """
        try:
            resource_name = resource.get("name", "Unknown")
            self.logger.info(f"Downloading resource: {resource_name}")

            with tempfile.TemporaryDirectory() as tmpdir:
                url, local_path = resource.download(folder=tmpdir)
                with open(local_path, "rb") as f:
                    data = f.read()
                # Compose the target path in the DataStore
                target_path = str(self.config.get_data_unit_path(resource))
                self.data_store.write_file(target_path, data)
                self.logger.info(
                    f"Downloaded resource: {resource_name} to {target_path}"
                )
                return target_path
        except Exception as e:
            self.logger.error(f"Error downloading resource {resource_name}: {str(e)}")
            return None


class HDXReader(BaseHandlerReader):
    """
    Reader for HDX datasets.

    Parses local HDX resource files (CSV, GeoJSON, etc.) into structured
    Python objects or dataframes.
    """

    def __init__(
        self,
        config: Optional[HDXConfig] = None,
        data_store: Optional[DataStore] = None,
        logger: Optional[logging.Logger] = None,
    ):
        config = config if isinstance(config, HDXConfig) else HDXConfig(**config)
        super().__init__(config=config, data_store=data_store, logger=logger)

    def load_from_paths(
        self, source_data_path: List[Union[str, Path]], **kwargs
    ) -> Any:
        """
        Load data from one or more local HDX resource paths.

        Args:
            source_data_path: List of local paths to resource files.
            **kwargs: Parameters passed to `read_dataset`.

        Returns:
            Aggregated data contents (typically a dict or single DataFrame).
        """
        if len(source_data_path) == 1:
            return read_dataset(
                source_data_path[0], data_store=self.data_store, **kwargs
            )

        all_data = {}
        for file_path in source_data_path:
            try:
                all_data[file_path] = read_dataset(
                    file_path, data_store=self.data_store, **kwargs
                )
            except Exception as e:
                raise ValueError(f"Could not read file {file_path}: {str(e)}")
        return all_data

    def load_all_resources(self):
        resources = self.config.list_resources()
        return self.load_from_paths(resources)


class HDXHandler(BaseHandler):
    """
    Unified handler for HDX datasets.

    Provides a streamlined API for interacting with the HDX platform,
    allowing for dataset-specific configuration and resource-level access.
    """

    def __init__(
        self,
        dataset_name: str,
        config: Optional[HDXConfig] = None,
        downloader: Optional[HDXDownloader] = None,
        reader: Optional[HDXReader] = None,
        data_store: Optional[DataStore] = None,
        logger: Optional[logging.Logger] = None,
        **kwargs,
    ):
        self._dataset_name = dataset_name
        super().__init__(
            config=config,
            downloader=downloader,
            reader=reader,
            data_store=data_store,
            logger=logger,
            **kwargs,
        )

    def create_config(
        self, data_store: DataStore, logger: logging.Logger, **kwargs
    ) -> HDXConfig:
        """
        Create an HDX configuration instance.

        Args:
            data_store: Storage backend for local files.
            logger: Component logger.
            **kwargs: Configuration overrides.

        Returns:
            A configured HDXConfig.
        """
        return HDXConfig(
            dataset_name=self._dataset_name,
            data_store=data_store,
            logger=logger,
            **kwargs,
        )

    def create_downloader(
        self,
        config: HDXConfig,
        data_store: DataStore,
        logger: logging.Logger,
        **kwargs,
    ) -> HDXDownloader:
        """
        Create an HDX downloader instance.

        Args:
            config: Handler configuration.
            data_store: Storage backend for local files.
            logger: Component logger.
            **kwargs: Downloader parameters.

        Returns:
            A configured HDXDownloader.
        """
        return HDXDownloader(
            config=config,
            data_store=data_store,
            logger=logger,
            **kwargs,
        )

    def create_reader(
        self,
        config: HDXConfig,
        data_store: DataStore,
        logger: logging.Logger,
        **kwargs,
    ) -> HDXReader:
        """
        Create an HDX reader instance.

        Args:
            config: Handler configuration.
            data_store: Storage backend for local files.
            logger: Component logger.
            **kwargs: Reader parameters.

        Returns:
            A configured HDXReader.
        """
        return HDXReader(
            config=config,
            data_store=data_store,
            logger=logger,
            **kwargs,
        )
