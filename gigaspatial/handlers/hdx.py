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
    """Configuration for HDX data access"""

    # User configuration
    dataset_name: str = Field(
        default=..., description="Name of the HDX dataset to download"
    )

    # Optional configuration with defaults
    base_path: Path = Field(default=global_config.get_path("hdx", "bronze"))
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
        """Search for datasets in HDX before initializing the class.

        Args:
            query: Search query string
            rows: Number of results per page. Defaults to all datasets (sys.maxsize).
            sort: Sort order - one of 'relevance', 'views_recent', 'views_total', 'last_modified' (default: 'relevance')
            hdx_site: HDX site to use - 'prod' or 'test' (default: 'prod')
            user_agent: User agent for HDX API requests (default: 'gigaspatial')

        Returns:
            List of dataset dictionaries containing search results

        Example:
            >>> results = HDXConfig.search_datasets("population", rows=5)
            >>> for dataset in results:
            >>>     print(f"Name: {dataset['name']}, Title: {dataset['title']}")
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

    def _match_pattern(self, value: str, pattern: str) -> bool:
        """Check if a value matches a pattern"""
        if isinstance(pattern, str):
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
        self, filter: Optional[Dict[str, Any]] = None, exact_match: bool = False
    ) -> List[Resource]:
        """Get resources from the HDX dataset

        Args:
            filter: Dictionary of key-value pairs to filter resources
            exact_match: If True, perform exact matching. If False, use pattern matching
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
                                self._match_pattern(str(res.data[key]), pattern)
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

    def get_relevant_data_units(
        self, source: Union[str, Dict], **kwargs
    ) -> List[Resource]:
        """Get relevant data units based on the source type

        Args:
            source: Either a country name/code (str) or a filter dictionary
            **kwargs: Additional keyword arguments passed to the specific method

        Returns:
            List of matching resources
        """
        if isinstance(source, str):
            # If source is a string, assume it's a country and use country-based filtering
            return self.get_relevant_data_units_by_country(source, **kwargs)
        elif isinstance(source, dict):
            # If source is a dict, use it directly as a filter
            return self.get_dataset_resources(filter=source, **kwargs)
        else:
            raise ValueError(f"Unsupported source type: {type(source)}")

    def get_relevant_data_units_by_geometry(
        self, geometry: Union[BaseGeometry, gpd.GeoDataFrame], **kwargs
    ) -> List[Resource]:
        raise NotImplementedError(
            "HDX does not support geometry-based filtering. "
            "Please use country-based filtering or direct resource filtering instead."
        )

    def get_relevant_data_units_by_points(
        self, points: List[Union[Point, tuple]], **kwargs
    ) -> List[Resource]:
        raise NotImplementedError(
            "HDX does not support point-based filtering. "
            "Please use country-based filtering or direct resource filtering instead."
        )

    def get_relevant_data_units_by_country(
        self,
        country: str,
        key: str = "url",
        **kwargs,
    ) -> Any:
        """Get relevant data units for a country

        Args:
            country: Country name or code
            key: The key to filter on in the resource data
            patterns: List of patterns to match against the resource data
            **kwargs: Additional keyword arguments
        """
        country = pycountry.countries.lookup(country)
        values = [country.alpha_3, country.alpha_2, country.name]
        return self.get_dataset_resources(
            filter={key: values},
        )

    def get_data_unit_path(self, unit: str, **kwargs) -> str:
        """Get the path for a data unit"""
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
    """Downloader for HDX datasets"""

    def __init__(
        self,
        config: Union[HDXConfig, dict],
        data_store: Optional[DataStore] = None,
        logger: Optional[logging.Logger] = None,
    ):
        config = config if isinstance(config, HDXConfig) else HDXConfig(**config)
        super().__init__(config=config, data_store=data_store, logger=logger)

    def download_data_unit(self, resource: str, **kwargs) -> str:
        """Download a single resource"""
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

    def download_data_units(self, resources: List[Resource], **kwargs) -> List[str]:
        """Download multiple resources sequentially

        Args:
            resources: List of HDX Resource objects
            **kwargs: Additional keyword arguments

        Returns:
            List of paths to downloaded files
        """
        if len(resources) == 0:
            self.logger.warning("There is no resource to download")
            return []

        downloaded_paths = []
        for resource in tqdm(resources, desc="Downloading resources"):
            path = self.download_data_unit(resource)
            if path:
                downloaded_paths.append(path)

        return downloaded_paths

    def download(self, source: Union[Dict, str], **kwargs) -> List[str]:
        """Download data for a source"""
        resources = self.config.get_relevant_data_units(source, **kwargs)
        return self.download_data_units(resources)


class HDXReader(BaseHandlerReader):
    """Reader for HDX datasets"""

    def __init__(
        self,
        config: Optional[HDXConfig] = None,
        data_store: Optional[DataStore] = None,
        logger: Optional[logging.Logger] = None,
    ):
        config = config if isinstance(config, HDXConfig) else HDXConfig(**config)
        super().__init__(config=config, data_store=data_store, logger=logger)

    def resolve_source_paths(
        self,
        source: Union[
            str,  # country code
            Dict,  # filter
            Path,  # path
            str,  # path
            List[Union[str, Path]],
        ],
        **kwargs,
    ) -> List[Union[str, Path]]:
        if isinstance(source, (str, Path)):
            # Could be a country code or a path
            if self.data_store.file_exists(str(source)) or str(source).endswith(
                (".csv", ".tif", ".json", ".parquet", ".gz", ".geojson", ".zip")
            ):
                source_data_paths = self.resolve_by_paths(source)
            else:
                source_data_paths = self.resolve_by_country(source, **kwargs)
        elif isinstance(source, Dict):
            resources = self.config.get_relevant_data_units(source=source, **kwargs)
            source_data_paths = self.config.get_data_unit_paths(resources, **kwargs)
        elif isinstance(source, Iterable) and all(
            isinstance(p, (str, Path)) for p in source
        ):
            source_data_paths = self.resolve_by_paths(source)
        else:
            raise NotImplementedError(f"Unsupported source type: {type(source)}")

        self.logger.info(f"Resolved {len(source_data_paths)} paths!")
        return source_data_paths

    def load_from_paths(
        self, source_data_path: List[Union[str, Path]], **kwargs
    ) -> Any:
        """Load data from paths"""
        if len(source_data_path) == 1:
            return read_dataset(self.data_store, source_data_path[0])

        all_data = {}
        for file_path in source_data_path:
            try:
                all_data[file_path] = read_dataset(self.data_store, file_path)
            except Exception as e:
                raise ValueError(f"Could not read file {file_path}: {str(e)}")
        return all_data

    def load_all_resources(self):
        resources = self.config.list_resources()
        return self.load_from_paths(resources)


class HDXHandler(BaseHandler):
    """Handler for HDX datasets"""

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
        """Create and return a HDXConfig instance"""
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
        """Create and return a HDXDownloader instance"""
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
        """Create and return a HDXReader instance"""
        return HDXReader(
            config=config,
            data_store=data_store,
            logger=logger,
            **kwargs,
        )
