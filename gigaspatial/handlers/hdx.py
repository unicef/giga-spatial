import os
import logging
from pathlib import Path
from typing import List, Optional, Union, Dict, Any
import tempfile

import pandas as pd
import geopandas as gpd
from pydantic import BaseModel, Field

from hdx.api.configuration import Configuration
from hdx.data.dataset import Dataset
from hdx.data.resource import Resource

from gigaspatial.core.io.data_store import DataStore
from gigaspatial.core.io.local_data_store import LocalDataStore
from gigaspatial.core.io.readers import read_dataset
from gigaspatial.config import config as global_config


class HDXConfig(BaseModel):
    """Configuration for HDX data access"""

    # User configuration
    dataset_name: str = Field(..., description="Name of the HDX dataset to download")
    base_path: Path = Field(default=global_config.get_path("hdx", "bronze"))
    user_agent: str = Field(
        default="gigaspatial", description="User agent for HDX API requests"
    )
    hdx_site: str = Field(default="prod", description="HDX site to use (prod or test)")
    resource_filter: Optional[Dict[str, Any]] = Field(
        default=None, description="Filter to apply to resources"
    )

    @property
    def output_dir_path(self) -> Path:
        """Path to save the downloaded HDX dataset"""
        return self.base_path / self.dataset_name

    def __repr__(self) -> str:
        return (
            f"HDXConfig(\n"
            f"  dataset_name='{self.dataset_name}'\n"
            f"  base_path='{self.base_path}'\n"
            f"  hdx_site='{self.hdx_site}'\n"
            f"  user_agent='{self.user_agent}'\n"
            f")"
        )


class HDXDownloader:
    """Downloader for HDX datasets"""

    def __init__(
        self,
        config: Union[HDXConfig, dict],
        data_store: Optional[DataStore] = None,
        logger: Optional[logging.Logger] = None,
    ):
        if isinstance(config, dict):
            self.config = HDXConfig(**config)
        else:
            self.config = config

        self.data_store = data_store or LocalDataStore()
        self.logger = logger or global_config.get_logger(self.__class__.__name__)
        try:
            Configuration.read()
            self._hdx_configured = True
        except:
            self._hdx_configured = False

    @classmethod
    def from_dataset_name(cls, dataset_name: str, **kwargs):
        """Create a downloader for a specific HDX dataset"""
        config = HDXConfig(dataset_name=dataset_name, **kwargs)
        return cls(config=config)

    def _configure_hdx(self):
        """Configure HDX API if not already configured"""
        if not self._hdx_configured:
            try:
                Configuration.create(
                    hdx_site=self.config.hdx_site,
                    user_agent=self.config.user_agent,
                    hdx_read_only=True,
                )
                self._hdx_configured = True
            except Exception as e:
                self.logger.error(f"Error configuring HDX API: {str(e)}")
                raise

    def get_dataset(self) -> Dataset:
        """Get the HDX dataset"""
        self._configure_hdx()

        try:
            self.logger.info(f"Fetching HDX dataset: {self.config.dataset_name}")
            dataset = Dataset.read_from_hdx(self.config.dataset_name)
            if not dataset:
                raise ValueError(
                    f"Dataset '{self.config.dataset_name}' not found on HDX"
                )
            return dataset
        except Exception as e:
            self.logger.error(f"Error fetching HDX dataset: {str(e)}")
            raise

    def get_dataset_resources(
        self, dataset: Optional[Dataset] = None
    ) -> List[Resource]:
        """Get resources from the HDX dataset"""
        dataset = dataset or self.get_dataset()

        try:
            resources = dataset.get_resources()

            # Apply resource filter if specified
            if self.config.resource_filter:
                filtered_resources = []
                for res in resources:
                    match = True
                    for key, value in self.config.resource_filter.items():
                        if key in res.data and res.data[key] != value:
                            match = False
                            break
                    if match:
                        filtered_resources.append(res)
                resources = filtered_resources

            return resources
        except Exception as e:
            self.logger.error(f"Error getting dataset resources: {str(e)}")
            raise

    def download_dataset(self) -> List[str]:
        """Download and save all resources from the HDX dataset into the data_store."""
        try:
            dataset = self.get_dataset()
            resources = self.get_dataset_resources(dataset)

            if not resources:
                self.logger.warning(
                    f"No resources found for dataset '{self.config.dataset_name}'"
                )
                return []

            self.logger.info(
                f"Found {len(resources)} resource(s) for dataset '{self.config.dataset_name}'"
            )

            downloaded_paths = []
            for res in resources:
                try:
                    resource_name = res.get("name", "Unknown")
                    self.logger.info(f"Downloading resource: {resource_name}")

                    # Download to a temporary directory
                    with tempfile.TemporaryDirectory() as tmpdir:
                        url, local_path = res.download(folder=tmpdir)
                        # Read the file and write to the DataStore
                        with open(local_path, "rb") as f:
                            data = f.read()
                        # Compose the target path in the DataStore
                        target_path = str(
                            self.config.output_dir_path / Path(local_path).name
                        )
                        self.data_store.write_file(target_path, data)
                        downloaded_paths.append(target_path)

                    self.logger.info(
                        f"Downloaded resource: {resource_name} to {target_path}"
                    )
                except Exception as e:
                    self.logger.error(
                        f"Error downloading resource {res.get('name', 'Unknown')}: {str(e)}"
                    )

            return downloaded_paths

        except Exception as e:
            self.logger.error(f"Error downloading dataset: {str(e)}")
            raise


class HDXReader:
    """Reader for HDX datasets"""

    def __init__(
        self,
        dataset_name: str,
        data_store: Optional[DataStore] = None,
        base_path: Optional[Path] = None,
    ):
        self.dataset_name = dataset_name
        self.data_store = data_store or LocalDataStore()
        self.base_path = base_path or global_config.get_path("hdx", "bronze")
        self.dataset_path = self.base_path / self.dataset_name

    def list_resources(self) -> List[str]:
        """List all resources in the dataset directory using the data_store."""
        # Check if the dataset directory exists in the data_store
        if not (
            self.data_store.is_dir(str(self.dataset_path))
            or self.data_store.file_exists(str(self.dataset_path))
        ):
            raise FileNotFoundError(
                f"HDX dataset '{self.dataset_name}' not found at {self.dataset_path}. "
                "Download the data first using HDXDownloader."
            )
        # List files using the data_store
        return self.data_store.list_files(str(self.dataset_path))

    def read_resource(
        self, resource_file: str
    ) -> Union[pd.DataFrame, gpd.GeoDataFrame]:
        """Read a specific resource file from the dataset using the data_store."""
        file_path = str(self.dataset_path / resource_file)

        if not self.data_store.file_exists(file_path):
            raise FileNotFoundError(
                f"Resource file {resource_file} not found in dataset {self.dataset_name}"
            )

        try:
            return read_dataset(self.data_store, file_path)
        except Exception as e:
            raise ValueError(f"Could not read file {file_path}: {str(e)}")

    def read_all_resources(self) -> Dict[str, Union[pd.DataFrame, gpd.GeoDataFrame]]:
        """Read all resources in the dataset directory using the data_store."""
        resources = self.list_resources()
        result = {}

        for resource in resources:
            try:
                result[resource] = self.read_resource(resource)
            except Exception as e:
                logging.warning(f"Could not read resource {resource}: {str(e)}")

        return result
