from pydantic.dataclasses import dataclass
from pydantic import (
    Field,
    field_validator,
    model_validator,
    ConfigDict,
)
from pathlib import Path
import functools
import multiprocessing
import os
from typing import Optional, Union, Literal, List, Dict, Any
import numpy as np
import pandas as pd
import geopandas as gpd
import pycountry
import requests
from shapely.geometry.base import BaseGeometry
from shapely.geometry import Point
from tqdm import tqdm
import logging

from gigaspatial.core.io.data_store import DataStore
from gigaspatial.processing.tif_processor import TifProcessor
from gigaspatial.handlers.base import (
    BaseHandlerConfig,
    BaseHandlerDownloader,
    BaseHandlerReader,
    BaseHandler,
)
from gigaspatial.config import config as global_config


class WorldPopRestClient:
    """
    REST API client for WorldPop data access.

    This class provides direct access to the WorldPop REST API without any
    configuration dependencies, allowing flexible integration patterns.
    """

    def __init__(
        self,
        base_url: str = "https://www.worldpop.org/rest/data",
        stats_url: str = "https://api.worldpop.org/v1/services/stats",
        api_key: Optional[str] = None,
        timeout: int = 30,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the WorldPop REST API client.

        Args:
            base_url: Base URL for the WorldPop REST API
            stats_url: URL for the WorldPop statistics API
            api_key: Optional API key for higher rate limits
            timeout: Request timeout in seconds
            logger: Optional logger instance
        """
        self.base_url = base_url.rstrip("/")
        self.stats_url = stats_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.logger = logger or logging.getLogger(self.__class__.__name__)

        # Setup session with default headers
        self.session = requests.Session()
        self.session.headers.update(
            {"Accept": "application/json", "User-Agent": "WorldPop-Python-Client/1.0"}
        )

        if self.api_key:
            self.session.headers["X-API-Key"] = self.api_key

    def get_available_projects(self) -> List[Dict[str, Any]]:
        """
        Get list of all available projects (e.g., population, births, pregnancies, etc.).

        Returns:
            List of project dictionaries with alias, name, title, and description
        """
        try:
            response = self.session.get(self.base_url, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            return data.get("data", [])
        except requests.RequestException as e:
            self.logger.error(f"Failed to fetch available project aliases: {e}")
            return []

    def get_project_sources(self, dataset_type: str) -> List[Dict[str, Any]]:
        """
        Get available sources for a specific project type.

        Args:
            dataset_type: Project type alias (e.g., 'pop', 'births', 'pregnancies')

        Returns:
            List of source dictionaries with alias and name
        """
        try:
            url = f"{self.base_url}/{dataset_type}"
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            return data.get("data", [])
        except requests.RequestException as e:
            self.logger.error(
                f"Failed to fetch project sources for {dataset_type}: {e}"
            )
            return []

    def get_source_entities(
        self, dataset_type: str, category: str
    ) -> List[Dict[str, Any]]:
        """
        Get list of entities (countries, global, continental) available for a specific project type and source.

        Args:
            dataset_type: Project type alias (e.g., 'pop', 'births')
            category: Source alias (e.g., 'wpgp', 'pic')

        Returns:
            List of entity dictionaries with id and iso3 codes (if applicable)
        """
        try:
            url = f"{self.base_url}/{dataset_type}/{category}"
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            return data.get("data", [])
        except requests.RequestException as e:
            self.logger.error(
                f"Failed to fetch entities for {dataset_type}/{category}: {e}"
            )
            return []

    def get_datasets(self, dataset_type: str, category: str, params: dict):
        """
        Get all datasets available for the params.

        Args:
            dataset_type: Dataset type alias (e.g., 'pop', 'births')
            category: Category alias (e.g., 'wpgp', 'pic')
            params: Query parameters (e.g., {'iso3`:'RWA'})

        Returns:
            List of dataset dictionaries with metadata and file information
        """
        try:
            url = f"{self.base_url}/{dataset_type}/{category}"
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            return data.get("data", [])
        except requests.RequestException as e:
            self.logger.error(f"Failed to fetch datasets for {params}: {e}")
            return []

    def get_datasets_by_country(
        self, dataset_type: str, category: str, iso3: str
    ) -> List[Dict[str, Any]]:
        """
        Get all datasets available for a specific country.

        Args:
            dataset_type: Dataset type alias (e.g., 'pop', 'births')
            category: Category alias (e.g., 'wpgp', 'pic')
            iso3: ISO3 country code (e.g., 'USA', 'BRA')

        Returns:
            List of dataset dictionaries with metadata and file information
        """
        params = {"iso3": iso3}
        return self.get_datasets(dataset_type, category, params)

    def get_dataset_by_id(
        self, dataset_type: str, category: str, dataset_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get dataset information by ID.

        Args:
            dataset_type: Dataset type alias (e.g., 'pop', 'births')
            category: Category alias (e.g., 'wpgp', 'pic')
            dataset_id: Dataset ID

        Returns:
            Dataset dictionary or None if not found
        """
        params = {"id": dataset_id}
        return self.get_datasets(dataset_type, category, params)

    def find_dataset(
        self,
        dataset_type: str,
        category: str,
        iso3: str,
        year: Union[str, int],
        **filters,
    ) -> Optional[Dict[str, Any]]:
        """
        Find a specific dataset by year and optional filters.

        Args:
            dataset_type: Dataset type alias
            category: Category alias
            iso3: ISO3 country code
            year: Year to search for
            **filters: Additional filters (e.g., gender='F', resolution='1km')

        Returns:
            Dataset dictionary or None if not found
        """
        datasets = self.get_country_datasets(dataset_type, category, iso3)
        year_str = str(year)

        for dataset in datasets:
            if dataset.get("popyear") == year_str:
                # Check additional filters
                match = True
                for key, value in filters.items():
                    if key in dataset and dataset[key] != value:
                        match = False
                        break

                if match:
                    return dataset

        return None

    def list_years_for_country(
        self, dataset_type: str, category: str, iso3: str
    ) -> List[int]:
        """
        List all available years for a specific country and dataset.

        Args:
            dataset_type: Dataset type alias
            category: Category alias
            iso3: ISO3 country code

        Returns:
            Sorted list of available years
        """
        datasets = self.get_datasets_by_country(dataset_type, category, iso3)
        years = []

        for dataset in datasets:
            try:
                year = int(dataset.get("popyear", 0))
                if year > 0:
                    years.append(year)
            except (ValueError, TypeError):
                continue

        return sorted(years)

    def search_datasets(
        self,
        dataset_type: Optional[str] = None,
        category: Optional[str] = None,
        iso3: Optional[str] = None,
        year: Optional[Union[str, int]] = None,
        **filters,
    ) -> List[Dict[str, Any]]:
        """
        Search for datasets with flexible filtering.

        Args:
            dataset_type: Optional dataset type filter
            category: Optional category filter
            iso3: Optional country filter
            year: Optional year filter
            **filters: Additional filters

        Returns:
            List of matching datasets
        """
        results = []

        if dataset_type:
            if category:
                # If we have country-specific filters
                if iso3:
                    datasets = self.get_datasets_by_country(
                        dataset_type, category, iso3
                    )
                    for dataset in datasets:
                        match = True

                        # Check year filter
                        if year and dataset.get("popyear") != str(year):
                            match = False

                        # Check additional filters
                        for key, value in filters.items():
                            if key in dataset and dataset[key] != value:
                                match = False
                                break

                        if match:
                            results.append(dataset)
                else:
                    return self.get_source_entities(dataset_type, category)
            else:
                return self.get_project_sources(dataset_type)
        else:
            return self.get_available_projects()

        return results

    def get_dataset_info(self, dataset: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract useful information from a dataset dictionary.

        Args:
            dataset: Dataset dictionary from API

        Returns:
            Cleaned dataset information
        """
        return {
            "id": dataset.get("id"),
            "title": dataset.get("title"),
            "description": dataset.get("desc"),
            "doi": dataset.get("doi"),
            "citation": dataset.get("citation"),
            "data_format": dataset.get("data_format"),
            "year": dataset.get("popyear"),
            "country": dataset.get("country"),
            "iso3": dataset.get("iso3"),
            "continent": dataset.get("continent"),
            "download_urls": dataset.get("files", []),
            "image_url": dataset.get("url_img"),
            "summary_url": dataset.get("url_summary"),
            "license": dataset.get("license"),
            "organization": dataset.get("organisation"),
            "author": dataset.get("author_name"),
            "maintainer": dataset.get("maintainer_name"),
            "project": dataset.get("project"),
            "category": dataset.get("category"),
            "date_created": dataset.get("date"),
            "public": dataset.get("public") == "Y",
            "archived": dataset.get("archive") == "Y",
        }

    def close(self):
        """Close the session."""
        self.session.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class WPPopulationConfig(BaseHandlerConfig):

    client = WorldPopRestClient()

    AVAILABLE_YEARS: List = Field(default=np.append(np.arange(2000, 2021), 2024))
    AVAILABLE_RESOLUTIONS: List = Field(default=[100, 1000])

    # user config
    base_path: Path = Field(default=global_config.get_path("worldpop", "bronze"))
    project: Literal["pop", "age_structures"] = Field(...)
    year: int = Field(...)
    resolution: int = Field(...)
    un_adjusted: bool = Field(...)
    constrained: bool = Field(...)
    school_age: bool = Field(...)

    @field_validator("year")
    def validate_year(cls, value: str) -> int:
        if value in cls.AVAILABLE_YEARS:
            return value
        raise ValueError(
            f"No datasets found for the provided year: {value}\nAvailable years are: {cls.AVAILABLE_YEARS}"
        )

    @field_validator("resolution")
    def validate_resolution(cls, value: str) -> int:
        if value in cls.AVAILABLE_RESOLUTIONS:
            return value
        raise ValueError(
            f"No datasets found for the provided resolution: {value}\nAvailable resolutions are: {cls.AVAILABLE_RESOLUTIONS}"
        )

    @model_validator(mode="after")
    def validate_configuration(self):
        """
        Validate that the configuration is valid based on dataset availability constraints.

        Specific rules:
        - For age_structures:
            - School age data is only available for 2020 at 1km resolution.
            - Non-school age data is only available at 100m resolution.
            - Unconstrained, non-school age data is only available without UN adjustment.
            - Constrained, non-school age data with UN adjustment is only available for 2020.
            - Constrained, non-school age data without UN adjustment is only available for 2020 and 2024.
        - For pop:
            - 2024 data is only available at 100m resolution and without UN adjustment.
            - Constrained data (other than 2024) is only available for 2020 at 100m resolution.
            - Unconstrained data at 100m or 1km is available for other years, with or without UN adjustment.
        """

        if self.project == "age_structures":

            if self.school_age:
                if self.resolution == 100:
                    self.logger.warning(
                        "School age population datasets are only available at 1km `resolution`, resolution is set as 1000"
                    )
                    self.resolution = 1000

                if self.year != 2020:
                    self.logger.warning(
                        "School age population datasets are only available for 2020, `year` is set as 2020"
                    )
                    self.year = 2020

                if self.un_adjusted:
                    self.logger.warning(
                        "School age population datasets are only available without UN adjustment, `un_adjusted` is set as False"
                    )
                    self.un_adjusted = False

                if self.constrained:
                    self.logger.warning(
                        "School age population datasets are only available unconstrained, `constrained` is set as False"
                    )
                    self.constrained = False

                self.dataset_category = "sapya1km"
            else:
                if self.resolution == 1000:
                    self.logger.warning(
                        "Age structures datasets are only available at 100m resolution, `resolution` is set as 100"
                    )
                    self.resolution = 100

                if not self.constrained:
                    if self.un_adjusted:
                        self.logger.warning(
                            "Age structures unconstrained datasets are only available without UN adjustment, `un_adjusted` is set as False"
                        )
                        self.un_adjusted = False

                    self.dataset_category = (
                        "G2_UC_Age_2024_100m" if self.year == 2024 else "aswpgp"
                    )
                else:
                    if self.un_adjusted:
                        if self.year != 2020:
                            self.logger.warning(
                                "Age structures constrained datasets with UN adjustment are only available for 2020, `year` is set as 2020"
                            )
                            self.year = 2020
                        self.dataset_category = "ascicua_2020"
                    else:
                        if self.year == 2024:
                            self.dataset_category = "G2_CN_Age_2024_100m"
                        elif self.year == 2020:
                            self.dataset_category = "ascic_2020"
                        else:
                            raise ValueError(
                                "Age structures constrained datasets without UN adjustment are only available for 2020 and 2024, please set `year` to one of the available options: 2020, 2024"
                            )

        elif self.project == "pop":

            if self.school_age:
                raise ValueError(
                    f"""
                    Received unexpected value of `{self.school_age}` for project: `{self.project}`.
                    For school age population datasets, please set project as `age_structures`.
                    """
                )

            if self.year == 2024:
                if self.resolution == 1000:
                    self.logger.warning(
                        "2024 datasets are only available at 100m resolution, `resolution` is set as 100m"
                    )
                    self.resolution = 100
                if self.un_adjusted:
                    self.logger.warning(
                        "2024 datasets are only available without UN adjustment, `un_adjusted` is set as False"
                    )
                    self.un_adjusted = False

                self.dataset_category = (
                    "G2_CN_POP_2024_100m" if self.constrained else "G2_UC_POP_2024_100m"
                )
            else:
                if self.constrained:
                    if self.year != 2020:
                        self.logger.warning(
                            "Population constrained datasets are only available for 2020, `year` is set as 2020"
                        )
                        self.year = 2020

                    if self.resolution != 100:
                        self.logger.warning(
                            "Population constrained datasets are only available at 100m resolution, `resolution` is set as 100"
                        )
                        self.resolution = 100

                    self.dataset_category = (
                        "cic2020_UNadj_100m" if self.un_adjusted else "cic2020_100m"
                    )
                else:
                    if self.resolution == 100:
                        self.dataset_category = (
                            f"wpgp{'unadj' if self.un_adjusted else ''}"
                        )
                    else:
                        self.dataset_category = (
                            "wpic1km" if not self.un_adjusted else "wpicuadj1km"
                        )

    def get_relevant_data_units_by_geometry(
        self, geometry: Union[BaseGeometry, gpd.GeoDataFrame], **kwargs
    ) -> List[Dict[str, Any]]:
        raise NotImplementedError(
            "WorldPop does not support geometry-based filtering. "
            "Please use country-based filtering or direct resource filtering instead."
        )

    def get_relevant_data_units_by_points(
        self, points: List[Union[Point, tuple]], **kwargs
    ) -> List[Dict[str, Any]]:
        raise NotImplementedError(
            "WorldPop does not support point-based filtering. "
            "Please use country-based filtering or direct resource filtering instead."
        )

    def get_relevant_data_units_by_country(
        self, country: str, **kwargs
    ) -> List[Dict[str, Any]]:
        iso3 = pycountry.countries.lookup(country).alpha_3

        datasets = self.client.search_datasets(
            self.project, self.dataset_category, iso3, self.year
        )

        files = [
            file
            for file in datasets[0].get("files", [])
            if ((self.dataset_category == "sapya1km") or file.endswith(".tif"))
        ]

        return files

    def get_data_unit_path(self, unit: str, **kwargs) -> Path:
        """
        Given a WP file url, return the corresponding path.
        """
        return self.base_path / unit.split("GIS/")[1]

    def __repr__(self) -> str:

        return (
            f"WPPopulationConfig(",
            f"project={self.project}, "
            f"year={self.year}, "
            f"resolution={self.resolution}, "
            f"un_adjusted={self.un_adjusted}, "
            f"constrained={self.constrained}, "
            f"school_age={self.school_age}, "
            f")",
        )


class WPPopulationDownloader(BaseHandlerDownloader):

    def __init__(
        self,
        config: Union[WPPopulationConfig, dict[str, Union[str, int]]],
        data_store: Optional[DataStore] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the downloader.

        Args:
            config: Configuration for the WorldPop dataset, either as a WPPopulationConfig object or a dictionary of parameters
            data_store: Optional data storage interface. If not provided, uses LocalDataStore.
            logger: Optional custom logger. If not provided, uses default logger.
        """
        config = (
            config
            if isinstance(config, WPPopulationConfig)
            else WPPopulationConfig(**config)
        )
        super().__init__(config=config, data_store=data_store, logger=logger)

    def download_data_unit(self, url, **kwargs):
        """Download data file for a url."""
        try:
            response = self.config.client.session.get(
                url, stream=True, timeout=self.config.client.timeout
            )
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))
            file_path = self.config.get_data_unit_path(url)

            with self.data_store.open(str(file_path), "wb") as file:
                with tqdm(
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    desc=f"Downloading {os.path.basename(file_path)}",
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            file.write(chunk)
                            pbar.update(len(chunk))

            self.logger.info(f"Successfully downloaded: {file_path}")
            return file_path

        except requests.RequestException as e:
            self.logger.error(f"Failed to download {url}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error downloading {url}: {e}")
            return None

    def download_data_units(
        self,
        urls: List[str],
    ) -> List[str]:
        """Download data files for multiple urls."""

        with multiprocessing.Pool(self.config.n_workers) as pool:
            download_func = functools.partial(self.download_data_unit)
            file_paths = list(
                tqdm(
                    pool.imap(download_func, urls),
                    total=len(urls),
                    desc=f"Downloading data",
                )
            )

        return [path for path in file_paths if path is not None]

    def download(self, source: str, **kwargs) -> List[str]:
        """Download data for a source"""
        resources = self.config.get_relevant_data_units(source, **kwargs)
        return self.download_data_units(resources)


class WPPopulationReader(BaseHandlerReader):

    def __init__(
        self,
        config: Union[WPPopulationConfig, dict[str, Union[str, int]]],
        data_store: Optional[DataStore] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the reader.

        Args:
            config: Configuration for the WorldPop dataset, either as a WPPopulationConfig object or a dictionary of parameters
            data_store: Optional data storage interface. If not provided, uses LocalDataStore.
            logger: Optional custom logger. If not provided, uses default logger.
        """
        config = (
            config
            if isinstance(config, WPPopulationConfig)
            else WPPopulationConfig(**config)
        )
        super().__init__(config=config, data_store=data_store, logger=logger)

    def load_from_paths(
        self, source_data_path: List[Union[str, Path]], **kwargs
    ) -> List[TifProcessor]:
        """
        Load TifProcessors of WP datasets.
        Args:
            source_data_path: List of file paths to load
        Returns:
            List[TifProcessor]: List of TifProcessor objects for accessing the raster data.
        """
        return self._load_raster_data(raster_paths=source_data_path)


class WPPopulationHandler(BaseHandler):
    """
    Handler for WorldPop Populations datasets.

    This class provides a unified interface for downloading and loading WP Population data.
    It manages the lifecycle of configuration, downloading, and reading components.
    """

    def __init__(
        self,
        project: Literal["pop", "age_structures"] = "pop",
        year: int = 2020,
        resolution: int = 1000,
        un_adjusted: bool = True,
        constrained: bool = False,
        school_age: bool = False,
        config: Optional[WPPopulationConfig] = None,
        downloader: Optional[WPPopulationDownloader] = None,
        reader: Optional[WPPopulationReader] = None,
        data_store: Optional[DataStore] = None,
        logger: Optional[logging.Logger] = None,
        **kwargs,
    ):
        self._project = project
        self._year = year
        self._resolution = resolution
        self._un_adjusted = un_adjusted
        self._constrained = constrained
        self._school_age = school_age
        super().__init__(
            config=config,
            downloader=downloader,
            reader=reader,
            data_store=data_store,
            logger=logger,
        )

    def create_config(
        self, data_store: DataStore, logger: logging.Logger, **kwargs
    ) -> WPPopulationConfig:
        """
        Create and return a WPPopulationConfig instance.

        Args:
            data_store: The data store instance to use
            logger: The logger instance to use
            **kwargs: Additional configuration parameters

        Returns:
            Configured WPPopulationConfig instance
        """
        return WPPopulationConfig(
            project=self._project,
            year=self._year,
            resolution=self._resolution,
            un_adjusted=self._un_adjusted,
            constrained=self._constrained,
            school_age=self._school_age,
            data_store=data_store,
            logger=logger,
            **kwargs,
        )

    def create_downloader(
        self,
        config: WPPopulationConfig,
        data_store: DataStore,
        logger: logging.Logger,
        **kwargs,
    ) -> WPPopulationDownloader:
        """
        Create and return a WPPopulationDownloader instance.

        Args:
            config: The configuration object
            data_store: The data store instance to use
            logger: The logger instance to use
            **kwargs: Additional downloader parameters

        Returns:
            Configured WPPopulationDownloader instance
        """
        return WPPopulationDownloader(
            config=config, data_store=data_store, logger=logger, **kwargs
        )

    def create_reader(
        self,
        config: WPPopulationConfig,
        data_store: DataStore,
        logger: logging.Logger,
        **kwargs,
    ) -> WPPopulationReader:
        """
        Create and return a WPPopulationReader instance.

        Args:
            config: The configuration object
            data_store: The data store instance to use
            logger: The logger instance to use
            **kwargs: Additional reader parameters

        Returns:
            Configured WPPopulationReader instance
        """
        return WPPopulationReader(
            config=config, data_store=data_store, logger=logger, **kwargs
        )

    def load_into_dataframe(
        self,
        source: str,
        ensure_available: bool = True,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Load GHSL data into a pandas DataFrame.

        Args:
            source: The data source specification
            ensure_available: If True, ensure data is downloaded before loading
            **kwargs: Additional parameters passed to load methods

        Returns:
            DataFrame containing the GHSL data
        """
        tif_processors = self.load_data(
            source=source, ensure_available=ensure_available, **kwargs
        )
        return pd.concat(
            [tp.to_dataframe() for tp in tif_processors], ignore_index=True
        )

    def load_into_geodataframe(
        self,
        source: str,
        ensure_available: bool = True,
        **kwargs,
    ) -> gpd.GeoDataFrame:
        """
        Load GHSL data into a geopandas GeoDataFrame.

        Args:
            source: The data source specification
            ensure_available: If True, ensure data is downloaded before loading
            **kwargs: Additional parameters passed to load methods

        Returns:
            GeoDataFrame containing the GHSL data
        """
        tif_processors = self.load_data(
            source=source, ensure_available=ensure_available, **kwargs
        )
        return pd.concat(
            [tp.to_geodataframe() for tp in tif_processors], ignore_index=True
        )
