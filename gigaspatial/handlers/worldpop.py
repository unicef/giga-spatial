from pydantic import BaseModel, Field, HttpUrl, field_validator, model_validator
from pathlib import Path
import os
from typing import Optional, Union, Literal, ClassVar
import pandas as pd
import pycountry
import requests
from tqdm import tqdm
from urllib.error import URLError
import logging

from gigaspatial.core.io.readers import *
from gigaspatial.core.io.writers import *
from gigaspatial.core.io.data_store import DataStore
from gigaspatial.core.io.local_data_store import LocalDataStore
from gigaspatial.config import config as global_config


class WorldPopConfig(BaseModel):
    # class variables
    _metadata_cache: ClassVar[Optional[pd.DataFrame]] = None

    # constants
    CURRENT_MAX_YEAR: int = 2022
    EARLIEST_YEAR: int = 2000
    SCHOOL_AGE_YEAR: int = 2020

    # base config
    WORLDPOP_DB_BASE_URL: HttpUrl = Field(default="https://data.worldpop.org/")
    SCHOOL_AGE_POPULATION_PATH: str = Field(
        default="GIS/AgeSex_structures/school_age_population/v1/2020/"
    )
    PPP_2021_2022_PATH: str = Field(
        default="GIS/Population/Global_2021_2022_1km_UNadj/"
    )
    DATASETS_METADATA_PATH: str = Field(default="assets/wpgpDatasets.csv")

    # user config
    base_path: Path = Field(default=global_config.get_path("worldpop", "bronze"))
    country: str = Field(...)
    year: int = Field(..., ge=EARLIEST_YEAR, le=CURRENT_MAX_YEAR)
    resolution: Literal["HIGH", "LOW"] = Field(
        default="LOW",
        description="Spatial resolution of the population grid: HIGH (100m) or LOW (1km)",
    )
    un_adjusted: bool = True
    constrained: bool = False
    school_age: Optional[Literal["PRIMARY", "SECONDARY"]] = None
    gender: Literal["F", "M", "F_M"] = "F_M"

    @field_validator("country")
    def validate_country(cls, value: str) -> str:
        try:
            return pycountry.countries.lookup(value).alpha_3
        except LookupError:
            raise ValueError(f"Invalid country code provided: {value}")

    @model_validator(mode="after")
    def validate_configuration(self):
        """
        Validate that the configuration is valid based on dataset availability constraints.

        Specific rules:
        - Post-2020 data is only available at 1km resolution with UN adjustment
        - School age population data is only available for 2020 at 1km resolution
        """
        if self.year > self.SCHOOL_AGE_YEAR:
            if self.resolution != "LOW":
                raise ValueError(
                    f"Data for year {self.year} is only available at LOW (1km) resolution"
                )

            if not self.un_adjusted:
                raise ValueError(
                    f"Data for year {self.year} is only available with UN adjustment"
                )

        if self.school_age:
            if self.resolution != "LOW":
                raise ValueError(
                    f"School age data is only available at LOW (1km) resolution"
                )

            if self.year != self.SCHOOL_AGE_YEAR:
                self.year = self.SCHOOL_AGE_YEAR
                raise ValueError(f"School age data is only available for 2020")

        return self

    @property
    def dataset_url(self) -> str:
        """Get the URL for the configured dataset. The URL is computed on first access and then cached for subsequent calls."""
        if not hasattr(self, "_dataset_url"):
            self._dataset_url = self._compute_dataset_url()
        return self._dataset_url

    @property
    def dataset_path(self) -> Path:
        """Construct and return the path for the configured dataset."""
        url_parts = self.dataset_url.split("/")
        file_path = (
            "/".join(
                [url_parts[4], url_parts[5], url_parts[7], self.country, url_parts[-1]]
            )
            if self.school_age
            else "/".join([url_parts[4], url_parts[6], self.country, url_parts[-1]])
        )
        return self.base_path / file_path

    def _load_datasets_metadata(self) -> pd.DataFrame:
        """Load and return the WorldPop datasets metadata, using cache if available."""
        if WorldPopConfig._metadata_cache is not None:
            return WorldPopConfig._metadata_cache

        try:
            WorldPopConfig._metadata_cache = pd.read_csv(
                str(self.WORLDPOP_DB_BASE_URL) + self.DATASETS_METADATA_PATH
            )
            return WorldPopConfig._metadata_cache
        except (URLError, pd.errors.EmptyDataError) as e:
            raise RuntimeError(f"Failed to load WorldPop datasets metadata: {e}")

    def _compute_dataset_url(self) -> str:
        """Construct and return the URL for the configured dataset."""
        # handle post-2020 datasets
        if self.year > self.SCHOOL_AGE_YEAR:
            return (
                str(self.WORLDPOP_DB_BASE_URL)
                + self.PPP_2021_2022_PATH
                + f"{'' if self.constrained else 'un'}constrained/{self.year}/{self.country}/{self.country.lower()}_ppp_{self.year}_1km_UNadj{'_constrained' if self.constrained else ''}.tif"
            )

        # handle school-age population datasets
        if self.school_age:
            return (
                str(self.WORLDPOP_DB_BASE_URL)
                + self.SCHOOL_AGE_POPULATION_PATH
                + f"{self.country}/{self.country}_SAP_1km_2020/{self.country}_{self.gender}_{self.school_age}_2020_1km.tif"
            )

        # handle standard population datasets
        wp_metadata = self._load_datasets_metadata()

        try:
            dataset_url = (
                self.WORLDPOP_DB_BASE_URL
                + wp_metadata[
                    (wp_metadata.ISO3 == self.country)
                    & (
                        wp_metadata.Covariate
                        == "ppp_"
                        + str(self.year)
                        + ("_UNadj" if self.un_adjusted else "")
                    )
                ].PathToRaster.values[0]
            )
        except IndexError:
            raise ValueError(
                f"No dataset found for country={self.country}, year={self.year}, un_adjusted={self.un_adjusted}"
            )

        # handle resolution conversion if needed
        if self.resolution == "HIGH":
            return dataset_url

        url_parts = dataset_url.split("/")
        url_parts[5] = (
            url_parts[5] + "_1km" + ("_UNadj" if self.un_adjusted else "")
        )  # get 1km folder with UNadj specification
        url_parts[8] = url_parts[8].replace(
            str(self.year), str(self.year) + "_1km_Aggregated"
        )  # get filename with 1km res
        dataset_url = "/".join(url_parts)

        return dataset_url

    def __repr__(self) -> str:

        parts = [
            f"WorldpopConfig(",
            f"  country='{self.country}'",
            f"  year={self.year}",
            f"  resolution={self.resolution}",
            f"  un_adjusted={self.un_adjusted}",
            f"  constrained={self.constrained}",
        ]

        if self.school_age:
            parts.append(f"  school_age='{self.school_age}'")
            parts.append(f"  gender='{self.gender}'")

        parts.append(")")

        return "\n".join(parts)


class WorldPopDownloader:
    """A class to handle downloads of WorldPop datasets."""

    def __init__(
        self,
        config: Union[WorldPopConfig, dict[str, Union[str, int]]],
        data_store: Optional[DataStore] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the downloader.

        Args:
            config: Configuration for the WorldPop dataset, either as a WorldPopConfig object or a dictionary of parameters
            data_store: Optional data storage interface. If not provided, uses LocalDataStore.
            logger: Optional custom logger. If not provided, uses default logger.
        """
        self.logger = logger or global_config.get_logger(self.__class__.__name__)
        self.data_store = data_store or LocalDataStore()
        self.config = (
            config if isinstance(config, WorldPopConfig) else WorldPopConfig(**config)
        )

    @classmethod
    def from_country_year(cls, country: str, year: int, **kwargs):
        """
        Create a downloader instance from country and year.

        Args:
            country: Country code or name
            year: Year of the dataset
            **kwargs: Additional parameters for WorldPopConfig or the downloader
        """
        return cls({"country": country, "year": year}, **kwargs)

    def download_dataset(self) -> str:
        """
        Download the configured dataset to the provided output path.
        """

        try:
            response = requests.get(self.config.dataset_url, stream=True)
            response.raise_for_status()

            output_path = str(self.config.dataset_path)

            total_size = int(response.headers.get("content-length", 0))

            with self.data_store.open(output_path, "wb") as file:
                with tqdm(
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    desc=f"Downloading {os.path.basename(output_path)}",
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            file.write(chunk)
                            pbar.update(len(chunk))

            self.logger.debug(f"Successfully downloaded dataset: {self.config}")

            return output_path

        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to download dataset {self.config}: {str(e)}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error downloading dataset: {str(e)}")
            return None
