import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import wkt
from datetime import datetime
import requests
from pathlib import Path
from pydantic import ConfigDict, Field
from typing import List, Literal, Optional, Tuple, Union
from pydantic.dataclasses import dataclass
from shapely.geometry.base import BaseGeometry
from shapely.geometry import Point

from gigaspatial.grid.mercator_tiles import MercatorTiles, CountryMercatorTiles
from gigaspatial.core.io.data_store import DataStore
from gigaspatial.config import config
from gigaspatial.handlers.base import BaseHandlerConfig
from gigaspatial.handlers.base import BaseHandlerDownloader
from gigaspatial.handlers.base import BaseHandlerReader
from gigaspatial.handlers.base import BaseHandler

import logging
from tqdm import tqdm


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class OoklaSpeedtestConfig(BaseHandlerConfig):
    """
    Configuration class for Ookla Speedtest data.

    This class defines the parameters for accessing and filtering Ookla Speedtest datasets,
    including available years, quarters, and how dataset URLs are constructed.
    """

    MIN_YEAR = 2019
    MAX_YEAR = datetime.today().year
    MAX_QUARTER = int(np.floor((datetime.today().month - 1) / 3))
    if MAX_QUARTER == 0:
        MAX_YEAR -= 1
        MAX_QUARTER = 4

    BASE_URL = "https://ookla-open-data.s3.amazonaws.com/parquet/performance"

    base_path: Path = Field(default=config.get_path("ookla_speedtest", "bronze"))

    type: Literal["fixed", "mobile"] = Field(...)
    year: Optional[int] = Field(default=None, ge=MIN_YEAR, le=MAX_YEAR)
    quarter: Optional[int] = Field(default=None, ge=0, le=4)

    def __post_init__(self):
        if self.year is None:
            self.year = self.MAX_YEAR
            self.logger.warning(
                "Year not provided. Using the latest available data year: %s", self.year
            )
        if self.quarter is None:
            self.quarter = self.MAX_QUARTER
            self.logger.warning(
                "Quarter not provided. Using the latest available data quarter for year %s: %s",
                self.year,
                self.quarter,
            )

        super().__post_init__()
        self.DATASET_URL = self._get_dataset_url(self.type, self.year, self.quarter)

    def _get_dataset_url(self, type, year, quarter):
        month = [1, 4, 7, 10]
        quarter_start = datetime(year, month[self.quarter - 1], 1)
        return f"{self.BASE_URL}/type={type}/year={quarter_start:%Y}/quarter={quarter}/{quarter_start:%Y-%m-%d}_performance_{type}_tiles.parquet"

    @staticmethod
    def get_available_datasets():
        start_year = 2019  # first data year
        max_year = datetime.today().year
        max_quarter = np.floor((datetime.today().month - 1) / 3)
        if max_quarter == 0:
            max_year -= 1
            max_quarter = 4

        ookla_tiles = []
        for year in range(start_year, max_year + 1):
            for quarter in range(1, 5):
                if year == max_year and quarter > max_quarter:
                    continue
                for type in ["fixed", "mobile"]:
                    ookla_tiles.append(
                        {"service_type": type, "year": year, "quarter": quarter}
                    )

        return ookla_tiles

    def get_relevant_data_units(self, source=None, **kwargs):
        return [self.DATASET_URL]

    def get_relevant_data_units_by_geometry(
        self, geometry: Union[BaseGeometry, gpd.GeoDataFrame] = None, **kwargs
    ) -> List[str]:
        return

    def get_data_unit_path(self, unit: str, **kwargs) -> Path:
        """
        Given a Ookla Speedtest file url, return the corresponding path.
        """
        return self.base_path / unit.split("/")[-1]


class OoklaSpeedtestDownloader(BaseHandlerDownloader):
    """
    A class to handle the downloading of Ookla Speedtest data.

    This downloader focuses on fetching parquet files based on the provided configuration
    and data unit URLs.
    """

    def __init__(
        self,
        config: Union[OoklaSpeedtestConfig, dict[str, Union[str, int]]],
        data_store: Optional[DataStore] = None,
        logger: Optional[logging.Logger] = None,
    ):
        config = (
            config
            if isinstance(config, OoklaSpeedtestConfig)
            else OoklaSpeedtestConfig(**config)
        )
        super().__init__(config=config, data_store=data_store, logger=logger)

    def download_data_unit(self, url: str, **kwargs) -> Optional[Path]:
        output_path = self.config.get_data_unit_path(url)

        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))

            with self.data_store.open(str(output_path), "wb") as file:
                for chunk in tqdm(
                    response.iter_content(chunk_size=8192),
                    total=total_size // 8192,
                    unit="KB",
                    desc=f"Downloading {output_path.name}",
                ):
                    file.write(chunk)

            self.logger.info(f"Successfully downloaded: {url} to {output_path}")
            return output_path

        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to download {url}: {str(e)}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error downloading {url}: {str(e)}")
            return None

    def download_data_units(self, urls: List[str], **kwargs) -> List[Optional[Path]]:
        # Ookla data is not parallelizable in a meaningful way beyond single file, so just iterate.
        results = [self.download_data_unit(url, **kwargs) for url in urls]
        return [path for path in results if path is not None]

    def download(
        self, source: Optional[Union[str, List[str]]] = None, **kwargs
    ) -> List[Optional[Path]]:
        urls = self.config.get_relevant_data_units(source)
        return self.download_data_units(urls, **kwargs)


class OoklaSpeedtestReader(BaseHandlerReader):
    """
    A class to handle reading Ookla Speedtest data.

    It loads parquet files into a DataFrame.
    """

    def __init__(
        self,
        config: Union[OoklaSpeedtestConfig, dict[str, Union[str, int]]],
        data_store: Optional[DataStore] = None,
        logger: Optional[logging.Logger] = None,
    ):
        config = (
            config
            if isinstance(config, OoklaSpeedtestConfig)
            else OoklaSpeedtestConfig(**config)
        )
        super().__init__(config=config, data_store=data_store, logger=logger)

    def load_from_paths(
        self, source_data_path: List[Union[str, Path]], **kwargs
    ) -> pd.DataFrame:
        result = self._load_tabular_data(file_paths=source_data_path)
        return result

    def load(
        self,
        source: Optional[
            Union[
                str,  # country
                List[Union[Tuple[float, float], Point]],  # points
                BaseGeometry,  # geometry
                gpd.GeoDataFrame,  # geodataframe
                Path,  # path
                str,  # path
                List[Union[str, Path]],
            ]
        ] = None,
        **kwargs,
    ) -> pd.DataFrame:
        return super().load(source=source, **kwargs)


class OoklaSpeedtestHandler(BaseHandler):
    """
    Handler for Ookla Speedtest data.

    This class orchestrates the configuration, downloading, and reading of Ookla Speedtest
    data, allowing for filtering by geographical sources using Mercator tiles.
    """

    def __init__(
        self,
        type: Literal["fixed", "mobile"],
        year: Optional[int] = None,
        quarter: Optional[int] = None,
        config: Optional[OoklaSpeedtestConfig] = None,
        downloader: Optional[OoklaSpeedtestDownloader] = None,
        reader: Optional[OoklaSpeedtestReader] = None,
        data_store: Optional[DataStore] = None,
        logger: Optional[logging.Logger] = None,
        **kwargs,
    ):
        self._type = type
        self._year = year
        self._quarter = quarter

        super().__init__(
            config=config,
            downloader=downloader,
            reader=reader,
            data_store=data_store,
            logger=logger,
        )

    def create_config(
        self, data_store: DataStore, logger: logging.Logger, **kwargs
    ) -> OoklaSpeedtestConfig:
        return OoklaSpeedtestConfig(
            type=self._type,
            year=self._year,
            quarter=self._quarter,
            data_store=data_store,
            logger=logger,
            **kwargs,
        )

    def create_downloader(
        self,
        config: OoklaSpeedtestConfig,
        data_store: DataStore,
        logger: logging.Logger,
        **kwargs,
    ) -> OoklaSpeedtestDownloader:
        return OoklaSpeedtestDownloader(
            config=config, data_store=data_store, logger=logger, **kwargs
        )

    def create_reader(
        self,
        config: OoklaSpeedtestConfig,
        data_store: DataStore,
        logger: logging.Logger,
        **kwargs,
    ) -> OoklaSpeedtestReader:
        return OoklaSpeedtestReader(
            config=config, data_store=data_store, logger=logger, **kwargs
        )

    def load_data(
        self,
        source: Union[
            str,  # country
            List[Union[Tuple[float, float], Point]],  # points
            BaseGeometry,  # geometry
            gpd.GeoDataFrame,  # geodataframe
            Path,  # path
            str,  # path
            List[Union[str, Path]],
        ] = None,
        process_geospatial: bool = False,
        ensure_available: bool = True,
        **kwargs,
    ) -> Union[pd.DataFrame, gpd.GeoDataFrame]:

        if source is None or (
            isinstance(source, (str, Path))
            and (
                self.data_store.file_exists(str(source))
                or str(source).endswith(".parquet")
            )
            or (
                isinstance(source, List)
                and all(isinstance(p, (str, Path)) for p in source)
            )
        ):
            # If no source or source is a direct path, load without filtering
            result = super().load_data(source, ensure_available, **kwargs)
        else:
            # Load the entire dataset and then apply Mercator tile filtering
            full_dataset = super().load_data(
                None, ensure_available, **kwargs
            )  # Load the full dataset (uses DATASET_URL)

            key = self.config._cache_key(source, **kwargs)

            # Check cache unless forced recompute
            if (
                not kwargs.get("force_recompute", False)
                and key in self.config._unit_cache
            ):
                self.logger.debug(
                    f"Using cached quadkeys for {key[0]}: {key[1][:50]}..."
                )
                quadkeys = self.config._unit_cache[key]

            else:

                if isinstance(source, str):  # country
                    mercator_tiles = CountryMercatorTiles.create(
                        source, zoom_level=16, **kwargs
                    )
                elif isinstance(source, (BaseGeometry, gpd.GeoDataFrame, List)):
                    mercator_tiles = MercatorTiles.from_spatial(
                        source, zoom_level=16, **kwargs
                    )
                else:
                    raise ValueError(
                        f"Unsupported source type for filtering: {type(source)}"
                    )

                quadkeys = mercator_tiles.quadkeys

                # Cache the result
                self.config._unit_cache[key] = quadkeys

            result = full_dataset[full_dataset["quadkey"].isin(quadkeys)].reset_index(
                drop=True
            )

        if process_geospatial:
            # Convert 'tile' column from WKT to geometry
            result["geometry"] = result["tile"].apply(wkt.loads)
            return gpd.GeoDataFrame(result, geometry="geometry", crs="EPSG:4326")

        return result
