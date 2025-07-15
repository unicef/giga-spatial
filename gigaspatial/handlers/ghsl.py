# from dataclasses import dataclass
from pathlib import Path
import functools
import multiprocessing
from typing import List, Optional, Union, Literal, Iterable, Tuple
import geopandas as gpd
import pandas as pd
import numpy as np
from pydantic.dataclasses import dataclass
from shapely.geometry import Point, MultiPoint
from shapely.geometry.base import BaseGeometry
from enum import Enum
import requests
from tqdm import tqdm
import zipfile
import tempfile
from pydantic import (
    HttpUrl,
    Field,
    model_validator,
    field_validator,
    ConfigDict,
)
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


class CoordSystem(int, Enum):
    """Enum for coordinate systems used by GHSL datasets."""

    WGS84 = 4326
    Mollweide = 54009


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class GHSLDataConfig(BaseHandlerConfig):
    # constants
    AVAILABLE_YEARS: List = Field(default=np.append(np.arange(1975, 2031, 5), 2018))
    AVAILABLE_RESOLUTIONS: List = Field(default=[10, 100, 1000])

    # base config
    GHSL_DB_BASE_URL: HttpUrl = Field(
        default="https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/"
    )
    TILES_URL: str = "https://ghsl.jrc.ec.europa.eu/download/GHSL_data_{}_shapefile.zip"

    # user config
    base_path: Path = Field(default=global_config.get_path("ghsl", "bronze"))
    coord_system: CoordSystem = CoordSystem.WGS84
    release: str = "R2023A"

    product: Literal[
        "GHS_BUILT_S",
        "GHS_BUILT_H_AGBH",
        "GHS_BUILT_H_ANBH",
        "GHS_BUILT_V",
        "GHS_POP",
        "GHS_SMOD",
    ] = Field(...)
    year: int = 2020
    resolution: int = 100

    def __post_init__(self):
        super().__post_init__()

    def _load_tiles(self):
        """Load GHSL tiles from tiles shapefile."""
        try:
            self.tiles_gdf = gpd.read_file(self.TILES_URL)
        except Exception as e:
            self.logger.error(f"Failed to download tiles shapefile: {e}")
            raise ValueError(
                f"Could not download GHSL tiles from {self.TILES_URL}"
            ) from e

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
        -
        """
        if self.year == 2018 and self.product in ["GHS_BUILT_V", "GHS_POP", "GHS_SMOD"]:
            raise ValueError(f"{self.product} product is not available for 2018")

        if self.resolution == 10 and self.product != "GHS_BUILT_H":
            raise ValueError(
                f"{self.product} product is not available at 10 (10m) resolution"
            )

        if "GHS_BUILT_H" in self.product:
            if self.year != 2018:
                self.logger.warning(
                    "Building height product is only available for 2018, year is set as 2018"
                )
                self.year = 2018

        if self.product == "GHS_BUILT_S":
            if self.year == 2018 and self.resolution != 10:
                self.logger.warning(
                    "Built-up surface product for 2018 is only available at 10m resolution, resolution is set as 10m"
                )
                self.resolution = 10

            if self.resolution == 10 and self.year != 2018:
                self.logger.warning(
                    "Built-up surface product at resolution 10 is only available for 2018, year is set as 2018"
                )
                self.year = 2018

            if self.resolution == 10 and self.coord_system != CoordSystem.Mollweide:
                self.logger.warning(
                    f"Built-up surface product at resolution 10 is only available with Mollweide ({CoordSystem.Mollweide}) projection, coordinate system is set as Mollweide"
                )
                self.coord_system = CoordSystem.Mollweide

        if self.product == "GHS_SMOD":
            if self.resolution != 1000:
                self.logger.warning(
                    f"Settlement model (SMOD) product is only available at 1000 (1km) resolution, resolution is set as 1000"
                )
                self.resolution = 1000

            if self.coord_system != CoordSystem.Mollweide:
                self.logger.warning(
                    f"Settlement model (SMOD) product is only available with Mollweide ({CoordSystem.Mollweide}) projection, coordinate system is set as Mollweide"
                )
                self.coord_system = CoordSystem.Mollweide

        self.TILES_URL = self.TILES_URL.format(self.coord_system.value)
        self._load_tiles()

        return self

    @property
    def crs(self) -> str:
        return "EPSG:4326" if self.coord_system == CoordSystem.WGS84 else "ESRI:54009"

    def get_relevant_data_units_by_geometry(
        self, geometry: Union[BaseGeometry, gpd.GeoDataFrame], **kwargs
    ) -> List[dict]:
        """
        Return intersecting tiles for a given geometry or GeoDataFrame.
        """
        return self._get_relevant_tiles(geometry)

    def get_relevant_data_units_by_points(
        self, points: Iterable[Union[Point, tuple]], **kwargs
    ) -> List[dict]:
        """
        Return intersecting tiles f or a list of points.
        """
        return self._get_relevant_tiles(points)

    def get_data_unit_path(self, unit: str = None, file_ext=".zip", **kwargs) -> Path:
        """Construct and return the path for the configured dataset or dataset tile."""
        info = self._get_product_info()

        tile_path = (
            self.base_path
            / info["product_folder"]
            / (
                f"{info['product_name']}_V{info['product_version']}_0"
                + (f"_{unit}" if unit else "")
                + file_ext
            )
        )

        return tile_path

    def compute_dataset_url(self, tile_id=None) -> str:
        """Compute the download URL for a GHSL dataset."""
        info = self._get_product_info()

        path_segments = [
            str(self.GHSL_DB_BASE_URL),
            info["product_folder"],
            info["product_name"],
            f"V{info['product_version']}-0",
            "tiles" if tile_id else "",
            f"{info['product_name']}_V{info['product_version']}_0"
            + (f"_{tile_id}" if tile_id else "")
            + ".zip",
        ]

        return "/".join(path_segments)

    def _get_relevant_tiles(
        self,
        source: Union[
            BaseGeometry,
            gpd.GeoDataFrame,
            Iterable[Union[Point, tuple]],
        ],
        crs="EPSG:4326",
    ) -> list:
        """
        Identify and return the GHSL tiles that spatially intersect with the given geometry.

        The input geometry can be a Shapely geometry object, a GeoDataFrame,
        or a list of Point objects or (lon, lat) tuples. The method ensures
        the input geometry is in GHSL tiles projection for the spatial intersection.

        Args:
            source: A Shapely geometry, a GeoDataFrame, or a list of Point
                      objects or (lat, lon) tuples representing the area of interest.

        Returns:
            A list the tile ids for the intersecting tiles.

        Raises:
            ValueError: If the input `source` is not one of the supported types.
        """
        if isinstance(source, gpd.GeoDataFrame):
            if source.crs != crs:
                source = source.to_crs(crs)
            search_geom = source.geometry.unary_union
        elif isinstance(
            source,
            BaseGeometry,
        ):
            search_geom = source
        elif isinstance(source, Iterable) and all(
            len(pt) == 2 or isinstance(pt, Point) for pt in source
        ):
            points = [
                pt if isinstance(pt, Point) else Point(pt[1], pt[0]) for pt in source
            ]
            search_geom = MultiPoint(points)
        else:
            raise ValueError(
                f"Expected Geometry, GeoDataFrame or iterable object of Points got {source.__class__}"
            )

        if self.tiles_gdf.crs != crs:
            search_geom = (
                gpd.GeoDataFrame(geometry=[search_geom], crs=crs)
                .to_crs(self.tiles_gdf.crs)
                .geometry[0]
            )

        # Find intersecting tiles
        mask = (
            tile_geom.intersects(search_geom) for tile_geom in self.tiles_gdf.geometry
        )

        intersecting_tiles = self.tiles_gdf.loc[mask, "tile_id"].to_list()

        return intersecting_tiles

    def _get_product_info(self) -> dict:
        """Generate and return common product information used in multiple methods."""
        resolution_str = (
            str(self.resolution)
            if self.coord_system == CoordSystem.Mollweide
            else ("3ss" if self.resolution == 100 else "30ss")
        )
        product_folder = f"{self.product}_GLOBE_{self.release}"
        product_name = f"{self.product}_E{self.year}_GLOBE_{self.release}_{self.coord_system.value}_{resolution_str}"
        product_version = 2 if self.product == "GHS_SMOD" else 1

        return {
            "resolution_str": resolution_str,
            "product_folder": product_folder,
            "product_name": product_name,
            "product_version": product_version,
        }

    def __repr__(self) -> str:
        """Return a string representation of the GHSL dataset configuration."""
        return (
            f"GHSLDataConfig("
            f"product='{self.product}', "
            f"year={self.year}, "
            f"resolution={self.resolution}, "
            f"coord_system={self.coord_system.name}, "
            f"release='{self.release}'"
            f")"
        )


class GHSLDataDownloader(BaseHandlerDownloader):
    """A class to handle downloads of GHSL datasets."""

    def __init__(
        self,
        config: Union[GHSLDataConfig, dict[str, Union[str, int]]],
        data_store: Optional[DataStore] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the downloader.

        Args:
            config: Configuration for the GHSL dataset, either as a GHSLDataConfig object or a dictionary of parameters
            data_store: Optional data storage interface. If not provided, uses LocalDataStore.
            logger: Optional custom logger. If not provided, uses default logger.
        """
        config = (
            config if isinstance(config, GHSLDataConfig) else GHSLDataConfig(**config)
        )
        super().__init__(config=config, data_store=data_store, logger=logger)

    def download_data_unit(
        self,
        tile_id: str,
        extract: bool = True,
        file_pattern: Optional[str] = r".*\.tif$",
        **kwargs,
    ) -> Optional[Union[Path, List[Path]]]:
        """
        Downloads and optionally extracts files for a given tile.

        Args:
            tile_id: tile ID to process.
            extract: If True and the downloaded file is a zip, extract its contents. Defaults to True.
            file_pattern: Optional regex pattern to filter extracted files (if extract=True).
            **kwargs: Additional parameters passed to download methods

        Returns:
            Path to the downloaded file if extract=False,
            List of paths to the extracted files if extract=True,
            None on failure.
        """
        url = self.config.compute_dataset_url(tile_id=tile_id)
        output_path = self.config.get_data_unit_path(tile_id)

        if not extract:
            return self._download_file(url, output_path)

        extracted_files: List[Path] = []
        temp_downloaded_path: Optional[Path] = None

        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as temp_file:
                temp_downloaded_path = Path(temp_file.name)
                self.logger.debug(
                    f"Downloading {url} to temporary file: {temp_downloaded_path}"
                )

                response = requests.get(url, stream=True)
                response.raise_for_status()

                total_size = int(response.headers.get("content-length", 0))

                with tqdm(
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    desc=f"Downloading {tile_id}",
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            temp_file.write(chunk)
                            pbar.update(len(chunk))

            self.logger.info(f"Successfully downloaded temporary file!")

            with zipfile.ZipFile(str(temp_downloaded_path), "r") as zip_ref:
                if file_pattern:
                    import re

                    pattern = re.compile(file_pattern)
                    files_to_extract = [
                        f for f in zip_ref.namelist() if pattern.match(f)
                    ]
                else:
                    files_to_extract = zip_ref.namelist()

                for file in files_to_extract:
                    extracted_path = output_path.parent / Path(file).name
                    with zip_ref.open(file) as source:
                        file_content = source.read()
                        self.data_store.write_file(str(extracted_path), file_content)
                    extracted_files.append(extracted_path)
                    self.logger.info(f"Extracted {file} to {extracted_path}")

            Path(temp_file.name).unlink()
            return extracted_files

        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to download {url} to temporary file: {e}")
            return None
        except zipfile.BadZipFile:
            self.logger.error(f"Downloaded file for {tile_id} is not a valid zip file.")
            return None
        except Exception as e:
            self.logger.error(f"Error downloading/extracting tile {tile_id}: {e}")
            return None
        finally:
            if temp_downloaded_path and temp_downloaded_path.exists():
                try:
                    temp_downloaded_path.unlink()
                    self.logger.debug(f"Deleted temporary file: {temp_downloaded_path}")
                except OSError as e:
                    self.logger.warning(
                        f"Could not delete temporary file {temp_downloaded_path}: {e}"
                    )

    def download_data_units(
        self,
        tile_ids: List[str],
        extract: bool = True,
        file_pattern: Optional[str] = r".*\.tif$",
        **kwargs,
    ) -> List[Optional[Union[Path, List[Path]]]]:
        """
        Downloads multiple tiles in parallel, with an option to extract them.

        Args:
            tile_ids: A list of tile IDs to download.
            extract: If True and the downloaded files are zips, extract their contents. Defaults to True.
            file_pattern: Optional regex pattern to filter extracted files (if extract=True).
            **kwargs: Additional parameters passed to download methods

        Returns:
            A list where each element corresponds to a tile ID and contains:
            - Path to the downloaded file if extract=False.
            - List of paths to extracted files if extract=True.
            - None if the download or extraction failed for a tile.
        """
        if not tile_ids:
            self.logger.warning("No tiles to download")
            return []

        with multiprocessing.Pool(processes=self.config.n_workers) as pool:
            download_func = functools.partial(
                self.download_data_unit, extract=extract, file_pattern=file_pattern
            )
            file_paths = list(
                tqdm(
                    pool.imap(download_func, tile_ids),
                    total=len(tile_ids),
                    desc=f"Downloading data",
                )
            )

        return file_paths

    def download(
        self,
        source: Union[
            str,  # country
            List[Union[Tuple[float, float], Point]],  # points
            BaseGeometry,  # shapely geoms
            gpd.GeoDataFrame,
        ],
        extract: bool = True,
        file_pattern: Optional[str] = r".*\.tif$",
        **kwargs,
    ) -> List[Optional[Union[Path, List[Path]]]]:
        """
        Download GHSL data for a specified geographic region.

        The region can be defined by a country code/name, a list of points,
        a Shapely geometry, or a GeoDataFrame. This method identifies the
        relevant GHSL tiles intersecting the region and downloads the
        specified type of data (polygons or points) for those tiles in parallel.

        Args:
            source: Defines the geographic area for which to download data.
                    Can be:
                      - A string representing a country code or name.
                      - A list of (latitude, longitude) tuples or Shapely Point objects.
                      - A Shapely BaseGeometry object (e.g., Polygon, MultiPolygon).
                      - A GeoDataFrame with geometry column in EPSG:4326.
            extract: If True and the downloaded files are zips, extract their contents. Defaults to True.
            file_pattern: Optional regex pattern to filter extracted files (if extract=True).
            **kwargs: Additional keyword arguments. These will be passed down to
                      `AdminBoundaries.create()` (if `source` is a country)
                      and to `self.download_data_units()`.

        Returns:
            A list of local file paths for the successfully downloaded tiles.
            Returns an empty list if no data is found for the region or if
            all downloads fail.
        """

        tiles = self.config.get_relevant_data_units(source, **kwargs)
        return self.download_data_units(
            tiles, extract=extract, file_pattern=file_pattern, **kwargs
        )

    def download_by_country(
        self,
        country_code: str,
        data_store: Optional[DataStore] = None,
        country_geom_path: Optional[Union[str, Path]] = None,
        extract: bool = True,
        file_pattern: Optional[str] = r".*\.tif$",
        **kwargs,
    ) -> List[Optional[Union[Path, List[Path]]]]:
        """
        Download GHSL data for a specific country.

        This is a convenience method to download data for an entire country
        using its code or name.

        Args:
            country_code: The country code (e.g., 'USA', 'GBR') or name.
            data_store: Optional instance of a `DataStore` to be used by
                        `AdminBoundaries` for loading country boundaries. If None,
                        `AdminBoundaries` will use its default data loading.
            country_geom_path: Optional path to a GeoJSON file containing the
                               country boundary. If provided, this boundary is used
                               instead of the default from `AdminBoundaries`.
            extract: If True and the downloaded files are zips, extract their contents. Defaults to True.
            file_pattern: Optional regex pattern to filter extracted files (if extract=True).
            **kwargs: Additional keyword arguments that are passed to
                      `download_data_units`. For example, `extract` to download and extract.

        Returns:
            A list of local file paths for the successfully downloaded tiles
            for the specified country.
        """
        return self.download(
            source=country_code,
            data_store=data_store,
            path=country_geom_path,
            extract=extract,
            file_pattern=file_pattern,
            **kwargs,
        )

    def _download_file(self, url: str, output_path: Path) -> Optional[Path]:
        """
        Downloads a file from a URL to a specified output path with a progress bar.

        Args:
            url: The URL to download from.
            output_path: The local path to save the downloaded file.

        Returns:
            The path to the downloaded file on success, None on failure.
        """
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))

            with self.data_store.open(str(output_path), "wb") as file:
                with tqdm(
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    desc=f"Downloading {output_path.name}",
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            file.write(chunk)
                            pbar.update(len(chunk))

            self.logger.debug(f"Successfully downloaded: {url} to {output_path}")
            return output_path

        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to download {url}: {str(e)}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error downloading {url}: {str(e)}")
            return None


class GHSLDataReader(BaseHandlerReader):

    def __init__(
        self,
        config: Union[GHSLDataConfig, dict[str, Union[str, int]]],
        data_store: Optional[DataStore] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the reader.

        Args:
            config: Configuration for the GHSL dataset, either as a GHSLDataConfig object or a dictionary of parameters
            data_store: Optional data storage interface. If not provided, uses LocalDataStore.
            logger: Optional custom logger. If not provided, uses default logger.
        """
        config = (
            config if isinstance(config, GHSLDataConfig) else GHSLDataConfig(**config)
        )
        super().__init__(config=config, data_store=data_store, logger=logger)

    def load_from_paths(
        self, source_data_path: List[Union[str, Path]], **kwargs
    ) -> List[TifProcessor]:
        """
        Load TifProcessors from GHSL dataset.
        Args:
            source_data_path: List of file paths to load
        Returns:
            List[TifProcessor]: List of TifProcessor objects for accessing the raster data.
        """
        return self._load_raster_data(raster_paths=source_data_path)

    def load(self, source, **kwargs):
        return super().load(source=source, file_ext=".tif")


class GHSLDataHandler(BaseHandler):
    """
    Handler for GHSL (Global Human Settlement Layer) dataset.

    This class provides a unified interface for downloading and loading GHSL data.
    It manages the lifecycle of configuration, downloading, and reading components.
    """

    def __init__(
        self,
        product: Literal[
            "GHS_BUILT_S",
            "GHS_BUILT_H_AGBH",
            "GHS_BUILT_H_ANBH",
            "GHS_BUILT_V",
            "GHS_POP",
            "GHS_SMOD",
        ],
        year: int = 2020,
        resolution: int = 100,
        config: Optional[GHSLDataConfig] = None,
        downloader: Optional[GHSLDataDownloader] = None,
        reader: Optional[GHSLDataReader] = None,
        data_store: Optional[DataStore] = None,
        logger: Optional[logging.Logger] = None,
        **kwargs,
    ):
        """
        Initialize the GHSLDataHandler.

        Args:
            product: The GHSL product to use. Must be one of:
                    - GHS_BUILT_S: Built-up surface
                    - GHS_BUILT_H_AGBH: Average building height
                    - GHS_BUILT_H_ANBH: Average number of building heights
                    - GHS_BUILT_V: Building volume
                    - GHS_POP: Population
                    - GHS_SMOD: Settlement model
            year: The year of the data (default: 2020)
            resolution: The resolution in meters (default: 100)
            config: Optional configuration object
            downloader: Optional downloader instance
            reader: Optional reader instance
            data_store: Optional data store instance
            logger: Optional logger instance
            **kwargs: Additional configuration parameters
        """
        self._product = product
        self._year = year
        self._resolution = resolution
        super().__init__(
            config=config,
            downloader=downloader,
            reader=reader,
            data_store=data_store,
            logger=logger,
        )

    def create_config(
        self, data_store: DataStore, logger: logging.Logger, **kwargs
    ) -> GHSLDataConfig:
        """
        Create and return a GHSLDataConfig instance.

        Args:
            data_store: The data store instance to use
            logger: The logger instance to use
            **kwargs: Additional configuration parameters

        Returns:
            Configured GHSLDataConfig instance
        """
        return GHSLDataConfig(
            product=self._product,
            year=self._year,
            resolution=self._resolution,
            data_store=data_store,
            logger=logger,
            **kwargs,
        )

    def create_downloader(
        self,
        config: GHSLDataConfig,
        data_store: DataStore,
        logger: logging.Logger,
        **kwargs,
    ) -> GHSLDataDownloader:
        """
        Create and return a GHSLDataDownloader instance.

        Args:
            config: The configuration object
            data_store: The data store instance to use
            logger: The logger instance to use
            **kwargs: Additional downloader parameters

        Returns:
            Configured GHSLDataDownloader instance
        """
        return GHSLDataDownloader(
            config=config, data_store=data_store, logger=logger, **kwargs
        )

    def create_reader(
        self,
        config: GHSLDataConfig,
        data_store: DataStore,
        logger: logging.Logger,
        **kwargs,
    ) -> GHSLDataReader:
        """
        Create and return a GHSLDataReader instance.

        Args:
            config: The configuration object
            data_store: The data store instance to use
            logger: The logger instance to use
            **kwargs: Additional reader parameters

        Returns:
            Configured GHSLDataReader instance
        """
        return GHSLDataReader(
            config=config, data_store=data_store, logger=logger, **kwargs
        )

    def load_data(
        self,
        source: Union[
            str,  # country
            List[Union[tuple, Point]],  # points
            BaseGeometry,  # geometry
            gpd.GeoDataFrame,  # geodataframe
            Path,  # path
            List[Union[str, Path]],  # list of paths
        ],
        ensure_available: bool = True,
        **kwargs,
    ):
        return super().load_data(
            source=source,
            ensure_available=ensure_available,
            file_ext=".tif",
            extract=True,
            file_pattern=r".*\.tif$",
            **kwargs,
        )

    def load_into_dataframe(
        self,
        source: Union[
            str,  # country
            List[Union[tuple, Point]],  # points
            BaseGeometry,  # geometry
            gpd.GeoDataFrame,  # geodataframe
            Path,  # path
            List[Union[str, Path]],  # list of paths
        ],
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
        source: Union[
            str,  # country
            List[Union[tuple, Point]],  # points
            BaseGeometry,  # geometry
            gpd.GeoDataFrame,  # geodataframe
            Path,  # path
            List[Union[str, Path]],  # list of paths
        ],
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

    def get_available_data_info(
        self,
        source: Union[
            str,  # country
            List[Union[tuple, Point]],  # points
            BaseGeometry,  # geometry
            gpd.GeoDataFrame,  # geodataframe
        ],
        **kwargs,
    ) -> dict:
        return super().get_available_data_info(source, file_ext=".tif", **kwargs)
