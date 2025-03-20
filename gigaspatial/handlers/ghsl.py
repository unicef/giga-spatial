from dataclasses import dataclass
from pathlib import Path
import functools
import multiprocessing
from typing import List, Optional, Union, Literal
import geopandas as gpd
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
from shapely.strtree import STRtree
from pydantic.dataclasses import dataclass
from shapely import wkt
from shapely.geometry import Polygon, MultiPolygon
from enum import Enum
import pycountry
import requests
from tqdm import tqdm
import zipfile
import tempfile
import shutil
from pydantic import (
    BaseModel,
    HttpUrl,
    Field,
    model_validator,
    field_validator,
    ConfigDict,
)
import logging
import os

from gigaspatial.core.io.data_store import DataStore
from gigaspatial.core.io.local_data_store import LocalDataStore
from gigaspatial.core.io.readers import read_dataset
from gigaspatial.handlers.boundaries import AdminBoundaries
from gigaspatial.config import config


class CoordSystem(int, Enum):
    """Enum for coordinate systems used by GHSL datasets."""

    WGS84 = 4326
    Mollweide = 54009


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class GHSLDataConfig:
    # constants
    AVAILABLE_YEARS: List = Field(default=np.append(np.arange(1975, 2031, 5), 2018))
    AVAILABLE_RESOLUTIONS: List = Field(default=[10, 100, 1000])

    # base config
    GHSL_DB_BASE_URL: HttpUrl = Field(
        default="https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/"
    )
    TILES_URL: str = "https://ghsl.jrc.ec.europa.eu/download/GHSL_data_{}_shapefile.zip"

    # user config
    base_path: Path = Field(default=config.get_path("ghsl", "bronze"))
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

    logger: Optional[logging.Logger] = config.get_logger(__name__)
    n_workers: int = 4

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

        self.TILES_URL = self.TILES_URL.format(self.coord_system)

        self._load_tiles()

        return self

    def _get_product_info(self) -> dict:
        """Generate and return common product information used in multiple methods."""
        resolution_str = (
            str(self.resolution)
            if self.coord_system == CoordSystem.Mollweide
            else ("3ss" if self.resolution == 100 else "30ss")
        )
        product_folder = f"{self.product}_GLOBE_{self.release}"
        product_name = f"{self.product}_E{self.year}_GLOBE_{self.release}_{self.coord_system}_{resolution_str}"
        product_version = 2 if self.product == "GHS_SMOD" else 1

        return {
            "resolution_str": resolution_str,
            "product_folder": product_folder,
            "product_name": product_name,
            "product_version": product_version,
        }

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

    def get_tile_path(self, tile_id=None) -> str:
        """Construct and return the path for the configured dataset."""
        info = self._get_product_info()

        tile_path = (
            self.base_path
            / info["product_folder"]
            / (
                f"{info['product_name']}_V{info['product_version']}_0"
                + (f"_{tile_id}" if tile_id else "")
                + ".zip"
            )
        )

        return tile_path

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


class GHSLDataDownloader:
    """A class to handle downloads of WorldPop datasets."""

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
        self.logger = logger or config.get_logger(__name__)
        self.data_store = data_store or LocalDataStore()
        self.config = (
            config if isinstance(config, GHSLDataConfig) else GHSLDataConfig(**config)
        )
        self.config.logger = self.logger

    def _get_country_tiles(
        self,
        country: str,
        data_store: Optional[DataStore] = None,
        country_geom_path: Optional[Union[Path, str]] = None,
    ) -> List:

        def _load_country_geometry(
            country: str,
            data_store: Optional[DataStore] = None,
            country_geom_path: Optional[Union[Path, str]] = None,
        ) -> Union[Polygon, MultiPolygon]:
            """Load country boundary geometry from DataStore or GADM."""

            gdf_admin0 = AdminBoundaries.create(
                country_code=pycountry.countries.lookup(country).alpha_3,
                admin_level=0,
                data_store=data_store,
                path=country_geom_path,
            ).to_geodataframe()

            return gdf_admin0.geometry.iloc[0]

        country_geom = _load_country_geometry(country, data_store, country_geom_path)

        s = STRtree(self.config.tiles_gdf.geometry)
        result = s.query(country_geom, predicate="intersects")

        intersection_tiles = self.config.tiles_gdf.iloc[result].reset_index(drop=True)

        return [tile for tile in intersection_tiles.tile_id]

    def _get_intersecting_tiles(
        self, geometry: Union[Polygon, MultiPolygon, gpd.GeoDataFrame]
    ) -> List[str]:
        """
        Find all GHSL tiles that intersect with the provided geometry.

        Args:
            geometry: A geometry or GeoDataFrame to check for intersection with GHSL tiles

        Returns:
            List of URLs for GHSL dataset tiles that intersect with the geometry
        """

        if isinstance(geometry, gpd.GeoDataFrame):
            search_geom = geometry.geometry.unary_union
        else:
            search_geom = geometry

        s = STRtree(self.config.tiles_gdf.geometry)
        result = s.query(search_geom, predicate="intersects")

        intersection_tiles = self.config.tiles_gdf.iloc[result].reset_index(drop=True)

        return [tile for tile in intersection_tiles.tile_id]

    def _download_tile(self, tile_id: str) -> str:
        """
        Download the configured dataset to the provided output path.

        Args:
            tile_id: tile ID to download

        Returns:
            path to extracted files
        """

        try:
            response = requests.get(
                self.config.compute_dataset_url(tile_id=tile_id), stream=True
            )
            response.raise_for_status()

            output_path = str(self.config.get_tile_path(tile_id=tile_id))

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

    def download_and_extract_tile(self, tile_id, file_pattern=None):
        """
        Download and extract specific files from GHSL dataset tile zip archives.

        Args:
            tile_id: tile ID to download and extract
            file_pattern: Optional regex pattern to filter which files to extract
                        (e.g., '.*\\.tif$' for only TIF files)

        Returns:
            path to extracted files
        """
        output_path = self.config.get_tile_path(tile_id=tile_id).parents[0]

        extracted_files = []

        url = self.config.compute_dataset_url(tile_id=tile_id)
        self.logger.info(f"Downloading zip from {url}")

        try:
            # download zip to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as temp_file:
                with requests.get(url, stream=True) as response:
                    response.raise_for_status()
                    shutil.copyfileobj(response.raw, temp_file)

            with zipfile.ZipFile(temp_file.name, "r") as zip_ref:
                # get list of files in the zip (filter if pattern provided)
                if file_pattern:
                    import re

                    pattern = re.compile(file_pattern)
                    files_to_extract = [
                        f for f in zip_ref.namelist() if pattern.match(f)
                    ]
                else:
                    files_to_extract = zip_ref.namelist()

                for file in files_to_extract:
                    extracted_path = output_path / Path(file).name
                    with zip_ref.open(file) as source, open(
                        extracted_path, "wb"
                    ) as target:
                        shutil.copyfileobj(source, target)
                    extracted_files.append(extracted_path)
                    self.logger.info(f"Extracted {file} to {extracted_path}")

            Path(temp_file.name).unlink()

        except Exception as e:
            self.logger.error(f"Error downloading/extracting tile {tile_id}: {e}")
            raise

        return extracted_files

    def download_by_country(
        self,
        country: str,
        data_store: Optional[DataStore] = None,
        country_geom_path: Optional[Union[str, Path]] = None,
        extract: bool = False,
    ) -> List[str]:
        """
        Download GHSL data for a specific country.

        Args:
            country: ISO 3166-1 alpha-3 country code, ISO alpha-2 country code or country name

        Returns:
            List of paths to downloaded files
        """

        # Get intersecting tiles
        country_tiles = self._get_country_tiles(
            country=country, data_store=data_store, country_geom_path=country_geom_path
        )

        if not country_tiles:
            self.logger.warning(f"There is no matching data for {country}")
            return []

        # Download tiles in parallel
        with multiprocessing.Pool(self.config.n_workers) as pool:
            download_func = functools.partial(
                self._download_tile if not extract else self.download_and_extract_tile
            )
            file_paths = list(
                tqdm(
                    pool.imap(download_func, country_tiles),
                    total=len(country_tiles),
                    desc=f"Downloading for {country}",
                )
            )

        # Filter out None values (failed downloads)
        return [path for path in file_paths if path is not None]

    def download_by_points(
        self, points_gdf: gpd.GeoDataFrame, extract: bool = False
    ) -> List[str]:
        """
        Download GHSL data for areas containing specific points.

        Args:
            points_gdf: GeoDataFrame containing points of interest

        Returns:
            List of paths to downloaded files
        """
        # Get intersecting tiles
        int_tiles = self._get_intersecting_tiles(points_gdf)

        if not int_tiles:
            self.logger.warning(f"There is no matching data for the points")
            return []

        # Download tiles in parallel
        with multiprocessing.Pool(self.config.n_workers) as pool:
            download_func = functools.partial(
                self._download_tile if not extract else self.download_and_extract_tile
            )
            file_paths = list(
                tqdm(
                    pool.imap(download_func, int_tiles),
                    total=len(int_tiles),
                    desc=f"Downloading for points dataset",
                )
            )

        # Filter out None values (failed downloads)
        return [path for path in file_paths if path is not None]
