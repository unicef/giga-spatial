from dataclasses import field
from pydantic.dataclasses import dataclass
from pydantic import ConfigDict
from pathlib import Path
import functools
import multiprocessing
from typing import List, Optional, Tuple, Union, Dict, Iterable
import numpy as np
import pandas as pd
from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry
from difflib import SequenceMatcher
import pycountry
import requests
from tqdm import tqdm
import logging
import geopandas as gpd

from gigaspatial.core.io.data_store import DataStore
from gigaspatial.core.io.local_data_store import LocalDataStore
from gigaspatial.grid.mercator_tiles import (
    MercatorTiles,
    CountryMercatorTiles,
)
from gigaspatial.handlers.base_reader import BaseHandlerReader
from gigaspatial.config import config as global_config


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class MSBuildingsConfig:
    """Configuration for Microsoft Global Buildings dataset files."""

    data_store: DataStore = field(
        default_factory=LocalDataStore
    )  # instance of DataStore for accessing data storage
    BASE_PATH: Path = global_config.get_path("microsoft_global_buildings", "bronze")

    TILE_URLS: str = (
        "https://minedbuildings.z5.web.core.windows.net/global-buildings/dataset-links.csv"
    )
    MERCATOR_ZOOM_LEVEL: int = 9

    LOCATION_MAPPING_FILE: Path = BASE_PATH / "location_mapping.json"
    SIMILARITY_SCORE: float = 0.8
    DEFAULT_MAPPING: Dict[str, str] = field(
        default_factory=lambda: {
            "Bonaire": "BES",
            "Brunei": "BRN",
            "IvoryCoast": "CIV",
            "CongoDRC": "COD",
            "DemocraticRepublicoftheCongo": "COD",
            "RepublicoftheCongo": "COG",
            "TheGambia": "GMB",
            "FYROMakedonija": "MKD",
            "SultanateofOman": "OMN",
            "StateofQatar": "QAT",
            "Russia": "RUS",
            "KingdomofSaudiArabia": "SAU",
            "Svalbard": "SJM",
            "Swaziland": "SWZ",
            "StMartin": "SXM",
            "leSaint-Martin": "MAF",
            "Turkey": "TUR",
            "VaticanCity": "VAT",
            "BritishVirginIslands": "VGB",
            "USVirginIslands": "VIR",
            "RepublicofYemen": "YEM",
            "CzechRepublic": "CZE",
            "French-Martinique": "MTQ",
            "French-Guadeloupe": "GLP",
            "UnitedStates": "USA",
        }
    )
    CUSTOM_MAPPING: Optional[Dict[str, str]] = None

    n_workers: int = 4  # number of workers for parallel processing
    logger: logging.Logger = None  # global_config.get_logger(__name__)

    def __post_init__(self):
        """Initialize the configuration, load tile URLs, and set up location mapping."""

        self.logger = global_config.get_logger(self.__class__.__name__)
        self._load_tile_urls()
        self.upload_date = self.df_tiles.upload_date[0]
        self._setup_location_mapping()

    def _load_tile_urls(self):
        """Load dataset links from csv file."""
        self.df_tiles = pd.read_csv(
            self.TILE_URLS,
            names=["location", "quadkey", "url", "size", "upload_date"],
            dtype={"quadkey": str},
            header=0,
        )

    def _setup_location_mapping(self):
        """Load or create the mapping between dataset locations and ISO country codes."""
        from gigaspatial.core.io.readers import read_json
        from gigaspatial.core.io.writers import write_json

        if self.data_store.file_exists(str(self.LOCATION_MAPPING_FILE)):
            self.location_mapping = read_json(
                self.data_store, str(self.LOCATION_MAPPING_FILE)
            )
        else:
            self.location_mapping = self.create_location_mapping(
                similarity_score_threshold=self.SIMILARITY_SCORE
            )
            self.location_mapping.update(self.DEFAULT_MAPPING)
            write_json(
                self.location_mapping, self.data_store, str(self.LOCATION_MAPPING_FILE)
            )

        self.location_mapping.update(self.CUSTOM_MAPPING or {})
        self._map_locations()
        self.df_tiles.loc[self.df_tiles.country.isnull(), "country"] = None

    def _map_locations(self):
        """Map the 'location' column in the tiles DataFrame to ISO country codes."""
        self.df_tiles["country"] = self.df_tiles.location.map(self.location_mapping)

    def create_location_mapping(self, similarity_score_threshold: float = 0.8):
        """
        Create a mapping between the dataset's location names and ISO 3166-1 alpha-3 country codes.

        This function iterates through known countries and attempts to find matching
        locations in the dataset based on string similarity.

        Args:
            similarity_score_threshold: The minimum similarity score (between 0 and 1)
                                        for a dataset location to be considered a match
                                        for a country. Defaults to 0.8.

        Returns:
            A dictionary where keys are dataset location names and values are
            the corresponding ISO 3166-1 alpha-3 country codes.
        """

        def similar(a, b):
            return SequenceMatcher(None, a, b).ratio()

        location_mapping = dict()

        for country in pycountry.countries:
            if country.name not in self.df_tiles.location.unique():
                try:
                    country_quadkey = CountryMercatorTiles.create(
                        country.alpha_3, self.MERCATOR_ZOOM_LEVEL
                    )
                except:
                    self.logger.warning(f"{country.name} is not mapped.")
                    continue
                country_datasets = country_quadkey.filter_quadkeys(
                    self.df_tiles.quadkey
                )
                matching_locations = self.df_tiles[
                    self.df_tiles.quadkey.isin(country_datasets.quadkeys)
                ].location.unique()
                scores = np.array(
                    [
                        (
                            similar(c, country.common_name)
                            if hasattr(country, "common_name")
                            else similar(c, country.name)
                        )
                        for c in matching_locations
                    ]
                )
                if any(scores > similarity_score_threshold):
                    matched = matching_locations[scores > similarity_score_threshold]
                    if len(matched) > 2:
                        self.logger.warning(
                            f"Multiple matches exist for {country.name}. {country.name} is not mapped."
                        )
                    location_mapping[matched[0]] = country.alpha_3
                    self.logger.debug(f"{country.name} matched with {matched[0]}!")
                else:
                    self.logger.warning(
                        f"No direct matches for {country.name}. {country.name} is not mapped."
                    )
                    self.logger.debug("Possible matches are: ")
                    for c, score in zip(matching_locations, scores):
                        self.logger.debug(c, score)
            else:
                location_mapping[country.name] = country.alpha_3

        return location_mapping

    def get_intersecting_tiles(
        self,
        source: Union[
            BaseGeometry,
            gpd.GeoDataFrame,
            List[Union[Point, Tuple[float, float]]],  # points]
        ],
    ) -> pd.DataFrame:
        """
        Get the DataFrame of Microsoft Buildings tiles that intersect with a given spatial geometry.

        Args:
            source: A Shapely geometry, a GeoDataFrame, or a list of Point
                    objects or (lat, lon) tuples representing the area of interest.
                    The coordinates are assumed to be in EPSG:4326.

        Returns:
            A pandas DataFrame containing the rows from the tiles list that
            spatially intersect with the `source`. Returns an empty DataFrame
            if no intersecting tiles are found.
        """
        source_tiles = MercatorTiles.from_spatial(
            source=source, zoom_level=self.MERCATOR_ZOOM_LEVEL
        ).filter_quadkeys(self.df_tiles.quadkey)

        if source_tiles:
            return self.df_tiles[self.df_tiles.quadkey.isin(source_tiles.quadkeys)]

        return pd.DataFrame(columns=self.df_tiles.columns)

    def get_tiles_for_country(self, country: str) -> pd.DataFrame:
        """
        Get the DataFrame of Microsoft Buildings tiles associated with a specific country code.

        This method first tries to find tiles directly mapped to the given country code.
        If no directly mapped tiles are found and the country is not in the location
        mapping, it attempts to find overlapping tiles by creating Mercator tiles
        for the country and filtering the dataset's tiles.

        Args:
            country: The country code or name.

        Returns:
            A pandas DataFrame containing the rows of tiles associated with the
            `country_code`. Returns an empty DataFrame if no tiles are found.
        """
        try:
            country_code = pycountry.countries.lookup(country).alpha_3
        except:
            raise ValueError("Invalid`country` value!")

        country_tiles = self.df_tiles[self.df_tiles["country"] == country_code]

        if not country_tiles.empty:
            return country_tiles

        self.logger.warning(
            f"The country code '{country_code}' is not directly in the location mapping. "
            "Manually checking for overlapping locations with the country boundary."
        )

        country_tiles = CountryMercatorTiles.create(
            country_code, self.MERCATOR_ZOOM_LEVEL
        ).filter_quadkeys(self.df_tiles.quadkey)

        if country_tiles:
            filtered_tiles = self.df_tiles[
                self.df_tiles.country.isnull()
                & self.df_tiles.quadkey.isin(country_tiles.quadkeys)
            ]
            return filtered_tiles

        return pd.DataFrame(columns=self.df_tiles.columns)

    def get_tile_path(self, quadkey: str, location: str) -> Path:
        """Construct the local file path for a downloaded Microsoft Buildings tile."""
        return self.BASE_PATH / location / self.upload_date / f"{quadkey}.csv.gz"

    def get_tile_paths(self, tiles: pd.DataFrame) -> List:
        if tiles.empty:
            return []

        return [
            self.get_tile_path(
                quadkey=tile["quadkey"],
                location=tile["country"] if tile["country"] else tile["location"],
            )
            for _, tile in tiles.iterrows()
        ]


class MSBuildingsDownloader:
    """A class to handle downloads of Microsoft's Global ML Building Footprints dataset."""

    def __init__(
        self,
        config: Optional[MSBuildingsConfig] = None,
        data_store: Optional[DataStore] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the downloader.

        Args:
            config: Optional configuration for customizing download behavior and file paths.
                    If None, a default `MSBuildingsConfig` is used.
            data_store: Optional instance of a `DataStore` for managing data storage.
                        If provided, it overrides the `data_store` in the `config`.
                        If None, the `data_store` from the `config` is used.
            logger: Optional custom logger instance. If None, a default logger
                    named after the module is created and used.
        """
        self.logger = logger or global_config.get_logger(self.__class__.__name__)
        self.data_store = data_store or LocalDataStore()
        self.config = config or MSBuildingsConfig(
            data_store=self.data_store, logger=self.logger
        )

    def _download_tile(
        self,
        tile_info: Union[pd.Series, dict],
    ) -> Optional[str]:
        """Download data file for a single tile."""

        # Modify URL based on data type if needed
        tile_url = tile_info["url"]

        try:
            response = requests.get(tile_url, stream=True)
            response.raise_for_status()

            file_path = str(
                self.config.get_tile_path(
                    quadkey=tile_info["quadkey"],
                    location=(
                        tile_info["country"]
                        if tile_info["country"]
                        else tile_info["location"]
                    ),
                )
            )

            with self.config.data_store.open(file_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)

                self.logger.debug(
                    f"Successfully downloaded tile: {tile_info['quadkey']}"
                )
                return file_path

        except requests.exceptions.RequestException as e:
            self.logger.error(
                f"Failed to download tile {tile_info['quadkey']}: {str(e)}"
            )
            return None

    def _download_tiles(self, tiles: pd.DataFrame):
        """Download data file for multiple tiles."""

        if tiles.empty:
            self.logger.warning(f"There is no matching data")
            return []

        with multiprocessing.Pool(self.config.n_workers) as pool:
            download_func = functools.partial(self._download_tile)
            file_paths = list(
                tqdm(
                    pool.imap(download_func, [row for _, row in tiles.iterrows()]),
                    total=len(tiles),
                    desc=f"Downloading polygons data",
                )
            )

        return [path for path in file_paths if path is not None]

    def download_data(
        self,
        source: Union[
            str,  # country
            List[Union[Tuple[float, float], Point]],  # points
            BaseGeometry,  # shapely geoms
            gpd.GeoDataFrame,
        ],
    ):
        """
        Download Microsoft Global ML Building Footprints data for a specified geographic region.

        The region can be defined by a country, a list of points,
        a Shapely geometry, or a GeoDataFrame. This method identifies the
        relevant data tiles intersecting the region and downloads them in parallel.

        Args:
            source: Defines the geographic area for which to download data.
                    Can be:
                      - A string representing a country code or name.
                      - A list of (latitude, longitude) tuples or Shapely Point objects.
                      - A Shapely BaseGeometry object (e.g., Polygon, MultiPolygon).
                      - A GeoDataFrame with a geometry column in EPSG:4326.

        Returns:
            A list of local file paths for the successfully downloaded tiles.
            Entries will be None for tiles that failed to download.
        """
        if isinstance(source, str):
            return self.download_by_country(country_code=source)
        elif isinstance(source, (BaseGeometry, Iterable)):
            source_tiles = self.config.get_intersecting_tiles(source)
            return self._download_tiles(source_tiles)

        raise ValueError(
            f"Data downloads supported for Country, Geometry, GeoDataFrame or iterable object of Points got {source.__class__}"
        )

    def download_by_country(self, country: str) -> List[str]:
        """
        Download Microsoft Global ML Building Footprints data for a specific country.

        This is a convenience method to download data for an entire country
        using its code or name.

        Args:
            country: The country code (e.g., 'USA', 'GBR') or name.

        Returns:
            A list of local file paths for the successfully downloaded tiles.
            Entries will be None for tiles that failed to download.
        """

        country_tiles = self.config.get_tiles_for_country(country)

        return self._download_tiles(country_tiles)


class MSBuildingsReader(BaseHandlerReader):
    """
    Reader for Microsoft Global Buildings data, supporting country, points, and geometry-based resolution.
    """

    def __init__(
        self,
        config: Optional[MSBuildingsConfig] = None,
        data_store: Optional[DataStore] = None,
    ):
        super().__init__(data_store=data_store)
        self.config = config or MSBuildingsConfig(data_store=self.data_store)

    def _post_load_hook(self, data, **kwargs) -> gpd.GeoDataFrame:
        """Post-processing after loading data files."""
        if data.empty:
            self.logger.warning("No data was loaded from the source files")
            return data

        self.logger.info(
            f"Post-load processing complete. {len(data)} valid building records."
        )
        return data

    def resolve_by_country(self, country: str, **kwargs) -> List[Union[str, Path]]:
        tiles = self.config.get_tiles_for_country(country)
        return self.config.get_tile_paths(tiles)

    def resolve_by_geometry(
        self, geometry: Union[BaseGeometry, gpd.GeoDataFrame], **kwargs
    ) -> List[Union[str, Path]]:
        tiles = self.config.get_intersecting_tiles(geometry)
        return self.config.get_tile_paths(tiles)

    def resolve_by_points(
        self, points: List[Union[Point, Tuple[float, float]]], **kwargs
    ) -> List[Union[str, Path]]:
        return self.resolve_by_geometry(points, **kwargs)

    def load_from_paths(
        self, source_data_path: List[Union[str, Path]], **kwargs
    ) -> gpd.GeoDataFrame:
        """
        Load building data from Microsoft Buildings dataset.
        Args:
            source_data_path: List of file paths to load
        Returns:
            GeoDataFrame containing building data
        """
        from gigaspatial.core.io.readers import read_gzipped_json_or_csv
        from shapely.geometry import shape

        def read_ms_dataset(data_store: DataStore, file_path: str):
            df = read_gzipped_json_or_csv(file_path=file_path, data_store=data_store)
            df["geometry"] = df["geometry"].apply(shape)
            return gpd.GeoDataFrame(df, crs=4326)

        result = self._load_tabular_data(
            file_paths=source_data_path, read_function=read_ms_dataset
        )
        return result
