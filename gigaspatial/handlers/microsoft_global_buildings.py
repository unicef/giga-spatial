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
from gigaspatial.grid.mercator_tiles import (
    MercatorTiles,
    CountryMercatorTiles,
)
from gigaspatial.handlers.base import (
    BaseHandlerReader,
    BaseHandlerConfig,
    BaseHandlerDownloader,
    BaseHandler,
)
from gigaspatial.config import config as global_config


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class MSBuildingsConfig(BaseHandlerConfig):
    """Configuration for Microsoft Global Buildings dataset files."""

    TILE_URLS: str = (
        "https://minedbuildings.z5.web.core.windows.net/global-buildings/dataset-links.csv"
    )
    MERCATOR_ZOOM_LEVEL: int = 9
    base_path: Path = global_config.get_path("microsoft_global_buildings", "bronze")

    LOCATION_MAPPING_FILE: Path = base_path / "location_mapping.json"
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

    def __post_init__(self):
        """Initialize the configuration, load tile URLs, and set up location mapping."""
        super().__post_init__()
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

    def get_relevant_data_units_by_geometry(
        self, geometry: Union[BaseGeometry, gpd.GeoDataFrame], **kwargs
    ) -> pd.DataFrame:
        """
        Return intersecting tiles for a given geometry or GeoDataFrame.
        """
        return self._get_relevant_tiles(geometry)

    def get_relevant_data_units_by_points(
        self, points: Iterable[Union[Point, tuple]], **kwargs
    ) -> pd.DataFrame:
        """
        Return intersecting tiles for a list of points.
        """
        return self._get_relevant_tiles(points)

    def get_relevant_data_units_by_country(
        self, country: str, **kwargs
    ) -> pd.DataFrame:
        """
        Return intersecting tiles for a given country.
        """
        return self._get_relevant_tiles(country)

    def get_data_unit_path(self, unit: Union[pd.Series, dict], **kwargs) -> Path:

        tile_location = unit["country"] if unit["country"] else unit["location"]

        return (
            self.base_path
            / tile_location
            / self.upload_date
            / f'{unit["quadkey"]}.csv.gz'
        )

    def get_data_unit_paths(
        self, units: Union[pd.DataFrame, Iterable[dict]], **kwargs
    ) -> List:
        if isinstance(units, pd.DataFrame):
            return [self.get_data_unit_path(row) for _, row in units.iterrows()]
        return super().get_data_unit_paths(units)

    def _get_relevant_tiles(
        self,
        source: Union[
            str,  # country
            BaseGeometry,  # shapely geoms
            gpd.GeoDataFrame,
            Iterable[Union[Point, Tuple[float, float]]],  # points
        ],
    ) -> pd.DataFrame:
        """
        Get the DataFrame of Microsoft Buildings tiles that intersect with a given source spatial geometry.

        In case country given, this method first tries to find tiles directly mapped to the given country.
        If no directly mapped tiles are found and the country is not in the location
        mapping, it attempts to find overlapping tiles by creating Mercator tiles
        for the country and filtering the dataset's tiles.

        Args:
            source: A country code/name, a Shapely geometry, a GeoDataFrame, or a list of Point
                    objects or (lat, lon) tuples representing the area of interest.
                    The coordinates are assumed to be in EPSG:4326.

        Returns:
            A pandas DataFrame containing the rows from the tiles list that
            spatially intersect with the `source`. Returns an empty DataFrame
            if no intersecting tiles are found.
        """
        if isinstance(source, str):
            try:
                country_code = pycountry.countries.lookup(source).alpha_3
            except:
                raise ValueError("Invalid`country` value!")

            mask = self.df_tiles["country"] == country_code

            if any(mask):
                return self.df_tiles.loc[
                    mask, ["quadkey", "url", "country", "location"]
                ].to_dict("records")

            self.logger.warning(
                f"The country code '{country_code}' is not directly in the location mapping. "
                "Manually checking for overlapping locations with the country boundary."
            )

            source_tiles = CountryMercatorTiles.create(
                country_code, self.MERCATOR_ZOOM_LEVEL
            )
        else:
            source_tiles = MercatorTiles.from_spatial(
                source=source, zoom_level=self.MERCATOR_ZOOM_LEVEL
            )

        filtered_tiles = source_tiles.filter_quadkeys(self.df_tiles.quadkey)

        mask = self.df_tiles.quadkey.isin(filtered_tiles.quadkeys)

        return self.df_tiles.loc[
            mask, ["quadkey", "url", "country", "location"]
        ].to_dict("records")


class MSBuildingsDownloader(BaseHandlerDownloader):
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
        config = config or MSBuildingsConfig()
        super().__init__(config=config, data_store=data_store, logger=logger)

    def download_data_unit(
        self,
        tile_info: Union[pd.Series, dict],
        **kwargs,
    ) -> Optional[str]:
        """Download data file for a single tile."""

        tile_url = tile_info["url"]

        try:
            response = requests.get(tile_url, stream=True)
            response.raise_for_status()

            file_path = str(self.config.get_data_unit_path(tile_info))

            with self.data_store.open(file_path, "wb") as file:
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
        except Exception as e:
            self.logger.error(f"Unexpected error downloading dataset: {str(e)}")
            return None

    def download_data_units(
        self,
        tiles: Union[pd.DataFrame, List[dict]],
        **kwargs,
    ) -> List[str]:
        """Download data files for multiple tiles."""

        if len(tiles) == 0:
            self.logger.warning(f"There is no matching data")
            return []

        with multiprocessing.Pool(self.config.n_workers) as pool:
            download_func = functools.partial(self.download_data_unit)
            file_paths = list(
                tqdm(
                    pool.imap(
                        download_func,
                        (
                            [row for _, row in tiles.iterrows()]
                            if isinstance(tiles, pd.DataFrame)
                            else tiles
                        ),
                    ),
                    total=len(tiles),
                    desc=f"Downloading polygons data",
                )
            )

        return [path for path in file_paths if path is not None]

    def download(
        self,
        source: Union[
            str,  # country
            List[Union[Tuple[float, float], Point]],  # points
            BaseGeometry,  # shapely geoms
            gpd.GeoDataFrame,
        ],
        **kwargs,
    ) -> List[str]:
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
            **kwargs: Additional parameters passed to data unit resolution methods

        Returns:
            A list of local file paths for the successfully downloaded tiles.
            Returns an empty list if no data is found for the region or if
            all downloads fail.
        """

        tiles = self.config.get_relevant_data_units(source, **kwargs)
        return self.download_data_units(tiles, **kwargs)

    def download_by_country(
        self,
        country: str,
        data_store: Optional[DataStore] = None,
        country_geom_path: Optional[Union[str, Path]] = None,
    ) -> List[str]:
        """
        Download Microsoft Global ML Building Footprints data for a specific country.

        This is a convenience method to download data for an entire country
        using its code or name.

        Args:
            country: The country code (e.g., 'USA', 'GBR') or name.
            data_store: Optional instance of a `DataStore` to be used by
                `AdminBoundaries` for loading country boundaries. If None,
                `AdminBoundaries` will use its default data loading.
            country_geom_path: Optional path to a GeoJSON file containing the
                country boundary. If provided, this boundary is used
                instead of the default from `AdminBoundaries`.

        Returns:
            A list of local file paths for the successfully downloaded tiles.
            Returns an empty list if no data is found for the country or if
            all downloads fail.
        """
        return self.download(
            source=country, data_store=data_store, path=country_geom_path
        )


class MSBuildingsReader(BaseHandlerReader):
    """
    Reader for Microsoft Global Buildings data, supporting country, points, and geometry-based resolution.
    """

    def __init__(
        self,
        config: Optional[MSBuildingsConfig] = None,
        data_store: Optional[DataStore] = None,
        logger: Optional[logging.Logger] = None,
    ):
        config = config or MSBuildingsConfig()
        super().__init__(config=config, data_store=data_store, logger=logger)

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


class MSBuildingsHandler(BaseHandler):
    """
    Handler for Microsoft Global Buildings dataset.

    This class provides a unified interface for downloading and loading Microsoft Global Buildings data.
    It manages the lifecycle of configuration, downloading, and reading components.
    """

    def create_config(
        self, data_store: DataStore, logger: logging.Logger, **kwargs
    ) -> MSBuildingsConfig:
        """
        Create and return a MSBuildingsConfig instance.

        Args:
            data_store: The data store instance to use
            logger: The logger instance to use
            **kwargs: Additional configuration parameters

        Returns:
            Configured MSBuildingsConfig instance
        """
        return MSBuildingsConfig(data_store=data_store, logger=logger, **kwargs)

    def create_downloader(
        self,
        config: MSBuildingsConfig,
        data_store: DataStore,
        logger: logging.Logger,
        **kwargs,
    ) -> MSBuildingsDownloader:
        """
        Create and return a MSBuildingsDownloader instance.

        Args:
            config: The configuration object
            data_store: The data store instance to use
            logger: The logger instance to use
            **kwargs: Additional downloader parameters

        Returns:
            Configured MSBuildingsDownloader instance
        """
        return MSBuildingsDownloader(
            config=config, data_store=data_store, logger=logger, **kwargs
        )

    def create_reader(
        self,
        config: MSBuildingsConfig,
        data_store: DataStore,
        logger: logging.Logger,
        **kwargs,
    ) -> MSBuildingsReader:
        """
        Create and return a MSBuildingsReader instance.

        Args:
            config: The configuration object
            data_store: The data store instance to use
            logger: The logger instance to use
            **kwargs: Additional reader parameters

        Returns:
            Configured MSBuildingsReader instance
        """
        return MSBuildingsReader(
            config=config, data_store=data_store, logger=logger, **kwargs
        )
