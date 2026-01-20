# gigaspatial/handlers/google_microsoft_combined/config.py

import pandas as pd
import geopandas as gpd
from pydantic.dataclasses import dataclass
from pydantic import Field, ConfigDict
from pathlib import Path
from typing import Optional, Literal, List, Union
import pyarrow.parquet as pq
import logging
import pycountry
import requests
from tqdm import tqdm
import re

from gigaspatial.core.io.data_store import DataStore
from gigaspatial.handlers.base import (
    BaseHandlerConfig,
    BaseHandlerDownloader,
    BaseHandlerReader,
    BaseHandler,
)
from gigaspatial.config import config as global_config


def get_available_iso3s(
    data_format: Literal["geoparquet", "flatgeobuf", "pmtiles"] = "geoparquet",
    url: str = "https://source.coop/vida/google-microsoft-open-buildings/{data_format}/by_country",
):
    url = url.format(data_format=data_format)

    response = requests.get(url, timeout=10)
    # Raise an HTTPError for bad responses (4xx or 5xx)
    response.raise_for_status()

    iso3_codes = set()
    iso_pattern = re.compile(r"country_iso=([A-Z]{3})")

    # Search the entire HTML content as a last resort
    matches = iso_pattern.findall(response.text)
    iso3_codes.update(matches)

    return sorted(list(iso3_codes))


def get_available_s2_files(
    iso3: str,
    data_format: Literal["geoparquet", "flatgeobuf"],
    url: str = "https://source.coop/vida/google-microsoft-open-buildings/{data_format}/by_country_s2/country_iso={iso3}",
) -> List[str]:
    """
    Fetches the content of a URL and searches for files matching the S2 grid cell ID pattern.

    Args:
        url: The URL pointing to the file listing or index.

    Returns:
        A sorted list of unique S2 grid cell file names found.
    """
    url = url.format(iso3=iso3, data_format=data_format)

    try:
        response = requests.get(url, timeout=10)
        # Raise an HTTPError for bad responses (4xx or 5xx)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error accessing URL {url}: {e}")
        return []

    s2_file_names = set()
    # Pattern to match a sequence of 17-20 digits followed by a file extension (e.g., .parquet, .fgb)
    # The S2 Cell ID is typically an unsigned 64-bit integer, which has 19-20 digits.
    # We capture the full filename.
    s2_pattern = re.compile(r"(\d{17,20}\.(?:parquet|fgb))")

    # Search the entire HTML or text content for matches
    matches = s2_pattern.findall(response.text)
    s2_file_names.update(matches)

    return sorted(list(s2_file_names))


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class GoogleMSBuildingsConfig(BaseHandlerConfig):
    """
    Configuration for Google-Microsoft Combined Buildings Handler.

    This handler provides access to the merged Google V3 Open Buildings
    and Microsoft Global Building Footprints dataset from VIDAsource.coop.

    Attributes:
        data_format: File format to access (default: 'geoparquet')
        partition_strategy: How to partition queries ('country' or 's2_grid')
        cache_dir: Directory to cache downloaded partitions
        storage_url: Base URL for cloud storage
        cloud_uri: Base URI for s3 access
    """

    base_path: Path = Field(
        default=global_config.get_path("google_microsoft_buildings", "bronze")
    )
    data_format: Literal["geoparquet", "flatgeobuf", "pmtiles"] = "geoparquet"
    partition_strategy: Literal["country", "s2_grid"] = "country"
    download_strategy: Literal["s3", "https"] = "s3"
    cache_dir: Optional[str] = None
    storage_url: str = "https://data.source.coop/vida/google-microsoft-open-buildings"
    cloud_uri: str = (
        "s3://us-west-2.opendata.source.coop/vida/google-microsoft-open-buildings"
    )

    # Metadata
    dataset_name: str = "Google-Microsoft Combined Buildings"
    dataset_id: str = "google_microsoft_combined_buildings"
    version: str = "2.0"
    license_info: str = "CC BY 4.0 + ODbL 1.0"

    def __post_init__(self):
        """Validate configuration parameters."""
        super().__post_init__()

        if self.data_format == "pmtiles" and self.partition_strategy == "s2_grid":
            raise ValueError(
                f"S2 level country data is only available for `geoparquet` and `flatgeobuf` data formats"
            )

        self.logger.info(
            f"GoogleMSBuildingsConfig initialized: format={self.data_format}, "
            f"partition_strategy={self.partition_strategy}"
        )

    def get_relevant_data_units_by_geometry(self, geometry, **kwargs):
        """
        Get relevant data partitions (by country or S2 grid) that intersect geometry.

        Args:
            geometry: Shapely geometry object

        Returns:
            List of partition identifiers (ISO codes or S2 grid IDs)
        """

        access_uri = (
            self.storage_url if self.download_strategy == "https" else self.cloud_uri
        )

        if self.partition_strategy == "s2_grid":
            s2_tiles = get_available_s2_files(
                iso3=geometry, data_format=self.data_format
            )
            return [
                f"{access_uri}/{self.data_format}/by_country_s2/country_iso={geometry}/{tile}"
                for tile in s2_tiles
            ]

        match self.data_format:
            case "geoparquet":
                file_ext = ".parquet"
            case "flatgeobuf":
                file_ext = ".fgb"
            case "pmtiles":
                file_ext = ".pmtiles"

        return [
            f"{access_uri}/{self.data_format}/by_country/country_iso={geometry}/{geometry}{file_ext}"
        ]

    def get_available_units(self, iso3: str = None) -> List[str]:
        """
        Get list of available data units (partitions).

        Args:
            iso3: Optional ISO3 country code to filter. If None, returns all.

        Returns:
            List of available units (URLs/URIs)
        """

        if iso3:
            # Get units for specific country
            return self.get_relevant_data_units_by_geometry(iso3)
        else:
            # Get all available countries
            available_iso3s = get_available_iso3s(data_format=self.config.data_format)
            self.logger.info(f"Found {len(available_iso3s)} available countries")

            all_units = []
            for country_iso3 in available_iso3s:
                units = self.get_relevant_data_units_by_geometry(country_iso3)
                all_units.extend(units)

            return all_units

    def get_data_unit_path(self, unit: str, **kwargs) -> Path:
        return self.base_path / unit.split("google-microsoft-open-buildings/")[
            1
        ].replace("country_iso=", "")

    def extract_search_geometry(self, source, **kwargs):
        """
        Override the method since geometry extraction does not apply.
        Returns country iso3 for url extraction
        """
        if not isinstance(source, str):
            raise ValueError(
                f"Unsupported source type: {type(source)}"
                "Please use country-based (str) filtering."
            )

        iso3 = pycountry.countries.lookup(source).alpha_3

        if iso3 not in get_available_iso3s(data_format=self.data_format):
            raise ValueError(f"Building data for country {iso3} is not available!")

        return iso3


class GoogleMSBuildingsDownloader(BaseHandlerDownloader):
    """
    Downloader for Google-Microsoft Combined Buildings from VIDAsource.coop.

    Handles:
    - Cloud-native GeoParquet/FlatGeobuf/PMTiles access via S3 or HTTPS
    - Partition discovery by country (ISO code) or S2 grid
    - Streaming and local caching
    - Flexible download strategy (S3 or HTTPS)

    Args:
        config: Optional configuration for file paths and download settings.
                If None, a default `GoogleMSBuildingsConfig` is used.
        data_store: Optional instance of a `DataStore` for managing data
                    storage. If None, a `LocalDataStore` is used.
        logger: Optional custom logger instance. If None, a default logger
                named after the module is created and used.
    """

    def __init__(
        self,
        config: Optional[GoogleMSBuildingsConfig] = None,
        data_store: Optional[DataStore] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """Initialize downloader with configuration."""
        config = config or GoogleMSBuildingsConfig()
        super().__init__(config=config, data_store=data_store, logger=logger)

        self.logger.info(
            f"GoogleMSBuildingsDownloader initialized with "
            f"format={config.data_format}, partition_strategy={config.partition_strategy}, "
            f"download_strategy={config.download_strategy}"
        )

    def download_data_unit(self, unit: str) -> Path:
        """
        Download a single data unit (partition file).

        Args:
            unit: Full URL/URI to the data unit (from get_relevant_data_units)

        Returns:
            Path to the downloaded file
        """

        file_path = str(self.config.get_data_unit_path(unit))

        try:
            if self.config.download_strategy == "s3":
                return self._download_from_s3(unit, file_path)
            else:  # https
                return self._download_from_https(unit, file_path)

        except Exception as e:
            self.logger.error(f"Error downloading unit {unit}: {e}")
            raise

    def download_data_units(
        self,
        units: List[str],
    ) -> List[Path]:
        """Download data files for multiple units."""

        if len(units) == 0:
            self.logger.warning(f"There is no matching data")
            return []

        return [
            self.download_data_unit(unit)
            for unit in tqdm(units, desc="Downloading building data")
        ]

    def _download_from_s3(self, s3_uri: str, save_path: Path) -> Path:
        """
        Download file from S3 using s3fs.

        Args:
            s3_uri: S3 URI (s3://bucket/path/to/file)
            save_path: Path to save file

        Returns:
            Path to downloaded file
        """
        import s3fs

        try:
            fs = s3fs.S3FileSystem(anon=True)  # Anonymous access to public bucket

            with fs.open(s3_uri, "rb") as remote_file:
                with self.data_store.open(save_path, "wb") as file:
                    # Stream in chunks to handle large files
                    chunk_size = 8192 * 1024  # 8MB chunks
                    while True:
                        chunk = remote_file.read(chunk_size)
                        if not chunk:
                            break
                        file.write(chunk)

            self.logger.info(f"Downloaded from S3: {save_path}")
            return save_path

        except Exception as e:
            self.logger.error(f"S3 download failed for {s3_uri}: {e}")
            # Clean up partial download
            if self.data_store.file_exists(save_path):
                self.data_store.remove(save_path)
            raise

    def _download_from_https(self, https_url: str, save_path: Path) -> Path:
        """
        Download file from HTTPS.

        Args:
            https_url: HTTPS URL
            save_path: Path to save file

        Returns:
            Path to downloaded file
        """
        import requests

        try:
            response = requests.get(https_url, stream=True, timeout=30)
            response.raise_for_status()

            with self.data_store.open(save_path, "wb") as file:
                # Stream in chunks
                for chunk in response.iter_content(chunk_size=8192 * 1024):
                    if chunk:
                        file.write(chunk)

            self.logger.info(f"Downloaded from HTTPS: {save_path}")
            return save_path

        except Exception as e:
            self.logger.error(f"HTTPS download failed for {https_url}: {e}")
            # Clean up partial download
            if self.data_store.file_exists(save_path):
                self.data_store.remove(save_path)
            raise

    def stream_unit(self, unit: str):
        """
        Stream a data unit directly without caching.

        Args:
            unit: Full URL/URI to the data unit

        Yields:
            PyArrow Table chunks (for Parquet) or file-like object
        """
        self.logger.info(f"Streaming from {unit}")

        if self.config.download_strategy == "s3":
            yield from self._stream_from_s3(unit)
        else:  # https
            yield from self._stream_from_https(unit)

    def _stream_from_s3(self, s3_uri: str):
        """
        Stream file from S3.

        Args:
            s3_uri: S3 URI

        Yields:
            PyArrow Table chunks for Parquet, raw bytes for other formats
        """
        import s3fs

        try:
            fs = s3fs.S3FileSystem(anon=True)

            if self.config.data_format == "geoparquet":
                # Stream Parquet by row groups
                import pyarrow.parquet as pq

                with fs.open(s3_uri, "rb") as f:
                    parquet_file = pq.ParquetFile(f)
                    for i in range(parquet_file.num_row_groups):
                        yield parquet_file.read_row_group(i)
            else:
                # For FlatGeobuf or PMTiles, yield the file object
                with fs.open(s3_uri, "rb") as f:
                    yield f

        except Exception as e:
            self.logger.error(f"S3 streaming failed for {s3_uri}: {e}")
            raise

    def _stream_from_https(self, https_url: str):
        """
        Stream file from HTTPS.

        Args:
            https_url: HTTPS URL

        Yields:
            Data chunks
        """
        import requests
        import tempfile

        try:
            response = requests.get(https_url, stream=True, timeout=30)
            response.raise_for_status()

            if self.config.data_format == "geoparquet":
                # For Parquet, we need to download to temp file first for PyArrow
                import pyarrow.parquet as pq

                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".parquet"
                ) as tmp:
                    for chunk in response.iter_content(chunk_size=8192 * 1024):
                        if chunk:
                            tmp.write(chunk)
                    tmp_path = tmp.name

                try:
                    parquet_file = pq.ParquetFile(tmp_path)
                    for i in range(parquet_file.num_row_groups):
                        yield parquet_file.read_row_group(i)
                finally:
                    Path(tmp_path).unlink()
            else:
                # For other formats, yield chunks
                for chunk in response.iter_content(chunk_size=8192 * 1024):
                    if chunk:
                        yield chunk

        except Exception as e:
            self.logger.error(f"HTTPS streaming failed for {https_url}: {e}")
            raise

    def validate_unit_exists(self, unit: str) -> bool:
        """
        Check if a data unit exists and is accessible.

        Args:
            unit: Full URL/URI to the data unit

        Returns:
            True if unit exists and is accessible
        """
        try:
            if self.config.download_strategy == "s3":
                import s3fs

                fs = s3fs.S3FileSystem(anon=True)
                return fs.exists(unit)
            else:  # https
                import requests

                response = requests.head(unit, timeout=10)
                return response.status_code == 200

        except Exception as e:
            self.logger.warning(f"Error validating unit {unit}: {e}")
            return False


class GoogleMSBuildingsReader(BaseHandlerReader):
    """
    Reader for Google-Microsoft Combined Buildings data.

    Handles:
    - Reading GeoParquet files
    - Source filtering (Google/Microsoft)
    - Confidence score filtering
    - Geometry cropping
    """

    def __init__(
        self,
        config: Optional[GoogleMSBuildingsConfig] = None,
        data_store: Optional[DataStore] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """Initialize reader with configuration."""
        config = config or GoogleMSBuildingsConfig()
        super().__init__(config=config, data_store=data_store, logger=logger)

    def load_from_paths(
        self, source_data_path: List[Union[str, Path]], **kwargs
    ) -> Union[pd.DataFrame, gpd.GeoDataFrame]:
        """
        Args:
            source_data_path: List of file paths to load
        Returns:
            DataFrame containing building data
        """
        result = self._load_tabular_data(file_paths=source_data_path, **kwargs)
        return result

    def _apply_filters(
        self,
        gdf: gpd.GeoDataFrame,
        source_filter: Literal["google", "microsoft"] = None,
        min_confidence: float = None,
    ) -> gpd.GeoDataFrame:
        """
        Apply source and confidence filters to GeoDataFrame.

        source_filter: Filter by source ('google', 'microsoft', or None for both)
        min_confidence: Minimum confidence score (0.0-1.0). None = no filter
        """
        original_count = len(gdf)

        # Source filter
        if source_filter:
            if "source" in gdf.columns:
                gdf = gdf[gdf["source"].str.lower() == source_filter.lower()]
                self.logger.debug(
                    f"Source filter applied: {source_filter} "
                    f"({len(gdf)}/{original_count} rows)"
                )

        # Confidence filter
        if min_confidence is not None:
            if "confidence" in gdf.columns:
                gdf = gdf[gdf["confidence"] >= min_confidence]
                self.logger.debug(
                    f"Confidence filter applied: >= {min_confidence} "
                    f"({len(gdf)}/{original_count} rows)"
                )

        return gdf

    def read_metadata(self, file_path: Union[str, Path]) -> dict:
        """
        Read metadata from GeoParquet file without loading full data.

        Args:
            file_path: Path to GeoParquet file

        Returns:
            Metadata dictionary
        """
        try:
            parquet_file = pq.read_table(file_path, memory_map=True)
            metadata = parquet_file.schema.metadata

            return {
                "num_rows": parquet_file.num_rows,
                "num_columns": parquet_file.num_columns,
                "columns": parquet_file.column_names,
                "metadata": metadata,
            }

        except Exception as e:
            self.logger.error(f"Error reading meta {e}")
            raise


class GoogleMSBuildingsHandler(BaseHandler):
    """
    Handler for Google-Microsoft Combined Buildings Dataset.

    Provides end-to-end workflow for downloading, caching, and reading
    merged Google V3 Open Buildings and Microsoft Global Building Footprints.

    Features:
    - Cloud-native GeoParquet streaming
    - Automatic partition discovery by geometry
    - Source filtering (Google/Microsoft)
    - Confidence score filtering
    - Local caching for repeated access

    Example:
        >>> config = GoogleMSBuildingsConfig()
        >>> handler = GoogleMSBuildingsHandler(config)
        >>> geometry = Polygon([...])  # Your area of interest
        >>> buildings = handler.load_data(geometry)
    """

    def __init__(
        self,
        data_format: Literal["geoparquet", "flatgeobuf", "pmtiles"] = "geoparquet",
        partition_strategy: Literal["country", "s2_grid"] = "country",
        download_strategy: Literal["s3", "https"] = "s3",
        config: Optional[GoogleMSBuildingsConfig] = None,
        downloader: Optional[GoogleMSBuildingsDownloader] = None,
        reader: Optional[GoogleMSBuildingsReader] = None,
        data_store: Optional[DataStore] = None,
        logger: Optional[logging.Logger] = None,
        **kwargs,
    ):
        self._data_format = data_format
        self._partition_strategy = partition_strategy
        self._download_strategy = download_strategy
        super().__init__(
            config=config,
            downloader=downloader,
            reader=reader,
            data_store=data_store,
            logger=logger,
        )

    def create_config(
        self, data_store: DataStore, logger: logging.Logger, **kwargs
    ) -> GoogleMSBuildingsConfig:
        """
        Create and return a GoogleMSBuildingsConfig instance.

        Args:
            data_store: The data store instance to use
            logger: The logger instance to use
            **kwargs: Additional configuration parameters

        Returns:
            Configured GoogleMSBuildingsConfig instance
        """
        return GoogleMSBuildingsConfig(
            data_format=self._data_format,
            partition_strategy=self._partition_strategy,
            download_strategy=self._download_strategy,
            data_store=data_store,
            logger=logger,
            **kwargs,
        )

    def create_downloader(
        self,
        config: GoogleMSBuildingsConfig,
        data_store: DataStore,
        logger: logging.Logger,
        **kwargs,
    ) -> GoogleMSBuildingsDownloader:
        """
        Create and return a GoogleMSBuildingsDownloader instance.

        Args:
            config: The configuration object
            data_store: The data store instance to use
            logger: The logger instance to use
            **kwargs: Additional downloader parameters

        Returns:
            Configured GoogleMSBuildingsDownloader instance
        """
        return GoogleMSBuildingsDownloader(
            config=config, data_store=data_store, logger=logger, **kwargs
        )

    def create_reader(
        self,
        config: GoogleMSBuildingsConfig,
        data_store: DataStore,
        logger: logging.Logger,
        **kwargs,
    ) -> GoogleMSBuildingsReader:
        """
        Create and return a GoogleMSBuildingsReader instance.

        Args:
            config: The configuration object
            data_store: The data store instance to use
            logger: The logger instance to use
            **kwargs: Additional reader parameters

        Returns:
            Configured GoogleMSBuildingsReader instance
        """
        return GoogleMSBuildingsReader(
            config=config, data_store=data_store, logger=logger, **kwargs
        )

    def get_dataset_info(self) -> dict:
        """Get information about the dataset."""
        return {
            "name": self.config.dataset_name,
            "id": self.config.dataset_id,
            "version": self.config.version,
            "license": self.config.license_info,
            "description": (
                "Merged Google V3 Open Buildings (1.8B footprints) and "
                "Microsoft Global Building Footprints (1.24B footprints). "
                "Total: 2.58B building detections across 185 countries."
            ),
            "data_source": "https://source.coop/vida/google-microsoft-open-buildings",
            "formats_available": ["geoparquet", "flatgeobuf", "pmtiles"],
            "num_partitions": 185,
        }

    def get_cell_ids(self, source, **kwargs) -> List[int]:
        paths = self.reader.resolve_source_paths(source, **kwargs)
        return [int(p.stem) for p in paths]
