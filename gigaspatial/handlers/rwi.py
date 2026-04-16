"""
Relative Wealth Index (RWI) data handler.

This module provides specialized handlers for the Meta/Facebook Relative Wealth
Index datasets hosted on HDX. It extends the HDX handler to support:
- Automatic selection of the latest RWI resources.
- Quadkey generation from point locations (Zoom 14).
- Seamless conversion to geospatial tiles.
"""

import logging
from typing import Optional, Union, Literal
from pydantic.dataclasses import dataclass
from datetime import datetime

from pydantic import Field, ConfigDict

from gigaspatial.core.io.data_store import DataStore
from gigaspatial.handlers.hdx import HDXConfig, HDXDownloader, HDXReader, HDXHandler
from gigaspatial.grid.mercator_tiles import MercatorTiles


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class RWIConfig(HDXConfig):
    """
    Configuration for Relative Wealth Index (RWI) data access.

    Attributes:
        dataset_name: Fixed to 'relative-wealth-index'.
        country: Optional ISO country code for resource filtering.
        latest_only: If True, filters for the most recent creation date.
    """

    # Override dataset_name to be fixed for RWI
    dataset_name: Literal["relative-wealth-index"] = Field(
        default="relative-wealth-index"
    )

    # Additional RWI-specific configurations
    country: Optional[str] = Field(
        default=None, description="Country ISO code to filter data for"
    )
    latest_only: bool = Field(
        default=True,
        description="If True, only get the latest resource for each country",
    )

    def __post_init__(self):
        super().__post_init__()

    def get_relevant_data_units(
        self, source: str, force_recompute: bool = False, **kwargs
    ):
        """
        Identify relevant RWI resources on HDX for a given source.

        Args:
            source: Geographic source (e.g., country name).
            force_recompute: If True, bypasses the internal cache.
            **kwargs: Additional filtering parameters.

        Returns:
            A list of matching HDX Resource objects.
        """
        key = self._cache_key(source, **kwargs)
        # Use token_match=True to prevent 'af' matching 'caf'
        resources = super().get_relevant_data_units(
            source, force_recompute, token_match=True, **kwargs
        )

        if self.latest_only and len(resources) > 1:
            # Find the resource with the latest creation date
            latest_resource = None
            latest_date = None

            for resource in resources:
                created = resource.get("created")
                if created:
                    try:
                        created_dt = datetime.fromisoformat(
                            created.replace("Z", "+00:00")
                        )
                        if latest_date is None or created_dt > latest_date:
                            latest_date = created_dt
                            latest_resource = resource
                    except ValueError:
                        self.logger.warning(
                            f"Could not parse creation date for resource: {created}"
                        )

            if latest_resource:
                resources = [latest_resource]

            # Update the cache to the latest only
            self._unit_cache[key] = (resources, self._unit_cache[key][1])
            return resources

        return resources


class RWIDownloader(HDXDownloader):
    """
    Downloader for RWI datasets.

    Handles acquisition of wealth index CSVs from the HDX platform.
    """

    def __init__(
        self,
        config: Union[RWIConfig, dict] = None,
        data_store: Optional[DataStore] = None,
        logger: Optional[logging.Logger] = None,
    ):
        config = config if isinstance(config, RWIConfig) else RWIConfig(**config)
        super().__init__(config=config, data_store=data_store, logger=logger)


class RWIReader(HDXReader):
    """
    Reader for RWI datasets.

    Parses wealth index CSVs into tabular formats, ensuring quadkey consistency.
    """

    def __init__(
        self,
        config: Union[RWIConfig, dict] = None,
        data_store: Optional[DataStore] = None,
        logger: Optional[logging.Logger] = None,
    ):
        config = config if isinstance(config, RWIConfig) else RWIConfig(**config)
        super().__init__(config=config, data_store=data_store, logger=logger)


class RWIHandler(HDXHandler):
    """
    Unified handler for Relative Wealth Index data.

    Coordinates acquisition, reading, and quadkey-based spatial processing
    of wealth index resources.
    """

    def __init__(
        self,
        config: Optional[RWIConfig] = None,
        downloader: Optional[RWIDownloader] = None,
        reader: Optional[RWIReader] = None,
        data_store: Optional[DataStore] = None,
        logger: Optional[logging.Logger] = None,
        **kwargs,
    ):
        super().__init__(
            dataset_name="relative-wealth-index",
            config=config,
            downloader=downloader,
            reader=reader,
            data_store=data_store,
            logger=logger,
            **kwargs,
        )

    def create_config(
        self, data_store: DataStore, logger: logging.Logger, **kwargs
    ) -> RWIConfig:
        """
        Create an RWI configuration instance.

        Args:
            data_store: Storage backend for local files.
            logger: Component logger.
            **kwargs: Configuration overrides.

        Returns:
            A configured RWIConfig.
        """
        return RWIConfig(
            data_store=data_store,
            logger=logger,
            **kwargs,
        )

    def create_downloader(
        self,
        config: RWIConfig,
        data_store: DataStore,
        logger: logging.Logger,
        **kwargs,
    ) -> RWIDownloader:
        """
        Create an RWI downloader instance.

        Args:
            config: Handler configuration.
            data_store: Storage backend for local files.
            logger: Component logger.
            **kwargs: Downloader parameters.

        Returns:
            A configured RWIDownloader.
        """
        return RWIDownloader(
            config=config,
            data_store=data_store,
            logger=logger,
            **kwargs,
        )

    def create_reader(
        self,
        config: RWIConfig,
        data_store: DataStore,
        logger: logging.Logger,
        **kwargs,
    ) -> RWIReader:
        """
        Create an RWI reader instance.

        Args:
            config: Handler configuration.
            data_store: Storage backend for local files.
            logger: Component logger.
            **kwargs: Reader parameters.

        Returns:
            A configured RWIReader.
        """
        return RWIReader(
            config=config,
            data_store=data_store,
            logger=logger,
            **kwargs,
        )

    def load_data(
        self,
        source,
        crop_to_source: bool = False,
        ensure_available: bool = True,
        **kwargs,
    ):
        """
        Acquire and process RWI data, ensuring quadkey availability.

        Args:
            source: Geographic source or direct file paths.
            crop_to_source: If True, filters results to the source boundary.
            ensure_available: If True, executes download if data is missing locally.
            **kwargs: Additional processing parameters.

        Returns:
            A pandas DataFrame of wealth index results with quadkeys.
        """
        data = super().load_data(source, crop_to_source, ensure_available, **kwargs)
        if "quadkey" not in data:
            quadkeys = MercatorTiles.get_quadkeys_from_points(
                data[["latitude", "longitude"]].to_numpy(), zoom_level=14
            )
            if len(quadkeys) != len(data):
                self.logger.warning(
                    "Number of data points does not match the quadkey count, returning original dataframe"
                )
                return data
            data["quadkey"] = quadkeys
            return data
        data["quadkey"] = data["quadkey"].apply(str)
        return data

    def load_as_geodataframe(
        self,
        source,
        crop_to_source: bool = False,
        ensure_available: bool = True,
        **kwargs,
    ):
        """
        Acquire and load RWI data as a geospatial GeoDataFrame of tiles.

        Args:
            source: Geographic source or direct file paths.
            crop_to_source: If True, filters results to the source boundary.
            ensure_available: If True, executes download if data is missing locally.
            **kwargs: Additional processing parameters.

        Returns:
            A GeoDataFrame containing wealth index data mapped to spatial tiles.
        """
        data = self.load_data(source, crop_to_source, ensure_available, **kwargs)
        tiles = MercatorTiles.from_quadkeys(data.quadkey.to_list()).to_geodataframe()
        gdf_rwi = tiles.merge(data, on="quadkey")
        return gdf_rwi
