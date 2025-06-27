import logging
from typing import List, Optional, Union, Literal
from pydantic.dataclasses import dataclass
from datetime import datetime
import pycountry

from hdx.data.resource import Resource

from pydantic import Field, ConfigDict

from gigaspatial.core.io.data_store import DataStore
from gigaspatial.handlers.hdx import HDXConfig, HDXDownloader, HDXReader, HDXHandler


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class RWIConfig(HDXConfig):
    """Configuration for Relative Wealth Index data access"""

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

    def get_relevant_data_units_by_country(
        self, country: str, **kwargs
    ) -> List[Resource]:
        """Get relevant data units for a country, optionally filtering for latest version"""
        country = pycountry.countries.lookup(country)
        values = [country.alpha_3]
        resources = self.get_dataset_resources(
            filter={"url": values},
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

        return resources


class RWIDownloader(HDXDownloader):
    """Specialized downloader for the Relative Wealth Index dataset from HDX"""

    def __init__(
        self,
        config: Union[RWIConfig, dict] = None,
        data_store: Optional[DataStore] = None,
        logger: Optional[logging.Logger] = None,
    ):
        config = config if isinstance(config, RWIConfig) else RWIConfig(**config)
        super().__init__(config=config, data_store=data_store, logger=logger)


class RWIReader(HDXReader):
    """Specialized reader for the Relative Wealth Index dataset from HDX"""

    def __init__(
        self,
        config: Union[RWIConfig, dict] = None,
        data_store: Optional[DataStore] = None,
        logger: Optional[logging.Logger] = None,
    ):
        config = config if isinstance(config, RWIConfig) else RWIConfig(**config)
        super().__init__(config=config, data_store=data_store, logger=logger)


class RWIHandler(HDXHandler):
    """Handler for Relative Wealth Index dataset"""

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
        """Create and return a RWIConfig instance"""
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
        """Create and return a RWIDownloader instance"""
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
        """Create and return a RWIReader instance"""
        return RWIReader(
            config=config,
            data_store=data_store,
            logger=logger,
            **kwargs,
        )
