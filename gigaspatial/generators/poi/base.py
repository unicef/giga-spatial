from abc import ABC, abstractmethod
from pydantic.dataclasses import dataclass
from pydantic import Field
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

import geopandas as gpd
import pandas as pd

from gigaspatial.core.io.data_store import DataStore
from gigaspatial.core.io.local_data_store import LocalDataStore
from gigaspatial.core.io.writers import write_dataset
from gigaspatial.config import config


@dataclass
class PoiViewGeneratorConfig:
    """Configuration for POI view generation."""

    base_path: Path = Field(default=config.get_path("poi", "views"))
    n_workers: int = 4


class PoiViewGenerator(ABC):
    """
    Base class for generating views from Points of Interest (POI) datasets.

    This class provides the structure for processing downloaded data sources
    and mapping them to POI data. Concrete implementations should extend this
    class for specific data sources.
    """

    def __init__(
        self,
        data_config: Optional[Any] = None,
        generator_config: Optional["PoiViewGeneratorConfig"] = None,
        data_store: Optional["DataStore"] = None,
    ):
        """
        Initialize the POI View Generator.

        Args:
            generator_config: Configuration for the view generator
            data_store: Data store for reading/writing data
        """
        self.data_config = data_config
        self.generator_config = generator_config or PoiViewGeneratorConfig()
        self.data_store = data_store or LocalDataStore()
        self.logger = config.get_logger(self.__class__.__name__)

    def resolve_source_paths(
        self,
        poi_data: Union[pd.DataFrame, gpd.GeoDataFrame],
        explicit_paths: Optional[Union[Path, str, List[Union[str, Path]]]] = None,
        **kwargs,
    ) -> List[Union[str, Path]]:
        """
        Resolve source data paths based on POI data or explicit paths.

        This method allows generators to dynamically determine source paths
        based on the POI data (e.g., by geographic intersection).

        Args:
            poi_data: POI data that may be used to determine relevant source paths
            explicit_paths: Explicitly provided source paths, if any
            **kwargs: Additional parameters for path resolution

        Returns:
            List of resolved source paths

        Notes:
            Default implementation returns explicit_paths if provided.
            Subclasses should override this to implement dynamic path resolution.
        """
        if explicit_paths is not None:
            if isinstance(explicit_paths, (str, Path)):
                return [explicit_paths]
            return list(explicit_paths)

        # Raises NotImplementedError if no explicit paths
        # and subclass hasn't overridden this method
        raise NotImplementedError(
            "This generator requires explicit source paths or a subclass "
            "implementation of resolve_source_paths()"
        )

    def _pre_load_hook(self, source_data_path, **kwargs) -> Any:
        """Hook called before loading data"""
        return source_data_path

    def _post_load_hook(self, data, **kwargs) -> Any:
        """Hook called after loading data"""
        return data

    @abstractmethod
    def load_data(self, source_data_path: List[Union[str, Path]], **kwargs) -> Any:
        """
        Load source data for POI processing. This method handles diverse source data formats.

        Args:
            source_data_path: List of source paths
            **kwargs: Additional parameters for data loading

        Returns:
            Data in its source format (DataFrame, GeoDataFrame, TifProcessor, etc.)
        """
        pass

    def process_data(self, data: Any, **kwargs) -> Any:
        """Process the source data to prepare it for POI view generation."""
        return data
        raise NotImplementedError("Subclasses must implement this method...")

    @abstractmethod
    def map_to_poi(
        self,
        processed_data: Any,
        poi_data: Union[pd.DataFrame, gpd.GeoDataFrame],
        **kwargs,
    ) -> Union[pd.DataFrame, gpd.GeoDataFrame]:
        """
        Map processed data to POI data.

        Args:
            processed_data: Processed source data as a GeoDataFrame
            poi_data: POI data to map to
            **kwargs: Additional mapping parameters

        Returns:
            (Geo)DataFrame with POI data mapped to source data
        """
        pass

    def generate_poi_view(
        self,
        poi_data: Union[pd.DataFrame, gpd.GeoDataFrame],
        source_data_path: Optional[Union[Path, str, List[Union[str, Path]]]] = None,
        custom_pipeline: bool = False,
        **kwargs,
    ) -> Union[pd.DataFrame, gpd.GeoDataFrame]:
        """
        Generate a POI view by running the complete pipeline.

        This method has been updated to make source_data_path optional.
        If not provided, it will use resolve_source_paths() to determine paths.

        Args:
            poi_data: POI data to map to
            source_data_path: Optional explicit path(s) to the source data
            **kwargs: Additional parameters for the pipeline

        Returns:
            DataFrame with the generated POI view
        """
        if custom_pipeline:
            return self._custom_pipeline(source_data_path, poi_data, **kwargs)

        self.logger.info("Starting POI view generation pipeline")

        # Resolve source paths if not explicitly provided
        resolved_paths = self.resolve_source_paths(poi_data, source_data_path, **kwargs)

        if not resolved_paths:
            self.logger.warning(
                "No source data paths resolved. Returning original POI data."
            )
            return poi_data

        # load data from resolved sources
        source_data = self.load_data(resolved_paths, **kwargs)
        self.logger.info("Source data loaded successfully")

        # process the data
        processed_data = self.process_data(source_data, **kwargs)
        self.logger.info("Data processing completed")

        # map to POI
        poi_view = self.map_to_poi(processed_data, poi_data, **kwargs)
        self.logger.info("POI mapping completed")

        return poi_view

    def save_poi_view(
        self,
        poi_view: Union[pd.DataFrame, gpd.GeoDataFrame],
        output_path: Union[Path, str],
        **kwargs,
    ) -> None:
        """
        Save the generated POI view to the data store.

        Args:
            poi_view: The POI view DataFrame to save
            output_path: Path where the POI view will be saved in DataStore
            **kwargs: Additional parameters for saving
        """
        self.logger.info(f"Saving POI view to {output_path}")
        write_dataset(poi_view, self.data_store, output_path, **kwargs)
