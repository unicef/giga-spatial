# gigaspatial/handlers/gee/profiler.py

import ee
import pandas as pd
import geopandas as gpd
import geemap
from typing import Optional, Union, List, Dict, Literal
from datetime import datetime
import os

from gigaspatial.config import config as globalconfig
from gigaspatial.core.io.data_store import DataStore
from gigaspatial.core.io.local_data_store import LocalDataStore
from .config import GEEConfig, get_default_registry

LOGGER = globalconfig.get_logger("GEEProfiler")


class GEEProfiler:
    """
    Google Earth Engine profiler for inspecting and mapping datasets.

    Provides comprehensive functionality to:
    - Inspect GEE collections (bands, dates, properties, metadata)
    - Map values to point locations with optional buffers
    - Map values to polygon zones with spatial aggregation
    - Extract temporal profiles and time series
    - Download data for offline use

    Examples
    --------
    >>> # Initialize with dataset ID (uses built-in registry)
    >>> profiler = GEEProfiler(dataset_id="nightlights")
    >>> profiler.display_collection_info()

    >>> # Map to schools with buffers
    >>> enriched = profiler.map_to_points(
    ...     gdf=schools,
    ...     band="avg_rad",
    ...     reducer="mean",
    ...     buffer_radius_m=1000,
    ...     start_date="2020-01-01",
    ...     end_date="2020-12-31"
    ... )

    >>> # Map to admin zones
    >>> zones = profiler.map_to_zones(
    ...     gdf=admin_boundaries,
    ...     band="avg_rad",
    ...     reducer="sum"
    ... )
    """

    def __init__(
        self,
        dataset_id: Optional[str] = None,
        collection: Optional[Union[str, ee.ImageCollection, ee.Image]] = None,
        service_account: Optional[str] = globalconfig.GOOGLE_SERVICE_ACCOUNT,
        key_path: Optional[str] = globalconfig.GOOGLE_SERVICE_ACCOUNT_KEY_PATH,
        project_id: Optional[str] = globalconfig.GOOGLE_CLOUD_PROJECT,
        data_store: Optional[DataStore] = None,
        **config_overrides,
    ):
        """
        Initialize GEE profiler.

        Parameters
        ----------
        dataset_id : str, optional
            Dataset ID from built-in registry (e.g., "nightlights", "population").
            If provided, automatically loads collection and metadata.
        collection : str, ee.ImageCollection, or ee.Image, optional
            Direct GEE collection ID or object. Overrides dataset_id if both provided.
        service_account : str, optional
            Google service account email for authentication
        key_path : str, optional
            Path to service account JSON key file
        project_id : str, optional
            Google Cloud project ID
        **config_overrides
            Additional configuration parameters to override defaults
            (e.g., band="avg_rad", reducer="median", scale=1000)

        Examples
        --------
        >>> # Use built-in dataset
        >>> profiler = GEEProfiler(dataset_id="nightlights")

        >>> # Use custom collection
        >>> profiler = GEEProfiler(collection="NOAA/VIIRS/DNB/MONTHLY_V1/VCMCFG")

        >>> # With authentication
        >>> profiler = GEEProfiler(
        ...     dataset_id="nightlights",
        ...     service_account="my-account@project.iam.gserviceaccount.com",
        ...     key_path="/path/to/key.json"
        ... )
        """
        self.data_store = data_store or LocalDataStore()
        # Initialize Earth Engine API
        self._initialize_ee(service_account, key_path, project_id, self.data_store)

        # Build config
        if dataset_id:
            self.config = GEEConfig.from_dataset_id(
                dataset_id=dataset_id,
                service_account=service_account,
                key_path=key_path,
                project_id=project_id,
                **config_overrides,
            )
        else:
            self.config = GEEConfig(
                service_account=service_account,
                key_path=key_path,
                project_id=project_id,
                **config_overrides,
            )

        # Set collection
        self._image_collection = None
        self._feature_collection = None
        self._number_of_images = 0
        self._number_of_features = 0

        if collection:
            self.image_collection = collection
        elif self.config.collection:
            self.image_collection = self.config.collection

    @staticmethod
    def _initialize_ee(
        service_account: str,
        key_path: str,
        project_id: str,
        data_store: str = LocalDataStore(),
    ):
        """Initialize Earth Engine API using Pydantic-style config fields."""

        if not project_id:
            raise ValueError(
                "GOOGLE_CLOUD_PROJECT must be set to initialize Earth Engine."
            )

        try:
            if service_account and key_path and data_store.file_exists(key_path):
                # Service Account flow
                credentials = ee.ServiceAccountCredentials(service_account, key_path)
                ee.Initialize(credentials, project=project_id)
                LOGGER.info(f"Initialized GEE with Service Account: {service_account}")
            else:
                # Fallback flow (looks for User Auth or ADC)
                # This is helpful for local dev where you might not have a JSON key
                ee.Initialize(project=project_id)
                LOGGER.info(
                    f"Initialized GEE with Default Credentials for project: {project_id}"
                )

        except Exception as e:
            LOGGER.error(f"Failed to initialize Earth Engine: {e}")
            raise

    # ========== Properties ==========

    @property
    def image_collection(self) -> Union[ee.ImageCollection, ee.Image]:
        """Get the current image collection."""
        return self._image_collection

    @image_collection.setter
    def image_collection(self, value: Union[str, ee.ImageCollection, ee.Image]):
        """Set image collection with validation."""
        value = self._init_image_collection(value)
        if self._is_valid_image_collection(value):
            self._image_collection = value
        else:
            raise ValueError("Given image collection is not valid!")

    @property
    def feature_collection(self) -> Union[gpd.GeoDataFrame, ee.FeatureCollection]:
        """Get the current feature collection."""
        return self._feature_collection

    @feature_collection.setter
    def feature_collection(self, value: Union[gpd.GeoDataFrame, ee.FeatureCollection]):
        """Set feature collection with validation."""
        if self._is_valid_feature_collection(value):
            self._feature_collection = value
        else:
            raise ValueError("Given feature collection is not valid!")

    # ========== Collection Initialization & Validation ==========

    def _init_image_collection(
        self, value: Union[str, ee.ImageCollection, ee.Image]
    ) -> Union[ee.ImageCollection, ee.Image]:
        """Initialize image collection from string, ImageCollection, or Image."""
        if isinstance(value, (ee.ImageCollection, ee.Image)):
            return value

        if isinstance(value, str):
            # Use ee.data.getAsset to check what this actually is on GEE servers
            try:
                asset_info = ee.data.getAsset(value)
                asset_type = asset_info.get("type")

                if asset_type == "IMAGE_COLLECTION":
                    return ee.ImageCollection(value)
                elif asset_type == "IMAGE":
                    return ee.Image(value)
                else:
                    raise ValueError(
                        f"Asset '{value}' is type {asset_type}, not an Image or ImageCollection."
                    )
            except Exception as e:
                # Fallback logic if getAsset fails (e.g., permissions)
                LOGGER.warning(
                    f"Could not verify asset type for {value}, attempting default load: {e}"
                )

                # Try Collection first, then Image
                try:
                    test_col = ee.ImageCollection(value)
                    # Hit to the server to check validity
                    test_col.limit(1).size().getInfo()
                    return test_col
                except:
                    return ee.Image(value)

        return value

    def _is_valid_image_collection(
        self, value: Union[ee.ImageCollection, ee.Image]
    ) -> bool:
        """Validate image collection."""
        if not isinstance(value, ee.Image):
            try:
                number_of_images = value.size().getInfo()
            except Exception as e:
                LOGGER.error(f"Collection validation failed: {e}")
                return False

            if number_of_images > 0:
                self._number_of_images = number_of_images
                return True
            else:
                LOGGER.error("There are no images in the image collection!")
                return False
        else:
            self._number_of_images = 1
            return True

    def _is_valid_feature_collection(
        self, value: Union[gpd.GeoDataFrame, ee.FeatureCollection]
    ) -> bool:
        """Validate feature collection."""
        if isinstance(value, ee.FeatureCollection):
            try:
                number_of_features = value.size().getInfo()
            except Exception as e:
                LOGGER.error(f"Feature collection validation failed: {e}")
                return False
        elif isinstance(value, gpd.GeoDataFrame):
            number_of_features = len(value)
        else:
            return False

        if number_of_features > 0:
            self._number_of_features = number_of_features
            return True

        LOGGER.error("There are no features in the feature collection!")
        return False

    # ========== Inspection Methods ==========

    def get_band_names(self) -> List[str]:
        """
        Get all band names from the collection or image.

        Returns
        -------
        list
            List of band names

        Examples
        --------
        >>> profiler = GEEProfiler(dataset_id="nightlights")
        >>> bands = profiler.get_band_names()
        >>> print(bands)
        ['avg_rad', 'cf_cvg']
        """
        if isinstance(self.image_collection, ee.ImageCollection):
            # Check if collection is empty first
            first_image = self.image_collection.first()
            if first_image is None:
                # Return bands from the registry metadata if available
                return self.config.supported_bands or []

            try:
                return ee.Image(first_image).bandNames().getInfo()
            except ee.EEException:
                # Fallback to config if the collection is empty/unreachable
                return self.config.supported_bands or []
        else:
            return self.image_collection.bandNames().getInfo()

    def display_band_names(self):
        """
        Print all available band names.

        Examples
        --------
        >>> profiler.display_band_names()
        Available bands (2):
          1. avg_rad
          2. cf_cvg
        """
        bands = self.get_band_names()
        print(f"Available bands ({len(bands)}):")
        for i, band in enumerate(bands, 1):
            print(f"  {i}. {band}")

    def get_date_range(self, date_format: str = "%Y-%m-%d") -> Dict[str, str]:
        """
        Get the date range of an ImageCollection.

        Parameters
        ----------
        date_format : str
            Date format string (default: "%Y-%m-%d")

        Returns
        -------
        dict
            Dictionary with 'min' and 'max' dates

        Raises
        ------
        TypeError
            If collection is an Image (not ImageCollection)

        Examples
        --------
        >>> date_range = profiler.get_date_range()
        >>> print(date_range)
        {'min': '2012-04-01', 'max': '2024-12-01'}
        """
        if not isinstance(self.image_collection, ee.ImageCollection):
            raise TypeError(
                "get_date_range() only works with ImageCollection, not Image"
            )

        date_range = self.image_collection.reduceColumns(
            ee.Reducer.minMax(), ["system:time_start"]
        ).getInfo()

        min_date = datetime.fromtimestamp(date_range["min"] / 1000).strftime(
            date_format
        )
        max_date = datetime.fromtimestamp(date_range["max"] / 1000).strftime(
            date_format
        )

        return {"min": min_date, "max": max_date}

    def display_date_range(self, date_format: str = "%Y-%m-%d"):
        """
        Print the date range of the ImageCollection.

        Parameters
        ----------
        date_format : str
            Date format string (default: "%Y-%m-%d")

        Examples
        --------
        >>> profiler.display_date_range()
        Date range:
          From: 2012-04-01
          To:   2024-12-01
        """
        try:
            date_range = self.get_date_range(date_format)
            print(
                f"Date range:\n  From: {date_range['min']}\n  To:   {date_range['max']}"
            )
        except TypeError as e:
            print(f"Cannot get date range: {e}")

    def get_collection_size(self) -> int:
        """
        Get the number of images in the collection.

        Returns
        -------
        int
            Number of images
        """
        return self._number_of_images

    def display_collection_info(self):
        """
        Print comprehensive information about the collection.

        Examples
        --------
        >>> profiler.display_collection_info()
        ============================================================
        GEE Collection Information
        ============================================================
        Dataset ID: nightlights
        Collection: NOAA/VIIRS/DNB/MONTHLY_V1/VCMCFG
        Type: ImageCollection
        Number of images: 154

        Bands:
        Available bands (2):
          1. avg_rad
          2. cf_cvg

        Temporal coverage:
        Date range:
          From: 2012-04-01
          To:   2024-12-01

        Configuration:
          Resolution: 463.83 m
          Default reducer: mean
          Temporal cadence: monthly
        ============================================================
        """
        print("=" * 60)
        print("GEE Collection Information")
        print("=" * 60)

        # Dataset metadata
        if self.config.dataset_id:
            print(f"Dataset ID: {self.config.dataset_id}")
        if self.config.collection:
            print(f"Collection: {self.config.collection}")

        # Type
        coll_type = (
            "ImageCollection"
            if isinstance(self.image_collection, ee.ImageCollection)
            else "Image"
        )
        print(f"Type: {coll_type}")

        # Size
        print(f"Number of images: {self.get_collection_size()}")

        # Bands
        print("\nBands:")
        self.display_band_names()

        # Date range (if ImageCollection)
        if isinstance(self.image_collection, ee.ImageCollection):
            print("\nTemporal coverage:")
            self.display_date_range()

        # Config info
        if self.config.dataset_id:
            print("\nConfiguration:")
            if self.config.scale:
                print(f"  Resolution: {self.config.scale} m")
            if self.config.reducer:
                print(f"  Default reducer: {self.config.reducer}")
            if self.config.temporal_cadence:
                print(f"  Temporal cadence: {self.config.temporal_cadence}")

        print("=" * 60)

    def get_properties(self, image_index: int = 0) -> Dict:
        """
        Get properties of an image from the collection.

        Parameters
        ----------
        image_index : int
            Index of image to inspect (default: 0 for first image)

        Returns
        -------
        dict
            Image properties
        """
        if isinstance(self.image_collection, ee.ImageCollection):
            image_list = self.image_collection.toList(self.image_collection.size())
            image = ee.Image(image_list.get(image_index))
        else:
            image = self.image_collection

        return image.getInfo()["properties"]

    def display_properties(self, image_index: int = 0):
        """
        Print properties of an image.

        Parameters
        ----------
        image_index : int
            Index of image to inspect (default: 0)
        """
        props = self.get_properties(image_index)
        print(f"Properties of image {image_index}:")
        for key, value in props.items():
            print(f"  {key}: {value}")

    # ========== Validation Helpers ==========

    def validate_date_range(self, start_date: str, end_date: str) -> bool:
        """
        Check if requested dates are within collection's date range.

        Parameters
        ----------
        start_date : str
            Start date in YYYY-MM-DD format
        end_date : str
            End date in YYYY-MM-DD format

        Returns
        -------
        bool
            True if dates are valid
        """
        try:
            coll_range = self.get_date_range()
            return (
                coll_range["min"] <= start_date <= coll_range["max"]
                and coll_range["min"] <= end_date <= coll_range["max"]
            )
        except TypeError:
            return True

    def validate_band(self, band_name: str) -> bool:
        """
        Check if band exists in collection.

        Parameters
        ----------
        band_name : str
            Band name to validate

        Returns
        -------
        bool
            True if band exists
        """
        return band_name in self.get_band_names()

    # ========== Profiling/Mapping Methods ==========

    ## ========== Helper Methods for Processing ==========

    def _prepare_image(
        self, band, start_date=None, end_date=None, temporal_reducer=None
    ) -> ee.Image:
        """
        Prepare an ee.Image from the collection with temporal filtering.

        Parameters
        ----------
        band : str
            Band to select
        start_date : str, optional
            Start date for temporal filtering
        end_date : str, optional
            End date for temporal filtering
        temporal_reducer : str, optional
            How to reduce temporally (mean, median, max, etc.)

        Returns
        -------
        ee.Image
            Processed image ready for spatial reduction
        """
        # Get values from config if not passed
        start_date = start_date or self.config.start_date
        end_date = end_date or self.config.end_date

        # Logic for ImageCollection
        if isinstance(self.image_collection, ee.ImageCollection):
            col = self.image_collection.select(band)

            # Apply date filters
            if start_date and end_date:
                col = col.filterDate(start_date, end_date)

            # CRITICAL: If the collection is empty after filtering (common in NRT),
            # fallback to the latest available image so the profiler doesn't crash.
            if col.size().getInfo() == 0:
                LOGGER.warning(
                    "Date filter returned empty collection. Falling back to latest image."
                )
                return ee.Image(
                    self.image_collection.select(band)
                    .sort("system:time_start", False)
                    .first()
                )

            # Apply temporal reduction
            t_reducer = temporal_reducer or self.config.temporal_reducer or "mean"
            reducer_method = getattr(col, t_reducer, col.mean)
            return reducer_method()

        return self.image_collection.select(band)

    def _chunk_geodataframe(
        self, gdf: gpd.GeoDataFrame, chunk_size: int
    ) -> List[gpd.GeoDataFrame]:
        """
        Split a GeoDataFrame into chunks.

        Parameters
        ----------
        gdf : GeoDataFrame
            GeoDataFrame to chunk
        chunk_size : int
            Number of features per chunk

        Returns
        -------
        list
            List of GeoDataFrame chunks
        """
        chunks = [gdf.iloc[i : i + chunk_size] for i in range(0, len(gdf), chunk_size)]
        return chunks

    def _process_chunk(
        self,
        chunk: gpd.GeoDataFrame,
        image: ee.Image,
        band: str,
        reducer: str,
        scale: float,
        max_pixels: int,
        tile_scale: int,
    ) -> gpd.GeoDataFrame:
        """
        Process a single chunk by reducing image over features.

        Parameters
        ----------
        chunk : GeoDataFrame
            Chunk of features to process
        image : ee.Image
            Prepared image to reduce
        band : str
            Band name (for output column naming)
        reducer : str
            Spatial reducer name
        scale : float
            Spatial resolution in meters
        max_pixels : int
            Maximum pixels per region
        tile_scale : int
            Tile scale factor

        Returns
        -------
        GeoDataFrame
            Processed chunk with reduction results
        """
        # Convert GeoDataFrame to ee.FeatureCollection
        # force the chunk to 4326 before sending to GEE to avoid CRS errors
        ee_features = geemap.geojson_to_ee(chunk.to_crs("EPSG:4326").__geo_interface__)

        # Get the appropriate reducer
        ee_reducer = self.config.get_ee_reducer()

        # Perform spatial reduction
        try:
            reduced = image.reduceRegions(
                collection=ee_features,
                reducer=ee_reducer,
                scale=scale,
                crs=self.config.crs,
                tileScale=tile_scale,
                maxPixelsPerRegion=max_pixels,  # Correct parameter name
            )

            # Convert back to GeoDataFrame
            processed_geojson = reduced.getInfo()
            gdf_chunk = gpd.GeoDataFrame.from_features(processed_geojson, crs=chunk.crs)

        except Exception as e:
            LOGGER.error(f"Error processing chunk: {e}")
            # Return empty chunk with same structure as input
            gdf_chunk = chunk.copy()
            gdf_chunk[f"{band}_{reducer}"] = None
            return gdf_chunk

        # Handle minMax reducer special case
        if reducer == "minMax":
            if "min" in gdf_chunk.columns and "max" in gdf_chunk.columns:
                gdf_chunk["minMaxDiff"] = gdf_chunk["max"] - gdf_chunk["min"]
                gdf_chunk = gdf_chunk.drop(columns=["max", "min"])
                output_col = f"{band}_minMaxDiff"
                gdf_chunk = gdf_chunk.rename(columns={"minMaxDiff": output_col})
            else:
                LOGGER.warning("minMax reducer did not return min/max columns")
        else:
            # Rename reducer output column
            if reducer in gdf_chunk.columns:
                gdf_chunk = gdf_chunk.rename(columns={reducer: f"{band}_{reducer}"})

        return gdf_chunk

    def _reduce_regions_with_chunking(
        self,
        gdf: gpd.GeoDataFrame,
        band: str,
        reducer: str,
        scale: float,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        temporal_reducer: Optional[str] = None,
        chunk_size: int = 1000,
        max_pixels: int = int(1e8),
        tile_scale: int = 1,
    ) -> gpd.GeoDataFrame:
        """
        Core method to reduce image over regions with automatic chunking.

        This is the unified processing pipeline used by both map_to_points
        and map_to_zones.
        """
        # Prepare the image
        LOGGER.info(f"Preparing image: band={band}, dates={start_date} to {end_date}")
        image = self._prepare_image(band, start_date, end_date, temporal_reducer)

        # Chunk the geodataframe
        chunks = self._chunk_geodataframe(gdf, chunk_size)
        num_chunks = len(chunks)
        LOGGER.info(f"Processing {len(gdf)} features in {num_chunks} chunks")

        # Process each chunk
        results = []
        for idx, chunk in enumerate(chunks):
            LOGGER.info(
                f"Processing chunk {idx + 1}/{num_chunks} ({len(chunk)} features)"
            )

            try:
                gdf_chunk = self._process_chunk(
                    chunk=chunk,
                    image=image,
                    band=band,
                    reducer=reducer,
                    scale=scale,
                    max_pixels=max_pixels,
                    tile_scale=tile_scale,
                )
                results.append(gdf_chunk)
                LOGGER.info(f"Chunk {idx + 1}/{num_chunks} completed")
            except Exception as e:
                LOGGER.error(f"Failed to process chunk {idx + 1}: {e}")
                # Append original chunk with null values
                chunk_copy = chunk.copy()
                chunk_copy[f"{band}_{reducer}"] = None
                results.append(chunk_copy)

        # Merge all chunks
        LOGGER.info("Merging all chunks...")
        merged_gdf = gpd.GeoDataFrame(
            pd.concat(results, ignore_index=True), crs=gdf.crs
        )

        LOGGER.info("Merging completed!")

        return merged_gdf

    ## ========== Public API Methods ==========

    def map_to_points(
        self,
        gdf: gpd.GeoDataFrame,
        band: Optional[str] = None,
        reducer: Optional[str] = None,
        buffer_radius_m: Optional[float] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        temporal_reducer: Optional[str] = None,
        chunk_size: Optional[int] = None,
        scale: Optional[float] = None,
    ) -> gpd.GeoDataFrame:
        """
        Map GEE values to point locations with optional circular buffers.

        This method extracts values from the Earth Engine image/collection
        at point locations. If buffer_radius_m is provided, circular buffers
        are created around points and spatial reduction is applied within
        each buffer.

        Parameters
        ----------
        gdf : GeoDataFrame
            Point locations to enrich (must have Point geometries)
        band : str, optional
            Band to extract (uses config default if None)
        reducer : str, optional
            Spatial reducer: mean, median, min, max, sum, etc.
            (uses config default if None)
        buffer_radius_m : float, optional
            Buffer radius in meters around each point.
            If None or 0, point values are extracted directly.
        start_date : str, optional
            Start date YYYY-MM-DD for temporal filtering
            (uses config if None)
        end_date : str, optional
            End date YYYY-MM-DD for temporal filtering
            (uses config if None)
        temporal_reducer : str, optional
            How to aggregate over time if multiple images: mean, median, max, etc.
            (uses config default if None)
        chunk_size : int, optional
            Features per chunk for API rate limiting
            (uses config default if None)
        scale : float, optional
            Spatial resolution in meters
            (uses config/dataset default if None)

        Returns
        -------
        GeoDataFrame
            Enriched GeoDataFrame with new column: {band}_{reducer}

        Examples
        --------
        >>> # Extract nightlight values at school points with 1km buffers
        >>> profiler = GEEProfiler(dataset_id="nightlights")
        >>> enriched = profiler.map_to_points(
        ...     gdf=schools,
        ...     band="avg_rad",
        ...     reducer="mean",
        ...     buffer_radius_m=1000,
        ...     start_date="2020-01-01",
        ...     end_date="2020-12-31"
        ... )
        >>> print(enriched[["school_id", "avg_rad_mean"]].head())
        """
        from gigaspatial.processing.geo import buffer_geodataframe

        # Validate input
        if not isinstance(gdf, gpd.GeoDataFrame):
            raise TypeError(f"gdf must be GeoDataFrame, got {type(gdf)}")

        if len(gdf) == 0:
            raise ValueError("Input GeoDataFrame is empty")

        # Set defaults from config
        band = band or self.config.band
        reducer = reducer or self.config.reducer
        chunk_size = chunk_size or self.config.chunk_size
        scale = scale or self.config.scale
        buffer_radius_m = buffer_radius_m or self.config.buffer_radius_m
        start_date = start_date or self.config.start_date
        end_date = end_date or self.config.end_date

        if band is None:
            raise ValueError(
                "band must be specified (either as parameter or in config)"
            )

        # Validate band
        if not self.validate_band(band):
            raise ValueError(
                f"Band '{band}' not found. Available: {self.get_band_names()}"
            )

        # Validate dates if ImageCollection
        if (
            start_date
            and end_date
            and isinstance(self.image_collection, ee.ImageCollection)
        ):
            if not self.validate_date_range(start_date, end_date):
                LOGGER.warning(
                    f"Requested dates ({start_date} to {end_date}) may be outside "
                    f"collection range. Available: {self.get_date_range()}"
                )

        # Apply buffers if requested
        if buffer_radius_m and buffer_radius_m > 0:
            gdf_to_process = buffer_geodataframe(gdf, buffer_radius_m)
        else:
            gdf_to_process = gdf.copy()

        # Process with chunking
        result = self._reduce_regions_with_chunking(
            gdf=gdf_to_process,
            band=band,
            reducer=reducer,
            scale=scale,
            start_date=start_date,
            end_date=end_date,
            temporal_reducer=temporal_reducer,
            chunk_size=chunk_size,
            max_pixels=self.config.max_pixels,
            tile_scale=self.config.tile_scale,
        )

        # Restore original geometry if buffers were applied
        if buffer_radius_m and buffer_radius_m > 0:
            result.geometry = gdf.geometry.values

        return result

    def map_to_zones(
        self,
        gdf: gpd.GeoDataFrame,
        band: Optional[str] = None,
        reducer: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        temporal_reducer: Optional[str] = None,
        chunk_size: Optional[int] = None,
        scale: Optional[float] = None,
    ) -> gpd.GeoDataFrame:
        """
        Map GEE values to polygon zones with spatial aggregation.

        This method aggregates values from the Earth Engine image/collection
        within polygon boundaries (e.g., administrative zones, grid cells).

        Parameters
        ----------
        gdf : GeoDataFrame
            Polygon zones to aggregate over
        band : str, optional
            Band to extract (uses config default if None)
        reducer : str, optional
            Spatial reducer: mean, median, min, max, sum, etc.
            (uses config default if None)
        start_date : str, optional
            Start date YYYY-MM-DD for temporal filtering
        end_date : str, optional
            End date YYYY-MM-DD for temporal filtering
        temporal_reducer : str, optional
            How to aggregate over time: mean, median, max, etc.
        chunk_size : int, optional
            Features per chunk for API rate limiting
        scale : float, optional
            Spatial resolution in meters

        Returns
        -------
        GeoDataFrame
            Enriched GeoDataFrame with new column: {band}_{reducer}

        Examples
        --------
        >>> # Aggregate population within admin boundaries
        >>> profiler = GEEProfiler(dataset_id="population")
        >>> zones_enriched = profiler.map_to_zones(
        ...     gdf=admin_boundaries,
        ...     band="population_density",
        ...     reducer="sum",
        ...     start_date="2020-01-01",
        ...     end_date="2020-12-31"
        ... )
        >>> print(zones_enriched[["admin_name", "population_density_sum"]].head())
        """
        # Validate input
        if not isinstance(gdf, gpd.GeoDataFrame):
            raise TypeError(f"gdf must be GeoDataFrame, got {type(gdf)}")

        if len(gdf) == 0:
            raise ValueError("Input GeoDataFrame is empty")

        # Set defaults from config
        band = band or self.config.band
        reducer = reducer or self.config.reducer
        chunk_size = chunk_size or self.config.chunk_size
        scale = scale or self.config.scale
        start_date = start_date or self.config.start_date
        end_date = end_date or self.config.end_date

        if band is None:
            raise ValueError(
                "band must be specified (either as parameter or in config)"
            )

        # Validate band
        if not self.validate_band(band):
            raise ValueError(
                f"Band '{band}' not found. Available: {self.get_band_names()}"
            )

        # Validate dates if ImageCollection
        if (
            start_date
            and end_date
            and isinstance(self.image_collection, ee.ImageCollection)
        ):
            if not self.validate_date_range(start_date, end_date):
                LOGGER.warning(
                    f"Requested dates ({start_date} to {end_date}) may be outside "
                    f"collection range. Available: {self.get_date_range()}"
                )

        # Process with chunking (no buffering for zones)
        result = self._reduce_regions_with_chunking(
            gdf=gdf,
            band=band,
            reducer=reducer,
            scale=scale,
            start_date=start_date,
            end_date=end_date,
            temporal_reducer=temporal_reducer,
            chunk_size=chunk_size,
            max_pixels=self.config.max_pixels,
            tile_scale=self.config.tile_scale,
        )

        return result
