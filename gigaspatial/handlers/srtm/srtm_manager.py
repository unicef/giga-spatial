import os
import re
from pathlib import Path
from typing import Tuple, List, Optional, Union
from functools import lru_cache
import numpy as np
import pandas as pd

from gigaspatial.handlers.srtm.srtm_parser import SRTMParser
from gigaspatial.handlers.srtm.nasa_srtm import NasaSRTMDownloader
from gigaspatial.core.io.data_store import DataStore
from gigaspatial.core.io.local_data_store import LocalDataStore


class SRTMManager:
    """
    Manager for accessing elevation data across multiple SRTM .hgt.zip files.

    Implements lazy loading with LRU caching for efficient memory usage.
    Automatically handles multiple tiles for elevation profiles.
    """

    def __init__(
        self,
        srtm_directory: Union[str, Path],
        downloader: NasaSRTMDownloader = None,
        cache_size: int = 10,
        data_store: Optional[DataStore] = None,
    ):
        """
        Initialize the SRTM Manager.

        Parameters
        ----------
        srtm_directory : str or Path
            Directory containing .hgt.zip files
        downloader : optional
            Downloader instance for auto-downloading missing tiles
        cache_size : int, default=10
            Maximum number of SRTM tiles to keep in memory (LRU cache)
        data_store : DataStore, optional
            Data store for reading files. Priority: provided data_store >
            downloader.data_store > LocalDataStore()
        """
        self.srtm_directory = Path(srtm_directory)
        self.downloader = downloader

        # Set data_store: use provided, otherwise downloader's, otherwise LocalDataStore
        if data_store is not None:
            self.data_store = data_store
        elif downloader is not None and hasattr(downloader, "data_store"):
            self.data_store = downloader.data_store
        else:
            self.data_store = LocalDataStore()

        # Check if directory exists
        if not self.data_store.is_dir(str(self.srtm_directory)):
            raise FileNotFoundError(f"Directory not found: {self.srtm_directory}")

        # Build index of available tiles
        self.tile_index = self._build_tile_index()

        # Set up LRU cache for lazy loading
        self._get_parser_cached = lru_cache(maxsize=cache_size)(self._load_parser)

    def _build_tile_index(self) -> dict:
        """
        Build an index of available SRTM tiles in the directory.

        Returns
        -------
        dict
            Mapping of (lat, lon) tuple to file path
        """
        tile_index = {}

        # Pattern to match SRTM filenames: N00E000 or S00W000
        pattern = re.compile(r"^([NS])(\d{2})([EW])(\d{3})")

        # List files using DataStore
        file_list = self.data_store.list_files(str(self.srtm_directory))

        for file_path_str in file_list:
            if file_path_str.endswith(".hgt.zip"):
                # Extract just the filename for pattern matching
                file_name = Path(file_path_str).name
                file_stem = Path(file_name).stem

                match = pattern.match(file_stem)

                if match:
                    lat_dir, lat_val, lon_dir, lon_val = match.groups()

                    lat = int(lat_val) if lat_dir == "N" else -int(lat_val)
                    lon = int(lon_val) if lon_dir == "E" else -int(lon_val)

                    # Use the path as returned by DataStore (will be used for reading)
                    tile_index[(lat, lon)] = file_path_str

        return tile_index

    def _get_tile_coordinates(
        self, latitude: float, longitude: float
    ) -> Tuple[int, int]:
        """
        Get the tile coordinates (southwest corner) for a given lat/lon.

        Parameters
        ----------
        latitude : float
            Latitude in decimal degrees
        longitude : float
            Longitude in decimal degrees

        Returns
        -------
        tuple of (lat_tile, lon_tile)
            Southwest corner coordinates of the tile
        """
        # SRTM tiles are 1x1 degree, named by their southwest corner
        lat_tile = int(np.floor(latitude))
        lon_tile = int(np.floor(longitude))

        return lat_tile, lon_tile

    def _load_parser(self, lat_tile: int, lon_tile: int):
        """
        Load a SRTMParser for a specific tile (used with LRU cache).

        Parameters
        ----------
        lat_tile : int
            Tile latitude (southwest corner)
        lon_tile : int
            Tile longitude (southwest corner)

        Returns
        -------
        SRTMParser
            Parser instance for the tile
        """
        tile_key = (lat_tile, lon_tile)

        if tile_key not in self.tile_index:
            if self.downloader:
                # Auto-download missing tile
                from shapely.geometry import box

                # Create tile_info following the pattern from NasaSRTMConfig
                tile_id = self.downloader.config._tile_name(lat_tile, lon_tile)
                tile_url = f"{self.downloader.config.BASE_URL}/{tile_id}.SRTMGL{self.downloader.config._res_arc}.hgt.zip"

                tile_info = {
                    "tile_id": tile_id,
                    "geometry": box(lon_tile, lat_tile, lon_tile + 1, lat_tile + 1),
                    "tile_url": tile_url,
                }

                # Use download_data_unit for direct download
                self.downloader.download_data_unit(tile_info)

                # Rebuild index to find new tile
                self.tile_index = self._build_tile_index()

                # Check if tile is now available
                if tile_key not in self.tile_index:
                    raise FileNotFoundError(
                        f"SRTM tile for ({lat_tile}, {lon_tile}) could not be downloaded to {self.srtm_directory}"
                    )
            else:
                raise FileNotFoundError(
                    f"SRTM tile for ({lat_tile}, {lon_tile}) not found in {self.srtm_directory}"
                )

        return SRTMParser(self.tile_index[tile_key], data_store=self.data_store)

    def get_elevation(self, latitude: float, longitude: float) -> float:
        """
        Get interpolated elevation for a specific coordinate.

        Automatically finds and loads the correct SRTM tile.

        Parameters
        ----------
        latitude : float
            Latitude in decimal degrees (-90 to 90)
        longitude : float
            Longitude in decimal degrees (-180 to 180)

        Returns
        -------
        float
            Interpolated elevation in meters

        Raises
        ------
        FileNotFoundError
            If the required SRTM tile is not available
        """
        # Get tile coordinates
        lat_tile, lon_tile = self._get_tile_coordinates(latitude, longitude)

        # Load parser (cached)
        parser = self._get_parser_cached(lat_tile, lon_tile)

        # Get elevation
        return parser.get_elevation(latitude, longitude)

    def get_elevation_batch(self, coordinates: np.ndarray) -> np.ndarray:
        """
        Get elevations for multiple coordinates efficiently.

        Groups coordinates by tile to minimize parser loads.

        Parameters
        ----------
        coordinates : np.ndarray of shape (n, 2)
            Array of (latitude, longitude) pairs

        Returns
        -------
        np.ndarray of shape (n,)
            Elevations in meters

        Raises
        ------
        FileNotFoundError
            If any required SRTM tile is not available
        """
        elevations = np.zeros(len(coordinates))

        # Group coordinates by tile
        tile_groups = {}
        for i, (lat, lon) in enumerate(coordinates):
            tile_key = self._get_tile_coordinates(lat, lon)
            if tile_key not in tile_groups:
                tile_groups[tile_key] = []
            tile_groups[tile_key].append((i, lat, lon))

        # Process each tile group
        for tile_key, coords_list in tile_groups.items():
            parser = self._get_parser_cached(*tile_key)

            # Extract coordinates for this tile
            indices = [c[0] for c in coords_list]
            tile_coords = np.array([[c[1], c[2]] for c in coords_list])

            # Get elevations
            tile_elevations = parser.get_elevation_batch(tile_coords)

            # Store results
            elevations[indices] = tile_elevations

        return elevations

    def get_elevation_profile(
        self,
        start_lat: float,
        start_lon: float,
        end_lat: float,
        end_lon: float,
        num_points: int = 100,
    ) -> pd.DataFrame:
        """
        Get elevation profile between two points.

        Uses linear interpolation between points and automatically handles multiple SRTM tiles.
        For more accurate great circle paths over long distances, consider using geopy.

        Parameters
        ----------
        start_lat : float
            Starting latitude in decimal degrees
        start_lon : float
            Starting longitude in decimal degrees
        end_lat : float
            Ending latitude in decimal degrees
        end_lon : float
            Ending longitude in decimal degrees
        num_points : int, default=100
            Number of sample points along the path

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: distance_km, latitude, longitude, elevation

        Raises
        ------
        FileNotFoundError
            If any required SRTM tile along the path is not available
        """
        # Generate points along the path (linear interpolation)
        lats = np.linspace(start_lat, end_lat, num_points)
        lons = np.linspace(start_lon, end_lon, num_points)

        coordinates = np.column_stack((lats, lons))

        # Get elevations for all points
        elevations = self.get_elevation_batch(coordinates)

        # Calculate distances using Haversine formula
        distances = self._calculate_cumulative_distances(lats, lons)

        # Create DataFrame
        profile = pd.DataFrame(
            {
                "distance_km": distances,
                "latitude": lats,
                "longitude": lons,
                "elevation": elevations,
            }
        )

        return profile

    @staticmethod
    def _calculate_cumulative_distances(
        lats: np.ndarray, lons: np.ndarray
    ) -> np.ndarray:
        """
        Calculate cumulative distances along a path using Haversine formula.

        Parameters
        ----------
        lats : np.ndarray
            Array of latitudes
        lons : np.ndarray
            Array of longitudes

        Returns
        -------
        np.ndarray
            Cumulative distances in kilometers
        """
        R = 6371.0  # Earth radius in km

        distances = np.zeros(len(lats))

        for i in range(1, len(lats)):
            # Haversine formula
            lat1, lon1 = np.radians(lats[i - 1]), np.radians(lons[i - 1])
            lat2, lon2 = np.radians(lats[i]), np.radians(lons[i])

            dlat = lat2 - lat1
            dlon = lon2 - lon1

            a = (
                np.sin(dlat / 2) ** 2
                + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
            )
            c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

            distances[i] = distances[i - 1] + R * c

        return distances

    def get_available_tiles(self) -> List[Tuple[int, int]]:
        """
        Get list of available SRTM tiles.

        Returns
        -------
        list of tuples
            List of (lat, lon) tile coordinates
        """
        return list(self.tile_index.keys())

    def check_coverage(self, latitude: float, longitude: float) -> bool:
        """
        Check if a specific coordinate has SRTM coverage.

        Parameters
        ----------
        latitude : float
            Latitude in decimal degrees
        longitude : float
            Longitude in decimal degrees

        Returns
        -------
        bool
            True if tile is available, False otherwise
        """
        tile_key = self._get_tile_coordinates(latitude, longitude)
        return tile_key in self.tile_index

    def clear_cache(self):
        """Clear the LRU cache of loaded parsers."""
        self._get_parser_cached.cache_clear()

    def get_cache_info(self):
        """
        Get cache statistics.

        Returns
        -------
        CacheInfo
            Named tuple with hits, misses, maxsize, currsize
        """
        return self._get_parser_cached.cache_info()

    def __repr__(self):
        return (
            f"SRTMManager(directory={self.srtm_directory}, "
            f"tiles={len(self.tile_index)}, "
            f"cache_size={self._get_parser_cached.cache_info().maxsize}, "
            f"data_store={type(self.data_store).__name__})"
        )
