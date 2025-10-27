import os
import struct
import zipfile
from pathlib import Path
from typing import Tuple, Optional, Union
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
import io

from gigaspatial.core.io.data_store import DataStore
from gigaspatial.core.io.local_data_store import LocalDataStore


class SRTMParser:
    """
    Efficient parser for NASA SRTM .hgt.zip files.

    Supports both SRTM-1 (3601x3601, 1 arc-second) and SRTM-3 (1201x1201, 3 arc-second) formats.
    Uses memory mapping for efficient handling of large files.
    """

    def __init__(
        self, hgt_zip_path: Union[str, Path], data_store: Optional[DataStore] = None
    ):
        """
        Initialize the SRTM parser.

        Parameters
        ----------
        hgt_zip_path : str or Path
            Path to the .hgt.zip file (e.g., 'S03E028.SRTMGL1.hgt.zip')
        data_store : DataStore, optional
            Data store for reading files. If None, uses LocalDataStore()
        """
        self.hgt_zip_path = Path(hgt_zip_path)
        self.data_store = data_store or LocalDataStore()

        # Check if file exists
        if not self.data_store.file_exists(str(self.hgt_zip_path)):
            raise FileNotFoundError(f"File not found: {self.hgt_zip_path}")

        # Extract tile coordinates from filename (e.g., S03E028)
        self._parse_filename()

        # Load the elevation data
        self.data = None
        self.resolution = None
        self.size = None
        self._load_data()

        # Set up interpolator for efficient querying
        self._setup_interpolator()

    def _parse_filename(self):
        """Extract latitude and longitude from the .hgt filename."""
        filename = self.hgt_zip_path.stem.split(".")[
            0
        ]  # Get base name without extensions

        # Parse latitude (first 3 characters: N/S + 2 digits)
        lat_str = filename[:3]
        lat_dir = lat_str[0]
        lat_val = int(lat_str[1:])
        self.lat_corner = lat_val if lat_dir == "N" else -lat_val

        # Parse longitude (next 4 characters: E/W + 3 digits)
        lon_str = filename[3:7]
        lon_dir = lon_str[0]
        lon_val = int(lon_str[1:])
        self.lon_corner = lon_val if lon_dir == "E" else -lon_val

    def _load_data(self):
        """Load elevation data from .hgt.zip file using memory-efficient approach."""
        # Read the zip file from DataStore
        zip_data = self.data_store.read_file(str(self.hgt_zip_path))

        # Create a BytesIO object from the zip data
        zip_file_obj = io.BytesIO(zip_data)

        # Extract .hgt file from zip
        with zipfile.ZipFile(zip_file_obj, "r") as zip_ref:
            # Find the .hgt file inside the zip
            hgt_files = [f for f in zip_ref.namelist() if f.endswith(".hgt")]

            if not hgt_files:
                raise ValueError(f"No .hgt file found in {self.hgt_zip_path}")

            hgt_filename = hgt_files[0]

            # Read the binary data
            with zip_ref.open(hgt_filename) as hgt_file:
                hgt_data = hgt_file.read()

        # Determine resolution based on file size
        file_size = len(hgt_data)

        if file_size == 25934402:  # 3601 * 3601 * 2 bytes (SRTM-1, 1 arc-second)
            self.size = 3601
            self.resolution = 1 / 3600  # degrees
        elif file_size == 2884802:  # 1201 * 1201 * 2 bytes (SRTM-3, 3 arc-second)
            self.size = 1201
            self.resolution = 3 / 3600  # degrees
        else:
            raise ValueError(f"Unexpected file size: {file_size} bytes")

        # Parse binary data as big-endian 16-bit signed integers
        # Using numpy for efficiency
        self.data = np.frombuffer(hgt_data, dtype=">i2").reshape((self.size, self.size))

        # Replace void values (-32768) with NaN
        self.data = self.data.astype(np.float32)
        self.data[self.data == -32768] = np.nan

    def _setup_interpolator(self):
        """Set up RegularGridInterpolator for efficient elevation queries."""
        # Create coordinate arrays
        # Note: SRTM data is stored from north to south (top to bottom)
        lats = np.linspace(
            self.lat_corner + 1, self.lat_corner, self.size  # North edge  # South edge
        )
        lons = np.linspace(
            self.lon_corner, self.lon_corner + 1, self.size  # West edge  # East edge
        )

        self.interpolator = RegularGridInterpolator(
            (lats, lons),
            self.data,
            method="linear",
            bounds_error=False,
            fill_value=np.nan,
        )

        # Store coordinate arrays for reference
        self.lats = lats
        self.lons = lons

    def to_dataframe(self, dropna=True) -> pd.DataFrame:
        """
        Convert elevation data to a DataFrame with coordinates.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: latitude, longitude, elevation
        """
        # Create meshgrid of coordinates
        lon_grid, lat_grid = np.meshgrid(self.lons, self.lats)

        # Flatten arrays
        df = pd.DataFrame(
            {
                "latitude": lat_grid.ravel(),
                "longitude": lon_grid.ravel(),
                "elevation": self.data.ravel(),
            }
        )

        return df.dropna(subset=["elevation"]) if dropna else df

    def to_array(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return elevation data in square array form with coordinate arrays.

        Returns
        -------
        tuple of (elevation_array, latitudes, longitudes)
            elevation_array : np.ndarray of shape (size, size)
                2D array of elevation values in meters
            latitudes : np.ndarray of shape (size,)
                Latitude values for each row (north to south)
            longitudes : np.ndarray of shape (size,)
                Longitude values for each column (west to east)
        """
        return self.data.copy(), self.lats.copy(), self.lons.copy()

    def get_elevation(self, latitude: float, longitude: float) -> float:
        """
        Get interpolated elevation for a specific coordinate.

        Uses bilinear interpolation for accurate elevation values between grid points.

        Parameters
        ----------
        latitude : float
            Latitude in decimal degrees
        longitude : float
            Longitude in decimal degrees

        Returns
        -------
        float
            Interpolated elevation in meters, or np.nan if outside tile bounds
        """
        # Check if coordinates are within tile bounds
        if not (self.lat_corner <= latitude <= self.lat_corner + 1):
            return np.nan
        if not (self.lon_corner <= longitude <= self.lon_corner + 1):
            return np.nan

        # Use interpolator for bilinear interpolation
        elevation = self.interpolator([[latitude, longitude]])[0]

        return float(elevation)

    def get_elevation_batch(self, coordinates: np.ndarray) -> np.ndarray:
        """
        Get interpolated elevations for multiple coordinates (vectorized).

        Parameters
        ----------
        coordinates : np.ndarray of shape (n, 2)
            Array of (latitude, longitude) pairs

        Returns
        -------
        np.ndarray of shape (n,)
            Interpolated elevations in meters
        """
        return self.interpolator(coordinates)

    def get_tile_info(self) -> dict:
        """
        Get information about the SRTM tile.

        Returns
        -------
        dict
            Dictionary containing tile metadata
        """
        return {
            "filename": self.hgt_zip_path.name,
            "lat_corner": self.lat_corner,
            "lon_corner": self.lon_corner,
            "lat_range": (self.lat_corner, self.lat_corner + 1),
            "lon_range": (self.lon_corner, self.lon_corner + 1),
            "resolution_arcsec": 1 if self.size == 3601 else 3,
            "resolution_deg": self.resolution,
            "size": (self.size, self.size),
            "min_elevation": float(np.nanmin(self.data)),
            "max_elevation": float(np.nanmax(self.data)),
            "mean_elevation": float(np.nanmean(self.data)),
            "void_percentage": float(np.isnan(self.data).sum() / self.data.size * 100),
        }

    def __repr__(self):
        return (
            f"SRTMParser(tile={self.lat_corner:+03d}{self.lon_corner:+04d}, "
            f"resolution={self.resolution*3600:.0f}arcsec, "
            f"size={self.size}x{self.size}, "
            f"data_store={type(self.data_store).__name__})"
        )
