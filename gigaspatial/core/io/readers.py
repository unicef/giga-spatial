import pandas as pd
import geopandas as gpd
from pathlib import Path
import json
import io
import zipfile

from .data_store import DataStore


def read_json(data_store: DataStore, path, **kwargs):
    with data_store.open(path, "r") as f:
        return json.load(f, **kwargs)


def read_kmz(file_obj, **kwargs):
    """Helper function to read KMZ files and return a GeoDataFrame."""
    try:
        with zipfile.ZipFile(file_obj) as kmz:
            # Find the KML file in the archive (usually doc.kml)
            kml_filename = next(
                name for name in kmz.namelist() if name.endswith(".kml")
            )

            # Read the KML content
            kml_content = io.BytesIO(kmz.read(kml_filename))

            gdf = gpd.read_file(kml_content)

            # Validate the GeoDataFrame
            if gdf.empty:
                raise ValueError(
                    "The KML file is empty or does not contain valid geospatial data."
                )

        return gdf

    except zipfile.BadZipFile:
        raise ValueError("The provided file is not a valid KMZ file.")
    except StopIteration:
        raise ValueError("No KML file found in the KMZ archive.")
    except Exception as e:
        raise RuntimeError(f"An error occurred: {e}")


def read_dataset(data_store: DataStore, path, **kwargs):
    """
    Read data from various file formats stored in both local and cloud-based storage.

    Parameters:
    ----------
    data_store : DataStore
        Instance of DataStore for accessing data storage.
    path : str
        Path to the file in data storage.
    **kwargs : dict
        Additional arguments passed to the specific reader function.

    Returns:
    -------
    pandas.DataFrame or geopandas.GeoDataFrame
        The data read from the file.

    Raises:
    ------
    FileNotFoundError
        If the file doesn't exist in blob storage.
    ValueError
        If the file type is unsupported or if there's an error reading the file.
    """

    # Define supported file formats and their readers
    BINARY_FORMATS = {
        ".shp",
        ".zip",
        ".parquet",
        ".gpkg",
        ".xlsx",
        ".xls",
        ".kmz",
        ".csv.gz",
    }

    PANDAS_READERS = {
        ".csv": pd.read_csv,
        ".xlsx": lambda f, **kw: pd.read_excel(f, engine="openpyxl", **kw),
        ".xls": lambda f, **kw: pd.read_excel(f, engine="xlrd", **kw),
        ".json": pd.read_json,
        ".csv.gz": lambda f, **kw: pd.read_csv(f, compression="gzip", **kw),
    }

    GEO_READERS = {
        ".shp": gpd.read_file,
        ".zip": gpd.read_file,
        ".geojson": gpd.read_file,
        ".gpkg": gpd.read_file,
        ".parquet": gpd.read_parquet,
        ".kmz": read_kmz,
    }

    try:
        # Check if file exists
        if not data_store.file_exists(path):
            raise FileNotFoundError(f"File '{path}' not found in blob storage")

        # Get file suffix and ensure it's lowercase
        suffix = Path(path).suffix.lower()

        # Special handling for compressed files
        if suffix == ".zip":
            # For zip files, we need to use binary mode
            with data_store.open(path, "rb") as f:
                return gpd.read_file(f)

        # Determine if we need binary mode based on file type
        mode = "rb" if suffix in BINARY_FORMATS else "r"

        # Try reading with appropriate reader
        if suffix in PANDAS_READERS:
            try:
                with data_store.open(path, mode) as f:
                    return PANDAS_READERS[suffix](f, **kwargs)
            except Exception as e:
                raise ValueError(f"Error reading file with pandas: {str(e)}")

        if suffix in GEO_READERS:
            try:
                with data_store.open(path, "rb") as f:
                    return GEO_READERS[suffix](f, **kwargs)
            except Exception as e:
                # For parquet files, try pandas reader if geopandas fails
                if suffix == ".parquet":
                    try:
                        with data_store.open(path, "rb") as f:
                            return pd.read_parquet(f, **kwargs)
                    except Exception as e2:
                        raise ValueError(
                            f"Failed to read parquet with both geopandas ({str(e)}) "
                            f"and pandas ({str(e2)})"
                        )
                raise ValueError(f"Error reading file with geopandas: {str(e)}")

        # If we get here, the file type is unsupported
        supported_formats = sorted(set(PANDAS_READERS.keys()) | set(GEO_READERS.keys()))
        raise ValueError(
            f"Unsupported file type: {suffix}\n"
            f"Supported formats: {', '.join(supported_formats)}"
        )

    except Exception as e:
        if isinstance(e, (FileNotFoundError, ValueError)):
            raise
        raise RuntimeError(f"Unexpected error reading dataset: {str(e)}")


def read_datasets(data_store: DataStore, paths, **kwargs):
    """
    Read multiple datasets from data storage at once.

    Parameters:
    ----------
    data_store : DataStore
        Instance of DataStore for accessing data storage.
    paths : list of str
        Paths to files in data storage.
    **kwargs : dict
        Additional arguments passed to read_dataset.

    Returns:
    -------
    dict
        Dictionary mapping paths to their corresponding DataFrames/GeoDataFrames.
    """
    results = {}
    errors = {}

    for path in paths:
        try:
            results[path] = read_dataset(data_store, path, **kwargs)
        except Exception as e:
            errors[path] = str(e)

    if errors:
        error_msg = "\n".join(f"- {path}: {error}" for path, error in errors.items())
        raise ValueError(f"Errors reading datasets:\n{error_msg}")

    return results
