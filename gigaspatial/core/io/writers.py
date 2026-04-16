"""
Module for writing datasets (DataFrames, GeoDataFrames, JSON) to a DataStore.
Provides high-level utilities for serializing geospatial and tabular data.
"""
import pandas as pd
import geopandas as gpd
from pathlib import Path
import json
import io

from .data_store import DataStore


def write_json(data, data_store: DataStore, path, **kwargs):
    """
    Write data to a JSON file in the data store.

    Args:
        data: Object to serialize to JSON.
        data_store: DataStore instance to use for writing.
        path: Destination path in the data store.
        **kwargs: Additional arguments passed to json.dump.
    """
    with data_store.open(path, "w") as f:
        json.dump(data, f, **kwargs)


def write_dataset(data, data_store: DataStore, path, **kwargs):
    """
    Write DataFrame, GeoDataFrame, or a generic object to various file formats.

    Supported formats include .csv, .xlsx, .json, .parquet for DataFrames,
    and .geojson, .gpkg, .parquet for GeoDataFrames.

    Args:
        data: The data to write. Can be pd.DataFrame, gpd.GeoDataFrame, or a JSON-serializable object.
        data_store: Instance of DataStore for accessing data storage.
        path: Path where the file will be written.
        **kwargs: Additional arguments passed to the specific writer function (e.g., index=False).

    Raises:
        ValueError: If the file type is unsupported or if there's an error writing the file.
        TypeError: If input data type is incompatible with the file extension.
        RuntimeError: For unexpected errors during the write process.
    """

    # Define supported file formats and their writers
    BINARY_FORMATS = {".shp", ".zip", ".parquet", ".gpkg", ".xlsx", ".xls"}

    PANDAS_WRITERS = {
        ".csv": lambda df, buf, **kw: df.to_csv(buf, **kw),
        ".xlsx": lambda df, buf, **kw: df.to_excel(buf, engine="openpyxl", **kw),
        ".json": lambda df, buf, **kw: df.to_json(buf, **kw),
        ".parquet": lambda df, buf, **kw: df.to_parquet(buf, **kw),
    }

    GEO_WRITERS = {
        ".geojson": lambda gdf, buf, **kw: gdf.to_file(buf, driver="GeoJSON", **kw),
        ".gpkg": lambda gdf, buf, **kw: gdf.to_file(buf, driver="GPKG", **kw),
        ".parquet": lambda gdf, buf, **kw: gdf.to_parquet(buf, **kw),
    }

    try:
        # Get file suffix and ensure it's lowercase
        suffix = Path(path).suffix.lower()

        # 1. Handle generic JSON data
        is_dataframe_like = isinstance(data, (pd.DataFrame, gpd.GeoDataFrame))
        if not is_dataframe_like:
            if suffix == ".json":
                try:
                    # Pass generic data directly to the write_json function
                    write_json(data, data_store, path, **kwargs)
                    return  # Successfully wrote JSON, so exit
                except Exception as e:
                    raise ValueError(f"Error writing generic JSON data: {str(e)}")
            else:
                # Raise an error if it's not a DataFrame/GeoDataFrame and not a .json file
                raise TypeError(
                    "Input data must be a pandas DataFrame or GeoDataFrame, "
                    "or a generic object destined for a '.json' file."
                )

        # 2. Handle DataFrame/GeoDataFrame
        # Determine if we need binary mode based on file type
        mode = "wb" if suffix in BINARY_FORMATS else "w"

        # Handle different data types and formats
        if isinstance(data, gpd.GeoDataFrame):
            if suffix not in GEO_WRITERS:
                supported_formats = sorted(GEO_WRITERS.keys())
                raise ValueError(
                    f"Unsupported file type for GeoDataFrame: {suffix}\n"
                    f"Supported formats: {', '.join(supported_formats)}"
                )

            try:
                # Write to BytesIO buffer first
                buffer = io.BytesIO()
                GEO_WRITERS[suffix](data, buffer, **kwargs)
                buffer.seek(0)

                # Then write buffer contents to data_store
                with data_store.open(path, "wb") as f:
                    f.write(buffer.getvalue())

                # with data_store.open(path, "wb") as f:
                #    GEO_WRITERS[suffix](data, f, **kwargs)
            except Exception as e:
                raise ValueError(f"Error writing GeoDataFrame: {str(e)}")

        else:  # pandas DataFrame
            if suffix not in PANDAS_WRITERS:
                supported_formats = sorted(PANDAS_WRITERS.keys())
                raise ValueError(
                    f"Unsupported file type for DataFrame: {suffix}\n"
                    f"Supported formats: {', '.join(supported_formats)}"
                )

            try:
                with data_store.open(path, mode) as f:
                    PANDAS_WRITERS[suffix](data, f, **kwargs)
            except Exception as e:
                raise ValueError(f"Error writing DataFrame: {str(e)}")

    except Exception as e:
        if isinstance(e, (TypeError, ValueError)):
            raise
        raise RuntimeError(f"Unexpected error writing dataset: {str(e)}")


def write_datasets(data_dict, data_store: DataStore, **kwargs):
    """
    Write multiple datasets to data storage at once.

    Args:
        data_dict: Dictionary mapping paths (str) to data objects.
        data_store: DataStore instance.
        **kwargs: Additional arguments passed to write_dataset for each item.

    Raises:
        ValueError: If one or more datasets fail to write, containing details of all errors.
    """
    errors = {}

    for path, data in data_dict.items():
        try:
            write_dataset(data, data_store, path, **kwargs)
        except Exception as e:
            errors[path] = str(e)

    if errors:
        error_msg = "\n".join(f"- {path}: {error}" for path, error in errors.items())
        raise ValueError(f"Errors writing datasets:\n{error_msg}")
