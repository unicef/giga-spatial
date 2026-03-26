import gzip
import io
import json
import os
import tempfile
import warnings
import zipfile
from pathlib import Path
from typing import Union

import geopandas as gpd
import pandas as pd

from .data_store import DataStore
from gigaspatial.config import config

logger = config.get_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def read_json(data_store: DataStore, path: str) -> dict:
    """
    Read a JSON file from a DataStore.

    Parameters
    ----------
    data_store : DataStore
        Instance of DataStore for accessing data storage.
    path : str
        Path to the JSON file.

    Returns
    -------
    dict
        Parsed JSON content.
    """
    # FIX: removed **kwargs forwarding - json.load only accepts cls/object_hook,
    # passing arbitrary kwargs (e.g. compression=...) raises TypeError.
    with data_store.open(path, "r") as f:
        return json.load(f)


def read_kmz(file_obj, **kwargs) -> gpd.GeoDataFrame:
    """
    Read a KMZ file and return a GeoDataFrame.

    Parameters
    ----------
    file_obj : file-like object
        A file-like object pointing to a KMZ archive.
    **kwargs
        Additional keyword arguments passed to ``geopandas.read_file``.

    Returns
    -------
    geopandas.GeoDataFrame

    Raises
    ------
    ValueError
        If the file is not a valid KMZ or contains no KML data.
    RuntimeError
        For unexpected errors during reading.
    """
    try:
        with zipfile.ZipFile(file_obj) as kmz:
            kml_filename = next(
                (name for name in kmz.namelist() if name.endswith(".kml")), None
            )
            if kml_filename is None:
                raise ValueError("No KML file found in the KMZ archive.")

            kml_content = io.BytesIO(kmz.read(kml_filename))
            gdf = gpd.read_file(kml_content, **kwargs)

            if gdf.empty:
                raise ValueError(
                    "The KML file is empty or does not contain valid geospatial data."
                )
        return gdf

    except zipfile.BadZipFile:
        raise ValueError("The provided file is not a valid KMZ file.")
    except ValueError:
        raise
    except Exception as e:
        raise RuntimeError(f"An error occurred reading KMZ: {e}") from e


def read_gzipped_json_or_csv(
    file_path: str, data_store: DataStore
) -> Union[pd.DataFrame, None]:
    """
    Read a gzipped file, attempting JSON (lines=True) first, then CSV.

    Parameters
    ----------
    file_path : str
        Path to the .gz file in the DataStore.
    data_store : DataStore
        Instance of DataStore for accessing data storage.

    Returns
    -------
    pandas.DataFrame or None
        Parsed DataFrame, or None if parsing fails.
    """
    with data_store.open(file_path, "rb") as f:
        text = gzip.GzipFile(fileobj=f).read().decode("utf-8")

    try:
        return pd.read_json(io.StringIO(text), lines=True)
    except (json.JSONDecodeError, ValueError):
        pass

    try:
        return pd.read_csv(io.StringIO(text))
    except pd.errors.ParserError:
        logger.error("Could not parse '%s' as JSON lines or CSV.", file_path)
        return None


# ---------------------------------------------------------------------------
# Format registries
# ---------------------------------------------------------------------------

BINARY_FORMATS = {
    ".shp",
    ".zip",
    ".parquet",
    ".gpkg",
    ".xlsx",
    ".xls",
    ".kmz",
    ".gz",
    ".fgb",
}

PANDAS_READERS = {
    ".csv": pd.read_csv,
    ".xlsx": lambda f, **kw: pd.read_excel(f, engine="openpyxl", **kw),
    ".xls": lambda f, **kw: pd.read_excel(f, engine="xlrd", **kw),
}

GEO_READERS = {
    ".shp": gpd.read_file,
    ".zip": gpd.read_file,
    ".geojson": gpd.read_file,
    ".json": gpd.read_file,
    ".geojsonl": gpd.read_file,
    ".ndjson": gpd.read_file,
    ".gpkg": gpd.read_file,
    ".fgb": gpd.read_file,
    ".kml": gpd.read_file,
    ".parquet": gpd.read_parquet,
    ".kmz": read_kmz,
}

COMPRESSION_FORMATS = {
    ".gz": "gzip",
    ".bz2": "bz2",
    ".xz": "xz",
}

# Sidecar extensions required alongside .shp
SHAPEFILE_SIDECAR_EXTENSIONS = [
    ".shp",
    ".shx",
    ".dbf",
    ".prj",
    ".cpg",
    ".sbn",
    ".sbx",
    ".fbn",
    ".fbx",
    ".ain",
    ".aih",
    ".atx",
    ".qix",
]


# ---------------------------------------------------------------------------
# Storage name helper
# ---------------------------------------------------------------------------


def _storage_display_name(data_store: DataStore) -> str:
    """Return a human-readable storage backend name for error messages."""
    class_name = data_store.__class__.__name__.lower()
    if "adls" in class_name:
        return "Azure Blob Storage"
    if "snowflake" in class_name:
        return "Snowflake stage"
    if "local" in class_name:
        return "local storage"
    return "storage"


# ---------------------------------------------------------------------------
# Main reader
# ---------------------------------------------------------------------------


def read_dataset(
    data_store: DataStore,
    path: str,
    compression: str = None,
    validate_crs: bool = True,
    **kwargs,
) -> Union[pd.DataFrame, gpd.GeoDataFrame]:
    """
    Read data from various file formats stored in local or cloud-based storage.

    Supports plain and compressed formats. For compound extensions such as
    ``.csv.gz`` or ``.geojson.gz``, the inner format is detected automatically.
    Compression can also be specified explicitly via the ``compression``
    parameter.

    Parameters
    ----------
    data_store : DataStore
        Instance of DataStore for accessing data storage.
    path : str
        Path to the file in data storage.
    compression : str, optional
        Explicit compression type (``"gzip"``, ``"bz2"``, ``"xz"``).
        When ``None``, compression is inferred from the file extension.
    validate_crs : bool, default True
        If ``True``, emit a ``UserWarning`` when a GeoDataFrame has no CRS.
    **kwargs
        Additional arguments forwarded to the underlying reader function.

    Returns
    -------
    pandas.DataFrame or geopandas.GeoDataFrame
        The data read from the file.

    Raises
    ------
    FileNotFoundError
        If the file does not exist in the storage backend.
    ValueError
        If the file type is unsupported or the file cannot be parsed.
    RuntimeError
        For unexpected errors not covered by the above.
    """
    try:
        # ------------------------------------------------------------------ #
        # 1. Existence check
        # ------------------------------------------------------------------ #
        if not data_store.file_exists(path):
            storage_name = _storage_display_name(data_store)
            raise FileNotFoundError(f"File '{path}' not found in {storage_name}.")

        path_obj = Path(path)
        suffixes = path_obj.suffixes
        file_extension = suffixes[-1].lower() if suffixes else ""

        # ------------------------------------------------------------------ #
        # 2. Compressed file handling (.gz / .bz2 / .xz)
        # FIX: .zip is no longer in COMPRESSION_FORMATS; it is handled as a
        #      geo container below. This removes the double-routing bug.
        # ------------------------------------------------------------------ #
        inferred_compression = COMPRESSION_FORMATS.get(file_extension)
        active_compression = compression or inferred_compression

        if active_compression and file_extension in COMPRESSION_FORMATS:
            if len(suffixes) > 1:
                inner_ext = suffixes[-2].lower()

                if inner_ext == ".tar":
                    raise ValueError(
                        "Tar archives (.tar.gz) are not directly supported."
                    )

                # Compressed tabular file (e.g. .csv.gz, .json.gz)
                if inner_ext in PANDAS_READERS:
                    try:
                        with data_store.open(path, "rb") as f:
                            return PANDAS_READERS[inner_ext](
                                f, compression=active_compression, **kwargs
                            )
                    except Exception as e:
                        raise ValueError(
                            f"Error reading compressed tabular file '{path}': {e}"
                        ) from e

                # Compressed geo file (e.g. .geojson.gz, .fgb.gz)
                if inner_ext in GEO_READERS and active_compression == "gzip":
                    try:
                        with data_store.open(path, "rb") as f:
                            decompressed = gzip.decompress(f.read())
                        result = GEO_READERS[inner_ext](
                            io.BytesIO(decompressed), **kwargs
                        )
                        return _maybe_warn_crs(result, path, validate_crs)
                    except Exception as e:
                        raise ValueError(
                            f"Error reading compressed geo file '{path}': {e}"
                        ) from e

                if inner_ext in GEO_READERS and active_compression != "gzip":
                    raise ValueError(
                        f"Compression '{active_compression}' is not supported "
                        f"for geo formats. Only gzip is supported."
                    )

            else:
                # Bare .gz with no inner extension hint - try JSON lines then CSV
                # FIX: delegates to read_gzipped_json_or_csv instead of silently
                #      assuming CSV, which could return garbage for JSON payloads.
                result = read_gzipped_json_or_csv(path, data_store)
                if result is None:
                    raise ValueError(
                        f"Could not parse '{path}' as JSON lines or CSV. "
                        f"Use a compound extension (e.g. '.json.gz') to specify "
                        f"the inner format explicitly."
                    )
                return result

        # ------------------------------------------------------------------ #
        # 3. ZIP container - always geo (shapefile zip or zipped vector)
        # FIX: .zip is no longer ambiguously caught by COMPRESSION_FORMATS.
        # ------------------------------------------------------------------ #
        if file_extension == ".zip":
            try:
                with data_store.open(path, "rb") as f:
                    result = gpd.read_file(f, **kwargs)
                return _maybe_warn_crs(result, path, validate_crs)
            except Exception as e:
                raise ValueError(
                    f"Error reading ZIP file '{path}' as geospatial data: {e}"
                ) from e

        # ------------------------------------------------------------------ #
        # 4. Shapefile - requires sidecar files
        # FIX: inner `suffixes` variable renamed to `sidecar_extensions` to
        #      avoid shadowing the outer `suffixes = path_obj.suffixes`.
        # ------------------------------------------------------------------ #
        if file_extension == ".shp":
            base_path = Path(path)
            with tempfile.TemporaryDirectory() as tmp_dir:
                local_shp_path = None
                for ext in SHAPEFILE_SIDECAR_EXTENSIONS:
                    part_path = str(base_path.with_suffix(ext))
                    if data_store.file_exists(part_path):
                        local_part = os.path.join(
                            tmp_dir, base_path.with_suffix(ext).name
                        )
                        content = data_store.read_file(part_path)
                        with open(local_part, "wb") as lf:
                            lf.write(content)
                        if ext == ".shp":
                            local_shp_path = local_part

                if local_shp_path is None:
                    raise FileNotFoundError(
                        f"Main .shp file could not be retrieved for path: {path}"
                    )
                result = gpd.read_file(local_shp_path, **kwargs)
            return _maybe_warn_crs(result, path, validate_crs)

        # ------------------------------------------------------------------ #
        # 5. Tabular formats (CSV, Excel)
        # ------------------------------------------------------------------ #
        if file_extension in PANDAS_READERS:
            mode = "rb" if file_extension in BINARY_FORMATS else "r"
            try:
                with data_store.open(path, mode) as f:
                    return PANDAS_READERS[file_extension](f, **kwargs)
            except Exception as e:
                raise ValueError(f"Error reading '{path}' with pandas: {e}") from e

        # ------------------------------------------------------------------ #
        # 6. Geospatial formats
        # ------------------------------------------------------------------ #
        if file_extension in GEO_READERS:
            try:
                with data_store.open(path, "rb") as f:
                    result = GEO_READERS[file_extension](f, **kwargs)
                return _maybe_warn_crs(result, path, validate_crs)
            except Exception as e:
                # FIX: For parquet, fall back to pandas if geopandas fails
                # (non-spatial parquet is common in GigaSpatial pipelines).
                if file_extension == ".parquet":
                    try:
                        with data_store.open(path, "rb") as f:
                            return pd.read_parquet(f, **kwargs)
                    except Exception as e2:
                        raise ValueError(
                            f"Failed to read parquet '{path}' with geopandas "
                            f"({e}) and pandas ({e2})."
                        ) from e2
                raise ValueError(f"Error reading '{path}' with geopandas: {e}") from e

        # ------------------------------------------------------------------ #
        # 7. Unsupported format
        # FIX: added missing newline between formats and compressions in message
        # ------------------------------------------------------------------ #
        supported_formats = sorted(set(PANDAS_READERS) | set(GEO_READERS))
        supported_compressions = sorted(COMPRESSION_FORMATS)
        raise ValueError(
            f"Unsupported file type: '{file_extension}'\n"
            f"Supported formats:      {', '.join(supported_formats)}\n"
            f"Supported compressions: {', '.join(supported_compressions)}"
        )

    except (FileNotFoundError, ValueError):
        raise
    except Exception as e:
        raise RuntimeError(f"Unexpected error reading dataset '{path}': {e}") from e


# ---------------------------------------------------------------------------
# Internal utility
# ---------------------------------------------------------------------------


def _maybe_warn_crs(
    result: Union[pd.DataFrame, gpd.GeoDataFrame],
    path: str,
    validate_crs: bool,
) -> Union[pd.DataFrame, gpd.GeoDataFrame]:
    """Emit a warning if a GeoDataFrame has no CRS defined."""
    if validate_crs and isinstance(result, gpd.GeoDataFrame) and result.crs is None:
        warnings.warn(
            f"File '{path}' was read as a GeoDataFrame but has no CRS defined. "
            f"Set the CRS explicitly with `.set_crs(epsg=...)` if known.",
            UserWarning,
            stacklevel=3,
        )
    return result


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
